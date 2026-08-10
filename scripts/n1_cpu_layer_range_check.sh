#!/usr/bin/env bash
# N1 oracle — CPU layer-range split == CPU unsplit, through the REAL PP path.
#
# docs/cpu-hybrid-design.md gate N1. The strong form: instead of comparing two
# code paths inside one process, run a genuine 2-stage cluster with the CPU
# backend and require it to reproduce the token ids that OFFICIAL single-machine
# ds4 produces on CPU. That exercises the whole new surface at once —
# encode/prefill_layer_range on [lo,hi), the HC hand-off across the PP boundary
# (read on one worker, write on the next), and logits_read on the last stage.
#
# Baseline: official ds4 CPU CLI, cached in /tmp keyed by prompt+tokens, same
# recipe run_single_infer.sh uses for E4 (-sys "" --nothink --temp 0), which
# builds the same token sequence coord's ds4_encode_chat_prompt does.
#
# Runs ON the DGX. Last line: N1_OK or N1_FAIL.
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD

GGUF="${IDLETOKEN_GGUF:-$HOME/work/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf}"
DS4_BIN="${IDLETOKEN_DS4_CPU_BIN:-/tmp/n0-cpu-refactor/tree/ds4}"   # N0 leaves a CPU CLI here
PROMPT="${IDLETOKEN_N1_PROMPT:-What is the capital of France?}"
NTOK="${IDLETOKEN_N1_TOKENS:-8}"
CTX="${IDLETOKEN_N1_CTX:-2048}"
WORKERS="${IDLETOKEN_N1_WORKERS:-2}"
LOG=/tmp/n1-cpu-range; mkdir -p "$LOG"
BASE="/tmp/n1-baseline-$(echo "$PROMPT|$NTOK|$(basename "$GGUF")" | cksum | cut -d' ' -f1).json"

fail() { echo "N1: $*" >&2; ( cd "$ROOT" && ./scripts/run_cluster.sh --stop >/dev/null 2>&1 ); echo N1_FAIL; exit 1; }

[ -f "$GGUF" ] || fail "no GGUF at $GGUF"

# --- 1. baseline: official ds4, CPU backend, greedy ------------------------
if [ ! -s "$BASE" ]; then
    [ -x "$DS4_BIN" ] || fail "no CPU ds4 CLI at $DS4_BIN (run scripts/n0_cpu_refactor_check.sh first, or set IDLETOKEN_DS4_CPU_BIN)"
    echo "N1: producing CPU greedy baseline with official ds4 (80G model on CPU — slow)" >&2
    "$DS4_BIN" -m "$GGUF" --backend cpu -c "$CTX" -sys "" --nothink --temp 0 \
        -n "$NTOK" -p "$PROMPT" --dump-logprobs "$BASE.tmp" > "$LOG/baseline.out" 2>&1 \
        || { tail -20 "$LOG/baseline.out" >&2; fail "baseline run failed"; }
    mv "$BASE.tmp" "$BASE"
fi
BASE_IDS=$(python3 -c '
import json,sys
print(" ".join(str(s["selected"]["id"]) for s in json.load(open(sys.argv[1]))["steps"]))' "$BASE") \
    || fail "could not parse baseline $BASE"
[ -n "$BASE_IDS" ] || fail "baseline produced no tokens"
echo "N1: baseline ids: $BASE_IDS"

# --- 2. 2-stage CPU cluster ------------------------------------------------
./scripts/run_cluster.sh --stop >/dev/null 2>&1
echo "N1: bringing up a $WORKERS-stage cluster with IDLETOKEN_DS4_CPU=1" >&2
IDLETOKEN_DS4_CPU=1 IDLETOKEN_E5_CTX="$CTX" \
    ./scripts/run_cluster.sh --serve --workers "$WORKERS" > "$LOG/cluster.out" 2>&1 \
    || { tail -30 "$LOG/cluster.out" >&2; fail "cluster bring-up failed"; }

# Every stage must actually be on the CPU backend — otherwise this gate would
# quietly re-test the GPU path and pass for the wrong reason.
CPU_STAGES=$(grep -c "IDLETOKEN_DS4_CPU=1" /tmp/idletoken-cluster/worker*.log 2>/dev/null | \
             awk -F: '{s+=$2} END{print s+0}')
[ "$CPU_STAGES" -ge "$WORKERS" ] \
    || fail "only $CPU_STAGES/$WORKERS workers reported the CPU backend — the gate would be testing the wrong path"

# --- 3. one greedy request through the API --------------------------------
API=$(grep -o 'API_READY .*' "$LOG/cluster.out" | tail -1 | awk '{print $2}')
API=${API:-127.0.0.1:8000}
curl -s -m 3600 "http://$API/v1/messages" -H 'content-type: application/json' \
     -d "{\"model\":\"deepseek-v4-flash\",\"max_tokens\":$NTOK,\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}]}" \
     > "$LOG/reply.json" 2>&1 || fail "API request failed"

CL_IDS=$(grep -o 'generated_ids:.*' /tmp/idletoken-cluster/coord.log 2>/dev/null | tail -1 | sed 's/generated_ids: *//' | xargs)
[ -n "$CL_IDS" ] || { tail -20 /tmp/idletoken-cluster/coord.log >&2; fail "no generated_ids trace in coord log"; }
echo "N1: cluster  ids: $CL_IDS"

# The cluster may stop early on EOS; compare only the overlapping prefix, but
# require at least 4 tokens of it — a 1-token "match" proves nothing.
NB=$(echo "$BASE_IDS" | wc -w); NC=$(echo "$CL_IDS" | wc -w)
N=$(( NB < NC ? NB : NC ))
[ "$N" -ge 4 ] || fail "only $N tokens to compare (baseline $NB, cluster $NC)"
B_CUT=$(echo "$BASE_IDS" | cut -d' ' -f1-"$N")
C_CUT=$(echo "$CL_IDS"   | cut -d' ' -f1-"$N")

./scripts/run_cluster.sh --stop >/dev/null 2>&1

if [ "$B_CUT" = "$C_CUT" ]; then
    echo "N1: $N tokens identical across a real $WORKERS-stage CPU pipeline"
    echo N1_OK
else
    python3 - "$B_CUT" "$C_CUT" <<'PY'
import sys
a=sys.argv[1].split(); b=sys.argv[2].split()
for i,(x,y) in enumerate(zip(a,b)):
    if x!=y:
        print(f"N1: first divergence at step {i}: baseline={x} cluster={y}", file=sys.stderr); break
PY
    echo N1_FAIL; exit 1
fi
