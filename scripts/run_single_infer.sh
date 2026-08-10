#!/usr/bin/env bash
# E4 oracle — single-node cluster inference correctness.
#
# Runs ON the coordinator node (DGX), from anywhere (cd's to repo root).
# Proves the acceptance-criteria E4 gate:
#
#   1. Produce a greedy-decode baseline with the OFFICIAL single-machine ds4
#      (--temp 0 --dump-logprobs → per-step selected token ids). Cached in
#      /tmp keyed by (prompt, n_tokens, gguf name) so re-runs are cheap.
#   2. Bring up a 1-node IdleToken cluster (coord + 1 worker, real GGUF load,
#      --n-predict 0 so no warmup step pollutes the KV cache / positions).
#   3. Drive one real chat request through the Anthropic endpoint.
#   4. Compare the cluster's greedy token-id sequence (coord log line
#      "generated_ids:") against the baseline. Exact match required.
#
# Prompt-template alignment: coord uses ds4_encode_chat_prompt(NULL system,
# DS4_THINK_NONE); the baseline uses `-sys "" --nothink` which builds the
# exact same token sequence (see vendor/ds4 encode_chat_prompt).
#
# Last line on success: SINGLE_INFER_OK   (acceptance.sh G4 contract)
set -u
cd "$(dirname "$0")/.." || exit 1

GGUF="${IDLETOKEN_GGUF:-$HOME/work/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf}"
DS4_BIN="${IDLETOKEN_DS4_BIN:-$HOME/work/ds4/ds4}"
PROMPT="${IDLETOKEN_E4_PROMPT:-List the first five prime numbers.}"
NTOK="${IDLETOKEN_E4_NTOK:-16}"
CTX="${IDLETOKEN_E4_CTX:-2048}"
NWORKERS="${IDLETOKEN_E4_WORKERS:-1}"   # >1 runs the same oracle against a PP cluster
COORD_BIND="127.0.0.1:14300"
API_BIND="127.0.0.1:14400"
LOG=/tmp/idletoken-e4
mkdir -p "$LOG"

fail() { echo "E4: $*" >&2; echo "SINGLE_INFER_FAIL"; exit 1; }

[ -f "$GGUF" ]        || fail "GGUF not found: $GGUF"
[ -x "$DS4_BIN" ]     || fail "official ds4 binary not found: $DS4_BIN"
[ -x ./idletoken-coord ] || fail "idletoken-coord missing (run: make all)"
[ -x ./idletoken-worker ] || fail "idletoken-worker missing (run: make all)"

# ---------------------------------------------------------------- baseline --
KEY=$(printf '%s|%s|%s|%s' "$PROMPT" "$NTOK" "$CTX" "$(basename "$GGUF")" \
      | sha256sum | cut -c1-16)
BASE="/tmp/idletoken-e4-baseline-$KEY.json"
if [ ! -s "$BASE" ]; then
    echo "E4: producing greedy baseline with official ds4 (loads 80G model)..."
    "$DS4_BIN" -m "$GGUF" --cuda -c "$CTX" -sys "" --nothink --temp 0 \
        -n "$NTOK" -p "$PROMPT" --dump-logprobs "$BASE.tmp" \
        > "$LOG/baseline.out" 2>&1 \
        || fail "baseline ds4 run failed (see $LOG/baseline.out)"
    mv "$BASE.tmp" "$BASE"
fi
BASE_IDS=$(python3 -c '
import json,sys
d = json.load(open(sys.argv[1]))
print(" ".join(str(s["selected"]["id"]) for s in d["steps"]))' "$BASE") \
    || fail "cannot parse baseline json $BASE"
[ -n "$BASE_IDS" ] || fail "baseline produced no decode steps"
echo "E4: baseline ids: $BASE_IDS"

# ------------------------------------------------------------- cluster up --
CPID=""; WPIDS=""
cleanup() {
    # SIGTERM first, then ESCALATE to -9: the engines install signal handlers
    # ("Ctrl-C to stop") and a coord blocked in accept() can outlive a single
    # TERM. A leaked 80GB cluster then poisons the NEXT model load (seen as a
    # flaky G4 on the acceptance ladder).
    for p in $WPIDS $CPID; do kill "$p" 2>/dev/null; done
    sleep 1
    for p in $WPIDS $CPID; do kill -9 "$p" 2>/dev/null; done
    wait 2>/dev/null
}
trap cleanup EXIT

./idletoken-coord --bind "$COORD_BIND" --api-bind "$API_BIND" --http \
    --num-workers "$NWORKERS" --n-predict 0 --ctx-size "$CTX" \
    --model-path "$GGUF" > "$LOG/coord.log" 2>&1 &
CPID=$!
sleep 1
for wi in $(seq 1 "$NWORKERS"); do
    ./idletoken-worker --coordinator "$COORD_BIND" --bind "127.0.0.1:$((14300 + wi))" \
        --gguf-dir "$(dirname "$GGUF")" \
        > "$LOG/worker$wi.log" 2>&1 &
    WPIDS="$WPIDS $!"
done

echo "E4: waiting for cluster (model load ~20-60s)..."
ready=0
for i in $(seq 1 180); do
    if curl -s -m 2 "http://$API_BIND/health" 2>/dev/null | grep -q '"status":"ok"'; then
        ready=1; break
    fi
    kill -0 "$CPID" 2>/dev/null || fail "coord exited early (see $LOG/coord.log)"
    for p in $WPIDS; do
        kill -0 "$p" 2>/dev/null || fail "a worker exited early (see $LOG/worker*.log)"
    done
    sleep 1
done
[ "$ready" = 1 ] || fail "cluster not ready within 180s (see $LOG/*.log)"

# Refuse to certify a mock: every worker must have loaded the real engine.
grep -q "model load MOCK" "$LOG"/worker*.log && \
    fail "a worker fell back to MOCK load — not a real inference (see $LOG/worker*.log)"

# ----------------------------------------------------------------- request --
ESC_PROMPT=$(printf '%s' "$PROMPT" | sed 's/\\/\\\\/g; s/"/\\"/g')
RESP=$(curl -s -m 300 -X POST "http://$API_BIND/v1/messages" \
    -H 'content-type: application/json' \
    -d "{\"max_tokens\":$NTOK,\"messages\":[{\"role\":\"user\",\"content\":\"$ESC_PROMPT\"}]}")
echo "$RESP" | grep -q '"type":"message"' || fail "API response invalid: $RESP"
echo "E4: api response: $RESP"

CLUSTER_IDS=$(grep -o 'generated_ids:.*' "$LOG/coord.log" | tail -1 \
              | sed 's/generated_ids: *//')
[ -n "$CLUSTER_IDS" ] || fail "no generated_ids trace in coord log"
echo "E4: cluster  ids: $CLUSTER_IDS"

# ----------------------------------------------------------------- compare --
if [ "$BASE_IDS" = "$CLUSTER_IDS" ]; then
    echo "E4: greedy token sequence matches official ds4 exactly (${NTOK} max tok)"
    echo "SINGLE_INFER_OK"
    exit 0
fi
echo "E4: MISMATCH" >&2
echo "  baseline: $BASE_IDS" >&2
echo "  cluster : $CLUSTER_IDS" >&2
fail "cluster tokens diverge from official ds4 greedy baseline"
