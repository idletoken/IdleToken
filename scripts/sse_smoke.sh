#!/usr/bin/env bash
# SSE smoke — engine-side true streaming (integration-plan 3.4).
#
# Starts a 1-worker mock cluster (absent GGUF + IDLETOKEN_MOCK_OK, same trick as
# platform_e2e_real_coord.sh), then asserts that `"stream":true` on the
# coord's OpenAI and Anthropic endpoints yields real SSE:
#
#   * `Content-Type: text/event-stream` head
#   * several `data:` frames (OpenAI: role preamble + word/token deltas +
#     final usage frame; Anthropic: the full message_start → ... →
#     message_stop event sequence)
#   * the protocol terminator (`data: [DONE]` / `event: message_stop`)
#
# A non-stream request is also checked to prove the default path regressed
# nowhere. Real model: set IDLETOKEN_GGUF=/path/to/ds4flash.gguf to rerun the
# same assertions against real per-token streaming (mock-marker checks are
# skipped, delta thresholds relaxed to >=1).
#
# Last-line contract (for acceptance wiring):
#   SSE_SMOKE_OK             all assertions green            (exit 0)
#   SSE_SMOKE_SKIP: reason   dependency missing              (exit 0)
#   SSE_SMOKE_FAIL           an assertion failed             (exit 1)
set -u
cd "$(dirname "$0")/.." || exit 1

# Same CUDA lib convention as pair_selftest.sh (harmless where absent).
export PATH="/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

skip() { echo "SSE_SMOKE_SKIP: $*"; exit 0; }
die()  { echo "sse_smoke: $*" >&2; echo "SSE_SMOKE_FAIL"; exit 1; }

[ -x ./idletoken-coord ]  || skip "idletoken-coord not built here (make all on an engine node)"
[ -x ./idletoken-worker ] || skip "idletoken-worker not built here (make all on an engine node)"
command -v curl >/dev/null 2>&1 || skip "curl not on PATH"

API_PORT="${IDLETOKEN_SSE_API_PORT:-18200}"
COORD_BIND="127.0.0.1:14360"
WORKER_BIND="127.0.0.1:14361"
GGUF="${IDLETOKEN_GGUF:-/tmp/idletoken-sse-nomodel-$$.gguf}"
REAL_MODEL=0; [ -f "$GGUF" ] && REAL_MODEL=1
LOGDIR=$(mktemp -d)

PIDS=()
cleanup() {
    for p in ${PIDS[@]+"${PIDS[@]}"}; do kill "$p" 2>/dev/null; done
    sleep 1
    for p in ${PIDS[@]+"${PIDS[@]}"}; do kill -9 "$p" 2>/dev/null; done
    echo "sse_smoke: logs kept in $LOGDIR (coord/worker + captured SSE)" >&2
}
trap cleanup EXIT

# Refuse to assert against someone else's live cluster.
if curl -s -m 2 -o /dev/null "http://127.0.0.1:${API_PORT}/health" 2>/dev/null; then
    die "port ${API_PORT} already serving — stop it or set IDLETOKEN_SSE_API_PORT"
fi

echo "== [1/4] coord + 1 worker (model: $([ "$REAL_MODEL" = 1 ] && echo "REAL $GGUF" || echo 'MOCK — absent GGUF'), --http on :${API_PORT}) =="
# WARNING: two switches at two layers, similarly named but **not the same thing**,
# and a mock cluster needs both:
#   IDLETOKEN_MOCK_OK    -> read by the coordinator: with no engine present, return
#                        an explicitly labelled mock completion
#   IDLETOKEN_ALLOW_MOCK -> read by the worker: permit a mock load when the weights
#                        are missing
# The worker one came later (the generality work deliberately removed the
# automatic mock fallback). This script used to set only the first, so the
# coordinator was satisfied, the worker refused to serve, and the cluster never
# came up -- G_PLAT stayed red for a while with a failure message that stopped at
# "coord API never became healthy", two layers away from the cause.
[ "$REAL_MODEL" = 1 ] || { export IDLETOKEN_MOCK_OK=1; export IDLETOKEN_ALLOW_MOCK=1; }
./idletoken-coord --bind "$COORD_BIND" --num-workers 1 --n-predict 0 \
    --model-path "$GGUF" --api-bind "127.0.0.1:${API_PORT}" --http \
    > "$LOGDIR/coord.log" 2>&1 &
PIDS+=($!)
sleep 2
./idletoken-worker --coordinator "$COORD_BIND" --bind "$WORKER_BIND" \
    --gguf-dir "$(dirname "$GGUF")" > "$LOGDIR/worker.log" 2>&1 &
PIDS+=($!)

up=""
WAIT=$([ "$REAL_MODEL" = 1 ] && echo 600 || echo 60)
for _ in $(seq 1 "$WAIT"); do
    curl -s -m 2 "http://127.0.0.1:${API_PORT}/health" 2>/dev/null | grep -q '"status":"ok"' && { up=1; break; }
    sleep 1
done
[ -n "$up" ] || { tail -30 "$LOGDIR/coord.log" >&2; die "coord API never became healthy"; }
echo "   coord /health ok"

TIMEOUT=$([ "$REAL_MODEL" = 1 ] && echo 600 || echo 30)
MIN_DELTAS=$([ "$REAL_MODEL" = 1 ] && echo 1 || echo 2)

echo "== [2/4] OpenAI /v1/chat/completions stream:true =="
OAI="$LOGDIR/openai.sse"; OAIH="$LOGDIR/openai.head"
curl -sN -m "$TIMEOUT" -D "$OAIH" -o "$OAI" \
    "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-v4-flash","stream":true,"max_tokens":16,"messages":[{"role":"user","content":"stream me some words please"}]}' \
    || die "curl to /v1/chat/completions (stream) failed"
grep -qi '^content-type: *text/event-stream' "$OAIH" \
    || die "OpenAI stream: content-type is not text/event-stream ($(tr -d '\r' < "$OAIH" | head -3 | tr '\n' ' '))"
ND=$(grep -c '^data: ' "$OAI" || true)
[ "$ND" -ge $((MIN_DELTAS + 2)) ] || { cat "$OAI" >&2; die "OpenAI stream: want >=$((MIN_DELTAS + 2)) data: frames, got $ND"; }
grep -q '"object":"chat.completion.chunk"' "$OAI" || die "OpenAI stream: no chat.completion.chunk frames"
grep -q '"delta":{"role":"assistant"}'     "$OAI" || die "OpenAI stream: missing role preamble frame"
grep -q '"delta":{"content":"'             "$OAI" || die "OpenAI stream: no content delta frames"
grep -q '"finish_reason":"'                "$OAI" || die "OpenAI stream: no finish_reason in the final frame"
grep -q '"usage":'                         "$OAI" || die "OpenAI stream: no usage in the final frame"
grep -q '^data: \[DONE\]'                  "$OAI" || die "OpenAI stream: missing data: [DONE]"
if [ "$REAL_MODEL" != 1 ]; then
    # Marker is word-chunked ACROSS frames — reassemble deltas before asserting.
    FULL=$(sed -n 's/.*"delta":{"content":"\(.*\)"}.*/\1/p' "$OAI" | tr -d '\n')
    printf '%s' "$FULL" | grep -q 'IDLETOKEN MOCK ENGINE' \
        || die "OpenAI stream: coord mock marker absent in reassembled deltas (got: ${FULL:0:80}) — origin unproven"
fi
echo "   $ND data frames + [DONE] ok"

echo "== [3/4] Anthropic /v1/messages stream:true =="
ANT="$LOGDIR/anthropic.sse"; ANTH="$LOGDIR/anthropic.head"
curl -sN -m "$TIMEOUT" -D "$ANTH" -o "$ANT" \
    "http://127.0.0.1:${API_PORT}/v1/messages" \
    -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-v4-flash","stream":true,"max_tokens":16,"messages":[{"role":"user","content":"stream me some words please"}]}' \
    || die "curl to /v1/messages (stream) failed"
grep -qi '^content-type: *text/event-stream' "$ANTH" \
    || die "Anthropic stream: content-type is not text/event-stream"
for ev in message_start content_block_start content_block_delta content_block_stop message_delta message_stop; do
    grep -q "^event: $ev" "$ANT" || { cat "$ANT" >&2; die "Anthropic stream: missing event: $ev"; }
done
NDD=$(grep -c '^event: content_block_delta' "$ANT" || true)
[ "$NDD" -ge "$MIN_DELTAS" ] || die "Anthropic stream: want >=$MIN_DELTAS content_block_delta events, got $NDD"
grep -q '"text_delta"'    "$ANT" || die "Anthropic stream: deltas are not text_delta"
grep -q '"output_tokens"' "$ANT" || die "Anthropic stream: message_delta carries no usage"
if [ "$REAL_MODEL" != 1 ]; then
    FULL=$(sed -n 's/.*"text_delta","text":"\(.*\)"}}.*/\1/p' "$ANT" | tr -d '\n')
    printf '%s' "$FULL" | grep -q 'IDLETOKEN MOCK ENGINE' \
        || die "Anthropic stream: coord mock marker absent in reassembled deltas (got: ${FULL:0:80}) — origin unproven"
fi
echo "   full event sequence ($NDD deltas) + message_stop ok"

echo "== [4/4] non-stream regression =="
NS=$(curl -s -m "$TIMEOUT" "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-v4-flash","max_tokens":16,"messages":[{"role":"user","content":"plain request"}]}')
printf '%s' "$NS" | grep -q '"object":"chat.completion"' || die "non-stream: not a chat.completion (got: ${NS:0:120})"
printf '%s' "$NS" | grep -q '"finish_reason":"'          || die "non-stream: no finish_reason"
if [ "$REAL_MODEL" != 1 ]; then
    printf '%s' "$NS" | grep -q 'IDLETOKEN MOCK ENGINE' || die "non-stream: coord mock marker absent"
fi
echo "   non-stream JSON reply intact"

echo ""
echo "SSE_SMOKE_OK"
exit 0
