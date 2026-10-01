#!/usr/bin/env bash
# Verify live speculative decoding against the target distribution. Logprobs
# requests intentionally disable drafting in the engine, so they CANNOT be the
# speculative arm. mtp_gate.py generates that arm without probabilities and
# teacher-forces its first divergence into a separate target-only scoring call.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/testbed-lib.sh"
BIN="${IDLETOKEN_SPEC_GATE_BIN:-${IDLETOKEN_LLAMA_SERVER_BIN:-$ROOT/vendor/llama.cpp/build/bin/llama-server}}"
GGUF="${IDLETOKEN_SPEC_GATE_GGUF:-${IDLETOKEN_SMOKE_GGUF:-}}"
PORT="${IDLETOKEN_SPEC_GATE_PORT:-18993}"
CTX="${IDLETOKEN_SPEC_GATE_CTX:-8192}"
SPEC_TYPE="${IDLETOKEN_SPEC_TYPE:-ngram-mod}"
MODEL="${IDLETOKEN_SPEC_GATE_MODEL:-${IDLETOKEN_MTP_GATE_MODEL:-}}"
QUANT="${IDLETOKEN_SPEC_GATE_QUANT:-${IDLETOKEN_MTP_GATE_QUANT:-}}"
DRAFT="${IDLETOKEN_SPEC_GATE_DRAFT:-${IDLETOKEN_MTP_GATE_DRAFT:-}}"
MMPROJ="${IDLETOKEN_SPEC_GATE_MMPROJ:-${IDLETOKEN_MTP_GATE_MMPROJ:-}}"
DEVICE="${IDLETOKEN_SPEC_GATE_DEVICE:-${IDLETOKEN_MTP_GATE_DEVICE:-}}"
CAPABILITY_ARGS=()
case ",$SPEC_TYPE," in *,draft-mtp,*) CAPABILITY_ARGS=(--idletoken-mtp-safe) ;; esac
GATE="${IDLETOKEN_SPEC_GATE_NAME:-G_SPEC_FAITHFUL}"
TMP="$(mktemp -d)"
OUT="${IDLETOKEN_SPEC_GATE_OUTPUT:-$TMP/evidence}"
ENGINE_PID=""
cleanup() {
    if [ -n "$ENGINE_PID" ]; then
        kill "$ENGINE_PID" 2>/dev/null || true
        wait "$ENGINE_PID" 2>/dev/null || true
    fi
    if [ -f "$TMP/engine.log" ] && [ -d "$OUT" ]; then
        cp "$TMP/engine.log" "$OUT/engine.log"
    fi
    rm -rf "$TMP"
}
trap cleanup EXIT
skip() { printf '%s_SKIP: %s\n' "$GATE" "$*"; exit 0; }
fail() { printf '%s_FAIL: %s\n' "$GATE" "$*"; exit 1; }
command -v python3 >/dev/null || skip 'python3 missing'
[ -x "$BIN" ] || skip 'engine binary missing; set IDLETOKEN_SPEC_GATE_BIN'
[ -n "$GGUF" ] || skip 'set IDLETOKEN_SPEC_GATE_GGUF to a readable GGUF'
[ -r "$GGUF" ] || fail 'the explicitly selected GGUF is not readable'
mkdir -p "$OUT"
python3 "$ROOT/scripts/mtp_gate.py" --self-test > "$OUT/checker-selftest.log" 2>&1 || fail 'checker controls failed'
ASSET_ARGS=()
ENGINE_ARGS=()
if [ -n "$DEVICE" ]; then
    [[ "$DEVICE" =~ ^(CUDA|MTL)[0-9]+$ ]] || fail 'select one local CUDA or Metal device'
    ENGINE_ARGS+=(--device "$DEVICE")
fi
if [ -n "$DRAFT" ] || [ -n "$MMPROJ" ]; then
    [ -n "$MODEL" ] && [ -n "$QUANT" ] || fail 'explicit dependencies require catalog MODEL and QUANT'
    [ -n "$DEVICE" ] || fail 'explicit dependencies require one local DEVICE'
fi
if [ -n "$DRAFT" ]; then
    case ",$SPEC_TYPE," in *,draft-mtp,*) ;; *) fail 'independent draft requires draft-mtp' ;; esac
    ASSET_ARGS+=(--draft "$DRAFT")
    ENGINE_ARGS+=(--idletoken-mtp-external-safe --spec-draft-model "$DRAFT"
        --spec-draft-device "$DEVICE" --spec-draft-ngl all)
fi
if [ -n "$MMPROJ" ]; then
    ASSET_ARGS+=(--mmproj "$MMPROJ")
    ENGINE_ARGS+=(--mmproj "$MMPROJ")
fi
if [ -n "$MODEL" ]; then
    python3 "$ROOT/scripts/mtp_gate.py" --verify-assets --catalog-model "$MODEL" \
        --quant "$QUANT" --target "$GGUF" --output "$OUT" \
        "${ASSET_ARGS[@]+"${ASSET_ARGS[@]}"}" || fail 'catalog compatibility or full asset hash verification failed'
fi
read -r _url PIN_SHA _tag _rest < "$ROOT/scripts/llamacpp-patches/UPSTREAM"
VERSION="$("$BIN" --version 2>&1)" || fail 'engine does not run'
case "$VERSION" in *"${PIN_SHA:0:7}"*) ;; *) fail 'engine does not match pin' ;; esac
python3 - "$PORT" <<'PY' || fail 'test port is occupied'
import socket,sys
with socket.socket() as s:
    s.bind(('127.0.0.1',int(sys.argv[1])))
PY
MTMD_BACKEND_DEVICE="$DEVICE" "$BIN" --idletoken-managed -m "$GGUF" --host 127.0.0.1 --port "$PORT" -ngl 99 --fit off \
    -c "$CTX" -np 1 --reasoning auto --spec-type "$SPEC_TYPE" \
    "${CAPABILITY_ARGS[@]+"${CAPABILITY_ARGS[@]}"}" \
    "${ENGINE_ARGS[@]+"${ENGINE_ARGS[@]}"}" \
    --spec-draft-n-max 3 --spec-ngram-mod-n-match 24 -ctkd f16 -ctvd f16 > "$TMP/engine.log" 2>&1 &
ENGINE_PID=$!
ready=0
for ((i=0; i<900; i++)); do
    kill -0 "$ENGINE_PID" 2>/dev/null || { tail -15 "$TMP/engine.log"; fail 'engine exited'; }
    body="$(curl -s --noproxy '*' -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null || true)"
    if [ "$body" = '{"status":"ok"}' ]; then ready=1; break; fi
    sleep 1
done
[ "$ready" = 1 ] || fail 'engine never ready'
CHECK_ARGS=()
[ "$GATE" = G_MTP ] || CHECK_ARGS=(--quality-only)
if [ "$GATE" = G_MTP ] && [ -n "$MMPROJ" ]; then CHECK_ARGS+=(--vision); fi
python3 "$ROOT/scripts/mtp_gate.py" --base "http://127.0.0.1:$PORT" \
    "${CHECK_ARGS[@]+"${CHECK_ARGS[@]}"}" --engine-log "$TMP/engine.log" --output "$OUT" || fail 'live drafting / target-distribution check failed'
