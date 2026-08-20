#!/usr/bin/env bash
# G-PRIV-7: raw embeddings never leave the coordinator (v2 rebuild plan §5.1 /
# WS-F2). The layer-0 privacy invariant, made checkable at the packet level.
#
# The design that matters is the POSITIVE CONTROL, run FIRST: before the good
# configuration is allowed to count as green, a deliberately BAD one must make
# the checker recover embedding rows from the wire. A checker that has never
# been shown to go red proves nothing -- the pivot doc calls this out twice as
# the mistake that made an earlier "nothing found" meaningless.
#
#   BAD  config: all layers remote (--device RPC0, so layer 0 + its embedding
#                input live on the worker) + GGML_RPC_ALLOW_PLAINTEXT=1 (no
#                TLS). Expectation: the attacker recovers EVERY prompt token's
#                embedding row from the captured stream, and ZERO decoy rows.
#   GOOD config: coordinator-first placement (--device <local>,RPC0 with a
#                split that keeps layer 0 local) + TLS (GGML_RPC_PSK). The
#                embedding never crosses the wire, and what does cross is
#                ciphertext. Expectation: ZERO prompt tokens recovered.
#
# The gate FAILS if the bad config recovers nothing (the checker is blind) OR
# the good config recovers anything (a real leak) OR a decoy is recovered in
# the bad run (the match is coincidence, not recovery).
#
# Topology: two processes on this machine over loopback, with a userspace tap
# (scripts/rpc_tap.py) in front of the rpc-server so the exact wire bytes are
# recorded (loopback packet capture needs BPF privileges this session lacks).
# The task allows loopback two-process on the Mac; we drive idletoken-server +
# idletoken-rpc-server DIRECTLY (not through the coordinator) precisely because the
# bad case must BYPASS the coordinator's layer-0 pin -- the coordinator refuses
# to produce the leaking configuration, which is the property under test.
#
# Last line contract: G_PRIV7_OK <summary> / G_PRIV7_FAIL: <reason> /
# G_PRIV7_SKIP: <reason>.
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENGINE_DIR="${IDLETOKEN_ENGINE_DIR:-$REPO/vendor/llama.cpp/build/bin}"
GGUF="${IDLETOKEN_SMOKE_GGUF:-}"
GGUF_PY="${IDLETOKEN_GGUF_PY:-$REPO/vendor/llama.cpp/gguf-py}"
BASE="${IDLETOKEN_PRIV7_PORT:-51880}"
RPC_PORT=$BASE                 # idletoken-rpc-server
TAP_PORT=$((BASE + 1))         # tap listen (idletoken-server dials this)
API_PORT=$((BASE + 2))         # idletoken-server HTTP
PROMPT="The quick brown fox jumps over the lazy dog near the river."
PREFIX_FLOATS="${IDLETOKEN_PRIV7_PREFIX:-8}"

say()  { printf '%s\n' "$*"; }
skip() { say "G_PRIV7_SKIP: $*"; exit 0; }
die()  { say "G_PRIV7_FAIL: $*"; cleanup; exit 1; }

# --- dependency checks (SKIP, not FAIL: a missing engine/model is a
#     provisioning gap, not a privacy failure) --------------------------------
command -v python3 >/dev/null 2>&1 || skip "python3 not available"
[ -x "$ENGINE_DIR/llama-server" ]    || skip "no idletoken-server in $ENGINE_DIR (scripts/build_llamacpp.sh)"
[ -x "$ENGINE_DIR/ggml-rpc-server" ] || skip "no idletoken-rpc-server in $ENGINE_DIR (build with -DGGML_RPC=ON)"
[ -n "$GGUF" ] || skip "set IDLETOKEN_SMOKE_GGUF to a small local GGUF (the public model the attack dequantizes)"
[ -r "$GGUF" ] || skip "IDLETOKEN_SMOKE_GGUF is set but unreadable: $GGUF"
[ -d "$GGUF_PY" ] || skip "no gguf-py at $GGUF_PY (needed to dequantize token_embd) -- set IDLETOKEN_GGUF_PY"
python3 -c "import numpy" 2>/dev/null || skip "python3 numpy not available (needed for the recovery attack)"
python3 -c "import sys; sys.path.insert(0,'$GGUF_PY'); from gguf import GGUFReader, dequantize" 2>/dev/null \
    || skip "gguf-py at $GGUF_PY has no GGUFReader/dequantize"

# Local compute device for the good-config split (embedding + layer 0 stay here).
case "$(uname -s)" in
    Darwin) LOCAL_DEV="MTL0"; RPC_DEV="MTL0" ;;
    *)      LOCAL_DEV="CUDA0"; RPC_DEV="CUDA0" ;;
esac
LOCAL_DEV="${IDLETOKEN_PRIV7_LOCAL_DEV:-$LOCAL_DEV}"
RPC_DEV="${IDLETOKEN_PRIV7_RPC_DEV:-$RPC_DEV}"

TMP="$(mktemp -d)"
RPC_PID=""; TAP_PID=""; LL_PID=""

cleanup() {
    for p in $LL_PID $TAP_PID $RPC_PID; do kill "$p" 2>/dev/null; done
    sleep 1
    pkill -f "ggml-rpc-serve[r].* -p $RPC_PORT" 2>/dev/null
    pkill -f "llama-serve[r].*--port $API_PORT" 2>/dev/null
    pkill -f "rpc_tap.py --listen $TAP_PORT" 2>/dev/null
    rm -rf "$TMP"
}
trap cleanup EXIT

start_stack() {  # start_stack <capture-file> <plaintext|tls> <device-args...>
    local cap="$1" mode="$2"; shift 2
    local psk_env=()
    if [ "$mode" = tls ]; then
        PSK="$(python3 -c 'import secrets;print(secrets.token_hex(32))')"
        psk_env=(GGML_RPC_PSK="$PSK")
    else
        psk_env=(GGML_RPC_ALLOW_PLAINTEXT=1)
    fi
    env "${psk_env[@]}" "$ENGINE_DIR/ggml-rpc-server" -H 127.0.0.1 -p "$RPC_PORT" \
        -d "$RPC_DEV" >"$TMP/rpc.log" 2>&1 &
    RPC_PID=$!
    sleep 2
    python3 "$REPO/scripts/rpc_tap.py" --listen "$TAP_PORT" --target "$RPC_PORT" \
        --out "$cap" >"$TMP/tap.log" 2>&1 &
    TAP_PID=$!
    sleep 1
    env "${psk_env[@]}" "$ENGINE_DIR/llama-server" -m "$GGUF" --host 127.0.0.1 \
        --port "$API_PORT" --rpc "127.0.0.1:$TAP_PORT" "$@" -ngl 99 \
        >"$TMP/llama.log" 2>&1 &
    LL_PID=$!
    local i b
    for i in $(seq 1 90); do
        b="$(curl -s -m 2 "http://127.0.0.1:$API_PORT/health" 2>/dev/null)"
        [ "$b" = '{"status":"ok"}' ] && return 0
        kill -0 "$LL_PID" 2>/dev/null || return 1
        sleep 1
    done
    return 1
}

stop_stack() {
    for p in $LL_PID $TAP_PID $RPC_PID; do kill "$p" 2>/dev/null; done
    # wait on each so bash's job monitor does not print "Terminated/Abort trap"
    # lines to stderr as the children die.
    for p in $LL_PID $TAP_PID $RPC_PID; do wait "$p" 2>/dev/null; done
    pkill -f "ggml-rpc-serve[r].* -p $RPC_PORT" 2>/dev/null
    pkill -f "llama-serve[r].*--port $API_PORT" 2>/dev/null
    pkill -f "rpc_tap.py --listen $TAP_PORT" 2>/dev/null
    LL_PID=""; TAP_PID=""; RPC_PID=""
    sleep 1
}

drive() {  # tokenize the prompt (-> $TMP/tok.json) and generate traffic
    curl -s -m 15 "http://127.0.0.1:$API_PORT/tokenize" \
        -H 'content-type: application/json' \
        -d "{\"content\":\"$PROMPT\"}" >"$TMP/tok.json" 2>/dev/null
    curl -s -m 90 "http://127.0.0.1:$API_PORT/completion" \
        -H 'content-type: application/json' \
        -d "{\"prompt\":\"$PROMPT\",\"n_predict\":8,\"temperature\":0}" \
        >/dev/null 2>&1
    sleep 1
}

# --- 1. POSITIVE CONTROL: all-remote + plaintext MUST leak -------------------
say "priv7: bad config (all layers remote, plaintext) -- the checker must recover the leak"
if ! start_stack "$TMP/bad.bin" plaintext --device RPC0; then
    die "the bad-config stack did not become ready (see $TMP/llama.log, $TMP/rpc.log)"
fi
drive
stop_stack

TOKENS="$(python3 -c "import json;print(','.join(str(x) for x in json.load(open('$TMP/tok.json'))['tokens']))" 2>/dev/null)"
[ -n "$TOKENS" ] || die "could not read prompt token ids (see $TMP/tok.json)"

# Decoys: 20 vocab ids not in the prompt (false-positive control).
DECOYS="$(python3 -c "
import json,random,sys
sys.path.insert(0,'$GGUF_PY')
from gguf import GGUFReader
r=GGUFReader('$GGUF')
nv=None
for t in r.tensors:
    if t.name=='token_embd.weight': nv=int(t.shape[1]); break
toks=set(json.load(open('$TMP/tok.json'))['tokens'])
random.seed(7); out=[]
while len(out)<20:
    d=random.randrange(nv)
    if d not in toks: out.append(d)
print(','.join(str(x) for x in out))
" 2>/dev/null)"
[ -n "$DECOYS" ] || die "could not build the decoy set"

BAD_JSON="$(python3 "$REPO/scripts/gpriv7_recover.py" --gguf "$GGUF" \
    --capture "$TMP/bad.bin" --tokens "$TOKENS" --decoys "$DECOYS" \
    --gguf-py "$GGUF_PY" --prefix-floats "$PREFIX_FLOATS" 2>&1)"
say "priv7: bad-config recovery: $BAD_JSON"
eval "$(python3 -c "
import json
d=json.loads('''$BAD_JSON''')
print('BN=%d;BR=%d;BD=%d;BH=%d'%(d.get('n_prompt',0),d.get('recovered',0),d.get('n_decoy',0),d.get('decoy_hits',0)))
" 2>/dev/null)" || die "could not parse the bad-config recovery result: $BAD_JSON"

# The whole gate rests on this: the checker CAN see a real leak.
[ "${BR:-0}" -gt 0 ] && [ "${BR:-0}" -eq "${BN:-0}" ] \
    || die "positive control failed: recovered ${BR:-?}/${BN:-?} prompt embeddings from a plaintext all-remote stream -- the checker cannot detect a leak it is supposed to catch, so a green good-config would mean nothing"
[ "${BH:-1}" -eq 0 ] \
    || die "positive control unreliable: ${BH} of ${BD} decoy tokens also 'recovered' -- the byte-prefix match is coincidental, not real recovery"
say "priv7: positive control OK -- recovered $BR/$BN prompt embeddings, 0/$BD decoys"

# --- 2. GOOD config: coordinator-first + TLS MUST NOT leak -------------------
say "priv7: good config (layer 0 local, TLS) -- no embedding may cross the wire"
if ! start_stack "$TMP/good.bin" tls --device "$LOCAL_DEV,RPC0" --tensor-split 0.7,0.3; then
    die "the good-config stack did not become ready (see $TMP/llama.log, $TMP/rpc.log)"
fi
drive
stop_stack

# Sanity: the good capture must actually be TLS (starts with a handshake
# record) -- otherwise 0 recoveries could just mean nothing was captured.
if ! head -c 3 "$TMP/good.bin" | od -An -tx1 | tr -d ' \n' | grep -q '^160303'; then
    die "the good-config capture does not begin with a TLS handshake -- the run may not have been encrypted (see $TMP/good.bin)"
fi

GOOD_JSON="$(python3 "$REPO/scripts/gpriv7_recover.py" --gguf "$GGUF" \
    --capture "$TMP/good.bin" --tokens "$TOKENS" --decoys "$DECOYS" \
    --gguf-py "$GGUF_PY" --prefix-floats "$PREFIX_FLOATS" 2>&1)"
say "priv7: good-config recovery: $GOOD_JSON"
GR="$(python3 -c "import json;print(json.loads('''$GOOD_JSON''').get('recovered',-1))" 2>/dev/null)"
[ "${GR:-1}" = 0 ] \
    || die "the product configuration LEAKED: recovered $GR prompt embeddings from a TLS coordinator-first stream (expected 0)"

say "G_PRIV7_OK positive-control recovered $BR/$BN (0/$BD decoys); product config recovered 0/$BN"
exit 0
