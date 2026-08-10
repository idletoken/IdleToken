#!/usr/bin/env bash
# Cross-machine verification (runs ON the DGX coord node): a real Windows worker
# joins the coordinator over the physical LAN using ONLY a verification code —
# no --coordinator. Proves engine-native discovery + mutual auth across two
# physical machines/OSes. Prints XMACHINE_PAIR_OK.
#
# Model: the SMALLEST one (qwen3.5-0.8b), pulled by the joiner from the
# coordinator's weight server. It used to run with the default model (DSv4,
# 82 GiB) and a comment saying "mock model, this is about discovery+join" —
# but since mock stopped being an automatic fallback (2026-07-29) the coord
# refuses on capacity BEFORE cluster_ready, so the check could only fail. A
# small real model makes the same assertion honest AND stronger: it now proves
# join *and* plan *and* weight fetch across the LAN, not just the handshake.
set -u
cd ~/work/IdleToken || exit 1
export PATH="/usr/local/cuda-13.0/bin:$PATH"

CODE="${1:-ABC234}"
DPORT="${2:-14097}"
WORKER_NODE="${WORKER_NODE:-${IDLETOKEN_WIN_BUILD_NODE:-}}"
if [ -z "$WORKER_NODE" ]; then
    echo "pair_xmachine_check.sh: needs WORKER_NODE=<ssh-alias> (or set it in testbed.env)" >&2
    exit 2
fi
# Per-node home: the accounts differ per machine, and defaulting every node to
# win-a's path means testing a directory that does not exist on the node you
# actually named (it fails in a way that looks like a pairing bug).
# shellcheck disable=SC1091
. "$(dirname "$0")/testbed-lib.sh"
WINHOME="${WINHOME:-$(testbed_repo_home "$WORKER_NODE")}"
if [ -z "$WINHOME" ]; then
    # The * branch here used to fall back to one specific machine's path, so the
    # script tested a directory that does not exist on the target and failed as if
    # it were a pairing bug. Better to say plainly what is missing.
    testbed_hint "$WORKER_NODE"; exit 2
fi
# v12.8 (bin\, not 13.x's bin\x64): must match what ds4cuda.dll was built
# against, or the DLL mismatch degrades to CPU without saying so.
CUDABIN="${IDLETOKEN_WIN_CUDA_BIN:-C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin}"
CLOG=/tmp/xcoord.log
SSHW="ssh -o BatchMode=yes -o ConnectTimeout=8 $WORKER_NODE"

cleanup() {
    pkill -f "idletoken-coor[d]" >/dev/null 2>&1
    pkill -f "idletoken-worker --serve-weight[s]" >/dev/null 2>&1
    $SSHW "powershell -NoProfile -Command \"Get-Process idletoken-worker -ErrorAction SilentlyContinue | Stop-Process -Force\"" >/dev/null 2>&1
}
trap cleanup EXIT
cleanup
rm -f "$CLOG"

# Small model + weight server: the joiner needs no local copy (it fetches its
# layer range over HTTP Range), so this works on a bare deploy machine too.
MODEL_ID="${MODEL_ID:-qwen3.5-0.8b}"
GGUF_DIR="${GGUF_DIR:-$HOME/work/qwen}"
GGUF_FILE="${GGUF_FILE:-Qwen3.5-0.8B-Q4_K_M.gguf}"
if [ ! -f "$GGUF_DIR/$GGUF_FILE" ]; then
    echo "no $GGUF_DIR/$GGUF_FILE on the coordinator — cannot run the cross-machine check"
    echo XMACHINE_PAIR_SKIP
    exit 0
fi
LAN=$(ip -4 -o addr show scope global | awk '{split($4,a,"/"); print a[1]}' \
      | grep -E '^(192\.168\.|10\.)' | head -1)

./idletoken-worker --serve-weights "$GGUF_DIR/$GGUF_FILE" --weights-port 8001 \
    > /tmp/xweights.log 2>&1 &
sleep 2

# Coordinator advertises the code on the LAN (udp/$DPORT), waits for one worker.
./idletoken-coord --pair-code "$CODE" --num-workers 1 --n-predict 0 \
    --model-id "$MODEL_ID" --model-path "$GGUF_DIR/$GGUF_FILE" --gguf-dir "$GGUF_DIR" \
    --discovery-port "$DPORT" > "$CLOG" 2>&1 &
sleep 3

# Windows worker joins by code ONLY (no --coordinator), discovering over the LAN.
# Run it in the FOREGROUND and stream its output back: the worker self-exits when
# it joins (coord closes the control conn after ready, n-predict 0) or after its
# 60s discovery timeout, so this returns on its own.
echo "======== WORKER ($WORKER_NODE, foreground) ========"
$SSHW "powershell -NoProfile -Command \"cd $WINHOME; \\\$env:PATH='$CUDABIN;'+\\\$env:PATH; & .\\idletoken-worker.exe --pair-code $CODE --shard-repo http://$LAN:8001/$GGUF_FILE --gguf-dir '$WINHOME' --discovery-port $DPORT 2>&1\"" 2>&1 | tr -d '\r' | tail -40

ok=""
grep -q "cluster ready" "$CLOG" 2>/dev/null && ok=1

echo "======== COORD ($CLOG) ========"
cat "$CLOG"
echo "==============================================="

if [ -n "$ok" ] && grep -q "passed pairing auth" "$CLOG"; then
    echo XMACHINE_PAIR_OK
else
    echo XMACHINE_PAIR_FAIL
fi
