#!/usr/bin/env bash
# G_PAIR self-test — LAN verification-code pairing, engine-native.
#
# Two engine processes (idletoken-coord + idletoken-worker) on ONE node self-assemble
# using ONLY a shared verification code — no manual --coordinator. The worker
# discovers the coordinator over the loopback UDP beacon, passes the mutual-auth
# preamble, and completes the existing HELLO -> RESOURCE -> ASSIGN -> LOAD_DONE
# handshake until the coordinator reports "cluster ready". This is the same flow
# two real machines take on the LAN, exercised locally as an oracle.
#
# Model load is mock here (no GGUF): G_PAIR proves discovery + code auth + join,
# not inference — inference correctness is G6/P6. Prints G_PAIR_OK / G_PAIR_FAIL.
#
# Usage: scripts/pair_selftest.sh [CODE]           (run on a node with the built
#        idletoken-coord + idletoken-worker binaries)
set -u

cd "$(dirname "$0")/.." || exit 1
# Make the CUDA runtime libs discoverable for the worker probe (harmless if the
# toolkit path differs; the worker still runs its probe).
export PATH="/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH"

CODE="${1:-ABC234}"
DPORT="${IDLETOKEN_DISC_PORT:-14097}"
COORD_LOG="$(mktemp)"; WORKER_LOG="$(mktemp)"
NOMODEL="/tmp/idletoken-nomodel-$$.gguf"   # intentionally absent -> mock load

cleanup() {
    pkill -f "idletoken-coor[d]" >/dev/null 2>&1
    pkill -f "idletoken-worke[r]" >/dev/null 2>&1
    rm -f "$COORD_LOG" "$WORKER_LOG"
}
trap cleanup EXIT
cleanup   # clear any stragglers from a prior run

# This gate tests **pairing** (discovery, join-code mutual recognition, then
# HELLO -> RESOURCE -> ASSIGN -> LOAD_DONE), not inference, so it deliberately
# points at a GGUF that does not exist and takes the mock load path. But a worker
# now **refuses** to serve from mock by default (the generality work removed the
# automatic fallback), and without an explicit opt-in LOAD_DONE comes back ok=0
# and the gate goes red. The coordinator needs IDLETOKEN_MOCK_OK for the same
# reason: two switches at two layers, similarly named but not the same thing.
export IDLETOKEN_ALLOW_MOCK=1
export IDLETOKEN_MOCK_OK=1

if [ ! -x ./idletoken-coord ] || [ ! -x ./idletoken-worker ]; then
    echo "pair_selftest: idletoken-coord/idletoken-worker not built here" >&2
    echo G_PAIR_FAIL
    exit 1
fi

# Coordinator advertises the code and waits for one worker.
./idletoken-coord --pair-code "$CODE" --num-workers 1 --n-predict 0 \
    --discovery-port "$DPORT" > "$COORD_LOG" 2>&1 &
sleep 3

# Worker joins by code ONLY (no --coordinator), discovering over the beacon.
timeout 90 ./idletoken-worker --pair-code "$CODE" --discovery-port "$DPORT" \
    --gguf-dir /tmp --model "$NOMODEL" > "$WORKER_LOG" 2>&1 &

ok=""
for _ in $(seq 1 40); do
    if grep -q "cluster ready" "$COORD_LOG" 2>/dev/null; then ok=1; break; fi
    if grep -qi "aborting\|refusing" "$COORD_LOG" 2>/dev/null; then break; fi
    sleep 1
done

echo "-------- coord --------"; cat "$COORD_LOG"
echo "-------- worker (tail) --------"; tail -20 "$WORKER_LOG"
echo "-----------------------"

if [ -n "$ok" ] \
   && grep -q "passed pairing auth" "$COORD_LOG" \
   && grep -q "pairing auth OK" "$WORKER_LOG" \
   && grep -q "discovered coordinator" "$WORKER_LOG"; then
    echo G_PAIR_OK
    exit 0
else
    echo G_PAIR_FAIL
    exit 1
fi
