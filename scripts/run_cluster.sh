#!/usr/bin/env bash
# E5/E6 oracle -- multi-worker cluster bring-up on one host (multi-process simulation).
#
# Runs ON the coordinator node (DGX), from anywhere (cd's to repo root).
#
#   --check-ready [--workers N]   E5 gate: coord + N workers (default 2) walk
#                                 HELLO → RESOURCE_REPORT → ASSIGN_PLAN →
#                                 LOAD_MODEL_DONE (real GGUF, layer-subset
#                                 load) + 1 warmup decode step through the
#                                 whole pipeline. Asserts: cluster ready, ≥2
#                                 stages, layer ranges cover [0,43) exactly,
#                                 every worker loaded a real (non-mock) subset.
#                                 Last line on success: CLUSTER_READY
#
#   --serve [--workers N]         E6 helper: same bring-up plus --http; waits
#                                 for /health, prints API_READY <addr>, leaves
#                                 the cluster running (pids in $RUN_DIR).
#
#   --stop                        Tear down a --serve cluster.
set -u
cd "$(dirname "$0")/.." || exit 1

GGUF="${IDLETOKEN_GGUF:-$HOME/work/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf}"
CTX="${IDLETOKEN_E5_CTX:-2048}"
N_LAYERS=43
COORD_BIND="127.0.0.1:14310"
API_BIND="${IDLETOKEN_API_BIND:-0.0.0.0:8000}"
API_CHECK="127.0.0.1:8000"
RUN_DIR=/tmp/idletoken-cluster
LOG=$RUN_DIR
mkdir -p "$RUN_DIR"

MODE=""
NWORKERS=2
while [ $# -gt 0 ]; do
    case "$1" in
        --check-ready) MODE=check ;;
        --serve)       MODE=serve ;;
        --stop)        MODE=stop ;;
        --workers)     shift; NWORKERS="$1" ;;
        *) echo "run_cluster.sh: unknown arg $1" >&2; exit 2 ;;
    esac
    shift
done
[ -n "$MODE" ] || { echo "usage: run_cluster.sh --check-ready|--serve|--stop [--workers N]" >&2; exit 2; }

fail() { echo "E5: $*" >&2; echo "CLUSTER_FAIL"; exit 1; }

stop_cluster() {
    local killed=0
    for pf in "$RUN_DIR"/*.pid; do
        [ -f "$pf" ] || continue
        local pid; pid=$(cat "$pf")
        kill "$pid" 2>/dev/null && killed=1
        rm -f "$pf"
    done
    [ "$killed" = 1 ] && sleep 1
    echo "CLUSTER_STOPPED"
}

if [ "$MODE" = stop ]; then stop_cluster; exit 0; fi

[ -f "$GGUF" ]         || fail "GGUF not found: $GGUF"
[ -x ./idletoken-coord ]  || fail "idletoken-coord missing (run: make all)"
[ -x ./idletoken-worker ] || fail "idletoken-worker missing (run: make all)"
[ "$NWORKERS" -ge 2 ] 2>/dev/null || fail "--workers must be >= 2 (E5 wants a real pipeline)"

# Tear down any leftover serve-mode cluster so ports are free.
stop_cluster >/dev/null

start_workers() {
    for i in $(seq 1 "$NWORKERS"); do
        local port=$((14310 + i))
        ./idletoken-worker --coordinator "$COORD_BIND" --bind "127.0.0.1:$port" \
            --gguf-dir "$(dirname "$GGUF")" \
            > "$LOG/worker$i.log" 2>&1 &
        echo $! > "$RUN_DIR/worker$i.pid"
    done
}

check_logs() {
    # ≥2 stages and contiguous [0,43) coverage, straight from the plan print.
    python3 - "$LOG/coord.log" "$N_LAYERS" <<'EOF' || return 1
import re, sys
log = open(sys.argv[1], errors="replace").read()
n_layers = int(sys.argv[2])
stages = re.findall(r"stage (\d+) -> \S+\s+layers \[(\d+),(\d+)\)", log)
if len(stages) < 2:
    sys.exit(f"only {len(stages)} stage(s) planned; E5 needs >= 2")
iv = sorted((int(lo), int(hi)) for _, lo, hi in stages)
cur = 0
for lo, hi in iv:
    if lo != cur or hi <= lo:
        sys.exit(f"layer coverage broken at [{lo},{hi}), expected start {cur}")
    cur = hi
if cur != n_layers:
    sys.exit(f"layers covered [0,{cur}), expected [0,{n_layers})")
print(f"E5: {len(stages)} stages cover [0,{n_layers}) contiguously")
EOF
    grep -q "cluster ready" "$LOG/coord.log" || { echo "no 'cluster ready' in coord log" >&2; return 1; }
    for i in $(seq 1 "$NWORKERS"); do
        grep -q "model load MOCK" "$LOG/worker$i.log" && \
            { echo "worker$i fell back to MOCK load" >&2; return 1; }
        # The layer-subset LOAD is a CUDA-path optimization: ds4's
        # accelerator_cache_model_tensors() returns immediately unless
        # backend == DS4_BACKEND_CUDA, so the "PP filter active" line cannot
        # appear under IDLETOKEN_DS4_CPU=1 — the CPU backend mmaps the GGUF and
        # touches pages on demand instead of caching chosen spans. Asserting it
        # there would fail a correct cluster; dropping it for everyone would
        # stop catching a real GPU regression (a worker loading the whole model
        # instead of its slice). So: keep it, scope it.
        if [ "${IDLETOKEN_DS4_CPU:-0}" = "1" ]; then
            grep -q "ds4 CPU backend" "$LOG/worker$i.log" || \
                { echo "worker$i did not report the CPU backend under IDLETOKEN_DS4_CPU=1" >&2; return 1; }
        else
            grep -Eq "PP filter active.*skipped" "$LOG/worker$i.log" || \
                { echo "worker$i did not layer-subset load (no PP filter line)" >&2; return 1; }
        fi
    done
    return 0
}

if [ "$MODE" = check ]; then
    # Foreground coord: with no --http it exits 0 after the warmup step, which
    # also proves HC tensors actually flowed across every stage boundary.
    ( sleep 2; start_workers ) &
    SPAWNER=$!
    trap 'kill $SPAWNER 2>/dev/null; stop_cluster >/dev/null' EXIT
    timeout 300 ./idletoken-coord --bind "$COORD_BIND" --num-workers "$NWORKERS" \
        --n-predict 1 --ctx-size "$CTX" --model-path "$GGUF" \
        > "$LOG/coord.log" 2>&1
    rc=$?
    wait "$SPAWNER" 2>/dev/null
    [ $rc -eq 0 ] || fail "coord exited rc=$rc (see $LOG/coord.log)"
    grep -q "warmup step 0" "$LOG/coord.log" || fail "warmup decode step did not run"
    check_logs || fail "readiness assertions failed (see $LOG/*.log)"
    echo "CLUSTER_READY"
    exit 0
fi

# ------------------------------------------------------------------ serve --
./idletoken-coord --bind "$COORD_BIND" --api-bind "$API_BIND" --http \
    --num-workers "$NWORKERS" --n-predict 0 --ctx-size "$CTX" \
    --model-path "$GGUF" > "$LOG/coord.log" 2>&1 &
echo $! > "$RUN_DIR/coord.pid"
sleep 2
start_workers

echo "E6: waiting for cluster + HTTP API (model load ~30-90s)..."
for i in $(seq 1 240); do
    if curl -s -m 2 "http://$API_CHECK/health" 2>/dev/null | grep -q '"status":"ok"'; then
        check_logs || { stop_cluster >/dev/null; fail "cluster up but readiness assertions failed"; }
        echo "API_READY $API_BIND"
        exit 0
    fi
    kill -0 "$(cat "$RUN_DIR/coord.pid")" 2>/dev/null || \
        { stop_cluster >/dev/null; fail "coord exited early (see $LOG/coord.log)"; }
    sleep 1
done
stop_cluster >/dev/null
fail "API did not come up within 240s (see $LOG/*.log)"
