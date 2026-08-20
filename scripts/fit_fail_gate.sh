#!/usr/bin/env bash
# Does the coordinator refuse when the engine says it could not fit the model
# into device memory — and only then?
#
# WHY (2026-08-18, win_PC). A 16 GiB card was started with 4 sequence slots of
# 40960 tokens each, ~22 GiB of KV. The engine warned on its first line that it
# could not fit the parameters into free device memory, then started anyway
# because we pin -ngl and -c. Windows WDDM did not fail the allocation either:
# it paged video memory out to system memory, and since the desktop compositor
# shares the card, the machine froze. Nothing in that chain reported an error.
#
# The budget fix (idletoken_llama_seq_slots, plan_test.c) is what keeps a
# machine out of that state. This gate covers the second line of defence: when
# the estimate is wrong anyway, the engine's own complaint must stop the start.
#
# THE FIXTURE. The coordinator spawns its engine, so the stub IS the engine:
# a shell script that writes the warning to stdout (which the sidecar redirects
# into the engine log, exactly where the real one landed) and then serves
# normally. Serving normally is the point — the failure being tested is an
# engine that works well enough to look fine.
#
# THREE RUNS, because one proves nothing:
#   1. warning present            -> refuse, exit 3
#   2. warning absent (CONTROL)   -> serves, stays up
#   3. warning + the escape hatch -> serves, stays up, says so loudly
# Run 2 is what makes run 1 meaningful: without it, a coordinator that refused
# every start would pass.
#
# Contract: last line FIT_GATE_OK, FIT_GATE_FAIL: <why>, or FIT_GATE_SKIP: <why>.
set -u

PORT="${1:-18779}"
cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD
# shellcheck disable=SC1091
[ -f scripts/testbed.env ] && . ./scripts/testbed.env
GGUF="${2:-${IDLETOKEN_SMOKE_GGUF:-}}"
[ -n "$GGUF" ] || { echo "FIT_GATE_SKIP: no GGUF: pass one, or set IDLETOKEN_SMOKE_GGUF in scripts/testbed.env"; exit 0; }
[ -f "$GGUF" ] || { echo "FIT_GATE_SKIP: $GGUF is not a file"; exit 0; }
[ -x ./idletoken-coord ] || { echo "FIT_GATE_SKIP: no ./idletoken-coord — run 'make coord' first"; exit 0; }
command -v python3 >/dev/null 2>&1 || { echo "FIT_GATE_SKIP: no python3 — the stub engine needs it"; exit 0; }

SETSID=$(command -v setsid || true)
LOG=/tmp/idletoken-fit-gate-engine.log

cleanup() {
    pkill -9 -f '[i]dletoken-coord --llama' 2>/dev/null
    pkill -9 -f '[s]tub_engine_dark' 2>/dev/null
}
fail() { echo "FIT_GATE_FAIL: $*"; cleanup; exit 1; }
cleanup; sleep 1

# The line is upstream's, copied from the machine that froze.
FIT_LINE='W common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort'

mk_stub() {   # $1 = stub path, $2 = "warn" | "quiet"
    {
        echo '#!/bin/sh'
        [ "$2" = warn ] && printf 'echo "%s"\n' "$FIT_LINE"
        # No chat is sent in this gate, so the stub's darkening never arms; it
        # is here purely as an engine that answers /health like a real one.
        printf 'exec python3 %s/scripts/stub_engine_dark.py "$@"\n' "$ROOT"
    } > "$1"
    chmod +x "$1"
}

# Start a coordinator on the stub; echoes its exit code once it is gone, or
# "alive" if it is still serving after $2 seconds.
# The memory figure is PINNED (8 GiB), and that is not a shortcut: what this
# gate asks is "does the engine's own complaint stop the start", and a
# coordinator that refuses first for its own memory reasons never gets to the
# question. Leaving it to the machine's mood also makes the gate pass or fail
# for reasons that have nothing to do with the code under test — measured, on a
# busy control Mac reporting 0.24 GiB usable.
run_coord() {   # $1 = stub, $2 = seconds to watch
    rm -f "$LOG"
    IDLETOKEN_TEST_USABLE_BYTES=$((8 * 1024 * 1024 * 1024)) \
    IDLETOKEN_LLAMA_LOG="$LOG" $SETSID nohup ./idletoken-coord \
        --llama-server-bin "$1" --llama-gguf "$GGUF" \
        --http --api-bind "127.0.0.1:$PORT" \
        --model-id qwen3.5-0.8b --ctx-size 4096 \
        </dev/null > /tmp/idletoken-fit-gate-coord.log 2>&1 &
    local pid=$!
    local i
    for i in $(seq 1 "$2"); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null
            echo "$?"
            return
        fi
        sleep 1
    done
    kill "$pid" 2>/dev/null
    echo alive
}

echo "  run 1: the engine reports it could not fit — the coordinator must refuse"
mk_stub /tmp/idletoken-stub-fitfail.sh warn
rc=$(run_coord /tmp/idletoken-stub-fitfail.sh 25)
[ "$rc" = 3 ] || fail "run 1: expected exit 3 (refuse), got '$rc' — see /tmp/idletoken-fit-gate-coord.log"
grep -q "refuse: the inference engine could not fit" /tmp/idletoken-fit-gate-coord.log \
    || fail "run 1: it exited 3 but not for this reason — the gate would pass for the wrong one"
grep -q "could not fit this model into free device memory" /tmp/idletoken-fit-gate-coord.log \
    || fail "run 1: the refusal does not say what happened"
echo "    refused, exit 3:"
grep -m1 "refuse: the inference engine" /tmp/idletoken-fit-gate-coord.log | cut -c1-140 | sed 's/^/      /'

echo "  run 2 (CONTROL): same stub, no warning — the coordinator must serve"
mk_stub /tmp/idletoken-stub-fitok.sh quiet
rc=$(run_coord /tmp/idletoken-stub-fitok.sh 20)
[ "$rc" = alive ] || fail "run 2: a clean start exited with '$rc' — the detector refuses starts that are fine"
echo "    still serving after 20s"

echo "  run 3: the escape hatch lets a measurement through, loudly"
rc=$(IDLETOKEN_ALLOW_VRAM_OVERCOMMIT=1 run_coord /tmp/idletoken-stub-fitfail.sh 20)
[ "$rc" = alive ] || fail "run 3: IDLETOKEN_ALLOW_VRAM_OVERCOMMIT=1 did not let it start (exit '$rc')"
grep -q "IDLETOKEN_ALLOW_VRAM_OVERCOMMIT=1" /tmp/idletoken-fit-gate-coord.log \
    || fail "run 3: it started but said nothing — a silent override is how this comes back"
echo "    started, with the banner"

cleanup
rm -f /tmp/idletoken-stub-fitfail.sh /tmp/idletoken-stub-fitok.sh
echo "FIT_GATE_OK"
