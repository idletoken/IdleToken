#!/usr/bin/env bash
# Does a dead engine take the whole coordinator down with it?
#
# WHY (2026-08-17, results/coord-wedge-20260817.md). A coordinator was found
# wedged in the field: its only executor thread parked in recv() on an engine
# that had stopped answering, with no timeout, forever. Because execution is
# serial that killed everything — even GET /health went unanswered, and only a
# restart recovered.
#
# THE FIXTURE, AND TWO THAT DID NOT WORK. The engine must go silent while
# STAYING a live, killable process that never closes the connection:
#   - SIGSTOP on the real engine is wrong: a stopped child cannot act on
#     SIGTERM either, so the sidecar monitor wedges too and the gate ends up
#     measuring a different hang. (Measured: it "failed" for that reason.)
#   - SIGKILL is wrong: it closes the socket, and a FIN is the one thing the
#     field failure never delivered — the coordinator would have recovered on
#     its own, so the gate would pass without testing anything.
# scripts/stub_engine_dark.py does exactly the right thing instead.
#
# READINESS IS A LITERAL COMPARE. The coordinator answers {"status":"ok",...}
# while the engine is still `starting`, and a chat posted then is refused
# outright — so an "ok"-only check produces a gate that tests nothing. (Also
# measured: the first version of this gate did that and reported a failure that
# was entirely its own.) Match on engine_state.
#
# Contract: last line WEDGE_GATE_OK, WEDGE_GATE_FAIL: <why>, or
# WEDGE_GATE_SKIP: <why>. SKIP is for a machine that cannot run the fixture at
# all (no smoke weights, no coordinator binary, no python3) — never for a
# fixture that armed and then did not hold. A gate that reports green on a
# machine where nothing ran is worse than no gate.
set -u

PORT="${1:-18777}"
# The weights are this machine's property, so they come from the testbed file
# rather than a path baked in for one node (scripts/testbed.env, gitignored).
cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD
# shellcheck disable=SC1091
[ -f scripts/testbed.env ] && . ./scripts/testbed.env
GGUF="${2:-${IDLETOKEN_SMOKE_GGUF:-}}"
# Missing prerequisites are a SKIP, not a FAIL: none of them says anything about
# whether a dark engine wedges the coordinator, which is the only question here.
[ -n "$GGUF" ] || { echo "WEDGE_GATE_SKIP: no GGUF: pass one, or set IDLETOKEN_SMOKE_GGUF in scripts/testbed.env"; exit 0; }
[ -f "$GGUF" ] || { echo "WEDGE_GATE_SKIP: $GGUF is not a file"; exit 0; }
[ -x ./idletoken-coord ] || { echo "WEDGE_GATE_SKIP: no ./idletoken-coord — run 'make coord' first"; exit 0; }
command -v python3 >/dev/null 2>&1 || { echo "WEDGE_GATE_SKIP: no python3 — scripts/stub_engine_dark.py is the fixture"; exit 0; }

# setsid is Linux-only; on macOS nohup alone is enough to survive this script.
SETSID=$(command -v setsid || true)

fail() { echo "WEDGE_GATE_FAIL: $*"; cleanup; exit 1; }
cleanup() {
    pkill -9 -f '[i]dletoken-coord --llama' 2>/dev/null
    pkill -9 -f '[s]tub_engine_dark' 2>/dev/null
}
cleanup; sleep 1

# The coordinator spawns its engine itself, so the stub is handed over as the
# engine binary. It ignores llama-server's arguments except --port.
STUB=/tmp/idletoken-stub-engine.sh
printf '#!/bin/sh\nexec python3 %s/scripts/stub_engine_dark.py "$@"\n' "$ROOT" > "$STUB"
chmod +x "$STUB"

$SETSID nohup ./idletoken-coord \
    --llama-server-bin "$STUB" --llama-gguf "$GGUF" \
    --http --api-bind "127.0.0.1:$PORT" \
    --model-id qwen3.5-0.8b --ctx-size 4096 </dev/null > /tmp/wedge-gate-coord.log 2>&1 &
sleep 1

ready=0
for _ in $(seq 1 60); do
    if curl -s -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"engine_state":"ready"'; then
        ready=1; break
    fi
    sleep 1
done
[ "$ready" = 1 ] || fail "the engine never reached ready (see /tmp/wedge-gate-coord.log)"
echo "  coordinator ready, engine_state=ready"

# The stub streams a few tokens and then goes dark mid-stream.
cat > /tmp/idletoken-wedge-body.json <<'JSON'
{"model":"qwen3.5-0.8b","stream":true,"max_tokens":400,
 "messages":[{"role":"user","content":"hello there"}]}
JSON
curl -s -m 120 -X POST "http://127.0.0.1:$PORT/v1/messages" \
     -H 'Content-Type: application/json' \
     --data-binary @/tmp/idletoken-wedge-body.json > /tmp/wedge-gate-turn.log 2>&1 &
turn_pid=$!
sleep 5
grep -q "went dark" "$HOME"/.idletoken/idletoken-server-*.log 2>/dev/null \
    || fail "the stub never went dark — the fixture did not arm, so nothing was tested"
echo "  engine went dark mid-stream: alive, silent, socket held open"

echo "  asking the coordinator for /health while the engine is dark..."
t0=$(date +%s)
served=0
for _ in $(seq 1 40); do
    if curl -s -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status"'; then
        served=1; break
    fi
    sleep 1
done
dt=$(( $(date +%s) - t0 ))
[ "$served" = 1 ] || fail "the coordinator never answered /health again (${dt}s) — still wedged behind the dark engine"
echo "  /health answered ${dt}s after the engine went dark"

# Since the pool landed (2026-08-18) /health answers straight away, because it
# was never behind the dark relay to begin with. That is the improvement — and
# it also means a fast /health no longer PROVES the liveness check ran. The
# proof is now the dark REQUEST itself: it must be given up on, by that
# mechanism, and say so. Budget = 15s of grace + the 10s probe interval + the
# probe's own timeout, plus slack.
echo "  waiting for the dark request itself to be given up on..."
t0=$(date +%s)
gave_up=0
for _ in $(seq 1 60); do
    if ! kill -0 "$turn_pid" 2>/dev/null; then gave_up=1; break; fi
    sleep 1
done
dtq=$(( $(date +%s) - t0 ))
kill "$turn_pid" 2>/dev/null
[ "$gave_up" = 1 ] || fail "the dark request was still parked after ${dtq}s — bounded waiting is not bounding anything"

# Recovering is not enough: it has to be THIS mechanism, and it has to say so.
# Without this the gate passes on any unrelated recovery — and a silent recovery
# is not diagnosable in the field either.
grep -q "stopped answering" /tmp/wedge-gate-coord.log \
    || fail "the request ended, but not through the liveness check (nothing in the log) — the gate would pass for the wrong reason"
echo "  dark request failed after ${dtq}s, by the liveness check:"
grep -m1 "stopped answering" /tmp/wedge-gate-coord.log | sed 's/^/    /'
cleanup; sleep 1

# ---------------------------------------------------------------------------
# Phase 2 (2026-08-18): with several slots, does ONE dead relay stay one dead
# request?
#
# Phase 1 cannot answer that. Its stub darkens the whole engine, so every
# stream ends at the same instant whether the coordinator isolates failures or
# not — a green run there would prove nothing about the multi-slot path. This
# phase uses --dark-mode one: the first chat connection goes silent and is held
# open forever, every later one is served normally.
#
# What must hold: the second stream, opened while the first is stuck, completes
# on its own schedule, and /health keeps answering throughout.
echo "  phase 2: one dark relay must not take the other slots with it"
STUB1=/tmp/idletoken-stub-engine-one.sh
printf '#!/bin/sh\nexec python3 %s/scripts/stub_engine_dark.py --dark-mode one "$@"\n' "$ROOT" > "$STUB1"
chmod +x "$STUB1"

IDLETOKEN_LLAMA_SLOTS=2 $SETSID nohup ./idletoken-coord \
    --llama-server-bin "$STUB1" --llama-gguf "$GGUF" \
    --http --api-bind "127.0.0.1:$PORT" \
    --model-id qwen3.5-0.8b --ctx-size 4096 </dev/null > /tmp/wedge-gate-coord2.log 2>&1 &
sleep 1
ready=0
for _ in $(seq 1 60); do
    if curl -s -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"engine_state":"ready"'; then
        ready=1; break
    fi
    sleep 1
done
[ "$ready" = 1 ] || fail "phase 2: the engine never reached ready (see /tmp/wedge-gate-coord2.log)"

grep -q '"seq_slots":2' <(curl -s -m 3 "http://127.0.0.1:$PORT/idletoken/v1/stats") \
    || fail "phase 2: the coordinator is not running 2 slots, so the isolation claim is untested"

# A: the request that gets the dark connection. It must NOT be waited on.
curl -s -m 120 -X POST "http://127.0.0.1:$PORT/v1/messages" \
     -H 'Content-Type: application/json' \
     --data-binary @/tmp/idletoken-wedge-body.json > /tmp/wedge-gate-turnA.log 2>&1 &
turnA=$!
sleep 2

# B: opened while A is stuck. Its own clock starts now.
t0=$(date +%s)
curl -s -m 30 -X POST "http://127.0.0.1:$PORT/v1/messages" \
     -H 'Content-Type: application/json' \
     --data-binary @/tmp/idletoken-wedge-body.json > /tmp/wedge-gate-turnB.log 2>&1
rcB=$?
dtB=$(( $(date +%s) - t0 ))

curl -s -m 3 "http://127.0.0.1:$PORT/health" | grep -q '"status"' \
    || fail "phase 2: /health stopped answering while one relay was dark"

kill "$turnA" 2>/dev/null
[ "$rcB" = 0 ] || fail "phase 2: the second stream did not complete (curl rc=$rcB) — the dark relay took it down too"
grep -q "message_stop" /tmp/wedge-gate-turnB.log \
    || fail "phase 2: the second stream returned without a terminator — it was cut, not served"
# A's grace period is 15s of silence before the first liveness probe, so a B
# that finished inside that window is proof it was never queued behind A.
[ "$dtB" -lt 15 ] || fail "phase 2: the second stream took ${dtB}s — it waited for the dark one instead of running beside it"
echo "  second stream completed in ${dtB}s while the first was dark"
cleanup
echo WEDGE_GATE_OK
