#!/usr/bin/env bash
# Does the coordinator's memory budget follow the GGUF it is actually going to
# open — and does it SAY where its number came from?
#
# WHY (2026-08-19, win_PC2, measured — results/llamacpp-multislot-big-win-20260819.md
# §4). `--model-id qwen3.5-27b --llama-gguf <...Q4_K_M.gguf>` planned against the
# manifest's DEFAULT quant (IQ2_XXS, 7.98 GiB) while the engine opened the real
# file (15.59 GiB), understating the weights by 7.6 GiB and deriving 2 sequence
# slots where 1 is correct. The engine's own fit check caught that machine, but
# it only fires when llama.cpp cannot fit the model AT ALL — it says nothing
# about a machine with room for the weights and not for the invented slots.
#
# THE FIXTURES. The sizes are the whole subject, and nothing here reads the
# bytes: each "GGUF" is a file of the right length with a hole in it (seek, put
# one byte), so a 15.59 GiB fixture costs no disk. The engine is the same stub
# the wedge and fit gates use — the coordinator prints its budget before it ever
# spawns one, so the stub only has to exist.
#
# FOUR RUNS, and run 2 is what makes run 1 mean anything:
#   1. file sized Q4_K_M, no --quant   -> weights 15.59, source names the file
#   2. file sized IQ2_XXS, no --quant  -> weights 7.98  (CONTROL: same command,
#                                         different file; a coordinator ignoring
#                                         the path would print 7.98 in BOTH and
#                                         run 1 would fail)
#   3. file sized IQ2_XXS + --quant Q4_K_M -> the FILE wins, and the
#                                         contradiction is named
#   4. file of a size in no menu       -> the real size, flagged as unrecognised
#
# Contract: last line BUDGET_GATE_OK, BUDGET_GATE_FAIL: <why>, or
# BUDGET_GATE_SKIP: <why>.
set -u

PORT="${1:-18781}"
cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD

[ -x ./idletoken-coord ] || { echo "BUDGET_GATE_SKIP: no ./idletoken-coord — run 'make coord' first"; exit 0; }
command -v python3 >/dev/null 2>&1 || { echo "BUDGET_GATE_SKIP: no python3 — the fixtures need it"; exit 0; }

# The registry's own numbers for qwen3.5-27b (src/common/model.c). Written out
# rather than derived so that a change to either side shows up as a gate
# failure to be read, not as two numbers agreeing with each other by
# construction.
IQ2_XXS_BYTES=8573593504     # variants[default_variant] — what the bug used
Q4_K_M_BYTES=16740812704     # the file win_PC2 was really serving
ODD_BYTES=13000000000        # ~12.11 GiB: between Q3_K_M and Q3_K_XL, matching
                             # neither within 1%

TMP=/tmp/idletoken-budget-gate
LOG=$TMP/coord.log
STUB=$TMP/stub-engine.sh

cleanup() {
    pkill -9 -f '[i]dletoken-coord --llama' 2>/dev/null
    pkill -9 -f '[s]tub_engine_dark' 2>/dev/null
}
fail() { echo "BUDGET_GATE_FAIL: $*"; cleanup; exit 1; }
cleanup
rm -rf "$TMP"; mkdir -p "$TMP" || { echo "BUDGET_GATE_SKIP: cannot create $TMP"; exit 0; }

{
    echo '#!/bin/sh'
    printf 'exec python3 %s/scripts/stub_engine_dark.py "$@"\n' "$ROOT"
} > "$STUB"
chmod +x "$STUB"

# A file of exactly $2 bytes without writing them. Sparse on APFS/ext4/tmpfs —
# checked, and the gate skips rather than filling a disk if it is not.
mk_gguf() {   # $1 = path, $2 = bytes
    python3 - "$1" "$2" <<'PY' || return 1
import os, sys
path, size = sys.argv[1], int(sys.argv[2])
with open(path, "wb") as f:
    f.truncate(size)
st = os.stat(path)
if st.st_size != size:
    sys.exit("size did not take")
# 512-byte blocks; a hole means near-zero. Bail loudly rather than silently
# consuming 15 GiB on a filesystem without sparse support.
if getattr(st, "st_blocks", 0) * 512 > 64 * 1024 * 1024:
    sys.exit("filesystem did not make the file sparse")
PY
}

SETSID=$(command -v setsid || true)

# Start a coordinator, let it print its plan, then stop it. The budget lines are
# written before the engine is spawned, so this never depends on the stub coming
# up. The memory figure is PINNED so that every run plans SINGLE and the only
# thing that varies between runs is the fixture's size — a coordinator refusing
# for its own memory reasons would never reach the question being asked.
run_coord() {   # $@ = extra coord args
    rm -f "$LOG"
    IDLETOKEN_TEST_USABLE_BYTES=$((60 * 1024 * 1024 * 1024)) \
    IDLETOKEN_LLAMA_LOG="$TMP/engine.log" $SETSID nohup ./idletoken-coord \
        --llama-server-bin "$STUB" \
        --http --api-bind "127.0.0.1:$PORT" \
        --model-id qwen3.5-27b --ctx-size 4096 \
        "$@" </dev/null > "$LOG" 2>&1 &
    local pid=$! i
    for i in $(seq 1 15); do
        grep -q "coord: scheduler:" "$LOG" 2>/dev/null && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    pkill -9 -f '[s]tub_engine_dark' 2>/dev/null
}

budget_line() { grep -m1 "coord: budget from:" "$LOG" 2>/dev/null; }
weights_gib()  { sed -n 's/.*(weights \([0-9.]*\) .*/\1/p' "$LOG" 2>/dev/null | head -1; }

BIG=$TMP/Qwen3.5-27B-Q4_K_M.gguf
SMALL=$TMP/Qwen3.5-27B-UD-IQ2_XXS.gguf
ODD=$TMP/Qwen3.5-27B-mystery.gguf
mk_gguf "$BIG"   "$Q4_K_M_BYTES"  || { echo "BUDGET_GATE_SKIP: cannot create a sparse 15.6 GiB fixture"; cleanup; exit 0; }
mk_gguf "$SMALL" "$IQ2_XXS_BYTES" || { echo "BUDGET_GATE_SKIP: cannot create the fixtures"; cleanup; exit 0; }
mk_gguf "$ODD"   "$ODD_BYTES"     || { echo "BUDGET_GATE_SKIP: cannot create the fixtures"; cleanup; exit 0; }

echo "  run 1: --llama-gguf at a Q4_K_M file, no --quant — the budget must be the file's"
run_coord --llama-gguf "$BIG"
b1=$(budget_line); w1=$(weights_gib)
[ -n "$b1" ] || fail "run 1: no 'budget from:' line — the coordinator does not say what it budgeted against"
# "manifest quant X", not a bare "X": the FIXTURE FILENAME also contains the
# quant, so a looser pattern is satisfied by the name the caller passed in and
# proves nothing about what the coordinator recognised. Found by injecting a
# wrong-label bug and watching this run pass anyway (2026-08-19).
case "$b1" in
    *"GGUF on disk"*"manifest quant Q4_K_M"*) : ;;
    *) fail "run 1: the source line does not name the file and the quant it recognised: $b1" ;;
esac
[ "$w1" = "15.59" ] || fail "run 1: planner saw 'weights $w1' GiB, expected 15.59"
echo "    $b1"

echo "  run 2 (CONTROL): same command, a file sized IQ2_XXS — the budget must FOLLOW it down"
run_coord --llama-gguf "$SMALL"
b2=$(budget_line); w2=$(weights_gib)
[ "$w2" = "7.98" ] || fail "run 2: planner saw 'weights $w2' GiB, expected 7.98 — the budget is not reading the file at all"
case "$b2" in
    *"manifest quant IQ2_XXS"*) : ;;
    *) fail "run 2: the source line does not name the quant it recognised: $b2" ;;
esac
[ "$w1" != "$w2" ] || fail "control: both runs budgeted $w1 GiB — the file is being ignored"
echo "    $b2"

echo "  run 3: --quant that contradicts the file — refuse, out loud, naming both"
# Contract since 9f0aa97 (2026-08-19): a flag/file precision mismatch is a
# startup REFUSAL, not a warn-and-continue. The served precision goes on the
# wire (stats, metering, the roster's quant broadcast) — running mislabeled
# would price and report a precision nobody is serving. The budget line with
# its WARNING still prints first (the budget half of the diagnosis), then the
# coordinator stops, so there must be NO scheduler line at all.
run_coord --llama-gguf "$SMALL" --quant Q4_K_M
b3=$(budget_line); w3=$(weights_gib)
case "$b3" in
    *WARNING*Q4_K_M*) : ;;
    *) fail "run 3: the budget line never said the flag disagreed: $b3" ;;
esac
grep -q "Refusing to label this run" "$LOG" \
    || fail "run 3: mismatch was not refused — a mislabeled precision would reach stats/metering"
grep -m1 "Refusing to label" "$LOG" | grep -q "Q4_K_M" \
    || fail "run 3: the refusal does not name the offending --quant"
grep -m1 "Refusing to label" "$LOG" | grep -q "IQ2_XXS" \
    || fail "run 3: the refusal does not name the precision actually on disk"
[ -z "$w3" ] || fail "run 3: refused and yet scheduled anyway (weights $w3 GiB) — the refusal is not a refusal"
echo "    $(grep -m1 'Refusing to label' "$LOG")"

echo "  run 4: a size in no menu — the real bytes, flagged as unrecognised"
run_coord --llama-gguf "$ODD"
b4=$(budget_line); w4=$(weights_gib)
[ "$w4" = "12.11" ] || fail "run 4: planner saw 'weights $w4' GiB, expected 12.11"
case "$b4" in
    *WARNING*"matches no quantization"*) : ;;
    *) fail "run 4: an unrecognised size was accepted without a word: $b4" ;;
esac
echo "    $b4"

cleanup
rm -rf "$TMP"
echo "BUDGET_GATE_OK"
