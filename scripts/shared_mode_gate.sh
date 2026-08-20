#!/usr/bin/env bash
# G-SHARED — the "a buyer's prompt is not readable on the host" gate.
#
# The bar is stated in docs/threat-model-shared-compute-2026-08.md: a provider
# using ORDINARY means — reading logs, capturing loopback, opening temp files,
# looking at a command line — finds nothing. Not "safe against a debugger".
#
# Four claims, from docs/shared-mode-plan-2026-08.md:
#
#   G-SHARED-1  the coordinator↔engine link is not readable traffic
#   G-SHARED-2  IDLETOKEN_LLAMA_ARGS is ignored, and SAYS it is ignored
#   G-SHARED-3  an engine binary that is not the shipped one refuses platform work
#   G-SHARED-4  after a request, the prompt is nowhere on disk
#
# **Every check carries its own positive control.** Most of them are the local
# (non-shared) run of the same code: it must come out the OTHER way. A "not
# found" from a checker never shown able to find anything is worth nothing —
# this repo has twice shipped exactly that (CLAUDE.md, "known traps:
# methodology"), and it happened a third time while writing this script: the
# first disk sweep used `grep -r`
# on /tmp, which BSD grep will not follow through the symlink, so a planted
# control string came back clean.
#
# Usage:  scripts/shared_mode_gate.sh <small-model.gguf> [engine-bin]
# Contract: prints exactly one G_SHARED_(OK|FAIL|SKIP) line last.
set -u
cd "$(dirname "$0")/.." || exit 1

# Same small model the other smoke gates use, so a testbed configured once
# (scripts/testbed.env) needs nothing extra for this one.
GGUF=${1:-${IDLETOKEN_SMOKE_GGUF:-}}
ENGINE=${2:-vendor/llama.cpp/build/bin/llama-server}
# Unique per run, and that is load-bearing. A fixed marker is also a string
# literal in this script and in the harness, so the disk sweep matches any COPY
# of our own source — which is exactly what happened the first time it ran
# under a stale /tmp/pubtest checkout: two "leaks" that were our own comments.
# A marker minted here cannot pre-exist anywhere.
MARKER="IdleTokenCanary$(date +%s)x$$"

FAILED=""
ok()  { echo "  [ok] $*"; }
bad() { echo "  [BAD] $*"; FAILED="${FAILED}
  - $*"; }

[ -n "$GGUF" ] || { echo "G_SHARED_SKIP: no GGUF given (usage: $0 <small-model.gguf> [engine-bin])"; exit 0; }
[ -f "$GGUF" ] || { echo "G_SHARED_SKIP: $GGUF is not a file"; exit 0; }
[ -x "$ENGINE" ] || { echo "G_SHARED_SKIP: no engine at $ENGINE — build the pinned llama.cpp first"; exit 0; }

# --- G-SHARED-1 / -2 / part of -5: the sidecar harness --------------------
# It starts the engine twice, shared and local, and every claim it makes about
# the shared run is paired with the opposite claim about the local one.
echo "G-SHARED-1/2: transport, /slots, and the environment"
if ! make -s sidecartest GGUF="$GGUF" ENGINE="$ENGINE" MARKER="$MARKER" 2>&1 | tee /tmp/g_shared_sidecar.txt | sed 's/^/  /'; then
    bad "the sidecar harness did not run"
fi
grep -q '^SIDECAR_OK$' /tmp/g_shared_sidecar.txt || bad "sidecar harness: see the [BAD] lines above"

# G-SHARED-2 wants BOTH halves: the variable ignored, and the log SAYING so.
# Testing only "the prompt did not appear" cannot tell "we ignored it" from
# "we never read the variable in the first place".
grep -q "IDLETOKEN_LLAMA_ARGS ignored" /tmp/g_shared_sidecar.txt \
    && ok "shared mode says out loud that IDLETOKEN_LLAMA_ARGS was ignored" \
    || bad "no line explaining that IDLETOKEN_LLAMA_ARGS was ignored — 'nothing leaked' and 'nothing was read' look identical without it"

# --- G-SHARED-3: the engine binary digest --------------------------------
# The verdict logic is judged against a PUBLISHED digest (the FIPS 180-4 vector
# for "abc"), not against our own hasher's output, inside coord --selftest.
echo "G-SHARED-3: engine binary integrity"
if [ -x ./idletoken-coord ]; then
    st=$(./idletoken-coord --selftest 2>&1 | grep "engine integrity")
    echo "$st" | sed 's/^/  /'
    echo "$st" | grep -q "FAIL" && bad "engine-integrity selftest failed"
    n=$(echo "$st" | grep -c "PASS")
    [ "$n" -ge 5 ] || bad "expected 5 engine-integrity assertions, saw $n (did they stop running?)"
else
    bad "no ./idletoken-coord built — cannot judge the integrity check"
fi

# --- G-SHARED-4: the prompt is nowhere on disk ---------------------------
# CONTROL FIRST. Plant the marker, prove the sweep finds it, remove it, then
# run the sweep for real. Absolute paths, not /tmp: on macOS that is a symlink
# and `grep -r` walks straight past it, which silently turns this whole gate
# green (found while writing it).
echo "G-SHARED-4: prompt text on disk"
SWEEP_DIRS="$HOME/.idletoken /private/tmp /private/var/tmp"
[ -d "$HOME/Library/Caches" ] && SWEEP_DIRS="$SWEEP_DIRS $HOME/Library/Caches"
sweep() {
    local total=0
    for d in $SWEEP_DIRS; do
        [ -d "$d" ] || continue
        local n
        n=$(grep -rlI "$MARKER" "$d" 2>/dev/null | wc -l | tr -d ' ')
        total=$((total + n))
    done
    echo "$total"
}
CONTROL="/private/tmp/g_shared_control.txt"
printf '%s\n' "$MARKER" > "$CONTROL"
if [ "$(sweep)" -ge 1 ]; then
    ok "the sweep can find a planted marker"
else
    bad "the sweep found nothing even with the marker planted — it proves nothing about the real run"
fi
rm -f "$CONTROL"
hits=$(sweep)
if [ "$hits" -eq 0 ]; then
    ok "no prompt text on disk after a shared-mode request"
else
    bad "$hits file(s) contain the prompt marker:"
    for d in $SWEEP_DIRS; do grep -rlI "$MARKER" "$d" 2>/dev/null | sed 's/^/      /'; done
fi

# --- what this run did NOT judge -----------------------------------------
# Naming the gaps is part of the result. A gate that stays quiet about what it
# skipped reads as "all covered".
echo "not judged here:"
echo "  - the refusal at the request boundary (G-SHARED-3, second half): needs a"
echo "    coordinator that actually starts an engine, which a machine with no"
echo "    usable compute memory refuses to do. Run it on a node that has a"
echo "    supported GPU and enough usable compute memory for the model."
echo "  - Windows: AF_UNIX exists in winsock since Windows 10 1803 and the code"
echo "    is written for it, but it has never been run there."
echo "  - a real loopback capture. The harness proves something stronger on the"
echo "    machines it runs on — the engine holds no listening TCP socket at all,"
echo "    so there are no loopback packets to capture — but tcpdump on a real"
echo "    provider machine is still the check a sceptic will ask for."

if [ -n "$FAILED" ]; then
    echo "G_SHARED_FAIL:$FAILED"
    exit 1
fi
echo "G_SHARED_OK"
