#!/usr/bin/env bash
# Stage the freshly built engine binaries as Tauri sidecars.
#
# WHY THIS IS ITS OWN SCRIPT (2026-08-04). The staging used to live ONLY inside
# build_client_release.sh, i.e. it happened when someone cut a release. But the
# P-gates (P1–P6, and therefore G-FINAL) drive the DEBUG client, which loads its
# sidecars from the very same directory. So the product gates were exercising
# whatever engine binaries the last release build happened to leave there.
#
# On 2026-08-04 that was six days old — old enough to predate protocol v5. The
# symptom was not a version warning: a freshly installed Windows worker joined
# the coordinator and the coordinator logged `recv HELLO: Protocol error` in a
# loop while the joiner's engine exited(1) with nothing on stderr. Same shape as
# the stale dist\ bundle found the same day: a gate certifying the wrong artifact.
#
# Usage:  scripts/stage_sidecars.sh          # on the node that runs the client
# Contract: last line STAGE_OK or STAGE_FAIL: <reason>.
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD

fail() { echo "STAGE_FAIL: $*"; exit 1; }

# rustup installs into ~/.cargo/bin, which a NON-INTERACTIVE ssh shell does not
# have on PATH (same trap as nvcc under /usr/local/cuda/bin). Without this the
# script fails with "no rustc" on a machine that plainly has it.
command -v rustc >/dev/null 2>&1 || PATH="$HOME/.cargo/bin:$PATH"
command -v rustc >/dev/null 2>&1 || fail "no rustc (looked on PATH and in ~/.cargo/bin)"
TRIPLE=$(rustc -vV | awk '/^host:/{print $2}')
[ -n "$TRIPLE" ] || fail "could not read the rustc host triple"

mkdir -p client/src-tauri/binaries

stage() {  # stage <src> <sidecar-name>
    [ -x "$1" ] || fail "missing engine binary $1 (build it first: make all / make -f Makefile.platform)"
    cp -f "$1" "client/src-tauri/binaries/$2-$TRIPLE" || fail "could not stage $2"
    # Print the mtime: the whole class of bug this script exists for is
    # invisible unless you look at dates.
    printf '  staged %-22s <- %s  (%s)\n' "$2" "$1" "$(date -r "$1" '+%Y-%m-%d %H:%M' 2>/dev/null || echo '?')"
}

stage "$ROOT/idletoken-worker" idletoken-worker
stage "$ROOT/idletoken-coord"  idletoken-coord

# llama-server (v2, WS-D): the coordinator's inference engine in llamacpp
# single-machine mode (coord --llama-server-bin). The client resolves it NEXT TO
# ITSELF at runtime, so it must be staged wherever the coord sidecar is staged.
# A missing engine is a hard failure, not a skip: a client without llama-server
# would offer "run this GGUF here" and then fail at the exact moment the user
# commits — the same "gate certifies the wrong artifact" shape this script
# exists to prevent.
LLAMA_SERVER="$ROOT/vendor/llama.cpp/build/bin/llama-server"
[ -x "$LLAMA_SERVER" ] || fail "missing engine binary $LLAMA_SERVER — build the pinned llama.cpp first: scripts/build_llamacpp.sh (Windows: scripts\\build_llamacpp_win.bat)"
stage "$LLAMA_SERVER" llama-server
GGML_RPC_SERVER="$ROOT/vendor/llama.cpp/build/bin/ggml-rpc-server"
[ -x "$GGML_RPC_SERVER" ] || fail "missing cluster engine binary $GGML_RPC_SERVER — build the pinned llama.cpp with GGML_RPC=ON"
stage "$GGML_RPC_SERVER" ggml-rpc-server
# The agent lives under build/ when made via Makefile.platform; accept either.
if [ -x "$ROOT/build/idletoken-platform-agent" ]; then
    stage "$ROOT/build/idletoken-platform-agent" idletoken-platform-agent
elif [ -x "$ROOT/idletoken-platform-agent" ]; then
    stage "$ROOT/idletoken-platform-agent" idletoken-platform-agent
else
    echo "  WARNING: no idletoken-platform-agent built — the Platform panel will not start it"
fi

# A DEBUG Tauri build resolves sidecars NEXT TO ITSELF (target/debug/<name>),
# not from src-tauri/binaries/ — that path is what the BUNDLER reads. The P
# gates and the install walkthrough all drive the debug client, so staging only
# binaries/ leaves them on whatever engine was there before.
#
# That is not hypothetical: on 2026-08-04 target/debug/ still held 07-29
# binaries, six days old and predating protocol v5. A freshly installed Windows
# worker (v5) then met a v4 coordinator and the coordinator logged
# `recv HELLO: Protocol error` in a loop while the joiner exited(1) silently.
# Staging binaries/ alone did NOT fix it — the first fix aimed at the wrong
# directory, which is exactly why both are done here now.
DBG="client/src-tauri/target/debug"
if [ -d "$DBG" ]; then
    for b in idletoken-worker idletoken-coord; do
        cp -f "$ROOT/$b" "$DBG/$b" || fail "could not stage $b into $DBG"
        printf '  staged %-22s -> %s\n' "$b" "$DBG/"
    done
    # The debug client resolves llama-server next to itself too (engine.rs
    # llamacpp_serve). Same hard-failure rule as above.
    cp -f "$LLAMA_SERVER" "$DBG/llama-server" || fail "could not stage llama-server into $DBG"
    printf '  staged %-22s -> %s\n' llama-server "$DBG/"
    cp -f "$GGML_RPC_SERVER" "$DBG/ggml-rpc-server" || fail "could not stage ggml-rpc-server into $DBG"
    printf '  staged %-22s -> %s\n' ggml-rpc-server "$DBG/"
    if [ -x "$ROOT/build/idletoken-platform-agent" ]; then
        cp -f "$ROOT/build/idletoken-platform-agent" "$DBG/idletoken-platform-agent" || true
        printf '  staged %-22s -> %s\n' idletoken-platform-agent "$DBG/"
    fi
else
    echo "  note: no $DBG yet (debug client never built here) — skipping"
fi

echo STAGE_OK
