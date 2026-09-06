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

# SHA-256 of a file, bare hex. macOS ships shasum, Linux sha256sum, and a
# Windows git-bash may have either — try all three rather than assume.
sha256_of() {
    if command -v shasum >/dev/null 2>&1;    then shasum -a 256 "$1" | awk '{print $1}'
    elif command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
    elif command -v openssl >/dev/null 2>&1;   then openssl dgst -sha256 "$1" | awk '{print $NF}'
    else return 1
    fi
}

stage() {  # stage <src> <sidecar-name>
    [ -x "$1" ] || fail "missing engine binary $1 (build it first: make all / make -f Makefile.platform)"
    dst="client/src-tauri/binaries/$2-$TRIPLE"
    cp -f "$1" "$dst" || fail "could not stage $2"
    # Record what these bytes are, beside them. In shared mode the coordinator
    # compares the engine against this before it will take other people's work
    # (docs/shared-mode-plan-2026-08.md P0-3) — a provider who swaps in a
    # self-built engine that logs prompts stops being dispatched to. Written
    # for EVERY sidecar, not just the engine: the file costs nothing, and a
    # rule with exceptions is a rule someone forgets to extend.
    #
    # `shasum -a 256` format on purpose, so a provider can reproduce the
    # number with one command instead of trusting our word for it.
    h=$(sha256_of "$dst") || fail "no sha256 tool (shasum/sha256sum/openssl) — the shared-mode integrity check has nothing to compare against"
    printf '%s  %s\n' "$h" "$2-$TRIPLE" > "$dst.sha256"
    # Print the mtime: the whole class of bug this script exists for is
    # invisible unless you look at dates.
    printf '  staged %-22s <- %s  (%s)  %s\n' "$2" "$1" \
        "$(date -r "$1" '+%Y-%m-%d %H:%M' 2>/dev/null || echo '?')" "${h%"${h#??????????}"}…"
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
# Staged under OUR name (2026-08-15): upstream builds "llama-server", but what
# a user sees in Task Manager, in a firewall prompt or in a crash dialog must
# say IdleToken — "llama-server.exe wants network access" reads as though some
# other product installed itself. The binary is unchanged; only the file name
# is ours, which the MIT licence allows (attribution stays in About + NOTICE).
stage "$LLAMA_SERVER" idletoken-server
GGML_RPC_SERVER="$ROOT/vendor/llama.cpp/build/bin/ggml-rpc-server"
[ -x "$GGML_RPC_SERVER" ] || fail "missing cluster engine binary $GGML_RPC_SERVER — build the pinned llama.cpp with GGML_RPC=ON"
stage "$GGML_RPC_SERVER" idletoken-rpc-server
# The agent lives under build/ when made via Makefile.platform; accept either.
if [ -x "$ROOT/build/idletoken-platform-agent" ]; then
    stage "$ROOT/build/idletoken-platform-agent" idletoken-platform-agent
elif [ -x "$ROOT/idletoken-platform-agent" ]; then
    stage "$ROOT/idletoken-platform-agent" idletoken-platform-agent
else
    echo "  WARNING: no idletoken-platform-agent built — the Platform panel will not start it"
fi

echo STAGE_OK
