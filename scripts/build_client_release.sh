#!/usr/bin/env bash
# Build the installable desktop client on a Linux compute node (.deb + .rpm).
#
# The engine and the client are two processes (design philosophy 17), but the
# *installer* has to carry both: Tauri ships the engine binaries as sidecars.
# This script stages those sidecars under the target-triple names Tauri expects
# and then runs the real bundler. It is the E3-for-the-product half: after this,
# a user installs one package and has a working node — no repo, no toolchain.
#
# Usage:  scripts/build_client_release.sh            # bundle everything
#         scripts/build_client_release.sh --no-bundle  # just compile the shell
#
# Contract: last line is CLIENT_RELEASE_OK (artifacts listed above it) or
# CLIENT_RELEASE_FAIL: <reason>.
set -u

cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
export PATH="$HOME/.cargo/bin:$PATH"
export CI=true          # pnpm refuses to touch node_modules without a TTY otherwise

BUNDLE=1
[ "${1:-}" = "--no-bundle" ] && BUNDLE=0

fail() { echo "CLIENT_RELEASE_FAIL: $1"; exit 1; }

command -v cargo >/dev/null 2>&1 || fail "no cargo on PATH (need the Rust toolchain)"
command -v pnpm  >/dev/null 2>&1 || fail "no pnpm on PATH (need node + pnpm for the frontend)"

TRIPLE=$(rustc -vV | awk '/^host:/{print $2}')
[ -n "$TRIPLE" ] || fail "could not read the rustc host triple"
echo "target triple: $TRIPLE"

# --- stage engine sidecars ------------------------------------------------
# scripts/stage_sidecars.sh is the single source of truth for what a client
# ships (coord, worker, platform-agent, and — since the v2 llama.cpp pivot —
# the pinned idletoken-server + idletoken-rpc-server). This script used to carry its own copy of the
# staging and it drifted: it kept shipping bundles without idletoken-server after
# stage_sidecars.sh already required it.
out=$(scripts/stage_sidecars.sh) || { echo "$out"; fail "sidecar staging failed"; }
echo "$out" | sed 's/^/  /'

# --- the coordinator must carry the pinned platform verify key --------------
# Without it, nobody who installs this package can ever switch sharing on
# (overflow.c RULE 3). Every release up to 0.1.5 shipped unpinned because
# nothing set the variable; 0.1.19's first macOS bundle shipped unpinned even
# though something did, because make had no reason to recompile overflow.c.
# So ask the binary rather than the build environment. Its only network target
# refuses instantly: a PINNED coordinator prints "overflow: on", an unpinned one
# refuses before any network I/O. Demanding the positive line means a probe that
# stops producing overflow output at all fails the build instead of passing it.
if [ -x ./idletoken-coord ]; then
    if ! ./idletoken-coord --overflow-url http://127.0.0.1:1 --overflow-key pin-probe \
           --model pin-probe-sentinel 2>&1 | grep -q "overflow: on"; then
        fail "idletoken-coord has no pinned platform verify key — rebuild with IDLETOKEN_PLATFORM_VERIFY_KEY_B64 set (scripts/platform-verify-key.b64 is the default)"
    fi
fi

# --- stage licences -----------------------------------------------------------
# The sidecars above carry vendored third-party code (ds4 = MIT, rax = BSD
# 3-Clause), and both licences require the notice to travel with a BINARY
# distribution; Apache-2.0 section 4(d) says the same about our own NOTICE. The
# installer is a binary distribution, so the texts have to be inside it —
# tauri.conf.json `bundle.resources` picks up this directory.
#
# Staged (copied) rather than referenced across the tree: Tauri resolves
# resources relative to src-tauri, and a `../../LICENSE` that silently resolves
# to nothing would ship an installer with no licences and no error. Copying
# fails loudly here instead.
LIC=client/src-tauri/licenses
mkdir -p "$LIC"
cp -f "$ROOT/LICENSE" "$LIC/LICENSE.txt"       || fail "could not stage LICENSE"
cp -f "$ROOT/NOTICE"  "$LIC/NOTICE.txt"        || fail "could not stage NOTICE"
cp -f "$ROOT/vendor/ds4/LICENSE" "$LIC/ds4-MIT.txt" || fail "could not stage the ds4 licence"
# llama.cpp is MIT and is shipped as the idletoken-server sidecar (v2 pivot).
cp -f "$ROOT/vendor/llama.cpp/LICENSE" "$LIC/llamacpp-MIT.txt" \
    || fail "could not stage the llama.cpp licence"
echo "  staged licences -> $LIC"

# --- build ----------------------------------------------------------------
cd client || fail "no client/ directory"
pnpm install >/tmp/client-release-install.log 2>&1 || fail "pnpm install failed (see /tmp/client-release-install.log)"

if [ "$BUNDLE" = 0 ]; then
    (cd src-tauri && cargo build --release) || fail "cargo build --release failed"
    echo "shell binary: $ROOT/client/src-tauri/target/release/idletoken-client"
    echo CLIENT_RELEASE_OK
    exit 0
fi

# `tauri build` runs beforeBuildCommand (pnpm build:release) itself — that is
# what injects the production platform URL into the shipped frontend.
#
# Linux releases are native package-manager installers only. Keep this list
# fail closed so a stale environment cannot quietly resurrect AppImage or an
# updater artifact in a public release.
BUNDLES="${IDLETOKEN_BUNDLES:-deb,rpm}"
case ",$BUNDLES," in
    ,deb,|,rpm,|,deb,rpm,|,rpm,deb,) ;;
    *) fail "IDLETOKEN_BUNDLES must contain only deb and/or rpm (got '$BUNDLES')" ;;
esac
# ⚠ not `| tail`: the pipe exit code is tail's, and a bundler that failed
# AFTER producing the .deb sailed through as CLIENT_RELEASE_OK (hit 2026-08-15;
# same trap as the repo-wide "never read an exit code through a pipe" rule).
BUILD_LOG=$(mktemp /tmp/idletoken-tauri-build.XXXXXX)
if ! pnpm tauri build --bundles "$BUNDLES" > "$BUILD_LOG" 2>&1; then
    tail -30 "$BUILD_LOG"; rm -f "$BUILD_LOG"
    fail "tauri build failed"
fi
tail -30 "$BUILD_LOG"; rm -f "$BUILD_LOG"

# --- report artifacts -----------------------------------------------------
BDIR="src-tauri/target/release/bundle"
[ -d "$BDIR" ] || fail "no bundle directory at $BDIR"
found=0
echo "--- artifacts ---"
while IFS= read -r f; do
    found=1
    printf '%s  %s  %s\n' "$(sha256sum "$f" | cut -c1-16)" "$(du -h "$f" | cut -f1)" "$ROOT/client/$f"
done < <(find "$BDIR" -type f \( -name '*.deb' -o -name '*.rpm' \) | sort)
[ "$found" = 1 ] || fail "bundler produced no Linux package"

# No automatic provenance, signature, updater archive or feed is emitted here.
# The release contract is the native installer files listed above, and only
# those files are uploaded to GitHub.

echo CLIENT_RELEASE_OK
