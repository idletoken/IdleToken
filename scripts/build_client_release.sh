#!/usr/bin/env bash
# Build the installable desktop client on a Linux compute node (.deb + AppImage).
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
# Names must match tauri.conf.json externalBin + the triple suffix, else the
# bundler silently ships nothing and the app fails at spawn time.
stage() {  # stage <src> <sidecar-name>
    [ -x "$1" ] || fail "missing engine binary $1 (build it first: make all / make -f Makefile.platform)"
    cp -f "$1" "client/src-tauri/binaries/$2-$TRIPLE" || fail "could not stage $2"
    echo "  staged $2 <- $1"
}
mkdir -p client/src-tauri/binaries
stage "$ROOT/idletoken-worker"               idletoken-worker
stage "$ROOT/idletoken-coord"                idletoken-coord
stage "$ROOT/build/idletoken-platform-agent" idletoken-platform-agent

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
# deb+rpm only by default: the AppImage target downloads linuxdeploy from
# GitHub release assets at bundle time and that download does not complete on
# this network (same failure as the Windows NSIS/WiX fetch — Tauri's downloader
# gives up with `timeout: global` while the same URL fetches fine otherwise).
# Set IDLETOKEN_BUNDLES=deb,rpm,appimage plus a working
# TAURI_BUNDLER_TOOLS_GITHUB_MIRROR to try it.
BUNDLES="${IDLETOKEN_BUNDLES:-deb,rpm}"
pnpm tauri build --bundles "$BUNDLES" 2>&1 | tail -30 || fail "tauri build failed"

# --- report artifacts -----------------------------------------------------
BDIR="src-tauri/target/release/bundle"
[ -d "$BDIR" ] || fail "no bundle directory at $BDIR"
found=0
echo "--- artifacts ---"
while IFS= read -r f; do
    found=1
    printf '%s  %s  %s\n' "$(sha256sum "$f" | cut -c1-16)" "$(du -h "$f" | cut -f1)" "$ROOT/client/$f"
done < <(find "$BDIR" -type f \( -name '*.deb' -o -name '*.AppImage' -o -name '*.rpm' \) | sort)
[ "$found" = 1 ] || fail "bundler produced no .deb/.AppImage"

# Put the tree back the way the acceptance gates expect it. `tauri build` ran
# beforeBuildCommand = `pnpm build:release`, which overwrites client/dist with a
# bundle carrying the PRODUCTION platform URL. The P-gates serve that same
# client/dist to the debug shell, so leaving it in place silently flips P2_auth
# from offline local identity to cloud auth against the live platform — the gate
# then fails with `auth.err.network` and nothing in the diff explains why.
pnpm build > /tmp/client-release-restore.log 2>&1 \
    || fail "release bundle is built, but restoring the non-release client/dist failed (see /tmp/client-release-restore.log)"
echo "restored client/dist to the non-release build (acceptance gates use it)"

echo CLIENT_RELEASE_OK
