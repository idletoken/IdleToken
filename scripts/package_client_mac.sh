#!/usr/bin/env bash
# Build the installable macOS client (.dmg) — WS-E E3.
#
# One command on the Mac: build the engine sidecars, stage them, bundle with
# Tauri, and then verify what actually shipped (sidecars + engine pin inside the
# mounted dmg). Verification is part of the contract, not a courtesy: a dmg that
# carries a stale idletoken-server looks identical to a good one from the
# outside.
#
# There is no update artifact and no signing step since 2026-09-02: the product
# has no in-app updater, so a release is just the installer.
#
# Usage:  scripts/package_client_mac.sh
# Env:    IDLETOKEN_MAC_BUNDLES            (default "dmg")
#
# Contract: last line is CLIENT_MAC_OK (artifacts listed above it) or
# CLIENT_MAC_FAIL: <reason>. Idempotent: safe to re-run; every step either
# rebuilds or reuses, nothing accumulates.
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD
export PATH="$HOME/.cargo/bin:$PATH"
export CI=true   # pnpm refuses to touch node_modules without a TTY otherwise

fail() { echo "CLIENT_MAC_FAIL: $1"; exit 1; }

[ "$(uname -s)" = "Darwin" ] || fail "this builds the macOS bundle; run it on a Mac"
command -v cargo >/dev/null 2>&1 || fail "no cargo on PATH (need the Rust toolchain)"
command -v pnpm  >/dev/null 2>&1 || fail "no pnpm on PATH (need node + pnpm for the frontend)"
TRIPLE=$(rustc -vV | awk '/^host:/{print $2}')
[ -n "$TRIPLE" ] || fail "could not read the rustc host triple"
echo "target triple: $TRIPLE"

# --- engine binaries (idempotent: make decides what to rebuild) -------------
# Pin the platform verify key by default (scripts/platform-verify-key.b64, the
# PUBLIC ed25519 verify half). An unpinned coord refuses to enable sharing on
# every user machine (overflow.c RULE 3, found 2026-08-21); the environment
# still wins for dev-gateway builds.
if [ -z "${IDLETOKEN_PLATFORM_VERIFY_KEY_B64:-}" ] && [ -f scripts/platform-verify-key.b64 ]; then
    IDLETOKEN_PLATFORM_VERIFY_KEY_B64=$(cat scripts/platform-verify-key.b64)
fi
IDLETOKEN_PLATFORM_VERIFY_KEY_B64="${IDLETOKEN_PLATFORM_VERIFY_KEY_B64:-}" \
make coord worker >/dev/null || fail "make coord worker failed"
make -f Makefile.platform >/dev/null || fail "make -f Makefile.platform failed"
# Ask the binary whether the pin actually took. Passing the variable is not the
# same as it reaching the object file — the first 0.1.19 macOS bundle set it and
# still shipped an unpinned coordinator, because make saw no reason to recompile
# overflow.c (fixed by COORD_PIN_STAMP in the Makefile; this is the check that
# would have caught it). Grepping the exe for the key is a documented dead end,
# so use the same offline probe the Windows lane uses: its only network target
# refuses instantly, a PINNED build prints "overflow: on", an unpinned one
# refuses before any network I/O. Demand the positive line, so a probe that
# stops producing overflow output at all fails the build.
if ! ./idletoken-coord --overflow-url http://127.0.0.1:1 --overflow-key pin-probe \
       --model pin-probe-sentinel 2>&1 | grep -q "overflow: on"; then
    fail "idletoken-coord has no pinned platform verify key — sharing could never be switched on by anyone who installs this build (expected the pin from scripts/platform-verify-key.b64)"
fi
# The pinned llama.cpp is a separate, expensive build; stage_sidecars.sh below
# hard-fails with the right instructions if it is missing. Do not build it here.

# --- stage sidecars (single source of truth; do not duplicate its logic) ----
out=$(scripts/stage_sidecars.sh) || { echo "$out"; fail "sidecar staging failed"; }
echo "$out" | sed 's/^/  /'

# --- stage licences (same obligation as build_client_release.sh) ------------
LIC=client/src-tauri/licenses
mkdir -p "$LIC"
cp -f "$ROOT/LICENSE" "$LIC/LICENSE.txt"            || fail "could not stage LICENSE"
cp -f "$ROOT/NOTICE"  "$LIC/NOTICE.txt"             || fail "could not stage NOTICE"
cp -f "$ROOT/vendor/ds4/LICENSE" "$LIC/ds4-MIT.txt" || fail "could not stage the ds4 licence"
awk '/^\/\* Rax/,/^ \*\/$/' "$ROOT/vendor/ds4/rax.c" > "$LIC/rax-BSD-3-Clause.txt" \
    || fail "could not extract the rax licence"
grep -q "Redistribution and use in source and binary forms" "$LIC/rax-BSD-3-Clause.txt" \
    || fail "extracted rax licence does not contain the BSD terms"
# llama.cpp is MIT and is shipped as the idletoken-server sidecar.
cp -f "$ROOT/vendor/llama.cpp/LICENSE" "$LIC/llamacpp-MIT.txt" \
    || fail "could not stage the llama.cpp licence"

# --- the signing key: READ-ONLY, NON-REGENERABLE ----------------------------
# No signing key (2026-09-02, user ruling): `createUpdaterArtifacts` is false,
# so `tauri build` produces only the .dmg and needs no minisign key. Updating
# means downloading the current installer.

# --- release preflight ------------------------------------------------------
# Refuses a dirty tree, so the provenance record below cannot name a commit
# that is not what was built. (It used to also prove the signing key was the
# RIGHT key; there is no signing key any more.)
# shellcheck disable=SC1091
. "$ROOT/scripts/release-provenance-lib.sh"
rp_preflight "macos" || fail "release preflight refused this build (see above)"

# --- build ------------------------------------------------------------------
cd client || fail "no client/ directory"
pnpm install >/tmp/client-mac-install.log 2>&1 || fail "pnpm install failed (see /tmp/client-mac-install.log)"
BUNDLES="${IDLETOKEN_MAC_BUNDLES:-dmg}"
# `tauri build` runs beforeBuildCommand (pnpm build:release) itself — that is
# what injects the production platform URL into the shipped frontend.
pnpm tauri build --bundles "$BUNDLES" 2>&1 | tail -25 || fail "tauri build failed"

BDIR=src-tauri/target/release/bundle
DMG=$(ls -t "$BDIR"/dmg/IdleToken_*.dmg 2>/dev/null | head -1)
[ -n "$DMG" ] || fail "no dmg produced under $BDIR/dmg"

# --- verify 1: sidecars + engine pin INSIDE the dmg -------------------------
# Verify the artifact users get, not the intermediate .app in the build tree.
MNT=$(mktemp -d /tmp/idletoken-dmg.XXXXXX)
hdiutil attach -readonly -nobrowse -mountpoint "$MNT" "$DMG" >/dev/null \
    || fail "could not mount $DMG"
trap 'hdiutil detach "$MNT" >/dev/null 2>&1; rmdir "$MNT" 2>/dev/null' EXIT
MACOS_DIR="$MNT/IdleToken.app/Contents/MacOS"
# idletoken-rpc-server is the other half of the engine: idletoken-server serves, the rpc
# server is what this machine runs when it joins someone else's cluster. Ship
# one without the other and the app offers a cluster mode it cannot join.
for b in idletoken-client idletoken-coord idletoken-worker idletoken-platform-agent idletoken-server idletoken-rpc-server; do
    [ -x "$MACOS_DIR/$b" ] || fail "dmg is missing $b in Contents/MacOS (bundler shipped an incomplete app)"
done
PIN_SHA=$(awk 'NR==1{print $2}' "$ROOT/scripts/llamacpp-patches/UPSTREAM")
[ -n "$PIN_SHA" ] || fail "cannot read the engine pin from scripts/llamacpp-patches/UPSTREAM"
# The bundled engine carries the PRODUCT name (idletoken-server), not the
# upstream one — probing llama-server here failed on a file that does not
# exist and read as "does not run" (rename leftover, caught 2026-08-20).
VERSION_LINE=$("$MACOS_DIR/idletoken-server" --version 2>&1 | grep -m1 'version:') \
    || fail "bundled idletoken-server does not run"
case "$VERSION_LINE" in
    *"${PIN_SHA:0:7}"*) echo "  bundled idletoken-server: $VERSION_LINE (matches pin ${PIN_SHA:0:7})" ;;
    *) fail "bundled idletoken-server is '$VERSION_LINE', not the pinned ${PIN_SHA:0:7} — a stale engine got staged" ;;
esac
for r in LICENSE.txt NOTICE.txt ds4-MIT.txt rax-BSD-3-Clause.txt llamacpp-MIT.txt; do
    [ -f "$MNT/IdleToken.app/Contents/Resources/licenses/$r" ] \
        || fail "dmg is missing licence $r in Contents/Resources/licenses"
done
hdiutil detach "$MNT" >/dev/null 2>&1
rmdir "$MNT" 2>/dev/null
trap - EXIT

# --- report artifacts -------------------------------------------------------
echo "--- artifacts ---"
for f in "$DMG"; do
    printf '%s  %s  %s\n' "$(shasum -a 256 "$f" | cut -c1-16)" "$(du -h "$f" | cut -f1)" "$ROOT/client/$f"
done

# No automatic provenance, signature, updater archive or feed is emitted here.
# The GitHub release contains the .dmg installer only.

# --- restore the non-release client/dist ------------------------------------
# `tauri build` ran beforeBuildCommand = `pnpm build:release`, which leaves a
# dist carrying the PRODUCTION platform URL; the acceptance gates serve that
# same dist to the debug shell. Same restore as build_client_release.sh.
pnpm build > /tmp/client-mac-restore.log 2>&1 \
    || fail "bundle is built, but restoring the non-release client/dist failed (see /tmp/client-mac-restore.log)"
echo "restored client/dist to the non-release build (acceptance gates use it)"

echo CLIENT_MAC_OK
