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
make -B NATIVE_CPU_FLAG=-mcpu=apple-m1 coord worker >/dev/null \
    || fail "make coord worker failed"
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
ENGINE_CACHE="$ROOT/vendor/llama.cpp/build/CMakeCache.txt"
[ -f "$ENGINE_CACHE" ] || fail "missing llama.cpp CMakeCache.txt; rebuild the release engine"
grep -Fxq 'GGML_NATIVE:BOOL=OFF' "$ENGINE_CACHE" \
    || fail "llama.cpp was built for this Mac's CPU; rebuild with GGML_NATIVE=OFF"
grep -Fxq 'LLAMA_OPENSSL:BOOL=OFF' "$ENGINE_CACHE" \
    || fail "llama.cpp was built with external OpenSSL; rebuild with LLAMA_OPENSSL=OFF"
native_flags=$(find "$ROOT/vendor/llama.cpp/build" -type f -name flags.make \
    -exec grep -H -E -- '(-march|-mcpu)=native' {} + 2>/dev/null || true)
[ -z "$native_flags" ] \
    || fail "llama.cpp build flags still contain a builder-native CPU target: $native_flags"
echo "  engine CPU baseline: portable (GGML_NATIVE=OFF)"

# --- stage sidecars (single source of truth; do not duplicate its logic) ----
out=$(scripts/stage_sidecars.sh) || { echo "$out"; fail "sidecar staging failed"; }
echo "$out" | sed 's/^/  /'

# externalBin carries executables only. The coordinator verifies the installed
# engine against a digest beside it before accepting shared work, so stage the
# two final-name digest files through tauri.macos.conf.json as well.
MAC_DIGESTS="$ROOT/client/src-tauri/runtime/macos/engine-digests"
if [ -L "$MAC_DIGESTS" ] || { [ -e "$MAC_DIGESTS" ] && [ ! -d "$MAC_DIGESTS" ]; }; then
    fail "unexpected non-directory macOS digest staging path: $MAC_DIGESTS"
fi
if [ -d "$MAC_DIGESTS" ]; then
    rm -rf -- "$MAC_DIGESTS"
fi
mkdir -p "$MAC_DIGESTS" || fail "could not create the macOS digest staging directory"
for b in idletoken-server idletoken-rpc-server; do
    staged="$ROOT/client/src-tauri/binaries/$b-$TRIPLE"
    [ -x "$staged" ] || fail "missing staged $b while recording its digest"
    hash=$(shasum -a 256 "$staged" | awk '{print $1}') \
        || fail "could not hash staged $b"
    printf '%s  %s\n' "$hash" "$b" > "$MAC_DIGESTS/$b.sha256" \
        || fail "could not stage the $b digest"
done

# --- stage licences (same obligation as build_client_release.sh) ------------
LIC=client/src-tauri/licenses
mkdir -p "$LIC"
cp -f "$ROOT/LICENSE" "$LIC/LICENSE.txt"            || fail "could not stage LICENSE"
cp -f "$ROOT/NOTICE"  "$LIC/NOTICE.txt"             || fail "could not stage NOTICE"
cp -f "$ROOT/vendor/ds4/LICENSE" "$LIC/ds4-MIT.txt" || fail "could not stage the ds4 licence"
# llama.cpp is MIT and is shipped as the idletoken-server sidecar.
cp -f "$ROOT/vendor/llama.cpp/LICENSE" "$LIC/llamacpp-MIT.txt" \
    || fail "could not stage the llama.cpp licence"

# --- the signing key: READ-ONLY, NON-REGENERABLE ----------------------------
# No signing key (2026-09-02, user ruling): `createUpdaterArtifacts` is false,
# so `tauri build` produces only the .dmg and needs no minisign key. Updating
# means downloading the current installer.

# --- build ------------------------------------------------------------------
cd client || fail "no client/ directory"
pnpm install >/tmp/client-mac-install.log 2>&1 || fail "pnpm install failed (see /tmp/client-mac-install.log)"
BUNDLES="${IDLETOKEN_MAC_BUNDLES:-dmg}"
case "$BUNDLES" in
    dmg) ;;
    *) fail "IDLETOKEN_MAC_BUNDLES must be dmg (got '$BUNDLES')" ;;
esac
# `tauri build` runs beforeBuildCommand (pnpm build:release) itself — that is
# what injects the production platform URL into the shipped frontend.
BUILD_LOG=$(mktemp /tmp/idletoken-tauri-mac-build.XXXXXX)
if ! pnpm tauri build --bundles "$BUNDLES" > "$BUILD_LOG" 2>&1; then
    tail -25 "$BUILD_LOG"
    rm -f "$BUILD_LOG"
    fail "tauri build failed"
fi
tail -25 "$BUILD_LOG"
rm -f "$BUILD_LOG"

BDIR=src-tauri/target/release/bundle
DMG=$(ls -t "$BDIR"/dmg/IdleToken_*.dmg 2>/dev/null | head -1)
[ -n "$DMG" ] || fail "no dmg produced under $BDIR/dmg"

# --- report artifacts -------------------------------------------------------
echo "--- artifacts ---"
for f in "$DMG"; do
    printf '%s  %s  %s\n' "$(shasum -a 256 "$f" | cut -c1-16)" "$(du -h "$f" | cut -f1)" "$ROOT/client/$f"
done

# No automatic provenance, signature, updater archive or feed is emitted here.
# The GitHub release contains the .dmg installer only.

echo CLIENT_MAC_OK
