#!/usr/bin/env bash
# Build the installable macOS client (.dmg + updater .app.tar.gz/.sig) — WS-E E3.
#
# One command on the Mac: build the engine sidecars, stage them, bundle with
# Tauri, sign the updater artifact with the NON-REGENERABLE minisign key, and
# then verify what actually shipped (sidecars + engine pin inside the mounted
# dmg, signature against the pubkey compiled into the client). Verification is
# part of the contract, not a courtesy: a dmg that carries a stale llama-server
# or an unverifiable signature looks identical to a good one from the outside.
#
# Usage:  scripts/package_client_mac.sh
# Env:    TAURI_SIGNING_PRIVATE_KEY_PATH   (default ~/.idletoken/updater.key)
#         TAURI_SIGNING_PRIVATE_KEY_PASSWORD (default empty)
#         IDLETOKEN_MAC_BUNDLES            (default "app,dmg")
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
make coord worker >/dev/null || fail "make coord worker failed"
make -f Makefile.platform >/dev/null || fail "make -f Makefile.platform failed"
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
# llama.cpp is MIT and is shipped as the llama-server sidecar.
cp -f "$ROOT/vendor/llama.cpp/LICENSE" "$LIC/llamacpp-MIT.txt" \
    || fail "could not stage the llama.cpp licence"

# --- the signing key: READ-ONLY, NON-REGENERABLE ----------------------------
# `createUpdaterArtifacts` is on, so `tauri build` refuses to run without the
# private key — wanted: an unsigned release is a dead update channel. The key
# is used strictly in place. NEVER generate a new one here: installed clients
# trust exactly one pubkey, and a fresh key would permanently orphan them.
KEY_PATH="${TAURI_SIGNING_PRIVATE_KEY_PATH:-$HOME/.idletoken/updater.key}"
if [ -z "${TAURI_SIGNING_PRIVATE_KEY:-}" ]; then
    [ -f "$KEY_PATH" ] || fail "no updater signing key at $KEY_PATH — RESTORE YOUR BACKUP (password manager / encrypted disk). Do NOT generate a new key: already-installed clients only trust the existing pubkey (28F23C3CE24BFDE9)"
    # Tauri v2 reads TAURI_SIGNING_PRIVATE_KEY (path or key material); the
    # _PATH spelling is our repo convention only. Pass the PATH so the key
    # material never enters the environment.
    export TAURI_SIGNING_PRIVATE_KEY="$KEY_PATH"
fi
export TAURI_SIGNING_PRIVATE_KEY_PASSWORD="${TAURI_SIGNING_PRIVATE_KEY_PASSWORD:-}"

# --- build ------------------------------------------------------------------
cd client || fail "no client/ directory"
pnpm install >/tmp/client-mac-install.log 2>&1 || fail "pnpm install failed (see /tmp/client-mac-install.log)"
BUNDLES="${IDLETOKEN_MAC_BUNDLES:-app,dmg}"
# `tauri build` runs beforeBuildCommand (pnpm build:release) itself — that is
# what injects the production platform URL into the shipped frontend.
pnpm tauri build --bundles "$BUNDLES" 2>&1 | tail -25 || fail "tauri build failed"

BDIR=src-tauri/target/release/bundle
DMG=$(ls -t "$BDIR"/dmg/IdleToken_*.dmg 2>/dev/null | head -1)
[ -n "$DMG" ] || fail "no dmg produced under $BDIR/dmg"
APPTAR=$(ls -t "$BDIR"/macos/IdleToken.app.tar.gz 2>/dev/null | head -1)
[ -n "$APPTAR" ] || fail "no updater artifact IdleToken.app.tar.gz under $BDIR/macos"
SIG="$APPTAR.sig"
[ -f "$SIG" ] || fail "no signature next to $APPTAR"

# --- verify 1: sidecars + engine pin INSIDE the dmg -------------------------
# Verify the artifact users get, not the intermediate .app in the build tree.
MNT=$(mktemp -d /tmp/idletoken-dmg.XXXXXX)
hdiutil attach -readonly -nobrowse -mountpoint "$MNT" "$DMG" >/dev/null \
    || fail "could not mount $DMG"
trap 'hdiutil detach "$MNT" >/dev/null 2>&1; rmdir "$MNT" 2>/dev/null' EXIT
MACOS_DIR="$MNT/IdleToken.app/Contents/MacOS"
# ggml-rpc-server is the other half of the engine: llama-server serves, the rpc
# server is what this machine runs when it joins someone else's cluster. Ship
# one without the other and the app offers a cluster mode it cannot join.
for b in idletoken-client idletoken-coord idletoken-worker idletoken-platform-agent llama-server ggml-rpc-server; do
    [ -x "$MACOS_DIR/$b" ] || fail "dmg is missing $b in Contents/MacOS (bundler shipped an incomplete app)"
done
PIN_SHA=$(awk '{print $2}' "$ROOT/scripts/llamacpp-patches/UPSTREAM")
[ -n "$PIN_SHA" ] || fail "cannot read the engine pin from scripts/llamacpp-patches/UPSTREAM"
VERSION_LINE=$("$MACOS_DIR/llama-server" --version 2>&1 | grep -m1 'version:') \
    || fail "bundled llama-server does not run"
case "$VERSION_LINE" in
    *"${PIN_SHA:0:7}"*) echo "  bundled llama-server: $VERSION_LINE (matches pin ${PIN_SHA:0:7})" ;;
    *) fail "bundled llama-server is '$VERSION_LINE', not the pinned ${PIN_SHA:0:7} — a stale engine got staged" ;;
esac
for r in LICENSE.txt NOTICE.txt ds4-MIT.txt rax-BSD-3-Clause.txt llamacpp-MIT.txt; do
    [ -f "$MNT/IdleToken.app/Contents/Resources/licenses/$r" ] \
        || fail "dmg is missing licence $r in Contents/Resources/licenses"
done
hdiutil detach "$MNT" >/dev/null 2>&1
rmdir "$MNT" 2>/dev/null
trap - EXIT

# --- verify 2: updater signature against the pubkey the client trusts -------
# The .sig is base64 of a minisign signature; the pubkey in tauri.conf.json is
# base64 of a minisign public-key file. Verify with the real tool so this
# cannot drift from what installed clients will do.
command -v minisign >/dev/null 2>&1 \
    || fail "minisign not installed (brew install minisign) — refusing to ship an unverified updater artifact"
VTMP=$(mktemp -d /tmp/idletoken-sigverify.XXXXXX)
python3 - "$ROOT/client/src-tauri/tauri.conf.json" > "$VTMP/updater.pub" <<'EOF' || fail "could not extract the updater pubkey from tauri.conf.json"
import base64, json, sys
conf = json.load(open(sys.argv[1]))
sys.stdout.write(base64.b64decode(conf["plugins"]["updater"]["pubkey"]).decode())
EOF
grep -q "28F23C3CE24BFDE9" "$VTMP/updater.pub" \
    || fail "extracted pubkey is not the known 28F23C3CE24BFDE9 — tauri.conf.json changed keys?"
base64 -D -i "$SIG" -o "$VTMP/apptar.minisig" 2>/dev/null \
    || base64 -d < "$SIG" > "$VTMP/apptar.minisig" \
    || fail "could not decode $SIG"
if minisign -Vm "$APPTAR" -x "$VTMP/apptar.minisig" -p "$VTMP/updater.pub" > "$VTMP/verify.out" 2>&1; then
    echo "  updater signature: $(head -1 "$VTMP/verify.out") (pubkey 28F23C3CE24BFDE9)"
else
    cat "$VTMP/verify.out"
    rm -rf "$VTMP"
    fail "updater signature does NOT verify against the client pubkey — the update channel would be dead"
fi
rm -rf "$VTMP"

# --- report artifacts -------------------------------------------------------
echo "--- artifacts ---"
for f in "$DMG" "$APPTAR" "$SIG"; do
    printf '%s  %s  %s\n' "$(shasum -a 256 "$f" | cut -c1-16)" "$(du -h "$f" | cut -f1)" "$ROOT/client/$f"
done

# --- restore the non-release client/dist ------------------------------------
# `tauri build` ran beforeBuildCommand = `pnpm build:release`, which leaves a
# dist carrying the PRODUCTION platform URL; the acceptance gates serve that
# same dist to the debug shell. Same restore as build_client_release.sh.
pnpm build > /tmp/client-mac-restore.log 2>&1 \
    || fail "bundle is built, but restoring the non-release client/dist failed (see /tmp/client-mac-restore.log)"
echo "restored client/dist to the non-release build (acceptance gates use it)"

echo CLIENT_MAC_OK
