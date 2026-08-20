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
# scripts/stage_sidecars.sh is the single source of truth for what a client
# ships (coord, worker, platform-agent, and — since the v2 llama.cpp pivot —
# the pinned idletoken-server + idletoken-rpc-server). This script used to carry its own copy of the
# staging and it drifted: it kept shipping bundles without idletoken-server after
# stage_sidecars.sh already required it.
out=$(scripts/stage_sidecars.sh) || { echo "$out"; fail "sidecar staging failed"; }
echo "$out" | sed 's/^/  /'

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
awk '/^\/\* Rax/,/^ \*\/$/' "$ROOT/vendor/ds4/rax.c" > "$LIC/rax-BSD-3-Clause.txt" \
    || fail "could not extract the rax licence"
grep -q "Redistribution and use in source and binary forms" "$LIC/rax-BSD-3-Clause.txt" \
    || fail "extracted rax licence does not contain the BSD terms (rax.c header changed shape)"
# llama.cpp is MIT and is shipped as the idletoken-server sidecar (v2 pivot).
cp -f "$ROOT/vendor/llama.cpp/LICENSE" "$LIC/llamacpp-MIT.txt" \
    || fail "could not stage the llama.cpp licence"
echo "  staged licences -> $LIC"

# --- the signing key: READ-ONLY, NON-REGENERABLE ----------------------------
# `createUpdaterArtifacts` is on in tauri.conf.json, so `tauri build` refuses
# to run without the private key — wanted: an unsigned release is a dead
# update channel and the build is the last place that can notice. Accept key
# MATERIAL over the environment (a control machine driving this over ssh must
# never write the key to this machine's disk) or a local key file path.
# NEVER generate a new key here: installed clients trust exactly one pubkey
# (28F23C3CE24BFDE9) and a fresh key would permanently orphan them.
if [ -z "${TAURI_SIGNING_PRIVATE_KEY:-}" ]; then
    KEY_PATH="${TAURI_SIGNING_PRIVATE_KEY_PATH:-$HOME/.idletoken/updater.key}"
    [ -f "$KEY_PATH" ] || fail "no updater signing key: set TAURI_SIGNING_PRIVATE_KEY (key material, e.g. passed over ssh from the machine that holds it) or put the key at $KEY_PATH — RESTORE YOUR BACKUP, do NOT generate a new key"
    # Tauri v2 reads TAURI_SIGNING_PRIVATE_KEY (path or key material); pass
    # the PATH so the key material never enters the environment.
    export TAURI_SIGNING_PRIVATE_KEY="$KEY_PATH"
fi
export TAURI_SIGNING_PRIVATE_KEY_PASSWORD="${TAURI_SIGNING_PRIVATE_KEY_PASSWORD:-}"

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
# All three bundles by default: the AppImage is the Linux in-app UPDATE
# vehicle (the updater cannot swap a dpkg/rpm install), so a Linux release
# without it has no update channel. Two real failures already hit here and
# both have their fix baked in below:
#   - linuxdeploy needs FUSE and the DGX has no libfuse2 — APPIMAGE_EXTRACT_AND_RUN=1
#     makes the AppImages self-extract instead (verified 2026-08-15);
#   - the tooling download can time out on this network — the cache at
#     ~/.cache/tauri survives, and TAURI_BUNDLER_TOOLS_GITHUB_MIRROR works too.
export APPIMAGE_EXTRACT_AND_RUN=1
BUNDLES="${IDLETOKEN_BUNDLES:-deb,rpm,appimage}"
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
done < <(find "$BDIR" -type f \( -name '*.deb' -o -name '*.AppImage' -o -name '*.rpm' \) | sort)
[ "$found" = 1 ] || fail "bundler produced no .deb/.AppImage"
# The AppImage is the Linux update vehicle: when it was requested, its absence
# is a dead update channel even if deb/rpm came out fine.
case ",$BUNDLES," in *,appimage,*)
    find "$BDIR" -type f -name '*.AppImage' | grep -q . \
        || fail "appimage was requested but the bundler produced none — Linux would have no update channel" ;;
esac

# Updater signatures. This Tauri version signs the Linux packages themselves
# (IdleToken_*.deb.sig / .rpm.sig, minisign format) — the update artifact IS
# the package. Report every .sig so the caller can verify them against the
# client pubkey; zero signatures means a dead update channel, fail there.
SIGS=$(find "$BDIR" -type f -name '*.sig' | sort)
[ -n "$SIGS" ] || fail "bundler produced no updater signatures — the update channel would be dead (signing key not applied?)"
echo "--- updater signatures ---"
for f in $SIGS; do
    printf '%s  %s  %s\n' "$(sha256sum "$f" | cut -c1-16)" "$(du -h "$f" | cut -f1)" "$ROOT/client/$f"
done

# --- verify: the engine inside the newest .deb is the pinned llama.cpp ------
# The whole point of the bundle is the engine it carries; a stale idletoken-server
# in the package is invisible to every gate that drives the repo binaries.
NEWEST_DEB=$(ls -t "$BDIR"/deb/*.deb 2>/dev/null | head -1)
if [ -n "$NEWEST_DEB" ]; then
    PIN_SHA=$(awk 'NR==1{print $2}' "$ROOT/scripts/llamacpp-patches/UPSTREAM")
    [ -n "$PIN_SHA" ] || fail "cannot read the engine pin from scripts/llamacpp-patches/UPSTREAM"
    XTMP=$(mktemp -d /tmp/idletoken-deb-verify.XXXXXX)
    dpkg-deb -x "$NEWEST_DEB" "$XTMP" || { rm -rf "$XTMP"; fail "could not extract $NEWEST_DEB"; }
    DEB_LS=$(find "$XTMP" -type f -name 'idletoken-server' | head -1)
    [ -n "$DEB_LS" ] || { rm -rf "$XTMP"; fail "the .deb does not contain idletoken-server (bundler shipped an incomplete app)"; }
    VERSION_LINE=$("$DEB_LS" --version 2>&1 | grep -m1 'version:') \
        || { rm -rf "$XTMP"; fail "idletoken-server inside the .deb does not run"; }
    case "$VERSION_LINE" in
        *"${PIN_SHA:0:7}"*) echo "deb engine check: $VERSION_LINE (matches pin ${PIN_SHA:0:7})" ;;
        *) rm -rf "$XTMP"; fail "idletoken-server inside the .deb is '$VERSION_LINE', not the pinned ${PIN_SHA:0:7} — a stale engine got staged" ;;
    esac
    for b in idletoken-coord idletoken-worker idletoken-platform-agent; do
        [ -n "$(find "$XTMP" -type f -name "$b" | head -1)" ] \
            || { rm -rf "$XTMP"; fail "the .deb does not contain $b"; }
    done
    rm -rf "$XTMP"
fi

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
