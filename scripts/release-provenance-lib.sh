#!/usr/bin/env bash
# Shared release preflight + provenance emission. Sourced by the per-platform
# build scripts so the rules live in ONE place.
#
# The rules are two, and both exist because their failure is silent:
#
#   rp_preflight   Refuse to start a release build from a source tree nobody
#                  can reconstruct, and refuse to start one whose signing key
#                  is not the key installed clients trust. The second is the
#                  nastier of the two: signing with the wrong key succeeds,
#                  verifies against its own public key, and is only discovered
#                  by users whose updater rejects the release (DIST-08, OPS-04).
#
#   rp_emit        Write and sign the provenance record for what was just
#                  built, and append it to the transparency log. Skipping this
#                  is invisible at build time and leaves users with no way to
#                  tell an official installer from a copy (DIST-03, OPS-12).
#
# Usage from a build script:
#     . "$(dirname "$0")/release-provenance-lib.sh"
#     rp_preflight "linux" || exit 1
#     ...build...
#     rp_emit "$VERSION" linux-x86_64 "$DEB" "$APPIMAGE"
#
# Both functions print their own diagnostics and return non-zero on failure;
# they never exit, so the caller keeps its own `fail` contract.
#
# Run this file directly with --self-test to exercise both without a build.

# shellcheck disable=SC2155
RP_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
RP_ROOT="$(cd "$RP_LIB_DIR/.." && pwd)"

rp_say()  { printf '  %s\n' "$*"; }
rp_warn() { printf '  !! %s\n' "$*"; }

# --- is this tree reconstructible? ------------------------------------------
rp_check_tree() {
    if ! git -C "$RP_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
        rp_warn "not a git worktree — a release from here has no commit to point at"
        [ "${IDLETOKEN_RELEASE_ALLOW_DIRTY:-0}" = "1" ] || return 1
        return 0
    fi
    if [ -n "$(git -C "$RP_ROOT" status --porcelain 2>/dev/null)" ]; then
        if [ "${IDLETOKEN_RELEASE_ALLOW_DIRTY:-0}" != "1" ]; then
            rp_warn "the worktree is DIRTY. A release built now cannot be reproduced from any commit,"
            rp_warn "and its provenance record would name a commit that is not what you built."
            rp_warn "Commit first, or set IDLETOKEN_RELEASE_ALLOW_DIRTY=1 to record the exception."
            return 1
        fi
        rp_warn "IDLETOKEN_RELEASE_ALLOW_DIRTY=1: building from a dirty tree, recorded as dirty=true"
    else
        rp_say "source tree: clean at $(git -C "$RP_ROOT" rev-parse --short HEAD)"
    fi
    return 0
}

# --- is the signing key the key installed clients trust? ---------------------
# Proved by signing a nonce and verifying it against the PINNED public key from
# tauri.conf.json, with our own verifier. Comparing key files would only prove
# two files match; this proves the key can produce signatures an installed
# client accepts, which is the property that matters.
rp_check_key() {
    local key="${TAURI_SIGNING_PRIVATE_KEY_PATH:-$HOME/.idletoken/updater.key}"
    local signer="$RP_ROOT/client/node_modules/.bin/tauri"
    local conf="$RP_ROOT/client/src-tauri/tauri.conf.json"
    if [ -n "${TAURI_SIGNING_PRIVATE_KEY:-}" ] && [ ! -f "${TAURI_SIGNING_PRIVATE_KEY:-}" ]; then
        rp_say "signing key supplied as material in the environment — key-match check skipped"
        return 0
    fi
    [ -n "${TAURI_SIGNING_PRIVATE_KEY:-}" ] && key="$TAURI_SIGNING_PRIVATE_KEY"
    if [ ! -r "$key" ]; then
        rp_warn "no updater signing key at $key — the build will fail later; restore your backup, do NOT generate a new key"
        return 1
    fi
    if [ ! -x "$signer" ] || [ ! -f "$conf" ]; then
        rp_say "signer or config unavailable — key-match check skipped"
        return 0
    fi
    local t; t=$(mktemp -d "${TMPDIR:-/tmp}/idletoken-rp.XXXXXX") || return 1
    head -c 128 /dev/urandom > "$t/nonce" 2>/dev/null
    # The bundle scripts set TAURI_SIGNING_PRIVATE_KEY to this file path for
    # `tauri build`. The signer subcommand also interprets that environment
    # variable as `--private-key`, which conflicts with our explicit `-f`.
    # Clear both signer inputs from its environment and pass the already
    # resolved path/password exactly once on the command line.
    if ! env -u TAURI_SIGNING_PRIVATE_KEY -u TAURI_SIGNING_PRIVATE_KEY_PASSWORD \
       "$signer" signer sign -f "$key" -p "${TAURI_SIGNING_PRIVATE_KEY_PASSWORD:-}" "$t/nonce" >/dev/null 2>&1; then
        rp_warn "the key at $key cannot sign (wrong password, or not a signing key)"
        rm -rf "$t"; return 1
    fi
    local pinned; pinned=$(python3 -c "import json;print(json.load(open('$conf'))['plugins']['updater']['pubkey'])" 2>/dev/null)
    if python3 "$RP_LIB_DIR/minisign_verify.py" --pubkey "$pinned" --sig "$t/nonce.sig" "$t/nonce" >/dev/null 2>&1; then
        rp_say "signing key verified against the pinned updater trust root"
        rm -rf "$t"; return 0
    fi
    rp_warn "THE SIGNING KEY IS NOT THE RELEASE KEY. Artifacts signed with it would be"
    rp_warn "rejected by every installed client. Restore the backed-up key; do not"
    rp_warn "generate a replacement — a new key orphans every existing installation."
    rm -rf "$t"; return 1
}

rp_preflight() {
    local label="${1:-release}"
    echo "== release preflight ($label) =="
    local rc=0
    rp_check_tree || rc=1
    rp_check_key  || rc=1
    [ "$rc" = 0 ] && rp_say "preflight ok"
    return "$rc"
}

# --- write, sign and log the provenance record -------------------------------
rp_emit() {
    local version="$1" platform="$2"; shift 2
    [ $# -gt 0 ] || { rp_warn "rp_emit: no artifacts"; return 1; }
    echo "== release provenance ($platform) =="
    local out
    if ! out=$("$RP_LIB_DIR/release_manifest.sh" --version "$version" --platform "$platform" --sign "$@" 2>&1); then
        printf '%s\n' "$out" | sed 's/^/  /'
        rp_warn "could not write the provenance record — users would have nothing to verify this release against"
        return 1
    fi
    printf '%s\n' "$out" | sed 's/^/  /'
    local prov; prov=$(printf '%s\n' "$out" | sed -n 's/^RELEASE_MANIFEST_OK //p' | tail -1)
    [ -n "$prov" ] || { rp_warn "release_manifest.sh printed no path"; return 1; }
    if ! out=$("$RP_LIB_DIR/release_transparency.sh" append "$prov" 2>&1); then
        printf '%s\n' "$out" | sed 's/^/  /'
        rp_warn "the release was NOT added to the transparency log — do this before publishing"
        return 1
    fi
    printf '%s\n' "$out" | sed 's/^/  /'
    return 0
}

# --- self-test ---------------------------------------------------------------
# Not a demo: it asserts that each refusal actually fires. A preflight that
# cannot say no is a preflight that certifies whatever it is handed.
if [ "${BASH_SOURCE[0]:-$0}" = "${0}" ] && [ "${1:-}" = "--self-test" ]; then
    bad=0
    echo "== release-provenance-lib self-test =="

    # Control 1: the dirty-tree refusal fires when the tree is dirty (this
    # checkout is; if it is clean the control is reported as not applicable so
    # nobody reads a skip as a pass).
    if [ -n "$(git -C "$RP_ROOT" status --porcelain 2>/dev/null)" ]; then
        if IDLETOKEN_RELEASE_ALLOW_DIRTY=0 rp_check_tree >/dev/null 2>&1; then
            echo "  [BAD] CONTROL: a dirty tree passed rp_check_tree"; bad=1
        else
            echo "  [ok] control: a dirty tree is refused"
        fi
        if IDLETOKEN_RELEASE_ALLOW_DIRTY=1 rp_check_tree >/dev/null 2>&1; then
            echo "  [ok] the documented override is honoured (and prints a warning)"
        else
            echo "  [BAD] IDLETOKEN_RELEASE_ALLOW_DIRTY=1 did not allow a dirty tree"; bad=1
        fi
    else
        echo "  [n/a] this checkout is clean — dirty-tree control not exercised"
    fi

    # Control 2: a valid but WRONG signing key must be refused. This is the
    # release-killing mistake nothing else notices.
    signer="$RP_ROOT/client/node_modules/.bin/tauri"
    if [ -x "$signer" ]; then
        t=$(mktemp -d "${TMPDIR:-/tmp}/idletoken-rpst.XXXXXX")
        "$signer" signer generate -w "$t/wrong.key" --password "" -f >/dev/null 2>&1
        if TAURI_SIGNING_PRIVATE_KEY_PATH="$t/wrong.key" TAURI_SIGNING_PRIVATE_KEY= \
           rp_check_key >/dev/null 2>&1; then
            echo "  [BAD] CONTROL: a foreign signing key passed rp_check_key"; bad=1
        else
            echo "  [ok] control: a valid-but-foreign signing key is refused"
        fi
        rm -rf "$t"
        if [ -r "${TAURI_SIGNING_PRIVATE_KEY_PATH:-$HOME/.idletoken/updater.key}" ]; then
            if rp_check_key >/dev/null 2>&1; then
                echo "  [ok] the real key on this machine passes"
            else
                echo "  [BAD] the real signing key did NOT verify against the pinned trust root"; bad=1
            fi
        else
            echo "  [n/a] no release key on this machine — positive case not exercised"
        fi
    else
        echo "  [n/a] no Tauri signer — key controls not exercised"
    fi

    [ "$bad" = 0 ] && echo "RELEASE_PROVENANCE_LIB_OK" || echo "RELEASE_PROVENANCE_LIB_FAIL"
    exit "$bad"
fi
