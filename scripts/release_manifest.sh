#!/usr/bin/env bash
# Write (and sign) the provenance record for one platform's release artifacts.
#
# WHAT PROBLEM THIS SOLVES
#
# A signature proves a key was used. It does not say which source tree, which
# engine pin or which toolchain produced the bytes, and it cannot be checked at
# all by someone who has not installed our software yet — which is precisely the
# person choosing between the official installer and a lookalike (DIST-03,
# DIST-06, DIST-09, OPS-12).
#
# So every release also publishes a small signed JSON that answers those
# questions, and `scripts/verify_release.sh` checks a downloaded file against
# it with nothing but python3 and shasum.
#
# WHAT IT DELIBERATELY DOES NOT CLAIM
#
# This is an attestation by the builder, not a reproducible-build proof. It says
# "this machine, holding this key, says these bytes came from this commit". A
# compromised build host produces a perfectly valid record (DIST-09/CHAIN-07).
# The defence against that is not this file — it is the transparency log
# (scripts/release_transparency.sh) plus a second builder's record for the same
# commit, both of which this format is shaped to support.
#
# Usage:
#   scripts/release_manifest.sh --version 0.1.30 --platform windows-x86_64 \
#       [--out DIR] [--sign] [--builder LABEL] ARTIFACT [ARTIFACT...]
#
# Env:
#   IDLETOKEN_RELEASE_ALLOW_DIRTY=1   record a dirty worktree instead of refusing
#   TAURI_SIGNING_PRIVATE_KEY_PATH    signing key (default ~/.idletoken/updater.key)
#   TAURI_SIGNING_PRIVATE_KEY_PASSWORD
#
# Contract: the last line is RELEASE_MANIFEST_OK <path> or
# RELEASE_MANIFEST_FAIL: <reason>.
set -u

cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD

fail() { echo "RELEASE_MANIFEST_FAIL: $*"; exit 1; }

VERSION="" PLATFORM="" OUTDIR="" SIGN=0 BUILDER=""
ARTIFACTS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --version)  VERSION="${2:-}"; shift 2 ;;
        --platform) PLATFORM="${2:-}"; shift 2 ;;
        --out)      OUTDIR="${2:-}"; shift 2 ;;
        --builder)  BUILDER="${2:-}"; shift 2 ;;
        --sign)     SIGN=1; shift ;;
        --help|-h)  sed -n '2,30p' "$0"; exit 0 ;;
        --*)        fail "unknown option $1" ;;
        *)          ARTIFACTS+=("$1"); shift ;;
    esac
done

[ -n "$VERSION" ]  || fail "--version is required"
[ -n "$PLATFORM" ] || fail "--platform is required (e.g. windows-x86_64, darwin-aarch64, linux-x86_64)"
[ "${#ARTIFACTS[@]}" -gt 0 ] || fail "no artifacts given"
for a in "${ARTIFACTS[@]}"; do [ -f "$a" ] || fail "not a file: $a"; done

# The builder label is a LABEL, never the hostname. Machine names are third
# -party-identifying on this testbed and are refused by the public-mirror scan
# (sync-public.sh FORBIDDEN); a provenance record that cannot be published is
# not a provenance record. Default to the platform tuple, which is the only
# part of "where was this built" that a reader can act on.
[ -n "$BUILDER" ] || BUILDER="$PLATFORM"
case "$BUILDER" in
    *" "*) fail "builder label must not contain spaces" ;;
esac

# --- the source tree ---------------------------------------------------------
# A release built from a tree nobody can reconstruct is a release nobody can
# audit. Refuse by default; the override exists because a genuine emergency
# rebuild should be possible, and it is recorded IN the artifact so the
# exception travels with it rather than being forgotten.
git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1 \
    || fail "not a git worktree — refusing to claim a commit for these artifacts"
HEAD_SHA=$(git -C "$ROOT" rev-parse HEAD) || fail "could not read HEAD"
DIRTY=false
if [ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]; then
    DIRTY=true
    if [ "${IDLETOKEN_RELEASE_ALLOW_DIRTY:-0}" != "1" ]; then
        fail "the worktree is dirty — these artifacts do not correspond to $HEAD_SHA. Commit first, or set IDLETOKEN_RELEASE_ALLOW_DIRTY=1 to record the exception in the manifest"
    fi
    echo "  !! IDLETOKEN_RELEASE_ALLOW_DIRTY=1: recording dirty=true. This release CANNOT be reproduced from $HEAD_SHA."
fi

# --- the engine pin ----------------------------------------------------------
# The cluster's one invariant is the llama.cpp version (hard constraint 4), and
# the patches are ours. Both go in: "which engine is inside this installer" is
# not answerable from the version string alone.
ENGINE_PIN=""
ENGINE_PATCHES="[]"
if [ -f "$ROOT/scripts/llamacpp-patches/UPSTREAM" ]; then
    ENGINE_PIN=$(awk 'NR==1{print $2}' "$ROOT/scripts/llamacpp-patches/UPSTREAM")
fi
if [ -d "$ROOT/scripts/llamacpp-patches" ]; then
    ENGINE_PATCHES=$(cd "$ROOT/scripts/llamacpp-patches" && \
        for p in *.patch; do
            [ -f "$p" ] || continue
            printf '{"file":"%s","sha256":"%s"}\n' "$p" "$(shasum -a 256 "$p" | cut -d' ' -f1)"
        done | paste -sd, - | sed 's/^/[/; s/$/]/')
    [ -n "$ENGINE_PATCHES" ] || ENGINE_PATCHES="[]"
fi

# --- toolchains --------------------------------------------------------------
# A non-login shell does not read the profile that puts cargo on PATH, so the
# record said `rustc: absent` on a machine that has it — a provenance field
# that is wrong is worse than one that is missing.
[ -x "$HOME/.cargo/bin/rustc" ] && PATH="$HOME/.cargo/bin:$PATH"
tool_ver() { command -v "$1" >/dev/null 2>&1 && "$@" 2>/dev/null | head -1 || echo "absent"; }
RUSTC_V=$(tool_ver rustc --version)
NODE_V=$(tool_ver node --version)
PNPM_V=$(tool_ver pnpm --version)

# --- the trust root this build's updater artifacts are signed with -----------
UPDATER_KEYID=$(python3 - "$ROOT/client/src-tauri/tauri.conf.json" <<'PY'
import base64, json, sys
try:
    pk = json.load(open(sys.argv[1]))["plugins"]["updater"]["pubkey"]
    raw = base64.b64decode(base64.b64decode(pk).decode().splitlines()[1])
    print(raw[2:10][::-1].hex().upper())
except Exception:
    print("")
PY
)
[ -n "$UPDATER_KEYID" ] || fail "could not read the updater public key from tauri.conf.json"

# --- artifact digests --------------------------------------------------------
sha256_of() {
    if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
    elif command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    else return 1; fi
}
ART_JSON=""
for a in "${ARTIFACTS[@]}"; do
    d=$(sha256_of "$a") || fail "no shasum/sha256sum available to digest $a"
    sz=$(wc -c < "$a" | tr -d ' ')
    # An updater signature sitting next to the artifact is recorded by NAME, so
    # a verifier can tell which file is supposed to carry one. Its presence is
    # not evidence of anything on its own; verify_release.sh checks it.
    sig="null"
    [ -f "$a.sig" ] && sig="\"$(basename "$a.sig")\""
    ART_JSON="$ART_JSON{\"name\":\"$(basename "$a")\",\"sha256\":\"$d\",\"bytes\":$sz,\"updaterSignature\":$sig},"
done
ART_JSON="[${ART_JSON%,}]"

BUILT_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
[ -n "$OUTDIR" ] || OUTDIR=$(dirname "${ARTIFACTS[0]}")
mkdir -p "$OUTDIR" || fail "cannot create $OUTDIR"
OUT="$OUTDIR/idletoken-release-$VERSION-$PLATFORM.provenance.json"

# python3 writes it, and every value arrives through the ENVIRONMENT rather
# than being pasted into the source text. Interpolating shell strings into a
# heredoc'd program is how `null` and `true` become Python NameErrors and how a
# stray quote in a version string produces a file no verifier can parse — both
# happened while writing this.
RM_VERSION="$VERSION" RM_PLATFORM="$PLATFORM" RM_BUILT_AT="$BUILT_AT" \
RM_BUILDER="$BUILDER" RM_HEAD="$HEAD_SHA" RM_DIRTY="$DIRTY" \
RM_ENGINE_PIN="$ENGINE_PIN" RM_ENGINE_PATCHES="$ENGINE_PATCHES" \
RM_RUSTC="$RUSTC_V" RM_NODE="$NODE_V" RM_PNPM="$PNPM_V" \
RM_KEYID="$UPDATER_KEYID" RM_ARTIFACTS="$ART_JSON" \
python3 - "$OUT" <<'PY' || fail "could not write $OUT"
import json, os, sys
env = os.environ
doc = {
    "schema": "idletoken-release-provenance/1",
    "version": env["RM_VERSION"],
    "platform": env["RM_PLATFORM"],
    "builtAt": env["RM_BUILT_AT"],
    "builder": env["RM_BUILDER"],
    "source": {"commit": env["RM_HEAD"], "dirty": env["RM_DIRTY"] == "true"},
    "engine": {
        "llamacppPin": env["RM_ENGINE_PIN"],
        "patches": json.loads(env["RM_ENGINE_PATCHES"] or "[]"),
    },
    "toolchain": {
        "rustc": env["RM_RUSTC"], "node": env["RM_NODE"], "pnpm": env["RM_PNPM"],
    },
    "updaterKeyId": env["RM_KEYID"],
    "artifacts": json.loads(env["RM_ARTIFACTS"]),
    "attestation": (
        "The builder asserts these artifacts were produced from the commit above. "
        "This is an attestation, not a reproducible-build proof: a compromised "
        "build host would produce a valid record. Cross-check it against the "
        "transparency log and, where two builders exist, against the other "
        "builder's record for the same commit."
    ),
}
json.dump(doc, open(sys.argv[1], "w"), indent=2, sort_keys=True)
open(sys.argv[1], "a").write("\n")
PY

echo "  wrote $OUT"
python3 -c "
import json,sys
d=json.load(open('$OUT'))
for a in d['artifacts']:
    print('    %s  %s  %d bytes' % (a['sha256'][:16], a['name'], a['bytes']))
"

# --- sign it -----------------------------------------------------------------
# Signed with the SAME key installed clients already trust, so a user has one
# fingerprint to know (28F23C3CE24BFDE9), not two. The signer is the Tauri CLI
# because that is the tool that already exists here; the VERIFIER is our own
# independent implementation, on purpose (see minisign_verify.py).
if [ "$SIGN" = 1 ]; then
    KEY_PATH="${TAURI_SIGNING_PRIVATE_KEY_PATH:-$HOME/.idletoken/updater.key}"
    [ -r "$KEY_PATH" ] || fail "no signing key at $KEY_PATH — restore your backup; do NOT generate a replacement (installed clients trust exactly one key)"
    SIGNER="$ROOT/client/node_modules/.bin/tauri"
    [ -x "$SIGNER" ] || fail "no Tauri signer at $SIGNER (run pnpm install in client/)"
    # Bundle scripts export the same path through TAURI_SIGNING_PRIVATE_KEY for
    # `tauri build`; leaving it set would make the signer receive both
    # `--private-key` and our explicit `-f` and reject an otherwise valid key.
    env -u TAURI_SIGNING_PRIVATE_KEY -u TAURI_SIGNING_PRIVATE_KEY_PASSWORD \
        "$SIGNER" signer sign -f "$KEY_PATH" -p "${TAURI_SIGNING_PRIVATE_KEY_PASSWORD:-}" "$OUT" >/dev/null 2>&1 \
        || fail "signing $OUT failed"
    [ -s "$OUT.sig" ] || fail "the signer produced no signature for $OUT"
    # Verify what we just signed, with the independent verifier and against the
    # PINNED public key rather than the private key's own pubkey file. Signing
    # with the wrong key produces a perfectly valid signature that no installed
    # client will accept — this is the step that notices.
    PINNED=$(python3 -c "import json;print(json.load(open('$ROOT/client/src-tauri/tauri.conf.json'))['plugins']['updater']['pubkey'])")
    python3 "$ROOT/scripts/minisign_verify.py" --pubkey "$PINNED" --sig "$OUT.sig" "$OUT" \
        || fail "the signature we just wrote does not verify against the pinned updater key $UPDATER_KEYID — the signing key on this machine is NOT the release key"
    echo "  signed and independently verified: $OUT.sig"
fi

echo "RELEASE_MANIFEST_OK $OUT"
