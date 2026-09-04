#!/usr/bin/env bash
# Verify a downloaded IdleToken artifact against the project's signed release
# provenance — WITHOUT installing anything.
#
# Since the in-app updater was retired (2026-09-02) this is the ONLY check
# standing between a user and a repacked installer, on both the first install
# and every upgrade: an installer downloaded from a mirror, a cloud-drive link
# or a search advertisement (threat register DIST-02, DIST-03, DIST-05) is
# indistinguishable from the official one until this runs. Until that installer
# runs, the only thing the user has is bytes — so the check has to work on bytes
# alone, with tools they already have.
#
# Requirements: python3 and one of shasum/sha256sum. Nothing else. In
# particular it does not require IdleToken, minisign, node, or network access.
#
# Usage:
#   scripts/verify_release.sh --provenance <file.json> [--sig <file.json.sig>] ARTIFACT...
#   scripts/verify_release.sh --provenance <file.json> --expect-version 0.1.30 ARTIFACT
#   scripts/verify_release.sh --self-test
#
# Options:
#   --pubkey V   trust root override (file, base64 blob, or 'none' to skip the
#                signature check). Skipping is LOUD and downgrades the verdict
#                to UNSIGNED, because an unsigned provenance file is exactly
#                what an attacker would supply next to their own installer.
#
# Contract: last line is VERIFY_RELEASE_OK, VERIFY_RELEASE_FAIL: <reason>, or
# VERIFY_RELEASE_UNSIGNED: <reason>.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

fail() { echo "VERIFY_RELEASE_FAIL: $*"; exit 1; }

PROV="" SIG="" PUBKEY="" EXPECT_VERSION="" SELFTEST=0
ARTIFACTS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --provenance) PROV="${2:-}"; shift 2 ;;
        --sig)        SIG="${2:-}"; shift 2 ;;
        --pubkey)     PUBKEY="${2:-}"; shift 2 ;;
        --expect-version) EXPECT_VERSION="${2:-}"; shift 2 ;;
        --self-test)  SELFTEST=1; shift ;;
        --help|-h)    sed -n '2,30p' "$0"; exit 0 ;;
        --*)          fail "unknown option $1" ;;
        *)            ARTIFACTS+=("$1"); shift ;;
    esac
done

command -v python3 >/dev/null 2>&1 || fail "python3 is required"

sha256_of() {
    if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
    elif command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    else return 1; fi
}

# ---------------------------------------------------------------------------
# Self-test. A verifier nobody has watched fail is a verifier nobody should
# trust: this builds a tiny release, verifies it, then breaks it four different
# ways and requires a refusal each time. It is the positive control for every
# claim this script makes.
# ---------------------------------------------------------------------------
if [ "$SELFTEST" = 1 ]; then
    command -v git >/dev/null 2>&1 || { echo "VERIFY_RELEASE_FAIL: git required for the self-test"; exit 1; }
    SIGNER="$ROOT/client/node_modules/.bin/tauri"
    [ -x "$SIGNER" ] || { echo "VERIFY_RELEASE_SKIP: no Tauri signer at $SIGNER (run pnpm install in client/)"; exit 0; }
    T=$(mktemp -d "${TMPDIR:-/tmp}/idletoken-verify-selftest.XXXXXX") || exit 1
    trap 'rm -rf "$T"' EXIT
    bad=0
    say() { printf '  %s\n' "$*"; }

    "$SIGNER" signer generate -w "$T/k" --password "" -f >/dev/null 2>&1 \
        || { echo "VERIFY_RELEASE_FAIL: could not make a throwaway key"; exit 1; }
    head -c 8192 /dev/urandom > "$T/IdleToken_0.0.0_selftest.exe"
    IDLETOKEN_RELEASE_ALLOW_DIRTY=1 "$HERE/release_manifest.sh" \
        --version 0.0.0 --platform selftest --out "$T" "$T/IdleToken_0.0.0_selftest.exe" >/dev/null \
        || { echo "VERIFY_RELEASE_FAIL: could not build a self-test provenance record"; exit 1; }
    P="$T/idletoken-release-0.0.0-selftest.provenance.json"
    "$SIGNER" signer sign -f "$T/k" -p "" "$P" >/dev/null 2>&1 \
        || { echo "VERIFY_RELEASE_FAIL: could not sign the self-test provenance"; exit 1; }

    if "$0" --provenance "$P" --pubkey "$T/k.pub" "$T/IdleToken_0.0.0_selftest.exe" >"$T/pos.log" 2>&1; then
        say "[ok] a correct artifact + signed provenance verifies"
    else
        say "[BAD] the happy path failed:"; sed 's/^/       /' "$T/pos.log"; bad=1
    fi

    # Control 1 — the artifact was swapped after the provenance was signed.
    # It keeps the NAME the record lists, so only the digest can catch it; a
    # checker that matched on file names would sail through this.
    mkdir -p "$T/swapdir"
    cp "$T/IdleToken_0.0.0_selftest.exe" "$T/swapdir/IdleToken_0.0.0_selftest.exe"
    printf 'malware' | dd of="$T/swapdir/IdleToken_0.0.0_selftest.exe" bs=1 seek=64 conv=notrunc 2>/dev/null
    if "$0" --provenance "$P" --pubkey "$T/k.pub" "$T/swapdir/IdleToken_0.0.0_selftest.exe" >"$T/c1.log" 2>&1; then
        say "[BAD] CONTROL: a swapped artifact with the right name verified"; bad=1
    else
        say "[ok] control: a swapped artifact is refused ($(tail -1 "$T/c1.log"))"
    fi

    # Control 2 — the provenance itself was edited to list the attacker's digest.
    cp "$P" "$T/edited.json"
    python3 - "$T/edited.json" "$(sha256_of "$T/swapdir/IdleToken_0.0.0_selftest.exe")" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
d["artifacts"][0]["sha256"] = sys.argv[2]
json.dump(d, open(sys.argv[1], "w"), indent=2, sort_keys=True)
PY
    cp "$P.sig" "$T/edited.json.sig"
    if "$0" --provenance "$T/edited.json" --pubkey "$T/k.pub" "$T/swapdir/IdleToken_0.0.0_selftest.exe" >"$T/c2.log" 2>&1; then
        say "[BAD] CONTROL: an edited provenance file verified"; bad=1
    else
        say "[ok] control: an edited provenance file is refused ($(tail -1 "$T/c2.log"))"
    fi

    # Control 3 — everything is internally consistent but signed by another key.
    # This is the fork case (DIST-05): the attacker signs their own provenance
    # for their own installer, correctly.
    if "$0" --provenance "$P" "$T/IdleToken_0.0.0_selftest.exe" >"$T/c3.log" 2>&1; then
        say "[BAD] CONTROL: provenance signed by a foreign key verified against the pinned key"; bad=1
    else
        say "[ok] control: a foreign signing key is refused ($(tail -1 "$T/c3.log"))"
    fi

    # Control 4 — no signature at all must not read as a pass.
    rm -f "$T/nosig.json.sig"; cp "$P" "$T/nosig.json"
    out=$("$0" --provenance "$T/nosig.json" --pubkey "$T/k.pub" "$T/IdleToken_0.0.0_selftest.exe" 2>&1 | tail -1)
    case "$out" in
        VERIFY_RELEASE_OK) say "[BAD] CONTROL: a provenance file with no signature reported OK"; bad=1 ;;
        *) say "[ok] control: a missing signature does not report OK ($out)" ;;
    esac

    [ "$bad" = 0 ] || { echo "VERIFY_RELEASE_FAIL: self-test controls did not all fire"; exit 1; }
    echo "VERIFY_RELEASE_SELFTEST_OK"
    exit 0
fi

# ---------------------------------------------------------------------------
[ -n "$PROV" ] || fail "--provenance is required (download it from the release page)"
[ -f "$PROV" ] || fail "no such provenance file: $PROV"
[ "${#ARTIFACTS[@]}" -gt 0 ] || fail "give at least one downloaded artifact to check"
for a in "${ARTIFACTS[@]}"; do [ -f "$a" ] || fail "no such file: $a"; done

[ -n "$SIG" ] || SIG="$PROV.sig"

# The default trust root is the project's release signing key, published in
# release-channels.json. Reading it from the file next to this script is a
# convenience, not the security property: someone who fetched both from the same
# fake page has verified nothing. The value of the published key id is that it
# can be compared against a copy obtained some other way — the source
# repository, an older download, a note somebody wrote down last month.
if [ -z "$PUBKEY" ]; then
    PUBKEY=$(python3 - "$ROOT" <<'PY'
import json, os, sys
root = sys.argv[1]
for path, keys in (
    (os.path.join(root, "scripts/release-channels.json"), ("releaseSigningKey", "publicKey")),
    # Pre-2026-09-02 name for the same key material, for a copy of this script
    # sitting next to an older channel record.
    (os.path.join(root, "scripts/release-channels.json"), ("updaterTrustRoot", "publicKey")),
):
    try:
        node = json.load(open(path))
        for k in keys:
            node = node[k]
        print(node)
        break
    except Exception:
        continue
PY
)
    [ -n "$PUBKEY" ] || fail "could not determine the official public key — pass --pubkey"
fi

echo "provenance: $PROV"

# --- 1. is the provenance file itself authentic? -----------------------------
SIGNED=0
if [ "$PUBKEY" = "none" ]; then
    echo "  !! --pubkey none: the provenance file is NOT being authenticated."
elif [ ! -f "$SIG" ]; then
    echo "  !! no signature file at $SIG"
else
    if out=$(python3 "$HERE/minisign_verify.py" --pubkey "$PUBKEY" --sig "$SIG" "$PROV" 2>&1); then
        echo "  signature: $out" | head -1
        SIGNED=1
    else
        echo "$out" | sed 's/^/  /'
        fail "the provenance file is not signed by the official release key — treat every artifact next to it as untrusted"
    fi
fi

# --- 2. do the downloaded bytes match what it lists? -------------------------
STATUS=$(python3 - "$PROV" "$EXPECT_VERSION" "${ARTIFACTS[@]}" <<'PY'
import hashlib, json, os, sys

prov_path, expect_version = sys.argv[1], sys.argv[2]
artifacts = sys.argv[3:]
try:
    doc = json.load(open(prov_path))
except Exception as exc:
    print(f"FAIL\tprovenance file is not valid JSON: {exc}")
    raise SystemExit(0)

if doc.get("schema") != "idletoken-release-provenance/1":
    print(f"FAIL\tunknown provenance schema {doc.get('schema')!r}")
    raise SystemExit(0)

listed = {a["name"]: a for a in doc.get("artifacts", [])}
problems, notes = [], []

notes.append(f"version {doc.get('version')} · platform {doc.get('platform')} "
             f"· commit {str(doc.get('source', {}).get('commit'))[:12]} "
             f"· engine pin {str(doc.get('engine', {}).get('llamacppPin'))[:7]}")
if doc.get("source", {}).get("dirty"):
    notes.append("!! built from a DIRTY worktree — these bytes cannot be reproduced from that commit")
if expect_version and doc.get("version") != expect_version:
    problems.append(f"provenance is for version {doc.get('version')}, you expected {expect_version}")

for path in artifacts:
    name = os.path.basename(path)
    entry = listed.get(name)
    if entry is None:
        problems.append(
            f"{name} is NOT listed in this provenance record "
            f"(it lists: {', '.join(sorted(listed)) or 'nothing'})"
        )
        continue
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != entry.get("sha256"):
        problems.append(
            f"{name} DIGEST MISMATCH\n      expected {entry.get('sha256')}\n      got      {got}"
        )
    else:
        size = os.path.getsize(path)
        if entry.get("bytes") is not None and size != entry["bytes"]:
            problems.append(f"{name} is {size} bytes, the record says {entry['bytes']}")
        else:
            notes.append(f"ok {name} sha256 {got[:16]}… ({size} bytes)")

for n in notes:
    print(f"NOTE\t{n}")
for p in problems:
    print(f"FAIL\t{p}")
if not problems:
    print("PASS\t")
PY
)

printf '%s\n' "$STATUS" | while IFS=$'\t' read -r kind msg; do
    case "$kind" in
        NOTE) printf '  %s\n' "$msg" ;;
        FAIL) printf '  [BAD] %s\n' "$msg" ;;
    esac
done

if printf '%s\n' "$STATUS" | grep -q '^FAIL'; then
    fail "the downloaded files do not match the signed release record"
fi

if [ "$SIGNED" = 1 ]; then
    echo "VERIFY_RELEASE_OK"
    exit 0
fi
echo "VERIFY_RELEASE_UNSIGNED: the digests match, but nothing proved this record is ours — get the .sig from the official release page"
exit 2
