#!/usr/bin/env bash
# G-RELEASE-ID — release identity and supply-chain gate.
#
# Threat register coverage: DIST-05/06/08/09/10, OPS-02, OPS-03, OPS-04,
# OPS-09, OPS-12, CHAIN-07.
#
# The question this gate answers is narrow and it is the one users cannot check
# for themselves: *is this tree still able to produce a release whose identity
# somebody outside the project can verify?* Everything downstream of that — the
# "did I download the official installer" check, the ability to notice a
# key-compromise release — is worthless if this drifts.
#
# 2026-09-02: the in-app updater was removed, so the signed provenance record is
# now the only verification path there is, for first installs and upgrades
# alike. Checks that existed only to guard the updater were not deleted with it;
# they were re-aimed at what still carries the property (R1, R2) or at keeping
# the removal true (R4). Deleting them would have quietly handed back the
# coverage the register claims for DIST-08, OPS-03, OPS-04 and OPS-12.
#
# **Every check carries its own positive control.** This repo has twice shipped
# a check that could not fail (a grep pattern that never matched, a `grep -P` on
# BSD grep counting nothing). A supply-chain check that cannot go red is worse
# than none, because it is quoted as evidence.
#
# Local gate: no cluster nodes, no GPU, no weights, no network.
#
# Contract: prints exactly one G_RELEASE_ID_(OK|FAIL|SKIP) line last.
set -u

cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD

FAILED=""
SKIPPED=""
ok()   { echo "  [ok] $*"; }
bad()  { echo "  [BAD] $*"; FAILED="${FAILED}
  - $*"; }
skip() { echo "  [skip] $*"; SKIPPED="${SKIPPED:+$SKIPPED; }$*"; }
note() { echo "  ---- $*"; }

command -v python3 >/dev/null 2>&1 || { echo "G_RELEASE_ID_SKIP: python3 not available"; exit 0; }

CONF="$ROOT/client/src-tauri/tauri.conf.json"
CHANNELS="$ROOT/scripts/release-channels.json"
REGISTRY="$ROOT/scripts/release-escape-hatches.tsv"
PINNED_KEYID="28F23C3CE24BFDE9"

# ===========================================================================
# R1 — the published release signing key is the one we actually sign with
# ===========================================================================
# Until 2026-09-02 this key was also the in-app updater's trust root, and the
# check compared the client's compiled-in pin against the published copy. The
# updater is gone; the key is not — it signs the provenance record, which is now
# the ONLY thing standing between a user and a repacked installer, on the first
# install and on every upgrade alike.
#
# That makes the published copy load-bearing in a way it was not before: it is
# no longer a fingerprint users can also read off their own client, it is the
# only place the fingerprint exists. So this checks the published record against
# itself — the advertised key id must be the key id embedded in the advertised
# key material — and proves it can notice tampering before its green counts.
# A swapped publicKey left under the familiar keyId is exactly the edit an
# attacker with commit access would make.
note "R1 published release signing key"
keyid_of() {  # $1 = a release-channels.json path -> key id, or empty
    python3 - "$1" <<'PY'
import base64, json, sys
try:
    d = json.load(open(sys.argv[1]))
    pk = (d.get("releaseSigningKey") or d["updaterTrustRoot"])["publicKey"]
    raw = base64.b64decode(base64.b64decode(pk).decode().splitlines()[1])
    if raw[:2] != b"Ed" or len(raw) != 42:
        print(""); raise SystemExit(0)
    print(raw[2:10][::-1].hex().upper())
except Exception:
    print("")
PY
}
if [ ! -f "$CHANNELS" ]; then
    bad "no scripts/release-channels.json — the release signing key is published nowhere (OPS-12: nothing for a user to compare against)"
else
    # Positive control FIRST: a copy with DIFFERENT key material must read
    # differently, even though its advertised keyId is untouched.
    ctl=$(mktemp -d "${TMPDIR:-/tmp}/idletoken-relid.XXXXXX")
    python3 - "$CHANNELS" "$ctl/tampered.json" <<'PY'
import base64, json, sys
d = json.load(open(sys.argv[1]))
node = d.get("releaseSigningKey") or d["updaterTrustRoot"]
pk = base64.b64decode(node["publicKey"]).decode()
head, body = pk.splitlines()[0], base64.b64decode(pk.splitlines()[1])
evil = body[:2] + bytes(b ^ 0xFF for b in body[2:10]) + body[10:]
node["publicKey"] = base64.b64encode(
    (head + "\n" + base64.b64encode(evil).decode() + "\n").encode()).decode()
json.dump(d, open(sys.argv[2], "w"))
PY
    if [ "$(keyid_of "$ctl/tampered.json")" = "$PINNED_KEYID" ]; then
        bad "CONTROL: swapped key material still read as $PINNED_KEYID — this check does not check"
    else
        ok "control: swapping the published key material changes what this check reads"
    fi
    rm -rf "$ctl"

    got=$(keyid_of "$CHANNELS")
    adv=$(python3 -c "
import json
try:
    d = json.load(open('$CHANNELS'))
    print((d.get('releaseSigningKey') or d['updaterTrustRoot'])['keyId'])
except Exception: print('')" 2>/dev/null)
    if [ -z "$got" ]; then
        bad "release-channels.json publishes no readable minisign public key — a downloader has nothing to verify a release against"
    elif [ "$got" != "$PINNED_KEYID" ]; then
        bad "release-channels.json publishes key material for $got, NOT $PINNED_KEYID — every key id users wrote down would stop matching"
    elif [ "$adv" != "$got" ]; then
        bad "release-channels.json advertises keyId $adv but its key material is $got — users would be told to compare the wrong fingerprint"
    else
        ok "the published key material is $PINNED_KEYID and its advertised key id agrees"
    fi

    # And the tool we tell people to run has to resolve to that key with no
    # arguments. If the lookup path breaks, verify_release.sh does not fail
    # open — but it does stop working, which is how a control stops being run.
    vt=$(mktemp -d "${TMPDIR:-/tmp}/idletoken-relid-vr.XXXXXX")
    printf '{"artifacts":[]}' > "$vt/p.json"; printf 'not-a-signature' > "$vt/p.json.sig"
    printf 'x' > "$vt/thing.bin"
    vout=$("$ROOT/scripts/verify_release.sh" --provenance "$vt/p.json" "$vt/thing.bin" 2>&1)
    case "$vout" in
        *"could not determine the official public key"*)
            bad "verify_release.sh cannot find the published key by itself — the command in release-channels.json fails for every user who follows it" ;;
        *)
            ok "verify_release.sh resolves the published key with no --pubkey argument" ;;
    esac
    rm -rf "$vt"
fi

# ===========================================================================
# R2 — private-key custody (OPS-04)
# ===========================================================================
# Retiring the in-app updater did NOT retire this key: it signs the provenance
# record, so it is still the project's single release identity. Two ways it ends
# a product: losing it (no release can ever be verified against the fingerprint
# people already have) and using the WRONG one — which is silent, because
# signing succeeds and verification succeeds against the signer's own pubkey,
# and only the users find out. So when a key is present, prove it is THE key by
# signing a nonce and verifying against the PUBLISHED public key.
#
# The file is still named updater.key. Renaming a non-regenerable key file buys
# tidiness and risks the backup nobody has tested; the name is not the identity.
note "R2 signing key custody"
KEY_PATH="${TAURI_SIGNING_PRIVATE_KEY_PATH:-$HOME/.idletoken/updater.key}"
SIGNER="$ROOT/client/node_modules/.bin/tauri"
if [ ! -f "$KEY_PATH" ]; then
    skip "no release private key at $KEY_PATH (normal on a build node; the control machine holds it)"
elif [ ! -x "$SIGNER" ]; then
    skip "no Tauri signer at $SIGNER (run pnpm install in client/) — key custody not proven"
else
    case "$KEY_PATH" in
        "$ROOT"/*) bad "the release private key is INSIDE the repository at $KEY_PATH — move it out; a key in the tree is one `git add -A` from being public forever" ;;
        *) ok "the private key lives outside the repository" ;;
    esac
    mode=$(stat -f '%Lp' "$KEY_PATH" 2>/dev/null || stat -c '%a' "$KEY_PATH" 2>/dev/null)
    case "$mode" in
        600|400) ok "key file mode is $mode" ;;
        "")      skip "could not read the key file mode on this platform" ;;
        *)       bad "the release private key is mode $mode — every local process can read it; chmod 600 $KEY_PATH" ;;
    esac
    T=$(mktemp -d "${TMPDIR:-/tmp}/idletoken-keymatch.XXXXXX")
    head -c 256 /dev/urandom > "$T/nonce.bin" 2>/dev/null
    if ! "$SIGNER" signer sign -f "$KEY_PATH" -p "${TAURI_SIGNING_PRIVATE_KEY_PASSWORD:-}" "$T/nonce.bin" >/dev/null 2>&1; then
        bad "the key at $KEY_PATH could not sign anything (wrong password, or the file is not a signing key)"
    else
        PINNED_PUB=$(python3 -c "import json;print(json.load(open('$CHANNELS'))['releaseSigningKey']['publicKey'])" 2>/dev/null)
        if python3 "$ROOT/scripts/minisign_verify.py" --pubkey "$PINNED_PUB" --sig "$T/nonce.bin.sig" "$T/nonce.bin" >/dev/null 2>&1; then
            ok "the private key on this machine IS the published release key $PINNED_KEYID (proved by signing a nonce)"
        else
            bad "the private key at $KEY_PATH is NOT the published release key $PINNED_KEYID — verify_release.sh would reject every artifact signed here"
        fi
        # Control: the same verification against a different key must refuse,
        # otherwise the line above proves nothing.
        "$SIGNER" signer generate -w "$T/other" --password "" -f >/dev/null 2>&1
        if python3 "$ROOT/scripts/minisign_verify.py" --pubkey "$T/other.pub" --sig "$T/nonce.bin.sig" "$T/nonce.bin" >/dev/null 2>&1; then
            bad "CONTROL: the nonce signature verified against an unrelated public key — the key-match check is not checking"
        else
            ok "control: the same signature is refused by an unrelated public key"
        fi
    fi
    rm -rf "$T"
fi

# ===========================================================================
# R3 — escape-hatch registry (OPS-03)
# ===========================================================================
# The property: nobody can add an environment variable that weakens a check
# without it appearing in scripts/release-escape-hatches.tsv. The registry is
# not the defence; the SCAN is. The registry is what the scan compares against.
note "R3 escape-hatch registry"
scan_hatches() {  # $1.. = roots -> one hatch-shaped env name per line
    local roots=("$@")
    for d in "${roots[@]}"; do
        [ -e "$d" ] || continue
        find "$d" -type f \( -name '*.c' -o -name '*.h' -o -name '*.m' -o -name '*.rs' \) \
             ! -path '*/ds4x/*' -print0
    done | xargs -0 grep -hoE '"(IDLETOKEN|GGML|LLAMA)_[A-Z0-9_]+"' 2>/dev/null \
      | tr -d '"' | sort -u \
      | grep -E '_(ALLOW|MOCK|FAKE|FORCE|SKIP|DISABLE|UNSAFE|INSECURE|PLAINTEXT|OVERRIDE|BYPASS|DEBUG|TEST|NO)(_|$)|_UPDATE_URL$|_UPDATE_PUBKEY$|_LOG_PROMPTS$|_FULL_LOGITS$|_UI_TEST$'
}
registry_names() {  # $1 = registry -> one registered name per line
    grep -v '^#' "$1" | awk -F'\t' 'NR>1 && $1 != "" {print $1}' | sort -u
}
unregistered_hatches() {  # $1 = registry; $2.. = source roots
    local registry=$1 h registered
    shift
    registered=$(registry_names "$registry")
    while IFS= read -r h; do
        [ -n "$h" ] || continue
        printf '%s\n' "$registered" | grep -qx "$h" || printf '%s\n' "$h"
    done <<EOF
$(scan_hatches "$@")
EOF
}
silent_runtime_hatches() {  # $1 = registry -> NAME=SURFACE per refused row
    awk -F'\t' '
        $1 !~ /^#/ && $1 != "NAME" && $1 != "" &&
        ($2 == "shipped-binary" || $2 == "production-runtime") &&
        $3 == "silent" {print $1 "=" $2}
    ' "$1"
}
registry_runtime_policy_ok() {  # $1 = registry
    [ -z "$(silent_runtime_hatches "$1")" ]
}
if [ ! -f "$REGISTRY" ]; then
    bad "no scripts/release-escape-hatches.tsv — nothing enumerates what an environment variable can do to a shipped build"
else
    # Positive controls FIRST, on a copied registry and throwaway source. The
    # first proves a future source hatch is refused until it is registered. The
    # second is deliberately a valid five-column row, so it can go red only
    # because shipped runtime hatches may not be classified as silent.
    ctl=$(mktemp -d "${TMPDIR:-/tmp}/idletoken-hatchctl.XXXXXX")
    cp "$REGISTRY" "$ctl/registry.tsv"
    printf 'int f(void){ return getenv("IDLETOKEN_ALLOW_UNREGISTERED_CONTROL") != 0; }\n' > "$ctl/fake.c"
    if unregistered_hatches "$ctl/registry.tsv" "$ctl" | grep -qx 'IDLETOKEN_ALLOW_UNREGISTERED_CONTROL'; then
        ok "control: a newly introduced source hatch is refused until it is registered"
    else
        bad "CONTROL: an unregistered planted source hatch stayed green — the registry comparison does not enforce registration"
    fi
    printf 'IDLETOKEN_ALLOW_CONFIG_CONTROL\tshipped-binary\tconfig\tcontrol/fake.c:2\tAllowed config control row.\n' >> "$ctl/registry.tsv"
    printf 'IDLETOKEN_FROZEN_SILENT_CONTROL\tfrozen\tsilent\tcontrol/fake.c:3\tAllowed frozen-surface control row.\n' >> "$ctl/registry.tsv"
    printf 'IDLETOKEN_ALLOW_LOUD_CONTROL\tproduction-runtime\tloud\tcontrol/fake.c:4\tAllowed loud runtime control row.\n' >> "$ctl/registry.tsv"
    if registry_runtime_policy_ok "$ctl/registry.tsv"; then
        ok "control: config enforcement, frozen surfaces and loud runtime hatches remain allowed"
    else
        bad "CONTROL: the narrow runtime policy rejected config, frozen or loud semantics"
    fi
    printf 'IDLETOKEN_ALLOW_SILENT_CONTROL\tshipped-binary\tsilent\tcontrol/fake.c:1\tSyntactically valid anti-vacuous control row.\n' >> "$ctl/registry.tsv"
    ctl_shape=$(awk -F'\t' '$1 == "IDLETOKEN_ALLOW_SILENT_CONTROL" {print NF ":" $2 ":" $3}' "$ctl/registry.tsv")
    if [ "$ctl_shape" != "5:shipped-binary:silent" ]; then
        bad "CONTROL: failed to plant an exact five-column shipped-binary/silent registry row"
    elif registry_runtime_policy_ok "$ctl/registry.tsv"; then
        bad "CONTROL: a syntactically valid shipped-binary/silent registry row stayed green — the runtime policy is not fail closed"
    elif silent_runtime_hatches "$ctl/registry.tsv" | grep -qx 'IDLETOKEN_ALLOW_SILENT_CONTROL=shipped-binary'; then
        ok "control: a syntactically valid shipped-binary/silent registry row makes the policy go red"
    else
        bad "CONTROL: the silent-row policy went red without identifying the planted shipped-binary hatch"
    fi
    rm -rf "$ctl"

    registered=$(registry_names "$REGISTRY")
    unregistered=$(unregistered_hatches "$REGISTRY" src include client/src-tauri/src)
    if [ -n "$unregistered" ]; then
        bad "escape hatches read by production code but not registered: $(echo "$unregistered" | tr '\n' ' ') — add a row to $REGISTRY saying what it does and whether it is compiled-out, loud or silent"
    else
        n=$(printf '%s\n' "$registered" | grep -c . || true)
        ok "every hatch-shaped variable in production sources is registered ($n rows)"
    fi

    # A registry row is only useful if its enforcement column is a real value.
    badrows=$(grep -v '^#' "$REGISTRY" | awk -F'\t' 'NR>1 && $1 != "" && $3 !~ /^(compiled-out|loud|silent|config)$/ {print $1"="$3}')
    if [ -n "$badrows" ]; then
        bad "registry rows with an unknown enforcement value: $(echo "$badrows" | tr '\n' ' ')"
    else
        ok "every registry row declares a known enforcement class"
    fi

    # A silent hatch in a shipped binary (or any future surface explicitly
    # classed as production-runtime) is a release blocker. Config rows remain
    # valid, as do frozen surfaces and loud/compiled-out enforcement.
    if registry_runtime_policy_ok "$REGISTRY"; then
        ok "no shipped-binary or production-runtime hatch is classified silent"
    else
        silent_runtime=$(silent_runtime_hatches "$REGISTRY")
        bad "registry classifies shipped runtime escape hatches as silent: $(echo "$silent_runtime" | tr '\n' ' ') — use compiled-out, loud or config enforcement before release"
    fi

    # Keep the existing open-defect warning for non-runtime surfaces without
    # weakening the fail-closed policy above.
    silent_other=$(grep -v '^#' "$REGISTRY" | awk -F'\t' 'NR>1 && $3 == "silent" && $2 !~ /^(shipped-binary|production-runtime)$/ {print $1}')
    if [ -n "$silent_other" ]; then
        echo "  [warn] non-runtime silent hatches (open, tracked as OPS-03 residual): $(echo "$silent_other" | tr '\n' ' ')"
    fi
fi

# ===========================================================================
# R4 — the client cannot install anything by itself (DIST-08, OPS-03)
# ===========================================================================
# This slot used to assert that the updater's env overrides were compiled out of
# release builds. The updater was removed on 2026-09-02, so that assertion is
# now vacuous — and a vacuous check is the failure mode this gate exists to
# avoid. What replaces it guards the property the removal bought:
#
#   an installed client has no code path that fetches, verifies or executes a
#   new build, so there is nothing to point at a hostile feed in the first place.
#
# That is worth a standing check rather than a changelog line, because it is
# easy to undo by accident. `tauri add updater` writes the plugin, the config
# and the capability in one command, and the resulting client would auto-install
# whatever a feed served — with none of R1/R2/R4's old checks watching, because
# they were deleted along with the code they guarded. Re-adding it has to be a
# deliberate act that turns this gate red first.
note "R4 no self-installing update path"
updater_traces() {  # $1 = a tree root -> one trace per line, silence = clean
    python3 - "$1" <<'PY'
import json, os, re, sys
root = sys.argv[1]
found = []
conf = os.path.join(root, "client/src-tauri/tauri.conf.json")
try:
    d = json.load(open(conf))
    if "updater" in (d.get("plugins") or {}):
        found.append("tauri.conf.json configures the updater plugin")
    if (d.get("bundle") or {}).get("createUpdaterArtifacts"):
        found.append("tauri.conf.json still asks for updater artifacts")
except FileNotFoundError:
    pass
cargo = os.path.join(root, "client/src-tauri/Cargo.toml")
try:
    # Comment lines are dropped first: this file explains that the dependency is
    # absent, and a substring scan reads that explanation as the dependency.
    t = "\n".join(l for l in open(cargo).read().splitlines()
                  if not l.lstrip().startswith("#"))
    if re.search(r'(^|[\s"\'{,])tauri-plugin-updater\s*[=\.]', t, re.M):
        found.append("Cargo.toml depends on tauri-plugin-updater")
    if re.search(r'^\s*updater-override\s*=', t, re.M):
        found.append("Cargo.toml still declares the updater-override feature")
except FileNotFoundError:
    pass
if os.path.exists(os.path.join(root, "client/src-tauri/src/update.rs")):
    found.append("client/src-tauri/src/update.rs is back")
for sub, pats in (("client/src-tauri/src", (r"\bupdater::", r"tauri_plugin_updater")),
                  ("client/src", (r"@tauri-apps/plugin-updater", r"\bcheckUpdate\b", r"\binstallUpdate\b")),
                  ("client/src-tauri/capabilities", (r'"updater:',))):
    base = os.path.join(root, sub)
    for dirpath, _, names in os.walk(base):
        for n in names:
            p = os.path.join(dirpath, n)
            try:
                body = open(p, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            for pat in pats:
                if re.search(pat, body):
                    found.append(f"{os.path.relpath(p, root)} references the updater ({pat})")
                    break
print("\n".join(found))
PY
}
# Positive control FIRST, on a copy with the plugin AND the crate planted back
# — the exact pair `tauri add updater` leaves behind. Both are asserted
# separately: the Cargo scan ignores comment lines (this file documents the
# dependency's ABSENCE, and a substring scan reads that as its presence), and a
# comment filter is one bad regex away from ignoring the real line too.
ctl=$(mktemp -d "${TMPDIR:-/tmp}/idletoken-r4.XXXXXX")
mkdir -p "$ctl/client/src-tauri"
if [ -f "$CONF" ] && cp "$CONF" "$ctl/client/src-tauri/tauri.conf.json"; then
    python3 - "$ctl/client/src-tauri/tauri.conf.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
d.setdefault("plugins", {})["updater"] = {"endpoints": ["https://example.invalid/latest.json"],
                                          "pubkey": "planted"}
json.dump(d, open(sys.argv[1], "w"))
PY
    traces=$(updater_traces "$ctl")
    case "$traces" in
        *"configures the updater plugin"*) ok "control: a re-added updater plugin is detected" ;;
        *) bad "CONTROL: a planted updater plugin was not detected — this check does not check" ;;
    esac
    printf '[dependencies]\ntauri-plugin-updater = "2"\n' > "$ctl/client/src-tauri/Cargo.toml"
    case "$(updater_traces "$ctl")" in
        *"depends on tauri-plugin-updater"*) ok "control: a re-added updater crate is detected" ;;
        *) bad "CONTROL: a planted tauri-plugin-updater dependency was not detected — the comment filter blinded the scan" ;;
    esac
else
    bad "could not build the R4 control (no $CONF) — the no-self-install check is unproven"
fi
rm -rf "$ctl"

r4=$(updater_traces "$ROOT")
if [ -n "$r4" ]; then
    while IFS= read -r line; do [ -n "$line" ] && bad "self-installing update path returned: $line"; done <<EOF
$r4
EOF
else
    ok "no updater plugin, dependency, module, capability or frontend call — the client installs nothing by itself"
fi

# ===========================================================================
# R5 — signing material cannot reach the public mirror (OPS-02)
# ===========================================================================
# The mirror's forbidden-pattern list is the only layer that catches a secret
# pasted into a file that is otherwise supposed to be public. It covers PEM
# private keys — and did NOT cover the format our updater key is actually in,
# which is what the updater key would leak as.
note "R5 signing material out of the public surface"
# The fixed header of a minisign/rsign SECRET key, and the base64 form the key
# file is stored in (Tauri base64-encodes the whole file, and that is the form a
# CI variable or a pasted TAURI_SIGNING_PRIVATE_KEY would take). The PUBLIC key
# shares the "untrusted comment:" prefix and must NOT match — it is published on
# purpose, and a pattern that flags it would red-flag every release.
#
# ⚠ The patterns are ASSEMBLED rather than written out, and the base64 forms are
# derived rather than pasted. This file is published (PUBLIC_SCRIPTS), and
# sync-public.sh's forbidden-pattern scan would — correctly — refuse a mirror
# containing the literal secret-key header. A checker that cannot be shipped is
# not a checker for the people who most need it.
_SK_TAIL='encrypted secret key'
b64_prefix() {  # base64 of the longest 3-byte-aligned prefix: a stable substring
    python3 -c "import base64,sys;s=sys.argv[1];print(base64.b64encode(s[:len(s)//3*3].encode()).decode())" "$1"
}
SECRET_PATTERNS=(
    "(minisign|rsign) $_SK_TAIL"
    "$(b64_prefix "untrusted comment: rsign $_SK_TAIL")"
    "$(b64_prefix "untrusted comment: minisign $_SK_TAIL")"
    'BEGIN [A-Z ]*PRIVATE KEY'
)
# Control first: a planted key must be found, and the published PUBLIC key must
# not be. Both halves matter — a pattern that matches everything is as useless
# as one that matches nothing, and it would red-flag every release.
ctl=$(mktemp -d "${TMPDIR:-/tmp}/idletoken-secretctl.XXXXXX")
printf 'untrusted comment: rsign %s\nRWRTY0Iyfake\n' "$_SK_TAIL" > "$ctl/planted.key"
python3 -c "
import base64
open('$ctl/planted_b64.txt','w').write(base64.b64encode(open('$ctl/planted.key','rb').read()).decode())
"
cp "$CONF" "$ctl/published-pubkey.json" 2>/dev/null || true
hit_plain=0 hit_b64=0 hit_pub=0
for pat in "${SECRET_PATTERNS[@]}"; do
    grep -rIlE "$pat" "$ctl/planted.key"     >/dev/null 2>&1 && hit_plain=1
    grep -rIlE "$pat" "$ctl/planted_b64.txt" >/dev/null 2>&1 && hit_b64=1
    [ -f "$ctl/published-pubkey.json" ] && grep -rIlE "$pat" "$ctl/published-pubkey.json" >/dev/null 2>&1 && hit_pub=1
done
[ "$hit_plain" = 1 ] && ok "control: a plaintext minisign secret key is detected" \
                     || bad "CONTROL: a planted minisign secret key was NOT detected"
[ "$hit_b64" = 1 ]   && ok "control: the base64 key-file form is detected too (this is the form a CI variable carries)" \
                     || bad "CONTROL: the base64 form of a secret key was NOT detected"
[ "$hit_pub" = 0 ]   && ok "control: the published PUBLIC key does not trip the secret patterns" \
                     || bad "CONTROL: the secret-key patterns also match the public key — every release would false-alarm"
rm -rf "$ctl"

# Now the real scan, over everything that can reach the mirror. The excluded
# directories are the ones the mirror never carries (VENDOR_EXCLUDES, build
# output, internal results); sync-public.sh is excluded because it is the file
# that DEFINES the patterns and is itself not published.
leaks=""
for pat in "${SECRET_PATTERNS[@]}"; do
    out=$(grep -rIlE "$pat" "$ROOT" \
            --exclude-dir=.git --exclude-dir=node_modules --exclude-dir=target \
            --exclude-dir=dist --exclude-dir=results --exclude-dir=llama.cpp \
            --exclude-dir=ds4 --exclude-dir=gguf-tools 2>/dev/null \
          | grep -v '/scripts/sync-public\.sh$' || true)
    [ -n "$out" ] && leaks="$leaks $out"
done
if [ -n "$leaks" ]; then
    bad "signing-key material found in the tree:$leaks"
else
    ok "no signing-key material anywhere in the mirror-eligible tree"
fi

# And the mirror's own list has to carry these patterns, or the check above only
# holds for as long as this gate is the thing that runs. sync-public.sh is what
# actually fails a publish; this gate only notices.
if [ ! -f "$ROOT/scripts/sync-public.sh" ]; then
    skip "no sync-public.sh in this checkout (mirror pattern coverage not checked)"
elif grep -q "$_SK_TAIL" "$ROOT/scripts/sync-public.sh"; then
    ok "sync-public.sh refuses a mirror containing minisign/rsign secret-key material"
else
    # Wording note: this message must not itself contain a forbidden pattern.
    # It did on the first run — the PEM header written out as prose made THIS
    # file the scan's only hit, which is a neat demonstration that the scan
    # works and a nuisance that would block every publish.
    bad "sync-public.sh has no forbidden pattern for minisign/rsign SECRET keys — the updater private key would be published without tripping anything, because it is not in PEM form and the PEM pattern is the only key pattern there"
fi

# ===========================================================================
# R6 — release identity metadata is self-consistent (OPS-12, DIST-06)
# ===========================================================================
note "R6 release identity metadata"
if [ ! -f "$CHANNELS" ]; then
    bad "no scripts/release-channels.json — there is no published statement of which download origins are official (DIST-02/03)"
else
    # Drift between what the client DOES and what we publish is the whole risk
    # here: a user told to compare against a fingerprint or an origin that the
    # binary no longer uses is being given a check that always passes.
    #
    # Also asserted: we never publish a signing status we do not have. Claiming
    # a signed Windows installer would train users to click through the very
    # SmartScreen box the official installer produces (DIST-04).
    r6_problems() {  # $1 = a tree root -> one problem per line, silence = consistent
        python3 - "$1" <<'PY'
import json, os, re, sys
root = sys.argv[1]
ch = json.load(open(os.path.join(root, "scripts/release-channels.json")))
problems = []
# The download page is where we send every user for every upgrade now that
# nothing is pushed to them. A published URL outside our own origins is the
# same failure as the old feed drift, with a bigger blast radius: it is the
# one address a user is told to trust without checking anything else.
dl = (ch.get("downloads") or {}).get("stable")
prefix = (ch.get("officialOrigins") or {}).get("releasePagePrefix")
if not dl:
    problems.append("release-channels.json publishes no downloads.stable — users are told to update but not from where")
elif not prefix or not dl.startswith(prefix):
    problems.append(f"downloads.stable {dl} is not under the published official release origin {prefix}")
if ch.get("updateFeeds"):
    problems.append("release-channels.json still publishes updateFeeds — the in-app updater was removed on 2026-09-02 and a live feed URL invites someone to serve it")
conf = json.load(open(os.path.join(root, "client/src-tauri/tauri.conf.json")))
versions = {"tauri.conf.json": conf["version"],
            "package.json": json.load(open(os.path.join(root, "client/package.json")))["version"]}
m = re.search(r'^version = "([^"]+)"', open(os.path.join(root, "client/src-tauri/Cargo.toml")).read(), re.M)
versions["Cargo.toml"] = m.group(1) if m else "?"
settings = open(os.path.join(root, "client/src/settings.ts")).read()
vm = re.search(r'import\s*\{\s*version\s+as\s+(\w+)\s*\}\s*from\s*["\']\.\./package\.json["\']', settings)
am = re.search(r'export\s+const\s+APP_VERSION\s*=\s*([^;]+);', settings)
if vm and am and am.group(1).strip() == vm.group(1):
    versions["About UI"] = versions["package.json"]
else:
    literal = re.fullmatch(r'["\']([^"\']+)["\']', am.group(1).strip()) if am else None
    versions["About UI"] = literal.group(1) if literal else "?"
if len(set(versions.values())) != 1:
    problems.append("version disagreement: " + ", ".join(f"{k}={v}" for k, v in versions.items()))
cs = ch.get("codeSigning", {})
if cs.get("windows", {}).get("authenticode"):
    problems.append("release-channels.json claims Windows Authenticode signing that the build scripts do not perform")
if cs.get("macos", {}).get("notarized"):
    problems.append("release-channels.json claims macOS notarization that the build scripts do not perform")
# DIST-07: the published support floor must be a version that exists. A floor
# ABOVE the current release would tell the platform to refuse every client,
# including the newest one.
def _v(s):
    return tuple(int(x) for x in re.findall(r"\d+", s or "0"))
msv = ch.get("minimumSupportedVersion")
if not msv:
    problems.append("release-channels.json publishes no minimumSupportedVersion (DIST-07: the platform has no floor to enforce against stale builds)")
elif _v(msv) > _v(versions["tauri.conf.json"]):
    problems.append(f"minimumSupportedVersion {msv} is newer than the current release {versions['tauri.conf.json']} — every client would be below the floor")
print("\n".join(problems))
PY
    }

    # Positive control FIRST, on a copy of the four files with the published
    # stable feed pointed somewhere else — the exact drift that would leave
    # users comparing against an origin the binary no longer uses.
    ctl=$(mktemp -d "${TMPDIR:-/tmp}/idletoken-r6.XXXXXX")
    mkdir -p "$ctl/scripts" "$ctl/client/src-tauri/src" "$ctl/client/src"
    cp "$CHANNELS" "$ctl/scripts/" && cp "$CONF" "$ctl/client/src-tauri/" \
        && cp "$ROOT/client/src-tauri/Cargo.toml" "$ctl/client/src-tauri/" \
        && cp "$ROOT/client/package.json" "$ctl/client/" \
        && cp "$ROOT/client/src/settings.ts" "$ctl/client/src/"
    python3 - "$ctl/scripts/release-channels.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
d["downloads"]["stable"] = "https://idletoken-downloads.example.invalid/latest"
json.dump(d, open(sys.argv[1], "w"))
PY
    if [ -n "$(r6_problems "$ctl")" ]; then
        ok "control: a download page outside the official origins is detected"
    else
        bad "CONTROL: download-origin drift was not detected — this check does not compare anything"
    fi
    rm -rf "$ctl"

    # A second control targets the bug this comparison is meant to prevent:
    # the shipped About note carrying a stale handwritten version while all
    # package metadata agrees.
    ctl=$(mktemp -d "${TMPDIR:-/tmp}/idletoken-r6-about.XXXXXX")
    mkdir -p "$ctl/scripts" "$ctl/client/src-tauri/src" "$ctl/client/src"
    cp "$CHANNELS" "$ctl/scripts/" && cp "$CONF" "$ctl/client/src-tauri/" \
        && cp "$ROOT/client/src-tauri/Cargo.toml" "$ctl/client/src-tauri/" \
        && cp "$ROOT/client/package.json" "$ctl/client/" \
        && cp "$ROOT/client/src/settings.ts" "$ctl/client/src/"
    python3 - "$ctl/client/src/settings.ts" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
s, n = re.subn(r'export\s+const\s+APP_VERSION\s*=\s*[^;]+;',
               'export const APP_VERSION = "0.0.0";', s, count=1)
if n != 1:
    raise SystemExit("could not plant stale About version control")
open(p, "w").write(s)
PY
    if [ -n "$(r6_problems "$ctl")" ]; then
        ok "control: a stale About-screen version is detected"
    else
        bad "CONTROL: a stale About-screen version was not detected"
    fi
    rm -rf "$ctl"

    r6=$(r6_problems "$ROOT")
    if [ -n "$r6" ]; then
        while IFS= read -r line; do [ -n "$line" ] && bad "$line"; done <<EOF
$r6
EOF
    else
        ok "the download origin, the version and the support floor agree across the client, the config and the published channel record"
    fi
fi

# ===========================================================================
# R7 — the verifier and the transparency log still work (their own controls)
# ===========================================================================
note "R7 verifier and transparency log"
if out=$(python3 "$ROOT/scripts/minisign_verify.py" --self-test 2>&1); then
    ok "independent signature verifier: RFC 8032 vectors pass and every tamper control refuses"
else
    printf '%s\n' "$out" | sed 's/^/       /'
    bad "the independent signature verifier failed its own vectors — verify_release.sh's answers mean nothing"
fi
if out=$(bash "$ROOT/scripts/release_transparency.sh" self-test 2>&1); then
    ok "transparency log: edit, delete and fork controls all fire"
else
    printf '%s\n' "$out" | sed 's/^/       /'
    bad "the release transparency log self-test failed"
fi
if [ -f "$ROOT/releases/transparency-log.jsonl" ]; then
    if out=$(bash "$ROOT/scripts/release_transparency.sh" verify 2>&1 | tail -1); then
        ok "published release log verifies: ${out#RELEASE_LOG_OK: }"
    else
        bad "the published release transparency log does NOT verify: $out"
    fi
fi
if [ -x "$SIGNER" ]; then
    if out=$(bash "$ROOT/scripts/verify_release.sh" --self-test 2>&1 | tail -1); then
        case "$out" in
            VERIFY_RELEASE_SELFTEST_OK) ok "end-user verifier: swap, edit, foreign-key and no-signature controls all refuse" ;;
            VERIFY_RELEASE_SKIP*)       skip "end-user verifier self-test skipped (${out#VERIFY_RELEASE_SKIP: })" ;;
            *)                          bad "end-user verifier self-test: $out" ;;
        esac
    else
        bad "scripts/verify_release.sh --self-test failed: $out"
    fi
else
    skip "no Tauri signer (end-user verifier self-test needs it to make a throwaway key)"
fi

# ===========================================================================
# R8 — the pinned dependencies have been looked at recently (OPS-09)
# ===========================================================================
note "R8 dependency review window"
REVIEW="$ROOT/scripts/dependency-review.json"
if [ ! -f "$REVIEW" ]; then
    bad "no scripts/dependency-review.json — nothing distinguishes 'the pin is deliberate' from 'nobody has looked at it in months'"
else
    r8=$(python3 - "$ROOT" <<'PY'
import datetime, json, os, sys
root = sys.argv[1]
d = json.load(open(os.path.join(root, "scripts/dependency-review.json")))
window = int(d.get("reviewWindowDays", 45))
today = datetime.date.today()
problems, notes = [], []
try:
    pin = open(os.path.join(root, "scripts/llamacpp-patches/UPSTREAM")).read().split()[1]
except Exception:
    pin = None
for c in d.get("components", []):
    try:
        last = datetime.date.fromisoformat(c["lastReviewed"])
    except Exception:
        problems.append(f"{c.get('name')}: lastReviewed is not an ISO date")
        continue
    age = (today - last).days
    if age > window:
        problems.append(f"{c['name']}: last reviewed {age} days ago, window is {window} — "
                        f"check {', '.join(c.get('advisorySources', []))} and update the record")
    else:
        notes.append(f"{c['name']}: reviewed {age}d ago")
    if c["name"] == "llama.cpp" and pin and c.get("reviewedPin") != pin:
        problems.append(f"llama.cpp pin moved to {pin[:12]} but the review record still refers to "
                        f"{str(c.get('reviewedPin'))[:12]} — a moved pin needs its own review")
print("\n".join(f"P\t{p}" for p in problems) + ("\n" if problems else "") +
      "\n".join(f"N\t{n}" for n in notes))
PY
)
    while IFS=$'\t' read -r kind msg; do
        [ -n "$msg" ] || continue
        case "$kind" in P) bad "$msg" ;; N) ok "$msg" ;; esac
    done <<EOF
$r8
EOF
    # Control: a record whose date is far in the past must be reported.
    ctl=$(mktemp -d "${TMPDIR:-/tmp}/idletoken-revctl.XXXXXX")
    mkdir -p "$ctl/scripts/llamacpp-patches"
    cp "$ROOT/scripts/llamacpp-patches/UPSTREAM" "$ctl/scripts/llamacpp-patches/" 2>/dev/null
    python3 - "$REVIEW" "$ctl/scripts/dependency-review.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for c in d["components"]:
    c["lastReviewed"] = "2000-01-01"
json.dump(d, open(sys.argv[2], "w"))
PY
    ctl_out=$(python3 - "$ctl" <<'PY'
import datetime, json, os, sys
root = sys.argv[1]
d = json.load(open(os.path.join(root, "scripts/dependency-review.json")))
window = int(d.get("reviewWindowDays", 45))
stale = [c["name"] for c in d["components"]
         if (datetime.date.today() - datetime.date.fromisoformat(c["lastReviewed"])).days > window]
print(",".join(stale))
PY
)
    rm -rf "$ctl"
    if [ -n "$ctl_out" ]; then
        ok "control: a stale review date is reported ($ctl_out)"
    else
        bad "CONTROL: a review record dated 2000-01-01 was not reported stale — the window is not enforced"
    fi
fi

# ===========================================================================
# R9 — impersonation-name checker still works (DIST-13)
# ===========================================================================
# Lives here rather than in the platform tests because the protected-name list
# it reads is release identity (release-channels.json). The checker itself is
# an admission control for the SERVER side; this only keeps it honest.
note "R9 impersonation-name checker"
if [ ! -f "$ROOT/scripts/impersonation_name_check.py" ]; then
    skip "scripts/impersonation_name_check.py not present"
elif out=$(python3 "$ROOT/scripts/impersonation_name_check.py" --self-test 2>&1); then
    printf '%s\n' "$out" | grep '\[ok\]' | sed 's/^ */  /'
else
    printf '%s\n' "$out" | sed 's/^/       /'
    bad "the impersonation-name checker fails its own table (homoglyph/leet cases, or it now flags legitimate names)"
fi

# ===========================================================================
# R10 — the verification tooling we tell users to run is actually publishable
# ===========================================================================
# SECURITY.md instructs a user to run verify_release.sh on a file they just
# downloaded. That instruction is worthless — worse, it is a false assurance —
# if the script is not in the public mirror. The mirror is a WHITELIST, so a
# new file is absent by default and nothing else would notice.
note "R10 user-facing verification tooling is in the public whitelist"
SYNC="$ROOT/scripts/sync-public.sh"
if [ ! -f "$SYNC" ]; then
    skip "no sync-public.sh (publish coverage not checked)"
else
    needed="minisign_verify.py verify_release.sh release-channels.json"
    # Everything SECURITY.md names must ship; the rest of the release tooling
    # is listed too because a gate nobody outside can run is our own word.
    wanted="$needed release_manifest.sh release_transparency.sh release_identity_gate.sh dependency_audit.sh release-escape-hatches.tsv dependency-review.json ops02_pattern_control.sh impersonation_name_check.py release-provenance-lib.sh"
    missing_pub=""
    for f in $wanted; do
        grep -qE "(^|[[:space:]])$(printf '%s' "$f" | sed 's/\./\\./g')([[:space:]]|$)" "$SYNC" || missing_pub="$missing_pub $f"
    done
    if [ -n "$missing_pub" ]; then
        bad "not in sync-public.sh's PUBLIC_SCRIPTS:$missing_pub — SECURITY.md tells users to run verify_release.sh, and a whitelist omission means it is simply absent from the public repository (AGENTS.md: missing whitelist = missing file). Add these lines to PUBLIC_SCRIPTS, and 'releases' to PUBLIC_PATHS for the transparency log."
    else
        ok "every script the public security documentation points at is whitelisted for the mirror"
    fi
fi

# ===========================================================================
# R11 — the server-enforced support floor ships in the gateway image (DIST-07)
# ===========================================================================
# Docker builds with platform/ as its context, so the canonical root-level
# scripts/release-channels.json is not present in the image. The gateway carries
# a generated snapshot and imports it into dist; this gate is the release-path
# bridge that refuses a stale snapshot before those self-consistent bytes ship.
note "R11 official-client support-floor snapshot"
VERSION_SNAPSHOT="$ROOT/platform/packages/gateway/src/providers/official-client-release-policy.json"
VERSION_POLICY="$ROOT/platform/packages/gateway/src/providers/official-client-version.ts"
version_snapshot_problems() { # $1 canonical channels; $2 bundled snapshot
    python3 - "$1" "$2" <<'PY'
import hashlib, json, sys
try:
    canonical = json.load(open(sys.argv[1]))
    snapshot = json.load(open(sys.argv[2]))
except Exception as e:
    print(f"unreadable version policy: {e}")
    raise SystemExit(0)
problems = []
want = hashlib.sha256(
    (str(canonical.get("schema", "")) + "\0" +
     str(canonical.get("minimumSupportedVersion", ""))).encode()
).hexdigest()
if snapshot.get("schema") != "idletoken-official-client-version-policy/1":
    problems.append("bundled policy schema is missing or unknown")
if snapshot.get("generatedFrom") != "scripts/release-channels.json":
    problems.append("bundled policy does not name the canonical source")
if snapshot.get("sourceSchema") != canonical.get("schema"):
    problems.append("bundled policy source schema drifted")
if snapshot.get("minimumSupportedVersion") != canonical.get("minimumSupportedVersion"):
    problems.append("bundled minimumSupportedVersion drifted")
if snapshot.get("sourcePolicySha256") != want:
    problems.append("bundled support-floor digest drifted")
print("\n".join(problems))
PY
}
if [ ! -d "$ROOT/platform" ]; then
    # The open-source mirror ships without platform/ on purpose, so this half
    # has nothing to compare and "missing" is not a finding there. Skipped, not
    # failed: a gate that can only ever be red in the public tree teaches
    # whoever cloned it to ignore red. Keyed on the DIRECTORY, so a private
    # checkout that really has lost the snapshot still gets the bad below.
    skip "no platform/ in this tree (open-source mirror) — the gateway support-floor snapshot is not part of it"
elif [ ! -f "$VERSION_SNAPSHOT" ]; then
    bad "no bundled gateway support-floor snapshot — the Docker image cannot read root scripts/release-channels.json"
elif [ ! -f "$VERSION_POLICY" ]; then
    bad "no gateway official-client version policy imports the bundled snapshot"
else
    ctl=$(mktemp -d "${TMPDIR:-/tmp}/idletoken-versionfloor.XXXXXX")
    cp "$VERSION_SNAPSHOT" "$ctl/snapshot.json"
    python3 - "$ctl/snapshot.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
p["minimumSupportedVersion"] = "0.0.0"
json.dump(p, open(sys.argv[1], "w"))
PY
    if [ -n "$(version_snapshot_problems "$CHANNELS" "$ctl/snapshot.json")" ]; then
        ok "control: a stale bundled support floor is refused"
    else
        bad "CONTROL: a bundled support floor that drifted to 0.0.0 stayed green"
    fi
    rm -rf "$ctl"

    r11=$(version_snapshot_problems "$CHANNELS" "$VERSION_SNAPSHOT")
    if [ -n "$r11" ]; then
        while IFS= read -r line; do [ -n "$line" ] && bad "$line"; done <<EOF
$r11
EOF
    elif ! grep -q "import bundledReleasePolicy from './official-client-release-policy.json'" "$VERSION_POLICY"; then
        bad "the gateway policy does not import the checked snapshot — tsc/Docker would not carry it into dist"
    elif ! grep -q 'COPY packages/gateway packages/gateway' "$ROOT/platform/packages/gateway/Dockerfile"; then
        bad "the gateway Dockerfile no longer copies the package that contains the support-floor snapshot"
    else
        ok "gateway Docker snapshot floor+digest match release-channels.json and the policy imports it into dist"
    fi
fi

# ===========================================================================
if [ -n "$FAILED" ]; then
    echo "G_RELEASE_ID_FAIL:$(echo "$FAILED" | tr '\n' ' ')"
    exit 1
fi
if [ -n "$SKIPPED" ]; then
    echo "G_RELEASE_ID_SKIP: $SKIPPED"
    exit 0
fi
echo "G_RELEASE_ID_OK: published signing key, key custody, escape-hatch registry, no self-installing update path, signing-material scan, identity metadata, verifier + transparency controls, dependency review window, bundled support-floor policy"
