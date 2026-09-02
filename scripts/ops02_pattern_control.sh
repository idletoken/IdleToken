#!/usr/bin/env bash
# OPS-02 — reverse verification for the signing-key patterns in sync-public.sh's
# FORBIDDEN list.
#
# AGENTS.md is explicit about this: after changing a forbidden pattern you must
# prove it with a line of text that is KNOWN to match. A pattern that matches
# nothing is a check that cannot fail, and this repo has shipped exactly that
# twice ("all green" while a real `C:\Users\<name>` sat in a published comment,
# and a `grep -P` on BSD grep counting nothing forever).
#
# The four patterns under test were added 2026-08-30 because the existing list
# only recognised PEM private keys — and the updater signing key, the single
# most damaging secret in this repository, is not PEM. It is minisign/rsign,
# and Tauri stores it base64-encoded on top. Before those patterns existed,
# pasting the updater private key into any file destined for the public mirror
# passed the scan clean.
#
# Three assertions, because two of them are the ones people forget:
#   1. KNOWN HIT   — each real-world shape of the secret is caught.
#   2. KNOWN MISS  — the deliberately PUBLISHED updater public key is NOT caught.
#                    It shares the "untrusted comment:" prefix; a pattern that
#                    flagged it would red-flag every single release.
#   3. NO FALSE POSITIVES — nothing already in the tree matches, so adding these
#                    does not break the next publish.
#
# Patterns are read FROM sync-public.sh rather than copied here, so this tests
# what is actually in force.
#
# Contract: last line is OPS02_PATTERN_CONTROL_OK, OPS02_PATTERN_CONTROL_FAIL,
# or an explicit SKIP when this public-mirror checkout does not contain the
# private mirror-generation policy that the control audits.
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SELF="$REPO/scripts/ops02_pattern_control.sh"
SYNC="$REPO/scripts/sync-public.sh"
if [ ! -f "$REPO/scripts/sync-public.sh" ]; then
    echo "OPS02_PATTERN_CONTROL_SKIP: scripts/sync-public.sh is private mirror policy and is not present in this public checkout"
    exit 0
fi
T=$(mktemp -d "${TMPDIR:-/tmp}/ops02.XXXXXX") || exit 1
trap 'rm -rf "$T"' EXIT

PATS=$(python3 - "$SYNC" <<'PY'
import re, sys
blk = re.search(r'^FORBIDDEN=\((.*?)^\)', open(sys.argv[1]).read(), re.S | re.M).group(1)
print("\n".join(re.findall(r"^\s*'([^']+)'", blk, re.M)))
PY
) || { echo "OPS02_PATTERN_CONTROL_FAIL: could not read FORBIDDEN from sync-public.sh"; exit 1; }

# The fixtures are ASSEMBLED, never written out as literals. This file is in
# scripts/ and may one day be published; a control script that trips the very
# scan it validates would block every publish, and the first draft of it did.
SK='encrypted secret key'
printf 'untrusted comment: rsign %s\nRWRTY0IyFAKE\n'    "$SK" > "$T/hit_rsign_plain.txt"
printf 'untrusted comment: minisign %s\nRWRTFAKE\n'     "$SK" > "$T/hit_minisign_plain.txt"
python3 -c "import base64,sys;open(sys.argv[2],'w').write(base64.b64encode(open(sys.argv[1],'rb').read()).decode())" \
        "$T/hit_rsign_plain.txt" "$T/hit_rsign_b64.txt"
python3 -c "import base64,sys;open(sys.argv[2],'w').write(base64.b64encode(open(sys.argv[1],'rb').read()).decode())" \
        "$T/hit_minisign_plain.txt" "$T/hit_minisign_b64.txt"
printf 'set "%s=%s"\n' "TAURI_SIGNING_PRIVATE_KEY" "dW50cnVzdGVkIGNvbW1lbnQ6" > "$T/hit_envvar.txt"

# Known-MISS fixture: the real, deliberately published updater PUBLIC key.
cp "$REPO/client/src-tauri/tauri.conf.json" "$T/miss_published_pubkey.json" 2>/dev/null

rc=0
echo "== known HITs (each must be caught by at least one FORBIDDEN pattern) =="
for f in "$T"/hit_*.txt; do
    found=""
    while IFS= read -r p; do
        [ -n "$p" ] || continue
        grep -rInE "$p" "$f" >/dev/null 2>&1 && { found="$p"; break; }
    done <<< "$PATS"
    if [ -n "$found" ]; then
        echo "  [ok]  $(basename "$f")  <-  $found"
    else
        echo "  [BAD] $(basename "$f") NOT CAUGHT — the pattern does not match the real shape"; rc=1
    fi
done

echo "== known MISS (the published PUBLIC key must trip nothing) =="
if [ ! -f "$T/miss_published_pubkey.json" ]; then
    echo "  [skip] no tauri.conf.json in this checkout"
else
    hit=0
    while IFS= read -r p; do
        [ -n "$p" ] || continue
        grep -rInE "$p" "$T/miss_published_pubkey.json" >/dev/null 2>&1 && {
            echo "  [BAD] tripped by: $p — this would red-flag every release"; hit=1; rc=1; }
    done <<< "$PATS"
    [ "$hit" = 0 ] && echo "  [ok]  tauri.conf.json (published updater public key) trips nothing"
fi

echo "== no false positives in the current tree (the four added patterns) =="
# Excluded: sync-public.sh (it DEFINES the patterns) and this file (it builds
# fixtures from them). Both exclusions are narrow and named, not a directory.
for p in "(minisign|rsign) $SK" \
         "$(python3 -c "import base64,sys;s=sys.argv[1];print(base64.b64encode(s[:len(s)//3*3].encode()).decode())" "untrusted comment: rsign $SK")" \
         "$(python3 -c "import base64,sys;s=sys.argv[1];print(base64.b64encode(s[:len(s)//3*3].encode()).decode())" "untrusted comment: minisign $SK")" \
         "TAURI_SIGNING_PRIVATE_KEY *= *[A-Za-z0-9+/]"; do
    out=$(grep -rIlE "$p" "$REPO" --exclude-dir=.git --exclude-dir=node_modules \
              --exclude-dir=target --exclude-dir=dist --exclude-dir=llama.cpp \
              --exclude-dir=results 2>/dev/null \
          | grep -v '/scripts/sync-public\.sh$' | grep -Fxv "$SELF" || true)
    if [ -n "$out" ]; then
        echo "  [BAD] $p also matches existing files:"; printf '        %s\n' $out; rc=1
    else
        echo "  [ok]  nothing in the tree matches: $p"
    fi
done

[ "$rc" = 0 ] && echo "OPS02_PATTERN_CONTROL_OK" || echo "OPS02_PATTERN_CONTROL_FAIL"
exit "$rc"
