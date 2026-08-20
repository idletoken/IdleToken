#!/usr/bin/env bash
# Read a node's log (both .out and .err) to stdout, whatever OS it runs.
#
#   scripts/node_read_log.sh <testbed-alias|local> <logbase-without-extension>
#
# Exists because reading a Windows log over ssh is a quoting minefield and the
# obvious tools mislead:
#   * `findstr` returns 0 when it FINDS a match -- the inverse of grep -- so a
#     caller checking the exit code reads "clean" exactly when it is not
#     (measured 2026-08-20; a privacy check passed for the wrong reason).
#   * ssh -> cmd.exe -> powershell mangles nested quotes into silence: the
#     command produced NO output at all and looked like an empty log.
# So: PowerShell driven by -EncodedCommand over UTF-16LE base64, which has no
# quoting layer to get wrong, and grep/`Select-String` decisions are left to the
# caller on plain text.
set -u
cd "$(dirname "$0")/.." || exit 1
# shellcheck disable=SC1091
. "$(dirname "$0")/testbed-lib.sh"

NODE="${1:-}"; BASE="${2:-}"
[ -n "$NODE" ] && [ -n "$BASE" ] || { echo "usage: node_read_log.sh <node> <logbase>" >&2; exit 2; }
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"

is_windows() { [ "$1" != local ] && [ -n "$(testbed_profile "$1")" ]; }

if [ "$NODE" = local ]; then
    cat "$BASE.out" "$BASE.err" 2>/dev/null
elif is_windows "$NODE"; then
    win="${BASE//\//\\}"
    ps_cmd="foreach (\$f in @('$win.out','$win.err')) { if (Test-Path \$f) { Get-Content -Raw -ErrorAction SilentlyContinue \$f } }"
    enc=$(printf '%s' "$ps_cmd" | iconv -f UTF-8 -t UTF-16LE | base64 | tr -d '\n')
    # Strip CR and the CLIXML progress noise PowerShell writes to stderr.
    $SSH "$NODE" "powershell -NoProfile -EncodedCommand $enc" 2>/dev/null \
        | tr -d '\r' | grep -v '^#< CLIXML' | grep -v '^<Objs '
else
    $SSH "$NODE" "cat '$BASE.out' '$BASE.err' 2>/dev/null"
fi
