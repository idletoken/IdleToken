#!/bin/sh
# Emit a source-provenance record for one sync, as JSON on stdout.
#
# Bench machines receive this repo by rsync/tar, so they have no `.git` and
# cannot answer the only question pricing calibration cares about: *which
# commit is this, and was it clean?*  The control machine can answer it, and
# it is the only place that can -- so it writes the answer down at the moment
# it ships the code, and the sync scripts drop the record next to the
# checkout as `provenance.json`.
#
# The platform-side pricing calibrator reads that file instead of probing git
# on the bench machine. The gate did not move: a dirty worktree is still
# refused, the check just got eyes in the right place.
#
# Usage:  scripts/sync-provenance.sh <target-label>
set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-}"
if [ -z "$TARGET" ]; then
    echo "usage: scripts/sync-provenance.sh <target-label>" >&2
    exit 2
fi

# No git, no claim. Writing dirty=false because we could not look would be the
# worst possible failure: it launders unknown code into a priced measurement.
if ! git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    echo "sync-provenance.sh: $REPO_ROOT is not a git worktree; refusing to claim a commit" >&2
    exit 1
fi

head=$(git -C "$REPO_ROOT" rev-parse HEAD)
# --porcelain covers staged, unstaged and untracked. Untracked counts on
# purpose: a stray .c in src/ changes what gets built just as much as an edit.
if [ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]; then
    dirty=true
else
    dirty=false
fi
synced_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
source_host=$(uname -n)

# The two free-text fields are the only injection surface in this file; keep
# them JSON-safe rather than trusting hostnames and ssh aliases to be tame.
escape() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }

printf '{\n'
printf '  "schema_version": 1,\n'
printf '  "head": "%s",\n' "$head"
printf '  "dirty": %s,\n' "$dirty"
printf '  "synced_at": "%s",\n' "$synced_at"
printf '  "source_host": "%s",\n' "$(escape "$source_host")"
printf '  "target": "%s"\n' "$(escape "$TARGET")"
printf '}\n'
