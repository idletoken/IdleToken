#!/usr/bin/env bash
# G3/E3 packaging helper — build the self-contained dist/ bundle on a POSIX
# engine node (coord node / Linux). Windows driver-only bundles are built by
# scripts/build_link.bat on the Windows box; this script covers the other half
# of the E3 gate: a dist/ folder that carries the engine binaries + the run
# scripts they need, with a MANIFEST and a real self-check (no fake pass).
#
# Contents (packaged only if built — honest, never stubbed):
#   dist/idletoken-coord              cluster coordinator + OpenAI/Anthropic API
#   dist/idletoken-worker             per-node worker
#   dist/idletoken-platform-agent     marketplace agent (from build/, Makefile.platform)
#   dist/scripts/*                 run scripts that expect binaries at ../
#   dist/LICENSE, dist/NOTICE      our own Apache-2.0 terms + attributions
#   dist/licenses/*                the third-party texts the binaries oblige us
#                                  to carry (see below)
#   dist/MANIFEST.txt              sha256 + size of every file in the bundle
#
# Layout invariant: the bundled run scripts cd to "$(dirname $0)/.." and expect
# ./idletoken-coord & ./idletoken-worker there — exactly how they run from the repo
# root — so dist/ is runnable in place with zero edits.
#
# Exit / last-line contract (used by acceptance.sh g3_package):
#   DIST_OK    bundle built, manifest written, every packaged binary passed
#              its --help self-check                              (exit 0)
#   DIST_SKIP  engine binaries not built on this host (e.g. macOS control
#              machine, no CUDA build) — nothing packaged, reason printed
#              above; this is a SKIP, not a pass                  (exit 0)
#   DIST_FAIL  a packaged binary failed its self-check, or copying broke
#              (reason printed above)                             (exit 1)
set -u
cd "$(dirname "$0")/.." || exit 1

# CUDA runtime libs for the worker binary's dynamic deps (harmless elsewhere;
# same convention as scripts/pair_selftest.sh).
export PATH="/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

DIST=dist

sha256_of() {  # portable: linux sha256sum / macOS shasum
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
    else shasum -a 256 "$1" | awk '{print $1}'; fi
}
size_of() { wc -c < "$1" | tr -d ' '; }

# --- locate the binaries (package what exists; never fabricate) -------------
find_bin() {  # echo the first existing path among the args, or nothing
    for p in "$@"; do [ -x "$p" ] && { echo "$p"; return 0; }; done
    return 1
}
COORD=$(find_bin ./idletoken-coord build/idletoken-coord || true)
WORKER=$(find_bin ./idletoken-worker build/idletoken-worker || true)
AGENT=$(find_bin build/idletoken-platform-agent || true)

if [ -z "$COORD" ] && [ -z "$WORKER" ]; then
    echo "package_dist: neither idletoken-coord nor idletoken-worker is built here" >&2
    echo "package_dist: build them first (make all) — refusing to package an empty engine dist" >&2
    echo "DIST_SKIP"
    exit 0
fi

# --- assemble ---------------------------------------------------------------
rm -rf "$DIST"
mkdir -p "$DIST/scripts" || { echo "DIST_FAIL"; exit 1; }

PACKAGED_BINS=()
for b in "$COORD" "$WORKER" "$AGENT"; do
    [ -n "$b" ] || continue
    cp "$b" "$DIST/$(basename "$b")" || { echo "package_dist: copy $b failed" >&2; echo "DIST_FAIL"; exit 1; }
    chmod +x "$DIST/$(basename "$b")"
    PACKAGED_BINS+=("$(basename "$b")")
done

# --- licences (MANDATORY, fail closed) ---------------------------------------
# The binaries above contain vendored third-party code, and both licences
# involved require their notice to travel with a BINARY distribution:
#
#   ds4  (vendor/ds4/ds4.c, ds4_cuda.cu / ds4_metal.m)  MIT
#   rax  (vendor/ds4/rax.c)                             BSD 3-Clause, clause 2
#
# plus Apache-2.0 §4(d) for our own NOTICE. Until 2026-08-13 this bundle shipped
# the binaries and none of the texts — the repo was compliant and the thing we
# actually hand people was not.
#
# `die` rather than "skip if missing": a licence file that quietly fails to copy
# produces exactly the bundle this block exists to prevent, and it would look
# identical to a good one.
mkdir -p "$DIST/licenses" || { echo "DIST_FAIL"; exit 1; }
for f in LICENSE NOTICE; do
    cp "$f" "$DIST/$f" || { echo "package_dist: copy $f failed" >&2; echo "DIST_FAIL"; exit 1; }
done
cp vendor/ds4/LICENSE "$DIST/licenses/ds4-MIT.txt" || {
    echo "package_dist: copy vendor/ds4/LICENSE failed" >&2; echo "DIST_FAIL"; exit 1; }
# rax carries its terms in its own header (there is no separate file upstream),
# so the text is extracted from the source we actually compile — it cannot go
# stale against a licence file nobody updates.
awk '/^\/\* Rax/,/^ \*\/$/' vendor/ds4/rax.c > "$DIST/licenses/rax-BSD-3-Clause.txt" || {
    echo "package_dist: extracting the rax licence failed" >&2; echo "DIST_FAIL"; exit 1; }
grep -q "Redistribution and use in source and binary forms" "$DIST/licenses/rax-BSD-3-Clause.txt" || {
    echo "package_dist: extracted rax licence does not contain the BSD terms" >&2
    echo "package_dist: (rax.c's header changed shape — fix the extraction, do not ship without it)" >&2
    echo "DIST_FAIL"; exit 1; }

# Run scripts the bundle needs (they resolve binaries at ../ relative to
# themselves, so they work unchanged from dist/scripts/).
RUN_SCRIPTS="run_cluster.sh run_single_infer.sh pair_selftest.sh weight_server.py"
for s in $RUN_SCRIPTS; do
    [ -f "scripts/$s" ] || continue
    cp "scripts/$s" "$DIST/scripts/$s" || { echo "package_dist: copy scripts/$s failed" >&2; echo "DIST_FAIL"; exit 1; }
done

# --- self-check: every packaged binary must actually run (--help) -----------
# The E3 bar is "runs with only the driver"; --help proves the binary loads and
# executes on this platform (a missing shared lib / wrong arch fails right
# here, red not fake green). GPU-touching --probe-only is exercised by G2 and
# the Windows-side E3 probe; --help keeps this half deterministic on GPU-less
# build hosts.
selfcheck() {  # selfcheck <path> <expect-substring>
    local out
    out=$("./$1" --help 2>&1 | head -3)
    case "$out" in
        *"$2"*) echo "package_dist: self-check ok: $1" ;;
        *) echo "package_dist: self-check FAILED: $1 --help gave '$out'" >&2; return 1 ;;
    esac
}
ok=1
[ -n "$COORD" ]  && { selfcheck "$DIST/idletoken-coord" "idletoken-coord" || ok=0; }
[ -n "$WORKER" ] && { selfcheck "$DIST/idletoken-worker" "idletoken-worker" || ok=0; }
[ -n "$AGENT" ]  && { selfcheck "$DIST/idletoken-platform-agent" "platform-agent" || ok=0; }
[ "$ok" = 1 ] || { echo "DIST_FAIL"; exit 1; }

# --- manifest ----------------------------------------------------------------
{
    echo "# IdleToken dist manifest"
    echo "# built: $(date -u '+%Y-%m-%dT%H:%M:%SZ')  host: $(hostname)  uname: $(uname -sm)"
    echo "# format: sha256  bytes  path"
    ( cd "$DIST" && find . -type f ! -name MANIFEST.txt | sort ) | while read -r f; do
        f="${f#./}"
        printf '%s  %s  %s\n' "$(sha256_of "$DIST/$f")" "$(size_of "$DIST/$f")" "$f"
    done
} > "$DIST/MANIFEST.txt" || { echo "DIST_FAIL"; exit 1; }

echo "package_dist: bundled [${PACKAGED_BINS[*]}] + scripts + licences + MANIFEST.txt -> $DIST/"
echo "DIST_OK"
exit 0
