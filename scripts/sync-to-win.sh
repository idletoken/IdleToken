#!/usr/bin/env bash
# Sync repo sources to a Windows node (build node or deploy node).
#
# Windows boxes have no rsync, so this ships a tarball and unpacks it with the
# bsdtar that Windows 10+ bundles. COPYFILE_DISABLE keeps macOS from injecting
# `._*` AppleDouble files — those make the MinGW/Tauri builds fail with
# "stream did not contain valid UTF-8", which is a confusing way to learn about
# resource forks.
#
# Usage:  scripts/sync-to-win.sh <ssh-alias> [--with-client-dist]
#         IDLETOKEN_SSH_HOSTNAME=<ip-or-dns> may override only the transport
#         address while retaining the alias's user, key and testbed mapping.
#         The alias comes from your own ~/.ssh/config; there is no default (see below).
set -eu

# No default node: the product targets "any number of machines", and a default
# could only ever be one maintainer's own box.
NODE="${1:-}"
if [ -z "$NODE" ]; then
    echo "usage: scripts/sync-to-win.sh <ssh-alias> [--with-client-dist]" >&2
    exit 2
fi
WITH_CLIENT_DIST=0
case "${2:-}" in
    "") ;;
    --with-client-dist) WITH_CLIENT_DIST=1 ;;
    *) echo "sync-to-win.sh: unknown option: $2" >&2; exit 2 ;;
esac
# Per-machine user directories come from scripts/testbed.env (not committed, see
# testbed.env.example). This used to be a third hardcoded copy -- the same fact
# spread across three scripts, which both drifts and pins the account name of
# whoever lent us the machine into the repository.
# shellcheck disable=SC1091
. "$(dirname "$0")/testbed-lib.sh"
WHOME="$(testbed_repo_home "$NODE")"
[ -n "$WHOME" ] || { echo "sync-to-win.sh:" >&2; testbed_hint "$NODE"; exit 2; }

SSH=(ssh)
if [ -n "${IDLETOKEN_SSH_HOSTNAME:-}" ]; then
    SSH+=(-o "HostName=$IDLETOKEN_SSH_HOSTNAME" -o "HostKeyAlias=$NODE")
fi
SSH+=("$NODE")

cd "$(dirname "$0")/.."
TAR=/tmp/idletoken-win-sync.tar.gz

# Record what we are about to ship BEFORE building the tarball, so the commit
# and the clean/dirty verdict describe the tree that actually leaves this
# machine. It rides inside the archive (one less ssh round trip, and it can
# never land without the sources it describes). See scripts/sync-provenance.sh.
PROVENANCE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/idletoken-provenance.XXXXXX")"
trap 'rm -rf "$PROVENANCE_DIR"' EXIT INT TERM
scripts/sync-provenance.sh "$NODE:$WHOME" > "$PROVENANCE_DIR/provenance.json"

# Sources only: the engine build inputs plus the scripts that drive it. No
# GGUFs, no build outputs, no node_modules.
#
# The Windows build and launch .bat files at the repo root are on the list. They
# **used not to be**, and the consequence was that the copies in the repo and on
# the machine drifted apart for a long time: build_ds4x.bat on the machine was a
# historical copy of build_ds4x_win.bat, and nobody knew which was newer. One fix
# existed only on the machine, and I overwrote it with the repo's older version.
# The repo is now the single source of truth. Since 2026-08-16 the pair shipped
# is build_worker_win.bat + build_coord_win.bat; the ds4x scripts they replaced
# are shelved and no longer synced, and leftovers on the machine are reported
# below (**without deleting** -- what is on someone's machine is theirs).
#
# `vendor/ds4/ds4_cuda.cu` must be on the list (added 2026-08-03). While it was
# not, kernels added on Linux could never reach the Windows ds4cuda.dll even
# though `ds4.c` (which is on the list) already referenced them -- so the Windows
# worker failed to link. That is exactly how the two DSpark kernels broke the
# Windows build for days without anyone noticing, because the ladder was running
# the older exe already on the machine.
#
# `client/` is on the list too (added 2026-08-04): the installer is built on this
# machine, and the client sources used to be **scp'd by hand** (one-off client
# source tarballs are still lying around there). Hand copies drift,
# and after the rename the staged sidecar on the build machine still had the old
# name, so a rebuilt package carried **both** engines. `client/dist` is
# **excluded by default**: it is a build artifact, and which one you carry
# matters (a release-mode dist points at the production platform address).
# Release packaging must use `--with-client-dist`; that mode first verifies the
# release provenance stamp and mirrors dist exactly. This explicit flag prevents
# an ordinary source sync from silently turning a development frontend into a
# release, while also preventing the Windows packager from reusing an old dist.
# Pricing calibration runs ON this machine (bench.py over loopback; from the
# control machine we would be pricing LAN latency). Only the wrapper, the
# workload and the rate card go -- the rest of platform/ is the commercial
# layer and has no business on a borrowed node. The [ -f ] guard makes them
# ride along only when present: the public mirror ships without platform/,
# and this script must keep working from that checkout too.
PRICING_SRCS=()
if [ -f platform/scripts/pricing-calibrate.py ] && [ -f platform/scripts/pricing-generate.py ]; then
    PRICING_SRCS=(platform/pricing platform/scripts/pricing-calibrate.py
                  platform/scripts/pricing-generate.py)
fi
DIST_EXCLUDES=(--exclude='client/dist')
if [ "$WITH_CLIENT_DIST" -eq 1 ]; then
    (cd client && node scripts/write_frontend_provenance.mjs --verify)
    DIST_EXCLUDES=()
fi
COPYFILE_DISABLE=1 tar czf "$TAR" \
    --exclude='._*' --exclude='.DS_Store' --exclude='*.o' --exclude='*.gguf' \
    --exclude='client/node_modules' --exclude='packages/shared-ui/node_modules' \
    --exclude='client/src-tauri/target' \
    --exclude='client/src-tauri/gen' --exclude='client/src-tauri/binaries' \
    ${DIST_EXCLUDES[@]+"${DIST_EXCLUDES[@]}"} \
    --exclude='platform/pricing/calibrations' \
    src include models scripts vendor/ds4/ds4.c vendor/ds4/ds4.h vendor/ds4/ds4_gpu.h \
    ${PRICING_SRCS[@]+"${PRICING_SRCS[@]}"} \
    vendor/ds4/ds4_cuda.cu \
    vendor/ds4/rax.c vendor/ds4/rax.h vendor/ds4/rax_malloc.h \
    vendor/tweetnacl vendor/blake2 packages/shared-ui \
    client \
    LICENSE NOTICE Makefile \
    build_worker_win.bat build_coord_win.bat build_agent_win.bat \
    -C "$PROVENANCE_DIR" provenance.json

echo "shipping $(du -h "$TAR" | cut -f1) to $NODE:$WHOME"
# The archive travels over STDIN and never touches the node's disk. With an
# on-disk archive, Windows bsdtar's "am I overwriting my own input?" check
# (pseudo dev/inode comparison) intermittently matched UNRELATED extracted
# files and refused with "Refusing to overwrite archive" — first with the
# archive inside the destination tree (robocopy staging worked around that),
# then, 2026-08-16, against a FRESH stage directory too (client/src/pairing.ts;
# a filter driver zeroing FileIndex makes every file look identical to the
# archive). No archive file, no comparison, no flake. cmd.exe hosts the
# extraction because PowerShell mangles binary stdin.
win_stage_path="${WHOME//\//\\}\\idletoken-sync-stage"
"${SSH[@]}" "cmd /c \"rmdir /s /q $win_stage_path 2>nul & mkdir $win_stage_path && tar xzf - -C $win_stage_path\"" < "$TAR"
tar_rc=$?
if [ "$tar_rc" -ne 0 ]; then
    echo "sync-to-win.sh: streamed extraction failed on $NODE (exit $tar_rc)" >&2
    rm -f "$TAR"
    exit 1
fi
# Merge stage -> checkout. Robocopy uses 0..7 for success and >=8 for failure,
# so translate that contract explicitly.
#
# The merge deliberately does NOT delete: what is on someone's machine is
# theirs. scripts/llamacpp-patches/ is the one exception, and it is mirrored
# with /PURGE. That directory is consumed by GLOB -- build_llamacpp_win.bat
# applies every *.patch it finds -- so a patch that the repo has RETIRED keeps
# being applied here long after it is gone everywhere else. Measured 2026-08-19
# while bumping the pin to b10502: 0002-rpc-memset-tensor was superseded
# upstream and deleted, DGX (rsync --delete) built fine, and both Windows nodes
# died with "FATAL: patch 0002-rpc-memset-tensor.patch does not apply" from a
# file no longer in the repo. A leftover here is not the machine owner's
# preference, it silently changes which engine gets built.
dist_merge=""
if [ "$WITH_CLIENT_DIST" -eq 1 ]; then
    dist_merge="& robocopy \"\$stage\\client\\src\" 'client\\src' /MIR /IS /IT /NFL /NDL /NJH /NJS /NP; \
\$frontendSrcRc=\$LASTEXITCODE; \
if (\$frontendSrcRc -ge 8) { exit \$frontendSrcRc }; \
& robocopy \"\$stage\\client\\public\" 'client\\public' /MIR /IS /IT /NFL /NDL /NJH /NJS /NP; \
\$frontendPublicRc=\$LASTEXITCODE; \
if (\$frontendPublicRc -ge 8) { exit \$frontendPublicRc }; \
& robocopy \"\$stage\\packages\\shared-ui\\src\" 'packages\\shared-ui\\src' /MIR /IS /IT /NFL /NDL /NJH /NJS /NP; \
\$sharedUiRc=\$LASTEXITCODE; \
if (\$sharedUiRc -ge 8) { exit \$sharedUiRc }; \
& robocopy \"\$stage\\client\\dist\" 'client\\dist' /MIR /IS /IT /NFL /NDL /NJH /NJS /NP; \
\$distRc=\$LASTEXITCODE; \
if (\$distRc -ge 8) { exit \$distRc };"
fi
remote_ps="\$ErrorActionPreference='Stop'; \
Set-Location '$WHOME'; \
\$stage='idletoken-sync-stage'; \
& robocopy \$stage . /E /IS /IT /NFL /NDL /NJH /NJS /NP; \
\$copyRc=\$LASTEXITCODE; \
if (\$copyRc -ge 8) { exit \$copyRc }; \
$dist_merge \
& robocopy \"\$stage\\scripts\\llamacpp-patches\" 'scripts\\llamacpp-patches' /E /PURGE /NFL /NDL /NJH /NJS /NP; \
\$purgeRc=\$LASTEXITCODE; \
if (\$purgeRc -ge 8) { exit \$purgeRc }; \
Remove-Item -Recurse -Force \$stage; \
Write-Output 'SYNC_OK'"
if sync_out=$("${SSH[@]}" "powershell -NoProfile -Command \"$remote_ps\"" 2>&1); then
    sync_rc=0
else
    sync_rc=$?
fi
printf '%s\n' "$sync_out" | tr -d '\r' | tail -1
if [ "$sync_rc" -ne 0 ] || ! printf '%s\n' "$sync_out" | tr -d '\r' | grep -qx 'SYNC_OK'; then
    echo "sync-to-win.sh: remote extraction failed on $NODE (ssh/tar exit $sync_rc)" >&2
    printf '%s\n' "$sync_out" >&2
    rm -f "$TAR"
    exit 1
fi
rm -f "$TAR"

if [ "$WITH_CLIENT_DIST" -eq 1 ]; then
    local_dist_hash=$(shasum -a 256 client/dist/frontend-provenance.json | cut -d' ' -f1)
    remote_dist_hash=$("${SSH[@]}" "powershell -NoProfile -Command \"(Get-FileHash '$WHOME/client/dist/frontend-provenance.json' -Algorithm SHA256).Hash\"" | tr -d '\r ' | tr 'A-F' 'a-f')
    if [ "$local_dist_hash" != "$remote_dist_hash" ]; then
        echo "sync-to-win.sh: client/dist provenance differs after sync" >&2
        exit 1
    fi
    echo "frontend release dist verified byte-identical on $NODE"
fi

if grep -q '"dirty": true' "$PROVENANCE_DIR/provenance.json"; then
    echo "note: this sync carried UNCOMMITTED changes; pricing calibration on $NODE will refuse it"
fi

# --- Shelved ds4x build scripts still on the machine -----------------------
# ds4/ds4x are shelved (2026-08-16): build_worker_win.bat replaces
# build_ds4x_win.bat, and the ds4x scripts are no longer shipped. Copies left on
# the machine still RUN, and running one produces a worker that links the
# shelved kernels and needs a CUDA Toolkit -- i.e. not the worker this repo
# builds anywhere else. Report them; deleting files on someone's machine is not
# this script's call.
stale=$("${SSH[@]}" "cd /d ${WHOME//\//\\} && for %f in (build_ds4x.bat build_ds4x_win.bat runworker-ds4x-win.bat) do @if exist %f echo %f" 2>/dev/null | tr -d '\r')
if [ -n "$stale" ]; then
    echo "note: shelved ds4x build scripts still on $NODE (no longer synced; use build_worker_win.bat):"
    printf '      %s\n' $stale
fi

# --- Verify the engine sources actually landed -----------------------------
# "The sync was sent" is not "the sync landed". The lesson of 2026-08-05: one
# machine's vendor/ds4/ds4.c was 1523 lines shorter than the repo's (the entire
# dspark patch had never been applied) while I spent a day treating them as the
# same file -- so the performance numbers measured on Windows had a broken source
# lineage, and a CPU code path that existed only in the repo was simply absent on
# that machine, surfacing as `cluster prefill failed`.
#
# Compare the few files that matter byte for byte. They are the entire basis for
# knowing what the engine built on this machine actually is.
verify_engine_sources() {
    local bad=0 f rl rr
    # ds4_stub.c is on the list because it is what the Windows worker and coord
    # now actually link (the ds4 sources below are shelved but still shipped for
    # the archaeology build).
    for f in vendor/ds4/ds4.c vendor/ds4/ds4.h vendor/ds4/ds4_cuda.cu \
             src/common/ds4_stub.c src/worker/worker_main.c; do
        rl=$(shasum -a 256 "$f" 2>/dev/null | cut -c1-16)
        rr=$("${SSH[@]}" "powershell -NoProfile -Command \"(Get-FileHash '${WHOME}/${f}' -Algorithm SHA256).Hash.Substring(0,16)\"" 2>/dev/null | tr -d '\r ' | tr 'A-F' 'a-f')
        if [ -z "$rr" ]; then
            echo "WARN: cannot read $f on $NODE" >&2; bad=1
        elif [ "$rl" != "$rr" ]; then
            echo "WARN: $f on $NODE **differs** from the repo (local $rl / remote $rr)" >&2; bad=1
        fi
    done
    if [ "$bad" = 0 ]; then
        echo "engine sources verified byte-identical on $NODE"
    else
        echo "ENGINE_SOURCE_DRIFT -- the engine built on this machine is not the repo's engine." >&2
        echo "  No performance number measured before this point can be attributed to the repo code." >&2
        return 1
    fi
}
verify_engine_sources || exit 1
