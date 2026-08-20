#!/usr/bin/env bash
# Build the Windows desktop client (.exe installer) on the Windows build node,
# driven from a machine that has Node.
#
# WHY THE SPLIT. The build node has Rust + cargo-tauri but **no Node/pnpm**, and
# `tauri.conf.json`'s beforeBuildCommand is `pnpm build:release`. So:
#   - the web assets are built HERE (they are platform-independent),
#   - they are shipped over,
#   - and the Windows side runs only `cargo tauri build`, with a config override
#     that blanks beforeBuildCommand so it never looks for pnpm.
# Installing Node on a borrowed machine would also work; this does not touch it.
#
# THE "REFUSING TO OVERWRITE ARCHIVE" TRAP (2026-08-11). bsdtar on Windows
# refuses to extract a member when it believes the target IS the archive it is
# reading. Windows stat() gives every file st_ino = 0, so that identity check
# misfires: with the tarball sitting in the extraction cwd, exactly one member
# (client/public/fonts/space-grotesk-600-latin-ext.woff2) was declared to be the
# archive and the whole extraction exited non-zero. Deterministic, and it cost a
# whole build — the notification said "exit 0" because the FAILURE was in the
# script's own contract line, not the exit status of the last command.
# Fix: keep the transfer archives OUT of the directory tar runs in. Proven by
# experiment — same archive, same tree, same command, extracting with the file
# one level up succeeds. (The dist tarball was already outside its cwd, which is
# why only the sources ever hit this.)
#
# THE dist/ TRAP (2026-08-10). `dist` is shipped as a tarball, and tar MERGES
# into whatever is already there. Vite emits content-hashed filenames, so an
# unclean target keeps every bundle from every previous build — the installer
# grows, and worse, grepping `dist/assets/*.js` to check "did my change ship?"
# reads a stale file and answers yes to anything you ever built. The remote
# dist is therefore DELETED, not overwritten.
#
# DEBUG MODE (--debug, added 2026-08-13). The product gates that drive the
# client (G-UPDATE, G-TRAY) need a **debug** shell on the Windows box, for one
# concrete reason: the updater plugin refuses a plain-http update endpoint in
# release builds, and those gates serve their signed test feed from the control
# machine over http. That refusal is a feature — it is exactly what stops a
# shipped build from being pointed at an unencrypted feed — so the gates use a
# debug binary rather than weakening it. --debug also ships the NON-release web
# assets (a release dist points at the production platform address).
#
# Usage:  scripts/build_client_win.sh [--debug] [<ssh-alias>]
#         Default node: IDLETOKEN_WIN_BUILD_NODE from scripts/testbed.env.
# Contract: last line is CLIENT_WIN_OK <path>, CLIENT_WIN_DEBUG_OK <path>,
#           or CLIENT_WIN_FAIL: <reason>.
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD

fail() { echo "CLIENT_WIN_FAIL: $*"; exit 1; }

# shellcheck disable=SC1091
. "$(dirname "$0")/testbed-lib.sh"
MODE=release
if [ "${1:-}" = "--debug" ]; then MODE=debug; shift; fi
NODE="${1:-${IDLETOKEN_WIN_BUILD_NODE:-}}"
[ -n "$NODE" ] || fail "no build node (pass an ssh alias, or set IDLETOKEN_WIN_BUILD_NODE in scripts/testbed.env)"
WHOME="$(testbed_repo_home "$NODE")"
[ -n "$WHOME" ] || { testbed_hint "$NODE"; exit 2; }

SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
WIN="${WHOME//\//\\}"                       # C:/Users/x/IdleToken -> C:\Users\x\IdleToken
TRIPLE=$($SSH "$NODE" 'rustc -vV' 2>/dev/null | tr -d '\r' | awk '/^host:/{print $2}')
[ -n "$TRIPLE" ] || fail "could not read the rustc host triple on $NODE (is Rust installed?)"

echo "== [1/4] build the web assets locally ($MODE mode) =="
# Clean first for the same reason the remote one is deleted: hashed filenames
# accumulate, and a stale bundle in dist/ ends up inside the installer.
rm -rf client/dist
if [ "$MODE" = release ]; then
    ( cd client && pnpm build:release ) >/tmp/client-win-frontend.log 2>&1 \
        || fail "pnpm build:release failed (see /tmp/client-win-frontend.log)"
else
    ( cd client && pnpm build ) >/tmp/client-win-frontend.log 2>&1 \
        || fail "pnpm build failed (see /tmp/client-win-frontend.log)"
fi
echo "   $(find client/dist -type f | wc -l | tr -d ' ') files"

echo "== [2/4] ship sources + dist to $NODE =="
COPYFILE_DISABLE=1 tar czf /tmp/idletoken-client-src.tgz \
    --exclude='._*' --exclude='.DS_Store' \
    --exclude='client/node_modules' --exclude='client/src-tauri/target' \
    --exclude='client/src-tauri/gen' --exclude='client/src-tauri/binaries' \
    --exclude='client/dist' client || fail "could not pack the client sources"
# The archives travel over STDIN and never touch the node's disk: even an
# archive staged OUTSIDE the repo hit "Refusing to overwrite archive" against
# unrelated extracted files (2026-08-16, client/src/pairing.ts — Windows
# bsdtar's pseudo dev/inode self-overwrite check collides when a filter driver
# zeroes FileIndex). No on-disk archive, nothing to collide with. cmd.exe
# hosts the extraction because PowerShell mangles binary stdin.
COPYFILE_DISABLE=1 tar czf /tmp/idletoken-client-dist.tgz --exclude='._*' -C client dist \
    || fail "could not pack dist"
err=$($SSH "$NODE" "cd /d $WIN && tar xzf - && echo UNPACK_OK" < /tmp/idletoken-client-src.tgz 2>&1 | tr -d '\r')
case "$err" in *UNPACK_OK*) ;; *) fail "could not unpack the sources on $NODE: $(echo "$err" | head -2 | tr '\n' ' ')" ;; esac
# Delete, do not merge — see the dist/ trap above.
$SSH "$NODE" "rmdir /s /q \"$WIN\\client\\dist\" 2>NUL & cd /d $WIN\\client && tar xzf -" < /tmp/idletoken-client-dist.tgz >/dev/null 2>&1
n=$($SSH "$NODE" "dir /b \"$WIN\\client\\dist\\assets\"" 2>/dev/null | tr -d '\r' | grep -c .)
[ "$n" -gt 0 ] || fail "dist did not land on $NODE"
echo "   dist/assets: $n files (a fresh build has ~4; more means the clean did not take)"

echo "== [3/4] stage the engine binaries as sidecars =="
# Real engines, not stubs: this is a compute node. Their dates are printed
# because a stale sidecar is invisible otherwise — a coordinator talking to an
# older worker just loops on `Protocol error` (see scripts/stage_sidecars.sh).
BIN="$WIN\\client\\src-tauri\\binaries"
$SSH "$NODE" "if not exist \"$BIN\" mkdir \"$BIN\"" >/dev/null 2>&1

# Where the pinned llama.cpp lands on Windows: the MSVC generator adds a
# per-config subdirectory, Ninja does not. Accept either instead of hardcoding
# one and failing on the next machine.
LLAMA_DIR=
for d in 'build\bin\Release' 'build\bin'; do
    if $SSH "$NODE" "if exist \"$WIN\\vendor\\llama.cpp\\$d\\llama-server.exe\" (exit 0) else (exit 1)" >/dev/null 2>&1; then
        LLAMA_DIR="$WIN\\vendor\\llama.cpp\\$d"; break
    fi
done
[ -n "$LLAMA_DIR" ] || fail "no llama-server.exe under $WIN\\vendor\\llama.cpp\\build\\bin[\\Release] on $NODE — build the pinned engine there first (scripts\\build_llamacpp_win.bat)"

# ALL FIVE externalBin entries are staged, every build. Staging only three of
# them (worker/coord/agent) is what this step used to do, and the other two —
# the actual inference engines — were then whatever an earlier build had left
# in the directory. That is the "gate certifies the wrong artifact" shape
# stage_sidecars.sh exists to prevent, except here it reaches a shipped .exe.
stage_win() {   # stage_win <src-exe> <sidecar-name>
    $SSH "$NODE" "copy /Y \"$1\" \"$BIN\\$2-$TRIPLE.exe\"" >/dev/null 2>&1 \
        || fail "could not stage $2 from $1 on $NODE — build it there first"
}
stage_win "$WIN\\idletoken-worker.exe"         idletoken-worker
stage_win "$WIN\\idletoken-coord.exe"          idletoken-coord
stage_win "$WIN\\idletoken-platform-agent.exe" idletoken-platform-agent
# Staged under OUR names, matching engine.rs's runtime lookup and
# stage_sidecars.sh. The binaries are upstream's, unchanged; only the file name
# is ours (MIT attribution stays in About + NOTICE).
stage_win "$LLAMA_DIR\\llama-server.exe"       idletoken-server
stage_win "$LLAMA_DIR\\ggml-rpc-server.exe"    idletoken-rpc-server
# Pre-rename leftovers: not in externalBin, so they never ship, but they sit
# next to the real ones and read as current.
$SSH "$NODE" "del /q \"$BIN\\ggml-rpc-server-$TRIPLE.exe\" \"$BIN\\llama-server-$TRIPLE.exe\" 2>NUL" >/dev/null 2>&1
$SSH "$NODE" "dir \"$BIN\\*.exe\"" 2>/dev/null | tr -d '\r' | grep -E '\.exe' | sed 's/^/   /'

if [ "$MODE" = debug ]; then
    echo "== [4/4] cargo build (debug shell for the product gates) =="
    # Plain cargo build, not `tauri build`: no bundling, and nothing runs
    # beforeBuildCommand, so the machine still needs no Node.
    # --features custom-protocol is NOT optional here. Without it a debug build
    # loads `devUrl` (http://localhost:1420) instead of the bundled dist — so on
    # a machine with no dev server running, the app opens the system browser at
    # a dead localhost address, the front end never starts, and every UI-test
    # directive silently produces nothing. That is what `tauri dev` turns on for
    # you and what a plain `cargo build` does not.
    out=$($SSH "$NODE" "cd /d $WIN\\client\\src-tauri && cargo build --features custom-protocol" 2>&1 | tr -d '\r')
    echo "$out" | tail -4 | sed 's/^/   /'
    EXE="$WHOME/client/src-tauri/target/debug/idletoken-client.exe"
    size=$($SSH "$NODE" "for %I in (\"${EXE//\//\\}\") do @echo %~zI" 2>/dev/null | tr -d '\r' | tail -1)
    case "$size" in ''|*[!0-9]*) fail "no debug binary produced (see the tail above)" ;; esac
    # The file existing proves nothing: a client still running holds a lock on
    # it, cargo fails with "Access is denied (os error 5)", and the OLD binary
    # sits there at its old size. Certifying that build would repeat this
    # repo's three-week-old-exe incident. Cargo prints "Finished" only when it
    # actually linked, so that is the check.
    case "$out" in
        *Finished*) ;;
        *"os error 5"*) fail "cargo could not replace the binary — a client is still running on $NODE (taskkill /IM idletoken-client.exe /F)" ;;
        *) fail "cargo build did not finish on $NODE (see the tail above)" ;;
    esac
    # The existence of the file proves nothing: a running client holds a lock on
    # it, cargo fails with "Access is denied (os error 5)", and the OLD binary
    # is still sitting there at its old size. This gate then certifies a build
    # that predates the change under test — the exact failure this repo has
    # already had once with a three-week-old exe. Cargo says "Finished" only
    # when it really linked, so that is what is checked.
    case "$out" in
        *Finished*) ;;
        *"os error 5"*) fail "cargo could not replace the binary — a client is still running on $NODE (taskkill /IM idletoken-client.exe /F)" ;;
        *) fail "cargo build did not finish on $NODE (see the tail above)" ;;
    esac
    echo "CLIENT_WIN_DEBUG_OK $NODE:$EXE ($size bytes)"
    exit 0
fi

echo "== [4/4] cargo tauri build (nsis) =="
# ⚠ KNOWN-BROKEN SIGNING PATH (2026-08-20). The inline-signing below wedges
# forever at Tauri's password prompt: cmd cannot represent an empty-valued
# environment variable (`set "X="` DELETES it), so the updater key's
# intentionally-empty password never reaches Tauri and it prompts — over a
# non-tty ssh that is an infinite hang, and no stdin trick reliably feeds it
# (the password reader flushes pending input). Use the DEFERRED-SIGNING flow
# instead: `IDLETOKEN_DEFER_UPDATER_SIGNING=1 scripts\build_client_release.bat`
# on the node, then sign the updater zip on the control machine and push the
# sig back — acceptance.sh's g_release() is the reference implementation and
# the certified path. This inline path is kept only for keys that carry a
# real, non-empty password.
# THE SIGNING KEY.
#
# `createUpdaterArtifacts` is on in tauri.conf.json, so this build also produces
# the update artifact and its minisign signature — and `tauri build` refuses to
# run without a private key. That refusal is wanted: the in-app updater only
# installs artifacts whose signature verifies against the pubkey compiled into
# the client, so an unsigned release ships a dead update channel, and the build
# is the last place that can notice.
#
# This is NOT a paid code-signing certificate (Authenticode). It is a locally
# generated keypair, free, no CA involved. The only real cost is custody: LOSE
# THE PRIVATE KEY AND ALREADY-INSTALLED CLIENTS CAN NEVER BE UPDATED AGAIN —
# they trust that one pubkey, so a new key means every user reinstalls by hand.
#
# The key is read HERE and passed to the build over ssh. It lives outside the
# repository (default ~/.idletoken/updater.key) and is never written to the
# build machine's disk by this script.
KEY_PATH="${TAURI_SIGNING_PRIVATE_KEY_PATH:-$HOME/.idletoken/updater.key}"
KEY_MATERIAL="${TAURI_SIGNING_PRIVATE_KEY:-}"
if [ -z "$KEY_MATERIAL" ]; then
    [ -f "$KEY_PATH" ] || fail "no updater signing key at $KEY_PATH — restore your backup, or (first time only) generate one: cd client && pnpm tauri signer generate -w ~/.idletoken/updater.key   [free, no certificate authority involved]"
    KEY_MATERIAL=$(cat "$KEY_PATH")
fi
KEY_PASSWORD="${TAURI_SIGNING_PRIVATE_KEY_PASSWORD:-}"
SIGN_ENV="set \"TAURI_SIGNING_PRIVATE_KEY=$KEY_MATERIAL\" && set \"TAURI_SIGNING_PRIVATE_KEY_PASSWORD=$KEY_PASSWORD\" && "

# The override lives on the machine only for the duration of the build.
printf '{"build":{"beforeBuildCommand":""}}' > /tmp/idletoken-nobuild.json
scp -q /tmp/idletoken-nobuild.json "$NODE:$WHOME/client/src-tauri/nobuild.conf.json" || fail "could not ship the config override"
out=$($SSH "$NODE" "cd /d $WIN\\client && ${SIGN_ENV}cargo tauri build --bundles nsis --config src-tauri\\nobuild.conf.json" 2>&1 | tr -d '\r')
echo "$out" | tail -4 | sed 's/^/   /'
$SSH "$NODE" "del /q \"$WIN\\client\\src-tauri\\nobuild.conf.json\"" >/dev/null 2>&1

SETUP="$WHOME/client/src-tauri/target/release/bundle/nsis/IdleToken_0.1.0_x64-setup.exe"
size=$($SSH "$NODE" "for %I in (\"${SETUP//\//\\}\") do @echo %~zI" 2>/dev/null | tr -d '\r' | tail -1)
case "$size" in ''|*[!0-9]*) fail "no installer produced (see the tail above)" ;; esac

# Put the local tree back to a NON-release dist. Same reason build_client_release.sh
# does it: the acceptance gates serve client/dist to the debug shell, and a
# release-mode dist points at the production platform address — leaving it here
# silently changes what P2_auth is testing.
( cd client && pnpm build ) >/tmp/client-win-restore.log 2>&1 \
    || fail "installer is built, but restoring the non-release client/dist failed (see /tmp/client-win-restore.log)"

echo "CLIENT_WIN_OK $NODE:$SETUP ($size bytes)"
