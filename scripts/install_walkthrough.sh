#!/usr/bin/env bash
# The literal install walkthrough (acceptance-criteria §5 G-FINAL, last line):
# take the shipped installer to a machine that has nothing but the NVIDIA
# driver, install it, and drive the whole product from there.
#
# Everything above this in the ladder runs from a repo checkout. This one does
# not: the joining machine only ever sees the package. It is the difference
# between "our tree works" and "what we ship works", and it has caught things
# no other gate could — a placeholder sidecar shipped under the agent's name, a
# stale CUDA DLL that made the packaged worker fall back to MOCK (cluster
# "ready", answers garbage), and missing inbound firewall rules that left the
# joiner stuck on "no cluster found for that code on this LAN".
#
# Runs from the control machine (Mac). Steps:
#   1. pull the installer off the Windows build node
#   2. push it to the target node and install silently
#   3. check every shipped binary runs with a driver-only PATH
#   4. start the coordinator's client (creator) and the TARGET's INSTALLED
#      client (joiner), paired by code only
#   5. assert the split covers all layers, the target fetched only its own
#      layers, its engine loaded REAL (not mock), and the API answers correctly
#
# Usage:  scripts/install_walkthrough.sh <target-node>      # alias of a clean target machine
# Contract: last line INSTALL_E2E_OK or INSTALL_E2E_FAIL: <reason>.
set -u

TARGET="${1:-${IDLETOKEN_INSTALL_TARGET:-}}"
BUILD_NODE="${IDLETOKEN_WIN_BUILD_NODE:-}"
COORD_NODE="${IDLETOKEN_COORD_NODE:-}"
COORD_HOME="${IDLETOKEN_COORD_HOME:-~/work/IdleToken}"
if [ -z "$TARGET" ] || [ -z "$BUILD_NODE" ] || [ -z "$COORD_NODE" ]; then
    echo "install_walkthrough.sh: needs three aliases -- target, build machine, coordinator." >&2
    echo "  scripts/install_walkthrough.sh <target>   and configure the rest in testbed.env" >&2
    echo "  IDLETOKEN_WIN_BUILD_NODE / IDLETOKEN_COORD_NODE" >&2
    exit 2
fi
CODE="${IDLETOKEN_PAIR_CODE:-INST01}"
READY_TRIES="${IDLETOKEN_READY_TRIES:-60}"   # × 15s ≈ 15 min (shard download + load)

# Everything machine-specific comes from scripts/testbed.env (not committed, see
# testbed.env.example) -- this used to be a third hardcoded copy of the usernames.
# shellcheck disable=SC1091
. "$(dirname "$0")/testbed-lib.sh"
THOME="$(testbed_profile "$TARGET")"
[ -n "$THOME" ] || { testbed_hint "$TARGET"; echo "INSTALL_E2E_FAIL: target $TARGET is not configured"; exit 1; }
TUSER="$(basename "$THOME")"
BUILD_HOME="$(testbed_profile "$BUILD_NODE")"
[ -n "$BUILD_HOME" ] || { testbed_hint "$BUILD_NODE"; echo "INSTALL_E2E_FAIL: build machine $BUILD_NODE is not configured"; exit 1; }

# The GGUF on the coordinator: expanded from $HOME by the remote shell rather than
# hardcoding anyone's home directory.
GGUF="${IDLETOKEN_GGUF_ABS:-\$HOME/work/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf}"
SETUP_WIN="$BUILD_HOME/IdleToken/client/src-tauri/target/release/bundle/nsis/IdleToken_0.1.0_x64-setup.exe"
APPDIR="$THOME/AppData/Local/IdleToken"
APPDIR_WIN="${APPDIR//\//\\}"

SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
SSHF="ssh -f -o BatchMode=yes -o ConnectTimeout=10"   # see acceptance.sh: backgrounding inside the remote shell hangs

# The GGUF path is **expanded to an absolute path on the coordinator** before it
# is used further down. Not expanding it repeats the 2026-07-29 trap: the path
# goes through IDLETOKEN_HEADLESS_PAIR straight into the sidecar's argv with no
# shell in between (and on the remote it sits inside single quotes), so the engine
# receives the literal `$HOME/...`, weight loading fails, it silently falls back
# to mock, and the cluster reports ready while answering with garbage tokens.
GGUF=$($SSH "$COORD_NODE" "echo $GGUF" 2>/dev/null | tr -d '\r')
case "$GGUF" in
    ""|*'$'*) echo "INSTALL_E2E_FAIL: could not expand the GGUF path on $COORD_NODE (got '$GGUF')"; exit 1 ;;
esac

die() { echo "INSTALL_E2E_FAIL: $*"; cleanup; exit 1; }
cleanup() {
    $SSH "$TARGET" "taskkill /IM idletoken-client.exe /F 2>NUL & taskkill /IM idletoken-worker.exe /F 2>NUL & exit 0" >/dev/null 2>&1
    $SSH "$COORD_NODE" "pkill -f 'idletoken-clien[t]'; pkill -9 -f 'idletoken-coor[d]'; pkill -9 -f 'idletoken-worke[r]'; true" >/dev/null 2>&1
}
trap cleanup EXIT
cleanup

echo "== [1/5] fetch the installer from $BUILD_NODE =="
$SSH "$COORD_NODE" "scp -q -o BatchMode=yes $BUILD_NODE:\"$SETUP_WIN\" /tmp/IdleToken-setup.exe" \
    || die "could not pull the installer from $BUILD_NODE (built it? scripts\\build_client_release.bat)"
size=$($SSH "$COORD_NODE" "stat -c %s /tmp/IdleToken-setup.exe 2>/dev/null" | tr -d '\r')
[ "${size:-0}" -gt 1000000 ] || die "installer looks empty ($size bytes)"
echo "   installer $((size / 1024 / 1024)) MiB"

echo "== [2/5] install silently on $TARGET (clean machine, driver only) =="
# Uninstall first. NSIS install-over does NOT remove files that left the
# package, so an older install keeps donating them to the new one — on
# 2026-08-04 the target still had January's cublas/cublasLt (769 MB) after we
# stopped shipping them, the file-count check happily said 9/9, and the run
# "proved" a self-contained bundle that no longer exists. A walkthrough whose
# whole point is "what we ship works" must start from what we ship, only.
$SSH "$TARGET" "if exist \"${APPDIR_WIN}\\uninstall.exe\" (\"${APPDIR_WIN}\\uninstall.exe\" /S)" >/dev/null 2>&1
sleep 5
$SSH "$TARGET" "if exist \"${APPDIR_WIN}\" rmdir /s /q \"${APPDIR_WIN}\"" >/dev/null 2>&1
$SSH "$COORD_NODE" "scp -q -o BatchMode=yes /tmp/IdleToken-setup.exe $TARGET:\"$THOME/IdleToken-setup.exe\"" \
    || die "could not push the installer to $TARGET"
$SSH "$TARGET" "${THOME//\//\\}\\IdleToken-setup.exe /S" >/dev/null 2>&1
# NSIS /S returns before it is done unpacking ~800 MB of DLLs, and the freshly
# written exe stays locked for a moment after that. Wait the installer out
# instead of racing it (a probe against a locked exe just prints nothing).
for _ in $(seq 1 60); do
    running=$($SSH "$TARGET" "powershell -NoProfile -Command \"(Get-Process IdleToken-setup -ErrorAction SilentlyContinue | Measure-Object).Count\"" 2>/dev/null | tr -d '\r ')
    [ "${running:-0}" = "0" ] && break
    sleep 5
done
sleep 3
present=$($SSH "$TARGET" "powershell -NoProfile -Command \"@('idletoken-client.exe','idletoken-worker.exe','idletoken-coord.exe','idletoken-platform-agent.exe','ds4cuda.dll','ds4xcuda.dll') | Where-Object { Test-Path (Join-Path '$APPDIR' \$_) } | Measure-Object | Select-Object -ExpandProperty Count\"" 2>/dev/null | tr -d '\r ')
[ "$present" = "6" ] || die "installed tree is incomplete ($present/6 expected files in $APPDIR)"
# ...and the 769 MB we deliberately stopped shipping must NOT be there. Counting
# what is present cannot catch a file that should be absent; this is the
# executable form of the 2026-08-04 packaging decision (philosophy 12 / E3).
extra=$($SSH "$TARGET" "powershell -NoProfile -Command \"@('cublas64_12.dll','cublasLt64_12.dll') | Where-Object { Test-Path (Join-Path '$APPDIR' \$_) } | Measure-Object | Select-Object -ExpandProperty Count\"" 2>/dev/null | tr -d '\r ')
[ "${extra:-0}" = "0" ] || die "cuBLAS is in the installed tree ($extra file(s)) — it is supposed to come from the user's CUDA Toolkit now"
echo "   installed 6/6 files into $APPDIR (no cuBLAS — user's Toolkit supplies it)"

echo "== [3/5] every shipped binary runs with a driver-only PATH =="
# The engines print their probe/selftest output on stderr — merge it on the
# REMOTE side, or the local 2>/dev/null swallows exactly what we assert on.
probe=$($SSH "$TARGET" "cd /d $APPDIR_WIN && set \"PATH=C:\\WINDOWS\\system32;C:\\WINDOWS\" && idletoken-worker.exe --probe-only 2>&1" 2>/dev/null | tr -d '\r')
echo "$probe" | grep -Eq "gpu:.*cc [1-9]" || die "packaged worker did not probe a GPU (missing DLL?): $(echo "$probe" | head -2)"
for exe in idletoken-coord idletoken-platform-agent; do
    st=$($SSH "$TARGET" "cd /d $APPDIR_WIN && set \"PATH=C:\\WINDOWS\\system32;C:\\WINDOWS\" && $exe.exe --selftest 2>&1" 2>/dev/null | tr -d '\r' | tail -1)
    case "$st" in *"ALL PASS"*) ;; *) die "$exe --selftest did not pass (got '$st')" ;; esac
done
echo "   worker probes a real GPU; coord + agent selftests pass"

echo "== [4/5] pair by code: coordinator client + the INSTALLED client =="
# Fresh shard cache so the weight fetch is exercised for real, not served from
# a previous run's leftovers.
$SSH "$TARGET" "powershell -NoProfile -Command \"Remove-Item '$APPDIR/L*.gguf*' -Force -ErrorAction SilentlyContinue\"" >/dev/null 2>&1
$SSHF "$COORD_NODE" "cd $COORD_HOME/client/src-tauri && exec env DISPLAY=:1 WEBKIT_DISABLE_DMABUF_RENDERER=1 no_proxy=localhost,127.0.0.1 IDLETOKEN_HEADLESS_PAIR='create:$CODE:coord-node:model=$GGUF' ./target/debug/idletoken-client < /dev/null > /tmp/install-creator.log 2>&1" >/dev/null 2>&1
sleep 6
$SSHF "$TARGET" "cd /d $APPDIR_WIN && set \"IDLETOKEN_HEADLESS_PAIR=join:$CODE:installed-node\" && idletoken-client.exe > $THOME/install-joiner.log 2>&1" >/dev/null 2>&1

status=""
for _ in $(seq 1 "$READY_TRIES"); do
    status=$($SSH "$COORD_NODE" "curl -s -m 3 http://127.0.0.1:8000/v1/cluster/status 2>/dev/null" | tr -d '\r')
    case "$status" in *'"phase":"ready"'*) break ;; esac
    sleep 15
done
case "$status" in
    *'"phase":"ready"'*) ;;
    *) die "cluster never reached ready (status='$status'; logs: $COORD_NODE:/tmp/install-creator.log, $TARGET:$THOME/install-joiner.log)" ;;
esac
echo "   $status"

# The installed node must have loaded a REAL engine — a MOCK fallback still
# reports "ready" and then answers with garbage tokens.
# findstr, not PowerShell: `$_` has to survive bash -> ssh -> cmd -> PS quoting
# and it does not. A mangled command yields an empty result, which reads as
# "fell back to MOCK" — a false red that costs a full 10-minute rerun.
jlog=$($SSH "$TARGET" "findstr /C:\"load REAL\" /C:\"load MOCK\" /C:\"fetching layers\" ${THOME//\//\\}\\install-joiner.log" 2>/dev/null | tr -d '\r')
echo "$jlog" | grep -q "load REAL"     || die "installed node fell back to MOCK (see $THOME/install-joiner.log)"
echo "$jlog" | grep -q "fetching layers" || die "installed node did not fetch a layer shard (did it have a local GGUF?)"
echo "   $(echo "$jlog" | grep 'fetching layers' | head -1 | sed 's/.*idletoken-weights: //')"

echo "== [5/5] real reply through the API =="
reply=$($SSH "$COORD_NODE" "curl -s -m 300 http://127.0.0.1:8000/v1/messages -H 'content-type: application/json' -d '{\"model\":\"deepseek-v4-flash\",\"max_tokens\":12,\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one word.\"}]}'" 2>/dev/null)
echo "$reply" | grep -qi paris || die "API did not answer correctly (got: ${reply:0:200})"
echo "   answered: Paris"

echo INSTALL_E2E_OK
