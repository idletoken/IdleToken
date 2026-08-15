# Shared runner for the product gates that drive the desktop client
# (client_update_gate.sh = G-UPDATE, client_tray_gate.sh = G-TRAY).
#
# Why it exists: both gates need the same four awkward things — a debug client
# binary on the machine under test, a way to launch it with environment
# variables and a timeout, a way to read its UI_TEST_REPORT lines back, and a
# way to touch the shell's window.json. Written twice, those drift; this repo
# has paid for that with three copies of a user-directory map.
#
# WHICH MACHINE. `IDLETOKEN_CLIENT_NODE` picks it:
#   unset      -> this machine (fine for a quick local check on a control Mac)
#   <ssh alias> -> that machine, over ssh. Windows nodes are the point: the tray
#                  is the notification area and the installer is NSIS, neither of
#                  which a macOS run says anything about.
#
# A debug build is required, not a preference: the updater plugin refuses a
# plain-http endpoint in release builds, and the update gate serves its signed
# test feed over http from the control machine. See build_client_win.sh --debug.

# shellcheck disable=SC1091
. "$(dirname "${BASH_SOURCE[0]:-$0}")/testbed-lib.sh"

CG_SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
CG_NODE="${IDLETOKEN_CLIENT_NODE:-}"
CG_KIND=local          # local | windows
CG_BIN=""              # path to the client binary, on the machine under test
CG_WIN_HOME=""         # C:\Users\x\IdleToken (windows only)
CG_PREFS=""            # window.json, on the machine under test
# Why a variable rather than an echoed message: cg_init sets the globals above,
# so it must run in THIS shell. Called as `why=$(cg_init)` it would run in a
# subshell and every global would be thrown away — which is exactly what
# happened the first time, and the gate silently tested the control machine
# while reporting the node's name.
CG_ERR=""

# Resolve everything above. Echoes a reason and returns 1 when the node is not
# usable, so each gate can turn that into its own SKIP line.
cg_init() {
    CG_ERR=""
    if [ -z "$CG_NODE" ]; then
        CG_KIND=local
        CG_BIN="$PWD/client/src-tauri/target/debug/idletoken-client"
        case "$(uname)" in
            Darwin) CG_PREFS="$HOME/Library/Application Support/ai.idletoken.client/window.json" ;;
            *)      CG_PREFS="${XDG_CONFIG_HOME:-$HOME/.config}/ai.idletoken.client/window.json" ;;
        esac
        return 0
    fi

    local home profile
    home="$(testbed_repo_home "$CG_NODE")"
    if [ -z "$home" ]; then
        CG_ERR="no testbed entry for $CG_NODE (add IDLETOKEN_PROFILE_$CG_NODE to scripts/testbed.env)"
        return 1
    fi
    $CG_SSH "$CG_NODE" "echo up" >/dev/null 2>&1 || { CG_ERR="cannot ssh to $CG_NODE"; return 1; }

    # Windows is the only remote kind for now; a Linux node would run the local
    # branch through ssh, which is a different (and unwritten) code path.
    local uname_out
    uname_out=$($CG_SSH "$CG_NODE" "ver" 2>/dev/null | tr -d '\r' | head -2 | tail -1)
    case "$uname_out" in
        *Windows*|*windows*) CG_KIND=windows ;;
        *) CG_ERR="$CG_NODE does not look like a Windows node (got: $uname_out); only Windows nodes are supported remotely"; return 1 ;;
    esac

    CG_WIN_HOME="${home//\//\\}"
    CG_BIN="$home/client/src-tauri/target/debug/idletoken-client.exe"
    profile="$(testbed_profile "$CG_NODE")"
    CG_PREFS="$profile/AppData/Roaming/ai.idletoken.client/window.json"

    # A GUI process needs a desktop to draw on. A locked or logged-out Windows
    # session has bitten this project before (WebView2 simply fails to create),
    # and the honest answer there is a SKIP, not a red gate.
    local sess
    sess=$($CG_SSH "$CG_NODE" "query session" 2>/dev/null | tr -d '\r' | grep -i "console" | grep -ci "active")
    if [ "${sess:-0}" -lt 1 ]; then
        CG_ERR="$CG_NODE has no active console session — log in on that machine (the client is a GUI app)"
        return 1
    fi

    # Deliberately NOT resolved here: "this machine's address as the node sees
    # it". It was, from %SSH_CLIENT%, and on this testbed that address is not
    # one we listen on — the update gate spent a run failing on a connection
    # refused that had nothing to do with the updater. Anything the node has to
    # fetch from us goes through an ssh reverse tunnel to 127.0.0.1 instead.
    return 0
}

# Make sure a debug binary exists on the machine under test.
cg_build() {
    CG_ERR=""
    if [ "$CG_KIND" = local ]; then
        [ -d client/dist ] || ( cd client && pnpm build ) >/tmp/idletoken-cg-web.log 2>&1 \
            || { CG_ERR="pnpm build failed (see /tmp/idletoken-cg-web.log)"; return 1; }
        [ -x "$CG_BIN" ] && return 0
        # --features custom-protocol: without it the debug binary loads devUrl
        # (a dev server that is not running) instead of the bundled dist, the
        # front end never starts, and every directive quietly reports nothing.
        ( cd client/src-tauri && cargo build --features custom-protocol ) >/tmp/idletoken-cg-cargo.log 2>&1 \
            || { CG_ERR="cargo build failed (see /tmp/idletoken-cg-cargo.log)"; return 1; }
        [ -x "$CG_BIN" ] || { CG_ERR="no client binary at $CG_BIN"; return 1; }
        return 0
    fi
    # Windows: one command, and it is the same one a human would run. It always
    # reships the sources — the gate must not certify a binary built before the
    # change under test.
    local out
    out=$(./scripts/build_client_win.sh --debug "$CG_NODE" 2>&1 | tail -1)
    case "$out" in
        CLIENT_WIN_DEBUG_OK*) return 0 ;;
        *) CG_ERR="could not build the debug client on $CG_NODE: $out"; return 1 ;;
    esac
}

# cg_run <directives> <timeout_s> <log> [VAR=VALUE ...]
# Runs the client, waits for it to exit (the `quit:<ms>` directive is what ends
# it), and leaves everything it printed in <log>. Kills it on timeout so a hung
# case costs one case, not the whole gate.
#
# WINDOWS: NOT a plain ssh command. Windows OpenSSH runs its shell in session 0
# (the services session), which has no desktop — a GUI process started there
# comes up with no window at all, runs forever and prints nothing, which reads
# exactly like "the feature is broken". The client has to be launched into the
# console session, and a scheduled task registered with /IT (interactive token)
# for the logged-in user is the way to do that without extra tools. Everything
# the client prints goes to a file on that machine, which is fetched afterwards.
cg_run() {
    local directives="$1" timeout_s="$2" log="$3"
    shift 3
    if [ "$CG_KIND" = local ]; then
        local envs=() pid waited=0
        for kv in "$@"; do envs+=("$kv"); done
        # ${envs[@]+...}: an empty array under `set -u` is an error in bash 3.2
        # (what macOS ships), and the gates call this with no extra variables.
        ( cd client && env ${envs[@]+"${envs[@]}"} IDLETOKEN_UI_TEST="$directives" "$CG_BIN" >"$log" 2>&1 ) &
        pid=$!
        while kill -0 $pid 2>/dev/null && [ "$waited" -lt "$timeout_s" ]; do sleep 1; waited=$((waited+1)); done
        if kill -0 $pid 2>/dev/null; then kill -9 $pid 2>/dev/null; echo "TIMEOUT"; fi
        wait $pid 2>/dev/null
        return 0
    fi

    local profile user runner_local runner_win remote_log task="IdleTokenGateRun"
    profile="$(testbed_profile "$CG_NODE")"
    user="${profile##*/}"
    remote_log="$profile/idletoken-gate-run.log"
    runner_local="${TMPDIR:-/tmp}/idletoken-gate-run.cmd"
    runner_win="${profile//\//\\}\\idletoken-gate-run.cmd"

    {
        echo "@echo off"
        echo "cd /d \"${CG_WIN_HOME}\\client\""
        for kv in "$@"; do echo "set \"$kv\""; done
        echo "set \"IDLETOKEN_UI_TEST=$directives\""
        echo "\"${CG_BIN//\//\\}\" > \"${remote_log//\//\\}\" 2>&1"
        # The marker is how the poll below tells "finished" from "still going"
        # without asking Windows about processes every second.
        echo "echo GATE_RUN_DONE=%ERRORLEVEL% >> \"${remote_log//\//\\}\""
    } > "$runner_local"
    scp -q "$runner_local" "$CG_NODE:$profile/idletoken-gate-run.cmd" || { echo "SCPFAIL"; return 0; }

    $CG_SSH "$CG_NODE" "del /q \"${remote_log//\//\\}\" 2>NUL & schtasks /create /tn $task /tr \"$runner_win\" /sc once /st 00:00 /ru $user /it /f >NUL && schtasks /run /tn $task >NUL" >/dev/null 2>&1

    local waited=0 done_marker=""
    while [ "$waited" -lt "$timeout_s" ]; do
        sleep 2
        waited=$((waited+2))
        done_marker=$($CG_SSH "$CG_NODE" "findstr /c:GATE_RUN_DONE \"${remote_log//\//\\}\"" 2>/dev/null | tr -d '\r')
        [ -n "$done_marker" ] && break
    done
    if [ -z "$done_marker" ]; then
        $CG_SSH "$CG_NODE" "taskkill /IM idletoken-client.exe /F" >/dev/null 2>&1
        echo "TIMEOUT"
    fi
    $CG_SSH "$CG_NODE" "type \"${remote_log//\//\\}\"" 2>/dev/null | tr -d '\r' > "$log"
    $CG_SSH "$CG_NODE" "schtasks /delete /tn $task /f >NUL 2>&1 & del /q \"$runner_win\" \"${remote_log//\//\\}\" 2>NUL & exit 0" >/dev/null 2>&1
    return 0
}

# cg_report <log> <tag> -> the JSON that tag reported, or empty.
cg_report() {
    grep "UI_TEST_REPORT $2 " "$1" 2>/dev/null | tr -d '\r' | head -1 | sed "s/.*UI_TEST_REPORT $2 //"
}

# cg_field <json> <key> -> the value python prints for it ("True"/"False" for
# booleans, empty when absent).
cg_field() {
    printf '%s' "$1" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get(sys.argv[1], ""))' "$2" 2>/dev/null
}

# ---- the shell's window.json, on the machine under test --------------------
# Read into a local file ($1). Returns 1 when there is none.
cg_prefs_pull() {
    if [ "$CG_KIND" = local ]; then
        [ -f "$CG_PREFS" ] || return 1
        cp "$CG_PREFS" "$1"
        return 0
    fi
    $CG_SSH "$CG_NODE" "type \"${CG_PREFS//\//\\}\"" 2>/dev/null | tr -d '\r' > "$1"
    [ -s "$1" ] || { rm -f "$1"; return 1; }
    return 0
}

# Write a local file ($1) to the machine under test.
cg_prefs_push() {
    if [ "$CG_KIND" = local ]; then
        mkdir -p "$(dirname "$CG_PREFS")" && cp "$1" "$CG_PREFS"
        return $?
    fi
    local dir_win="${CG_PREFS%/*}"
    $CG_SSH "$CG_NODE" "if not exist \"${dir_win//\//\\}\" mkdir \"${dir_win//\//\\}\"" >/dev/null 2>&1
    scp -q "$1" "$CG_NODE:$CG_PREFS"
}

cg_prefs_rm() {
    if [ "$CG_KIND" = local ]; then rm -f "$CG_PREFS"; return 0; fi
    $CG_SSH "$CG_NODE" "del /q \"${CG_PREFS//\//\\}\"" >/dev/null 2>&1
    return 0
}

# A human-readable name for the machine under test, for the gate's own output.
cg_where() {
    if [ "$CG_KIND" = local ]; then echo "this machine"; else echo "$CG_NODE (Windows)"; fi
}
