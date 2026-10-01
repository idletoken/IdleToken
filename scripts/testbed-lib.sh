# Shared loader for the testbed configuration, sourced by acceptance.sh,
# sync-to-win.sh, install_walkthrough.sh, topology_matrix.sh and pair_*.sh.
#
# Why one library instead of a copy in every script: this repo has been bitten
# several times by two or three copies of the same fact (two Windows build
# scripts drifting apart, three hardcoded user-directory maps, two price seeds).
# The symptom is identical every time -- one copy gets changed while the other
# keeps quietly running the old behaviour.
#
# Usage:
#   . "$(dirname "$0")/testbed-lib.sh"
#   home=$(testbed_repo_home my-win-box)   # -> C:/Users/<user>/IdleToken
#   prof=$(testbed_profile   my-win-box)   # -> C:/Users/<user>
# Both return an empty string when nothing is configured, and the caller decides
# how to report it (the contracts differ: the acceptance ladder wants exit code 2,
# the installation walkthrough wants its INSTALL_E2E_FAIL line).

# Environment variables given on the command line win over the file -- a full
# acceptance run often narrows the node set through IDLETOKEN_WORKER_NODES, and
# the config file must not clobber that.
_tb_pre_coord="${IDLETOKEN_COORD_NODE:-}"
_tb_pre_workers="${IDLETOKEN_WORKER_NODES:-}"
_tb_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# `set -a` so every setting in the file is EXPORTED, not merely set in this
# shell. Several gates delegate to a child script (ppl_gate.sh,
# gpriv7_embedding_check.sh, model_fetch.sh...), and a child only inherits
# exported variables: on 2026-08-15 the ladder had IDLETOKEN_SMOKE_GGUF set and
# used it fine inline, while ppl_gate.sh saw nothing and skipped "because it is
# not configured". A config file that only half-reaches its consumers is worse
# than no config file -- it skips quietly.
# shellcheck disable=SC1090
if [ -f "$_tb_dir/testbed.env" ]; then
    set -a
    . "$_tb_dir/testbed.env"
    set +a
fi
# It only counts as an override when the command-line value differs from the
# file. Callers use this to decide whether to say so loudly -- the point of that
# notice is to make **running less than usual** conspicuous, and printing it every
# round means nobody reads it (when the config file already named the same node
# set, it had become constant noise).
IDLETOKEN_WORKERS_OVERRIDDEN=""
IDLETOKEN_COORD_OVERRIDDEN=""
[ -n "$_tb_pre_workers" ] && [ "$_tb_pre_workers" != "${IDLETOKEN_WORKER_NODES:-}" ] && IDLETOKEN_WORKERS_OVERRIDDEN=1
[ -n "$_tb_pre_coord" ]   && [ "$_tb_pre_coord"   != "${IDLETOKEN_COORD_NODE:-}" ]   && IDLETOKEN_COORD_OVERRIDDEN=1
[ -n "$_tb_pre_coord" ]   && IDLETOKEN_COORD_NODE="$_tb_pre_coord"
[ -n "$_tb_pre_workers" ] && IDLETOKEN_WORKER_NODES="$_tb_pre_workers"
unset _tb_pre_coord _tb_pre_workers

# alias -> that machine's user directory (e.g. C:/Users/someone). Empty when
# unconfigured.
testbed_profile() {
    local key; key="IDLETOKEN_PROFILE_$(printf '%s' "$1" | tr -c 'A-Za-z0-9' '_')"
    printf '%s' "${!key:-}"
}

# alias -> the repository checkout on that machine. By convention
# <user directory>/IdleToken.
testbed_repo_home() {
    local p; p="$(testbed_profile "$1")"
    if [ -n "$p" ]; then printf '%s' "$p/IdleToken"; fi
    # The explicit `return 0` is required: written as `[ -n "$p" ] && printf ...`
    # an unconfigured node makes the function return non-zero, and the caller
    # sync-to-win.sh runs with `set -e` -- so the script dies **before** it reaches
    # its own check and prints no hint at all. Missing configuration should say
    # what is missing, not exit silently.
    return 0
}

# alias -> the weight directories to search on that machine, in order, space
# separated. Falls back to <repo checkout>/gguf when unconfigured: one machine may
# keep different model families in different places (DSv4 next to the ds4
# checkout, small models elsewhere), so this is a list per node rather than one
# directory.
testbed_gguf_dirs() {
    local key; key="IDLETOKEN_GGUF_DIRS_$(printf '%s' "$1" | tr -c 'A-Za-z0-9' '_')"
    local v="${!key:-}"
    if [ -n "$v" ]; then printf '%s' "$v"
    else
        local h; h="$(testbed_repo_home "$1")"
        if [ -n "$h" ]; then printf '%s' "$h/gguf"; fi
    fi
    return 0
}

# One wording for missing configuration -- three scripts each writing their own
# version means the same problem grows three different faces.
testbed_hint() {  # testbed_hint <alias>
    # Print the **normalized** variable name. Hyphens and other characters in an
    # alias are turned into underscores by tr (see the two lookup functions
    # above), while this used to echo the raw alias -- so for an alias like
    # `my-win-box` the hint told you to add `IDLETOKEN_PROFILE_my-win-box=`, which
    # is not a valid shell variable name and **is still unreadable if you follow
    # it**. A hint that does not work when followed costs more time than none.
    local key; key="IDLETOKEN_PROFILE_$(printf '%s' "$1" | tr -c 'A-Za-z0-9' '_')"
    echo "no user directory configured for node $1. Add a line to scripts/testbed.env" >&2
    echo "(template: scripts/testbed.env.example):" >&2
    echo "    $key='C:/Users/<the username on that machine>'" >&2
}

# ---------------------------------------------------------------------------
# Which OS family a node runs. Asked of the machine, never guessed from its name
# -- the whole premise of the product is "any machines", and a wrong guess does
# not report itself, it reports a probe failure far from the cause.
#
# `uname -s` succeeds on Linux and macOS and fails under Windows OpenSSH (whose
# default shell is cmd or powershell), so that is the test. Memoized in a string
# (the bash 3.2 macOS ships has no associative arrays), so each machine is asked
# once. It SETS a variable instead of echoing: a `$(...)` call would memoize
# inside a subshell and re-ssh every time.
#
# Caveat: an unreachable machine also produces no `uname` output and is therefore
# read as Windows. Callers that care must check liveness first.
_TB_NODE_OS=" "
NODE_OS=""
testbed_node_os() {  # testbed_node_os <alias> -> sets NODE_OS to windows|linux|macos|unknown
    local rest u
    case "$_TB_NODE_OS" in
        *" $1:"*) rest="${_TB_NODE_OS#*" $1:"}"; NODE_OS="${rest%% *}"; return 0 ;;
    esac
    u=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "uname -s" 2>/dev/null | tr -d '\r')
    case "$u" in
        Linux)  NODE_OS=linux ;;
        Darwin) NODE_OS=macos ;;
        "")     NODE_OS=windows ;;
        *)      NODE_OS=unknown ;;
    esac
    _TB_NODE_OS="$_TB_NODE_OS$1:$NODE_OS "
}

# ---------------------------------------------------------------------------
# The pair of machines used by the cross-machine benchmarks and concurrency
# checks (xmachine_bench, xmachine_concurrent_check, fullchain_concurrent_check,
# pair_xmachine_infer). First entry is the coordinator.
#
# Why this exists as its own setting rather than "the coordinator plus the first
# worker": since 2026-08-12 a cluster must be HOMOGENEOUS, and on a typical
# testbed the coordinator node is the one strong Linux box while the workers are
# Windows. That combination is no longer a legal cluster -- the coordinator
# refuses the second machine at HELLO -- so these scripts need a same-OS pair,
# which is generally NOT the pair the rest of the ladder uses.
#
# Default: the first two IDLETOKEN_WORKER_NODES. The caller must still verify
# they agree (testbed_xm_check below) -- a default that happens to be illegal
# should say so, not fail somewhere downstream.
testbed_xm_nodes() {
    if [ -n "${IDLETOKEN_XM_NODES:-}" ]; then printf '%s' "$IDLETOKEN_XM_NODES"; return 0; fi
    set -- ${IDLETOKEN_WORKER_NODES:-}
    [ $# -ge 2 ] && printf '%s %s' "$1" "$2"
    return 0
}

# Resolve + validate the pair. On success sets XM_COORD / XM_WORKER / XM_OS and
# returns 0; otherwise explains which line to add and returns 1.
testbed_xm_check() {
    local nodes; nodes=$(testbed_xm_nodes)
    set -- $nodes
    if [ $# -lt 2 ]; then
        echo "this check needs TWO compute nodes of the same OS family." >&2
        echo "Add to scripts/testbed.env (template: testbed.env.example):" >&2
        echo "    IDLETOKEN_XM_NODES=\"<coordinator-alias> <worker-alias>\"" >&2
        return 1
    fi
    XM_COORD="$1"; XM_WORKER="$2"
    testbed_node_os "$XM_COORD"; local a="$NODE_OS"
    testbed_node_os "$XM_WORKER"; local b="$NODE_OS"
    if [ "$a" != "$b" ]; then
        echo "IDLETOKEN_XM_NODES is $XM_COORD ($a) + $XM_WORKER ($b) -- a cluster must be" >&2
        echo "homogeneous (CLAUDE.md #2); the coordinator refuses a mixed-OS join at HELLO." >&2
        echo "Name two machines of the same OS family in scripts/testbed.env." >&2
        return 1
    fi
    XM_OS="$a"
    return 0
}

# A node's LAN IPv4. Workers must be told an address their peers can dial, and
# the control machine cannot know it -- it changes, and on Windows there may be
# several adapters. Ask the machine.
testbed_lan_ip() {  # testbed_lan_ip <alias>
    testbed_node_os "$1"
    if [ "$NODE_OS" = windows ]; then
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" \
            "powershell -NoProfile -Command \"(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { \$_.IPAddress -like '192.168.*' -or \$_.IPAddress -like '10.*' } | Select-Object -First 1).IPAddress\"" \
            2>/dev/null | tr -d '\r ' | tail -1
    else
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" \
            "ip -4 -o addr show scope global | awk '{split(\$4,a,\"/\"); print a[1]}' | grep -E '^(192\.168\.|10\.)' | head -1" \
            2>/dev/null | tr -d '\r'
    fi
}

# ---------------------------------------------------------------------------
# Issue an API request **on the node itself** rather than connecting from the
# control machine.
#
#   testbed_api_post <ssh-alias> <port> <path> <json>
#   -> prints "TIME=<seconds> CODE=<http code>"; on anything but 200 it returns 1
#      and writes the response body to stderr.
#
# Why it has to work this way: the control machine (a development Mac or laptop)
# has a changing IP and is **not necessarily on the cluster's subnet**. Connecting
# directly to the coordinator once made an entire benchmark round read "11.04s" --
# that was curl's connection timeout, not inference time. It looked far too much
# like a plausible performance number and fooled six rounds of debugging on real
# hardware: I suspected and reverted timing changes in the engine, suspected a
# forced split, suspected the CUDA driver state, suspected a coordinator crash.
# None of them. ping does not help either -- Windows blocks ICMP by default, so
# no reply proves nothing.
#
# Two design constraints; drop either one and the trap comes back:
#   1. The request is issued on the node itself against 127.0.0.1, so the control
#      machine's network position never enters the measurement path.
#   2. **No timing may be reported unless the HTTP code is 200.** The previous
#      version fooled us for so long precisely because it printed the duration of
#      a failure as though it were the duration of a success.
testbed_api_post() {  # <alias> <port> <path> <json>
    local node="$1" port="$2" path="$3" body="$4"
    local out code time
    out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" \
        "curl.exe -s -m 900 -o %TEMP%\\idletoken_api_resp.json \
         -w \"TIME=%{time_total} CODE=%{http_code}\" \
         -X POST http://127.0.0.1:$port$path \
         -H \"content-type: application/json\" -d \"$(printf '%s' "$body" | sed 's/"/\\"/g')\"" \
        2>/dev/null | tr -d '\r')
    time=$(printf '%s' "$out" | grep -o 'TIME=[0-9.]*' | cut -d= -f2)
    code=$(printf '%s' "$out" | grep -o 'CODE=[0-9]*' | cut -d= -f2)
    if [ "${code:-000}" != "200" ]; then
        echo "testbed_api_post: $node:$port$path returned HTTP ${code:-no response} (took ${time:-?}s)" >&2
        echo "  That duration is **not** inference time; do not use it as a performance number. Response body:" >&2
        ssh -o BatchMode=yes "$node" 'type %TEMP%\idletoken_api_resp.json' 2>/dev/null | head -c 400 >&2
        echo >&2
        return 1
    fi
    printf 'TIME=%s CODE=%s\n' "$time" "$code"
}

# Fetch the response body of the node's last request (used with testbed_api_post).
testbed_api_last_body() {  # <alias>
    ssh -o BatchMode=yes "$1" 'type %TEMP%\idletoken_api_resp.json' 2>/dev/null | tr -d '\r'
}
