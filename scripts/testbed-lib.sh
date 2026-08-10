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
# shellcheck disable=SC1090
[ -f "$_tb_dir/testbed.env" ] && . "$_tb_dir/testbed.env"
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
