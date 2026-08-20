#!/usr/bin/env bash
# Bring up a REAL llama.cpp cluster and hand the gates something to assert on.
#
# This is the replacement vehicle for scripts/run_cluster.sh, whose INFER_* wire
# protocol and ds4 engine were retired (2026-08-14 pivot / 2026-08-16 shelving).
# That file stays on disk unchanged for the archaeology; the ladder no longer
# calls it.
#
#   --serve         bring the cluster up, print CLUSTER_LLAMACPP_READY <json>,
#                   LEAVE IT RUNNING. State lands in the run dir so --stop works
#                   from a different shell than the one that started it.
#   --check-ready   --serve, assert the readiness facts, tear down, print
#                   CLUSTER_READY. The G5 shape.
#   --facts         print the recorded facts of a running cluster (no bring-up).
#   --stop          tear down whatever --serve left behind.
#
# WHAT IT DRIVES is the product path, same as the topology matrix cell:
# idletoken-coord in llamacpp cluster mode (pairing by code, PSK minted by the
# coordinator and wrapped over the pairing channel, scheduler-produced
# tensor-split) plus idletoken-worker --rpc-supervisor on each worker node.
# Never a hand-built `llama-server --rpc`: a gate measured by a different code
# path than the product uses proves nothing.
#
# RELATIONSHIP TO matrix_cell_cluster_llamacpp.sh (testbed tooling, private)
#   The machine orchestration below (per-node OS/home resolution, the ssh-held
#   process model, the teardown) is LIFTED from that script, which has been
#   green on three cells at b10502. It is deliberately a copy, not a shared
#   library: the matrix cell is the ruler and this is the judge, and the plan
#   for this work says to keep them independent so re-greening one cannot be
#   held hostage by the other. If you fix a process-handling bug here, look at
#   that file too -- and vice versa. Converging them is a follow-up that has to
#   re-green the matrix cells, not a drive-by.
#
# THE HOOKS the gates need (this is the part the matrix cell does not have):
#   * IDLETOKEN_CLUSTER_COORD_ENV / _WORKER_ENV -- arbitrary `K=V K=V` forwarded
#     across the ssh boundary to the coordinator / every worker. This is how a
#     gate stages its RED control (no PSK, plaintext allowed, faked engine
#     version, an IDLETOKEN_LLAMA_ARGS override). The matrix cell forwards a
#     FIXED list of four variables, and on 2026-08-20 that cost a whole run: the
#     escape hatch the coordinator names in its own refusal message never
#     reached the coordinator, so following the printed advice changed nothing.
#     A gate that cannot set the environment cannot build a negative control.
#   * --coord-arg / --worker-arg -- extra argv, for the same reason (the overlay
#     refusal control needs `--rpc-host 100.64.0.1`).
#   * --expect-refuse -- invert the verdict: the run is a PASS when the
#     coordinator or a worker refuses. Without this every red control has to
#     re-implement "did it fail for the reason I wanted?" by hand.
#   * the facts json -- coord/worker log paths, engine log path, the api base,
#     the rpc endpoints, the tensor split. Assertions read files this names;
#     none of them re-derive where things are.
#
# Contract: last line is CLUSTER_LLAMACPP_READY <json> / CLUSTER_READY /
# CLUSTER_STOPPED / CLUSTER_REFUSED <reason> / CLUSTER_FAIL: <reason>.
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD
# shellcheck disable=SC1091
. "$(dirname "$0")/testbed-lib.sh"

MODE=""
COORD=""; WORKERS=(); MODEL=""; QUANT=""; GGUF=""
COORD_ARGS_EXTRA=""; WORKER_ARGS_EXTRA=""
EXPECT_REFUSE=0
TAG="${IDLETOKEN_CLUSTER_TAG:-gate}"
API_PORT="${IDLETOKEN_CLUSTER_API_PORT:-18530}"
CTX="${IDLETOKEN_CLUSTER_CTX:-4096}"
READY_WAIT_S="${IDLETOKEN_CLUSTER_READY_S:-420}"

while [ $# -gt 0 ]; do
    case "$1" in
        --serve)        MODE=serve ;;
        --check-ready)  MODE=check ;;
        --facts)        MODE=facts ;;
        --stop)         MODE=stop ;;
        --coord)        shift; COORD="${1:-}" ;;
        --worker)       shift; WORKERS+=("${1:-}") ;;
        --model)        shift; MODEL="${1:-}" ;;
        --quant)        shift; QUANT="${1:-}" ;;
        --gguf)         shift; GGUF="${1:-}" ;;
        --api-port)     shift; API_PORT="${1:-}" ;;
        --ctx)          shift; CTX="${1:-}" ;;
        --tag)          shift; TAG="${1:-}" ;;
        --coord-arg)    shift; COORD_ARGS_EXTRA="$COORD_ARGS_EXTRA ${1:-}" ;;
        --worker-arg)   shift; WORKER_ARGS_EXTRA="$WORKER_ARGS_EXTRA ${1:-}" ;;
        --expect-refuse) EXPECT_REFUSE=1 ;;
        --ready-wait)   shift; READY_WAIT_S="${1:-}" ;;
        -h|--help)      sed -n '2,55p' "$0"; exit 0 ;;
        *) echo "CLUSTER_FAIL: unknown argument $1"; exit 2 ;;
    esac
    shift
done
[ -n "$MODE" ] || { echo "CLUSTER_FAIL: one of --serve/--check-ready/--facts/--stop is required"; exit 2; }

RUNDIR="${TMPDIR:-/tmp}/idletoken-cluster-llamacpp/$TAG"
PROCS="$RUNDIR/procs"       # one "<node> <pid> <logbase>" per line, start order
FACTS="$RUNDIR/facts.json"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"

say()  { printf '%s\n' "$*"; }
fail() { say "CLUSTER_FAIL: $1"; teardown >/dev/null 2>&1; exit 1; }

# ---------------------------------------------------------------- per-node ---
# Lifted from matrix_cell_cluster_llamacpp.sh; see the header note.
node_os() {  # testbed alias (or local) -> windows|linux|macos
    if [ "$1" = local ]; then
        [ "$(uname -s)" = Darwin ] && echo macos || echo linux
    elif [ -n "$(testbed_profile "$1")" ]; then echo windows; else echo linux; fi
}
node_home() {  # repo checkout on that node -- ALWAYS absolute (a relative home
    # resolved against the post-cd cwd and broke both the binary and log paths)
    if [ "$1" = local ]; then echo "$ROOT"
    elif [ "$(node_os "$1")" = windows ]; then testbed_repo_home "$1"
    else echo "$($SSH "$1" 'printf %s "$HOME"')/${IDLETOKEN_REMOTE_DIR:-work/IdleToken}"; fi
}
node_sh() {  # run a shell snippet on a non-windows node
    local n="$1"; shift
    if [ "$n" = local ]; then bash -c "$1"; else $SSH "$n" "$1"; fi
}
node_log() {  # cat a remote log (both streams merged)
    local n="$1" p="$2"
    if [ "$(node_os "$n")" = windows ]; then
        $SSH "$n" "cmd /c \"type ${p//\//\\}.out 2>nul & type ${p//\//\\}.err 2>nul\"" 2>/dev/null | tr -d '\r'
    else
        node_sh "$n" "cat '$p.out' '$p.err' 2>/dev/null"
    fi
}

# --------------------------------------------------------- process control ---
# Windows: the remote process is HELD BY A PERSISTENT LOCAL SSH, because Windows
# OpenSSH puts session processes in a kill-on-close job object -- any "start
# detached and return" scheme dies the moment the ssh returns. Killing the local
# ssh is therefore also the teardown. POSIX is symmetric on purpose (nohup+&
# hung for the real worker), with one addition: the coordinator SURVIVES the
# sshd HUP, so its remote pid goes in a file next to the logs and teardown kills
# it by pid. An orphaned coordinator is worse than an orphaned worker -- its
# llama-server keeps an RPC connection open and the next run's rpc-server then
# serves two engines at once and reports nonsense.
track_proc() { mkdir -p "$RUNDIR"; printf '%s %s %s\n' "$1" "$2" "$3" >> "$PROCS"; }
start_proc() {  # start_proc <node> <exe> <args> <logbase> <envlist> -> pid
    local n="$1" exe="$2" args="$3" logbase="$4" envlist="$5" home kv
    home=$(node_home "$n")
    if [ "$(node_os "$n")" = windows ]; then
        local wexe="${exe//\//\\}" whome="${home//\//\\}" wenv=""
        for kv in $envlist; do wenv="${wenv}set ${kv}&& "; done
        $SSH "$n" "cmd /c \"cd /d $whome && ${wenv}$wexe $args > $logbase.out 2> $logbase.err\"" >/dev/null 2>&1 &
        echo "local:$!"
    else
        # The brace group matters: without it `&` binds the whole `cd && ...`
        # list, so `echo $!` runs in the login directory and reports the
        # subshell's pid instead of the process we started. `wait` keeps the
        # session held, so the foreground semantics are unchanged.
        $SSH "$n" "cd '$home' && { $envlist '$exe' $args > '$logbase.out' 2> '$logbase.err' & echo \$! > '$logbase.pid'; wait \$!; }" >/dev/null 2>&1 &
        echo "local:$!"
    fi
}
stop_pid() {  # stop_pid <node> <pid>   ("local:N" = kill the local ssh holder)
    local n="$1" pid="$2" home wroot
    [ -n "$pid" ] || return 0
    case "$pid" in
        local:*) kill "${pid#local:}" 2>/dev/null; wait "${pid#local:}" 2>/dev/null ;;
        ''|*[!0-9]*) return 0 ;;
        *)
            if [ "$(node_os "$n")" = windows ]; then
                home=$(node_home "$n"); wroot="${home//\//\\}"
                $SSH "$n" "powershell -NoProfile -ExecutionPolicy Bypass -File $wroot\\scripts\\matrix_cluster_proc_win.ps1 -Action stop -TargetPid $pid" >/dev/null 2>&1
            else
                node_sh "$n" "kill $pid 2>/dev/null; sleep 1; kill -9 $pid 2>/dev/null; true" >/dev/null 2>&1
            fi ;;
    esac
}
teardown() {
    [ -f "$PROCS" ] || { rm -rf "$RUNDIR"; return 0; }
    # Reverse start order: the workers' supervisors die before the coordinator,
    # so the coordinator does not respawn an engine against half a cluster
    # mid-teardown.
    # No `mapfile`: the control machine is a Mac and ships bash 3.2, where that
    # builtin does not exist -- teardown would silently become a no-op and every
    # run would leak a coordinator holding the model in memory.
    local lines=() n pid logbase home rpid kids k i line
    while IFS= read -r line; do [ -n "$line" ] && lines+=("$line"); done < "$PROCS"
    for (( i=${#lines[@]}-1; i>=0; i-- )); do
        # shellcheck disable=SC2086
        set -- ${lines[$i]}; n="$1"; pid="$2"; logbase="$3"
        stop_pid "$n" "$pid"
        sleep 1
        home=$(node_home "$n")
        if [ "$(node_os "$n")" != windows ]; then
            rpid=$(node_sh "$n" "cat '$home/$logbase.pid' 2>/dev/null" | tr -dc '0-9')
            [ -n "$rpid" ] && stop_pid "$n" "$rpid"
        fi
        # Backstop on every OS: children the supervisor's log names, BY PID
        # (never by image name -- another session's engine is not ours to kill).
        kids=$(node_log "$n" "$home/$logbase" | sed -n 's/.*(pid \([0-9][0-9]*\)).*/\1/p' | sort -u)
        for k in $kids; do stop_pid "$n" "$k"; done
    done
    rm -rf "$RUNDIR"
}

if [ "$MODE" = stop ];  then teardown; say "CLUSTER_STOPPED"; exit 0; fi
if [ "$MODE" = facts ]; then
    [ -f "$FACTS" ] || { say "CLUSTER_FAIL: no running cluster recorded under $RUNDIR"; exit 1; }
    cat "$FACTS"; exit 0
fi

# ------------------------------------------------------------- bring-up ------
[ -n "$COORD" ] || COORD="${IDLETOKEN_COORD_NODE:-}"
[ -n "$COORD" ] || fail "no coordinator: pass --coord or set IDLETOKEN_COORD_NODE in scripts/testbed.env"
if [ ${#WORKERS[@]} -eq 0 ]; then
    # shellcheck disable=SC2086
    for w in ${IDLETOKEN_WORKER_NODES:-}; do WORKERS+=("$w"); done
fi
[ ${#WORKERS[@]} -gt 0 ] || fail "no worker nodes: pass --worker or set IDLETOKEN_WORKER_NODES"
[ -n "$MODEL" ] || MODEL="${IDLETOKEN_CLUSTER_MODEL:-}"
[ -n "$MODEL" ] || fail "no model: pass --model (a models/<id>.json id)"
[ -n "$QUANT" ] || QUANT="${IDLETOKEN_CLUSTER_QUANT:-}"

# Refuse to run over somebody else's live cluster rather than asserting against
# it -- the T16 window and this one can overlap on the same machines.
if $SSH "$COORD" "curl -s -m 2 -o /dev/null http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
    fail "port $API_PORT on $COORD is already serving -- another cluster is live there; stop it or pass --api-port"
fi

teardown >/dev/null 2>&1        # clear a previous run's leftovers
mkdir -p "$RUNDIR"

COORD_OS=$(node_os "$COORD"); COORD_HOME=$(node_home "$COORD")
if [ "$COORD_OS" = windows ]; then
    COORD_BIN="$COORD_HOME/idletoken-coord.exe"
    COORD_ENGINE="$COORD_HOME/vendor/llama.cpp/build/bin/Release/llama-server.exe"
else
    COORD_BIN="$COORD_HOME/idletoken-coord"
    COORD_ENGINE="$COORD_HOME/vendor/llama.cpp/build/bin/llama-server"
fi

# The weight file on the COORDINATOR (it must hold the full GGUF -- invariant).
if [ -z "$GGUF" ]; then
    dirs_var="IDLETOKEN_GGUF_DIRS_${COORD}"
    eval "dirs=\${$dirs_var:-}"
    [ -n "${dirs:-}" ] || fail "no GGUF directory for $COORD -- set $dirs_var in scripts/testbed.env or pass --gguf"
    fname=$(python3 - "models/$MODEL.json" "$QUANT" <<'PY' 2>/dev/null
import json, sys
man = json.load(open(sys.argv[1])); want = sys.argv[2]
variants = man.get("variants") or []
v = None
if variants:
    v = next((x for x in variants if x.get("quant") == want), None) if want else None
    if v is None:
        dq = man.get("default_quant")
        v = next((x for x in variants if x.get("quant") == dq), variants[0])
print((v or {}).get("gguf") or man.get("default_gguf", ""))
PY
)
    [ -n "$fname" ] || fail "models/$MODEL.json names no GGUF for quant '$QUANT' -- pass --gguf"
    for d in $dirs; do
        if $SSH "$COORD" "test -r '$d/$fname'" >/dev/null 2>&1; then GGUF="$d/$fname"; break; fi
    done
    [ -n "$GGUF" ] || fail "$fname not found on $COORD in: $dirs"
fi

for n in "$COORD" "${WORKERS[@]}"; do
    if [ "$(node_os "$n")" = windows ]; then
        bash "$ROOT/scripts/sync-to-win.sh" "$n" >/dev/null 2>&1 || fail "could not sync the repo to $n"
    fi
done

# Join code from the unambiguous alphabet -- no O/0/I/1, and the coordinator
# VALIDATES it (a code with an 'O' in it is refused, which once read as a
# pairing failure).
AB="ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
CODE="GT"; for _ in 1 2 3 4; do CODE="$CODE${AB:$((RANDOM % ${#AB})):1}"; done

# The small curated models fit one machine, so the scheduler would -- correctly
# -- release the workers ("don't go networked if it fits"). This is the
# documented test-vehicle escape hatch, and the facts json records that it was
# used so no reader mistakes this for a production-shaped cluster.
COORD_ENV="IDLETOKEN_ALLOW_SMALL_CLUSTER=${IDLETOKEN_ALLOW_SMALL_CLUSTER:-1} ${IDLETOKEN_CLUSTER_COORD_ENV:-}"
WORKER_ENV="${IDLETOKEN_CLUSTER_WORKER_ENV:-}"

CLOGBASE="itc-$TAG-coord"
CARGS="--model-id $MODEL --llama-server-bin $COORD_ENGINE --llama-gguf $GGUF"
CARGS="$CARGS --num-workers ${#WORKERS[@]} --pair-code $CODE --http"
CARGS="$CARGS --api-bind 127.0.0.1:$API_PORT --ctx-size $CTX$COORD_ARGS_EXTRA"
CPID=$(start_proc "$COORD" "$COORD_BIN" "$CARGS" "$CLOGBASE" "$COORD_ENV")
track_proc "$COORD" "$CPID" "$CLOGBASE"
say "coordinator $CPID on $COORD ($COORD_OS), api 127.0.0.1:$API_PORT, code $CODE"
say "  gguf  $GGUF"
say "  log   $COORD_HOME/$CLOGBASE.{out,err}"

sleep 3

WLOGS=()
for w in "${WORKERS[@]}"; do
    wos=$(node_os "$w"); whome=$(node_home "$w")
    if [ "$wos" = windows ]; then
        WBIN="$whome/idletoken-worker.exe"; WENG="$whome/vendor/llama.cpp/build/bin/Release"
    else
        WBIN="$whome/idletoken-worker"; WENG="$whome/vendor/llama.cpp/build/bin"
    fi
    ip_var="IDLETOKEN_LAN_IP_${w}"; eval "wip=\${$ip_var:-}"
    WARGS="--rpc-supervisor --pair-code $CODE --engine-dir $WENG"
    [ -n "${wip:-}" ] && WARGS="$WARGS --rpc-host $wip"
    WARGS="$WARGS$WORKER_ARGS_EXTRA"
    WLOG="itc-$TAG-worker-$w"
    WPID=$(start_proc "$w" "$WBIN" "$WARGS" "$WLOG" "$WORKER_ENV")
    track_proc "$w" "$WPID" "$WLOG"
    WLOGS+=("$w:$whome/$WLOG")
    say "worker $WPID on $w ($wos), log $whome/$WLOG.{out,err}"
done

# ------------------------------------------------------------- readiness -----
# Readiness is the coordinator's OWN "idletoken-server ready" line: it polls
# /health for the literal {"status":"ok"}, so the 503-while-loading trap is
# handled there, once, instead of re-implemented here.
deadline=$(( $(date +%s) + READY_WAIT_S ))
STATE=""; REFUSAL=""
while [ "$(date +%s)" -lt "$deadline" ]; do
    clog=$(node_log "$COORD" "$COORD_HOME/$CLOGBASE")
    if printf '%s' "$clog" | grep -q "idletoken-server ready on"; then STATE=ready; break; fi
    REFUSAL=$(printf '%s' "$clog" | grep -E "refuse|refusing|dropping this worker|did not reach RPC_READY|attempt 5/5|FATAL" | head -1)
    [ -n "$REFUSAL" ] && { STATE=refused; break; }
    # A refusal can also land on the worker (overlay address, no PSK, version
    # mismatch), and those never reach the coordinator's log at all.
    for wl in "${WLOGS[@]}"; do
        wnode="${wl%%:*}"; wpath="${wl#*:}"
        REFUSAL=$(node_log "$wnode" "$wpath" | grep -E "JOIN_REFUSED|refusing|is an overlay address|refuses to" | head -1)
        [ -n "$REFUSAL" ] && { REFUSAL="$wnode: $REFUSAL"; break; }
    done
    [ -n "$REFUSAL" ] && { STATE=refused; break; }
    sleep 5
done

if [ "$EXPECT_REFUSE" = 1 ]; then
    # Red-control mode: a refusal IS the pass. Nothing is torn down silently --
    # the caller still gets the reason, because "it failed" and "it failed for
    # the reason I staged" are different claims.
    teardown >/dev/null 2>&1
    if [ "$STATE" = refused ]; then say "CLUSTER_REFUSED $REFUSAL"; exit 0; fi
    say "CLUSTER_FAIL: --expect-refuse, but the cluster came up ${STATE:-not at all} with no refusal — the negative control did not fire"
    exit 1
fi
[ "$STATE" = refused ] && fail "refused: $REFUSAL"
[ "$STATE" = ready ] || fail "engine not ready within ${READY_WAIT_S}s — coord log tail: $(node_log "$COORD" "$COORD_HOME/$CLOGBASE" | tail -3 | tr '\n' ' | ')"

# ------------------------------------------------------------- the facts -----
clog_all=$(node_log "$COORD" "$COORD_HOME/$CLOGBASE")
splits=$(printf '%s' "$clog_all" | sed -n 's/.*share=\([0-9.]*\).*/\1/p' | tr '\n' ',' | sed 's/,$//')
endpoints=$(printf '%s' "$clog_all" | sed -n 's/.*rpc worker [0-9]* ready: *\([^ ]*\) *\([^ ]*\).*/\1@\2/p' | tr '\n' ' ' | sed 's/ $//')
elog=$(printf '%s' "$clog_all" | sed -n 's/.*engine log: \(.*\)$/\1/p' | tail -1)
PIN=$(awk 'NR==1{print $2}' "$ROOT/scripts/llamacpp-patches/UPSTREAM")

WJSON=""
for wl in "${WLOGS[@]}"; do
    wnode="${wl%%:*}"; wpath="${wl#*:}"
    WJSON="$WJSON{\"node\":\"$wnode\",\"os\":\"$(node_os "$wnode")\",\"log\":\"$wpath\"},"
done
WJSON="${WJSON%,}"

cat > "$FACTS" <<JSON
{
  "tag": "$TAG",
  "coord": {"node": "$COORD", "os": "$COORD_OS", "home": "$COORD_HOME",
            "log": "$COORD_HOME/$CLOGBASE", "engine_log": "$elog"},
  "workers": [$WJSON],
  "api_base": "http://127.0.0.1:$API_PORT",
  "api_port": $API_PORT,
  "model": "$MODEL", "quant": "$QUANT", "gguf": "$GGUF", "ctx": $CTX,
  "pair_code": "$CODE",
  "engine_pin": "$PIN",
  "tensor_split": "$splits",
  "rpc_endpoints": "$endpoints",
  "allow_small_cluster": true,
  "rundir": "$RUNDIR"
}
JSON

if [ "$MODE" = serve ]; then
    say "CLUSTER_LLAMACPP_READY $(tr -d '\n' < "$FACTS" | tr -s ' ')"
    exit 0
fi

# ------------------------------------------------------- --check-ready -------
# The G5 shape: prove the cluster is REAL, then tear it down. "Real" is not
# "the port answers": a coordinator serving alone would answer just as well.
[ -n "$endpoints" ] || fail "no rpc worker endpoint in the coordinator log — the coordinator may be serving ALONE, which is not a cluster"
n_ep=$(printf '%s' "$endpoints" | wc -w | tr -d ' ')
[ "$n_ep" -ge "${#WORKERS[@]}" ] || fail "only $n_ep of ${#WORKERS[@]} workers reached RPC_READY"
case "$splits" in
    *,*) : ;;
    *) fail "tensor split '$splits' names fewer than 2 nodes — nothing was split" ;;
esac
teardown >/dev/null 2>&1
say "CLUSTER_READY"
