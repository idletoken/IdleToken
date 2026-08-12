#!/usr/bin/env bash
# Topology × model matrix (acceptance G-TOPO).
#
# The product claims "any handful of machines running the SAME OS, any supported
# model". Every other gate proves ONE topology with ONE model. This walks the
# claim: every non-empty subset of the real machines, times every model the
# registry says is runnable, plus simulated 5-10 node clusters that no amount of
# borrowed hardware would let us build.
#
# Since 2026-08-12 a cluster must be homogeneous (CLAUDE.md hard constraint #2),
# so subsets that span OS families are recorded SKIP with that reason — they are
# no longer a product configuration and the coordinator refuses them at HELLO.
# On the usual testbed (a Linux coordinator + Windows workers) that removes most
# of the multi-machine cells; what remains is the all-Windows household. Read
# the SKIP reasons before reading the pass count.
#
# Design notes that matter when you rerun it:
#   - RESUMABLE. Each cell appends to a TSV; a rerun skips cells already PASSed.
#     A full sweep runs for hours, so "start over from scratch" is not an option.
#   - Only the coordinator needs the weights on disk. Every other machine pulls
#     just its own layers from the coordinator's weight server, which is what
#     makes 15 subsets affordable at all.
#   - Simulated cells are LABELLED simulated. They prove the split/protocol/
#     roster hold at N=10; they do not prove ten real GPUs work (design
#     philosophy 13 — never let a simulation read as real hardware).
#
# Usage:
#   scripts/topology_matrix.sh                 # everything not yet passed
#   scripts/topology_matrix.sh --models qwen3.5-0.8b     # one model
#   scripts/topology_matrix.sh --max-nodes 2   # only 1- and 2-machine subsets
#   scripts/topology_matrix.sh --no-sim        # skip the simulated 5-10 cells
#   scripts/topology_matrix.sh --report        # print the matrix and exit
#
# Contract: last line TOPO_MATRIX_OK / TOPO_MATRIX_FAIL: <n> failing cells.
set -u

cd "$(dirname "$0")/.." || exit 1

COORD_NODE="${IDLETOKEN_COORD_NODE:-}"
# Where the repository is checked out on the coordinator. Configurable; the
# default is a conventional path that is **independent of any one machine**.
COORD_HOME="${IDLETOKEN_COORD_HOME:-~/work/IdleToken}"
RESULTS="${IDLETOKEN_TOPO_RESULTS:-$PWD/build/topology-matrix.tsv}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"

# Real machines. `home` is where the engine lives; `gguf` is that machine's
# local weight dir (only the coordinator's copy has to hold the model).
# The nodes and their per-machine paths come from scripts/testbed.env (not
# committed) -- this used to be yet another hardcoded username map, the fourth.
# shellcheck disable=SC1091
. "$(dirname "$0")/testbed-lib.sh"
read -r -a NODES <<< "${IDLETOKEN_TOPO_NODES:-${IDLETOKEN_COORD_NODE:-} ${IDLETOKEN_WORKER_NODES:-}}"
if [ -z "$COORD_NODE" ] || [ ${#NODES[@]} -eq 0 ] || [ -z "${NODES[0]}" ]; then
    echo "topology_matrix.sh: no nodes configured -- copy scripts/testbed.env.example to scripts/testbed.env and fill in" >&2
    echo "  IDLETOKEN_COORD_NODE / IDLETOKEN_WORKER_NODES (the values are aliases from your ~/.ssh/config)" >&2
    exit 2
fi
node_home()  {
    # The coordinator's checkout follows the POSIX convention; Windows machines
    # derive it from the user directory.
    if [ "$1" = "$COORD_NODE" ]; then echo "$COORD_HOME"; return 0; fi
    testbed_repo_home "$1"
}
# Weight directories to search, in order. A machine keeps different model
# families in different places (DSv4 lives beside the ds4 checkout, the small
# models in a qwen dir), so one fixed path per node silently skipped whole
# model rows.
node_gguf_dirs() { testbed_gguf_dirs "$1"; }

# First directory on <node> that actually holds <file>, or empty.
node_gguf_for() {  # node_gguf_for <node> <file>
    local d
    for d in $(node_gguf_dirs "$1"); do
        if is_win "$1"; then
            $SSH "$1" "cmd /c \"if exist ${d//\//\\}\\$2 (exit 0) else (exit 1)\"" >/dev/null 2>&1 && { echo "$d"; return 0; }
        else
            $SSH "$1" "test -f '$d/$2'" >/dev/null 2>&1 && { echo "$d"; return 0; }
        fi
    done
    return 1
}
# Whether a machine is Windows is answered by **asking the machine**, never
# guessed from its name.
#
# 2026-08-08: this used to read `[ "$1" != "linux-coord" ]` -- "anything not
# called linux-coord is Windows". That happened to hold for one maintainer's set
# of machines and is **wrong for every machine in anyone else's set**, while the
# entire premise of the product is "any number of machines". It is also load
# bearing: ten call sites use it to choose between cmd and sh syntax, taskkill and
# pkill -- and a wrong guess does not report "wrong guess", it reports a probe
# failure far from the cause.
#
# `uname -s` succeeds on Linux and macOS and fails under Windows OpenSSH (whose
# default shell is cmd or powershell), so that is the test. Results are memoized
# in a string (the bash 3.2 that ships with macOS has no associative arrays), so
# each machine is asked only once.
#
# node_os distinguishes all three families (is_win only ever needed Windows vs
# not; homogeneity needs Linux vs macOS too). It SETS a variable instead of
# echoing: a `$(...)` call would memoize inside a subshell and re-ssh every time.
#
# Caveat inherited from the original: an unreachable machine also produces no
# `uname` output and is therefore read as Windows. The sweep checks liveness
# (subset_offline) before it consults the family, so this never decides a cell.
_NODE_OS=" "
NODE_OS=""
node_os() {  # node_os <alias> -> sets NODE_OS to windows|linux|macos|unknown
    local rest u
    case "$_NODE_OS" in
        *" $1:"*) rest="${_NODE_OS#*" $1:"}"; NODE_OS="${rest%% *}"; return 0 ;;
    esac
    u=$($SSH "$1" "uname -s" 2>/dev/null | tr -d '\r')
    case "$u" in
        Linux)  NODE_OS=linux ;;
        Darwin) NODE_OS=macos ;;
        "")     NODE_OS=windows ;;
        *)      NODE_OS=unknown ;;
    esac
    _NODE_OS="$_NODE_OS$1:$NODE_OS "
}
is_win() { node_os "$1"; [ "$NODE_OS" = windows ]; }

MODELS=""
MAX_NODES=4
DO_SIM=1
REPORT_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --models)    shift; MODELS="${1:-}" ;;
        --max-nodes) shift; MAX_NODES="${1:-4}" ;;
        --no-sim)    DO_SIM=0 ;;
        --report)    REPORT_ONLY=1 ;;
        *) echo "topology_matrix.sh: unknown arg $1" >&2; exit 2 ;;
    esac
    shift
done

mkdir -p "$(dirname "$RESULTS")"
[ -f "$RESULTS" ] || printf 'status\tmodel\tsubset\tcoord\tkind\tdetail\n' > "$RESULTS"

report() {
    printf '\n%-6s %-20s %-34s %-10s %-9s %s\n' STATUS MODEL MACHINES COORD KIND DETAIL
    tail -n +2 "$RESULTS" | while IFS=$'\t' read -r st mo su co ki de; do
        printf '%-6s %-20s %-34s %-10s %-9s %s\n' "$st" "$mo" "$su" "$co" "$ki" "$de"
    done
    printf '\n  %s cells: %s pass, %s skip, %s fail\n' \
        "$(tail -n +2 "$RESULTS" | wc -l | tr -d ' ')" \
        "$(grep -c '^PASS' "$RESULTS" || true)" \
        "$(grep -c '^SKIP' "$RESULTS" || true)" \
        "$(grep -c '^FAIL' "$RESULTS" || true)"
}
[ "$REPORT_ONLY" = 1 ] && { report; exit 0; }

# Pair codes must come from the engine's alphabet (no 0/1/I/O — they are
# meant to be read aloud). A naive "TM%04d" produced codes with a zero in them
# and the coordinator rejected every one of them with "invalid join code".
CODE_ALPHABET=ABCDEFGHJKLMNPQRSTUVWXYZ23456789
mint_code() {
    local out="" i
    for i in 1 2 3 4 5 6; do
        out="$out${CODE_ALPHABET:$((RANDOM % 32)):1}"
    done
    echo "$out"
}

# HTTP against a node's own coordinator. The redirect has to match the node's
# shell: `2>/dev/null` is a bogus PATH to cmd, which then fails the whole
# command — the poller read nothing and every Windows-coordinator cell was
# recorded as "never reached ready" while the cluster was in fact serving.
node_get() {  # node_get <node> <url-path>
    if is_win "$1"; then
        $SSH "$1" "curl -s -m 5 \"http://127.0.0.1:8000$2\" 2>NUL" 2>/dev/null | tr -d '\r'
    else
        $SSH "$1" "curl -s -m 5 'http://127.0.0.1:8000$2' 2>/dev/null" 2>/dev/null | tr -d '\r'
    fi
}
# 160 tokens, not 12: reasoning models (Qwen3 family) open with a <think> block
# and spend the whole budget inside it — 12 tokens produced "Okay, the user is
# asking for the capital of" and the cell read as a wrong answer when the model
# was working perfectly. The budget has to outlast the preamble.
node_post_chat() {  # node_post_chat <node> <model> <timeout>
    local body='{"model":"'"$2"'","max_tokens":160,"messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}]}'
    if is_win "$1"; then
        # cmd needs the JSON in double quotes with inner quotes escaped as \".
        local esc; esc=$(printf '%s' "$body" | sed 's/"/\\"/g')
        $SSH "$1" "curl -s -m $3 http://127.0.0.1:8000/v1/messages -H \"content-type: application/json\" -d \"$esc\"" 2>/dev/null | tr -d '\r'
    else
        $SSH "$1" "curl -s -m $3 http://127.0.0.1:8000/v1/messages -H 'content-type: application/json' -d '$body'" 2>/dev/null | tr -d '\r'
    fi
}

# One row per (model, subset, coord): a re-run REPLACES the old verdict instead
# of appending. Append-only meant a harness bug's FAIL row stayed in the file
# forever and no later PASS could clear it — the audit gate would stay red for a
# cell that now works.
record() {
    local tmp; tmp=$(mktemp)
    awk -F'\t' -v m="$2" -v s="$3" -v c="$4" \
        '!(NR>1 && $2==m && $3==s && $4==c)' "$RESULTS" > "$tmp" && mv "$tmp" "$RESULTS"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" >> "$RESULTS"
}
done_already() {  # done_already <model> <subset> <coord>
    awk -F'\t' -v m="$1" -v s="$2" -v c="$3" \
        '$1=="PASS" && $2==m && $3==s && $4==c {found=1} END{exit !found}' "$RESULTS"
}

# --- which models to test ---------------------------------------------------
# The registry decides, not a hardcoded list: adding a manifest adds columns.
if [ -z "$MODELS" ]; then
    MODELS=$(python3 - <<'PY'
import glob, json
ids = []
for f in sorted(glob.glob("models/*.json")):
    m = json.load(open(f))
    if m.get("available"):
        ids.append(m["id"])
print(" ".join(ids))
PY
)
fi
echo "models under test: $MODELS"

# Resolve a model's GGUF filename (default precision) from its manifest.
gguf_name() {
    python3 - "models/$1.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
vs = m.get("variants") or []
dq = m.get("default_quant")
v = next((x for x in vs if x.get("quant") == dq), vs[0] if vs else None)
print((v or {}).get("gguf") or m.get("default_gguf", ""))
PY
}

# --- capacity pre-check via the advisor -------------------------------------
# A subset that genuinely cannot hold a model is an EXPECTED outcome, not a
# failure — recording it as FAIL would make the matrix cry wolf, and finding out
# by watching a 15-minute load fail is a waste. So ask the advisor first (the
# same code the planner uses, G-ADVISE), and record its "no" with its own
# number. If the advisor says yes and the cell then fails, that is a real bug.
probe_mem() {  # probe_mem <node> -> "vramGiB:ramGiB:unified"
    local cache="build/.probe-$1.txt"
    if [ ! -s "$cache" ]; then
        local js
        if is_win "$1"; then
            js=$($SSH "$1" "cd /d $(node_home "$1" | sed 's|/|\\|g') && idletoken-worker.exe --probe-json 2>NUL" 2>/dev/null | tr -d '\r' | tail -1)
        else
            js=$($SSH "$1" "cd $(node_home "$1") && ./idletoken-worker --probe-json 2>/dev/null | tail -1" 2>/dev/null | tr -d '\r')
        fi
        [ -n "${IDLETOKEN_TOPO_DEBUG:-}" ] && echo "   [debug] peers='$peers' json=${js:0:120}" >&2
    printf '%s' "$js" | python3 -c '
import json, sys
d = json.load(sys.stdin)
print("%.2f:%.2f:%d" % (d["vram_usable"] / 1073741824.0,
                        d["ram_usable"] / 1073741824.0,
                        1 if d.get("unified_memory") else 0))' > "$cache" 2>/dev/null || return 1
    fi
    cat "$cache"
}

# "gpu_only" | "hybrid" | "no" | "unavailable" for <model> on <coord + peers>.
advisor_verdict() {  # advisor_verdict <model> <coord> <members...>
    local model="$1" coord="$2"; shift 2
    local peers="" m
    for m in "$@"; do
        [ "$m" = "$coord" ] && continue
        local mm; mm=$(probe_mem "$m") || return 1
        peers="${peers:+$peers,}$mm"
    done
    local js
    if is_win "$coord"; then
        js=$($SSH "$coord" "cd /d $(node_home "$coord" | sed 's|/|\\|g') && idletoken-worker.exe --advise-json ${peers:+--advise-peers $peers} 2>NUL" 2>/dev/null | tr -d '\r' | tail -1)
    else
        js=$($SSH "$coord" "cd $(node_home "$coord") && ./idletoken-worker --advise-json ${peers:+--advise-peers $peers} 2>/dev/null | tail -1" 2>/dev/null | tr -d '\r')
    fi
    printf '%s' "$js" | python3 -c '
import json, sys
want = sys.argv[1]
d = json.load(sys.stdin)
rows = [m for m in d["models"] if m["id"] == want]
if not rows:
    print("unknown"); raise SystemExit
# The matrix runs the default precision, which is the first row for that model.
r = rows[0]
gb = r["shortfall_bytes"] / 1073741824.0
print("%s %.0f" % (r["mode"], gb))' "$model" 2>/dev/null
}

cleanup_all() {
    # `local` matters: bash scopes dynamically, so an undeclared loop variable
    # here would overwrite the caller's. This function is called from inside
    # run_cell, which uses `n` for the worker count — without `local _n` the
    # coordinator got started with `--num-workers win-c`.
    local _n
    $SSH "$COORD_NODE" "pkill -9 -f 'idletoken-coor[d]'; pkill -9 -f 'idletoken-worke[r]'; true" >/dev/null 2>&1
    for _n in "${NODES[@]}"; do
        is_win "$_n" && $SSH "$_n" "taskkill /IM idletoken-worker.exe /F 2>NUL & taskkill /IM idletoken-coord.exe /F 2>NUL & exit 0" >/dev/null 2>&1
    done
}
trap cleanup_all EXIT

# --- one cell ---------------------------------------------------------------
# Coordinator runs on $coord with the model on disk; the other machines join by
# pair code and fetch only their layers. Returns 0 on a correct reply.
run_cell() {  # run_cell <model> <coord> <all-nodes...>
    local model="$1" coord="$2"; shift 2
    local members=("$@")
    local n=${#members[@]}
    local file; file=$(gguf_name "$model")
    local coord_gguf; coord_gguf=$(node_gguf_for "$coord" "$file") || coord_gguf=""
    local code; code=$(mint_code)

    cleanup_all
    sleep 2

    # Coordinator side: weight server + coord + its own co-located worker.
    local chome; chome=$(node_home "$coord")
    # The coordinator is the one machine that must hold the weights (everyone
    # else pulls layer ranges from it). No copy anywhere -> honest SKIP.
    [ -n "$coord_gguf" ] || { echo "SKIPREASON:no $file on $coord"; return 2; }

    # The coordinator may itself be a Windows box (that is the whole point of
    # rotating the role): its shell is cmd, so every launch below has two forms.
    if is_win "$coord"; then
        local cwin="${chome//\//\\}" gwin="${coord_gguf//\//\\}"
        ssh -f -o BatchMode=yes "$coord" "cmd /c \"cd /d $cwin & idletoken-worker.exe --serve-weights $gwin\\$file --weights-port 8001 > tm-weights.log 2>&1\"" >/dev/null 2>&1
        sleep 2
        ssh -f -o BatchMode=yes "$coord" "cmd /c \"cd /d $cwin & idletoken-coord.exe --pair-code $code --num-workers $n --http --model-id $model --model-path $gwin\\$file --gguf-dir $gwin --n-predict 0 --ctx-size 8192 > tm-coord.log 2>&1\"" >/dev/null 2>&1
        sleep 3
        ssh -f -o BatchMode=yes "$coord" "cmd /c \"cd /d $cwin & idletoken-worker.exe --pair-code $code --model $gwin\\$file --gguf-dir $gwin --bind 0.0.0.0:14102 > tm-w0.log 2>&1\"" >/dev/null 2>&1
    else
        ssh -f -o BatchMode=yes "$coord" "cd $chome && exec ./idletoken-worker --serve-weights '$coord_gguf/$file' --weights-port 8001 > /tmp/tm-weights.log 2>&1" >/dev/null 2>&1
        sleep 2
        ssh -f -o BatchMode=yes "$coord" "cd $chome && exec ./idletoken-coord --pair-code $code --num-workers $n --http --model-id $model --model-path '$coord_gguf/$file' --gguf-dir '$coord_gguf' --n-predict 0 --ctx-size 8192 > /tmp/tm-coord.log 2>&1" >/dev/null 2>&1
        sleep 3
        ssh -f -o BatchMode=yes "$coord" "cd $chome && exec ./idletoken-worker --pair-code $code --model '$coord_gguf/$file' --gguf-dir '$coord_gguf' --bind 0.0.0.0:14102 > /tmp/tm-w0.log 2>&1" >/dev/null 2>&1
    fi

    # Every other machine: join by code, pull its layers from the coordinator.
    local lan
    if is_win "$coord"; then
        lan=$($SSH "$coord" "powershell -NoProfile -Command \"(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { \$_.IPAddress -like '192.168.*' } | Select-Object -First 1).IPAddress\"" 2>/dev/null | tr -d '\r ')
    else
        lan=$($SSH "$coord" "ip -4 -o addr show scope global | awk '{split(\$4,a,\"/\"); print a[1]}' | grep -E '^(192\.168\.|10\.)' | head -1" 2>/dev/null | tr -d '\r')
    fi
    for m in "${members[@]}"; do
        [ "$m" = "$coord" ] && continue
        local mhome; mhome=$(node_home "$m")
        local mgguf; mgguf=$(node_gguf_dirs "$m" | awk '{print $1}')
        if is_win "$m"; then
            # --shard-repo as a FLAG, not an env var. `set VAR=value & next` in cmd
            # swallows the space before `&` into the value (the worker then asked
            # for ".../model.gguf .idx" and got "bad index format"), and quoting
            # `set "VAR=value"` through bash -> ssh -> cmd is a losing game. The
            # flag has none of those problems.
            ssh -f -o BatchMode=yes "$m" "cmd /c \"cd /d ${mhome//\//\\} & idletoken-worker.exe --pair-code $code --shard-repo http://$lan:8001/$file --gguf-dir ${mgguf//\//\\} --bind 0.0.0.0:14323 > tm-worker.log 2>&1\"" >/dev/null 2>&1
        else
            ssh -f -o BatchMode=yes "$m" "cd $mhome && exec ./idletoken-worker --pair-code $code --shard-repo http://$lan:8001/$file --gguf-dir '$mgguf' --bind 0.0.0.0:14101 > /tmp/tm-worker.log 2>&1" >/dev/null 2>&1
        fi
    done

    # Wait for ready, then ask something with one right answer.
    local ready="" i
    for i in $(seq 1 "${IDLETOKEN_TOPO_TRIES:-80}"); do
        local st; st=$(node_get "$coord" /v1/cluster/status)
        case "$st" in *'"phase":"ready"'*) ready="$st"; break ;; esac
        sleep 10
    done
    [ -n "$ready" ] || { echo "FAILREASON:never reached ready"; return 1; }

    # Layer coverage: the plan must cover the model end to end, not "most of it".
    local covered
    covered=$(printf '%s' "$ready" | python3 -c '
import json, sys
d = json.load(sys.stdin)
ms = d.get("members") or []
if not ms:
    print("NOMEMBERS"); raise SystemExit
spans = sorted((m["layer_lo"], m["layer_hi"]) for m in ms)
gap = any(spans[i][1] != spans[i + 1][0] for i in range(len(spans) - 1))
print("%d-%d%s" % (spans[0][0], spans[-1][1], " GAP" if gap else ""))' 2>/dev/null)
    # An empty result means the extractor itself broke — treat that as a failure
    # rather than silently "no gap found" (a check that cannot fail is not a check).
    case "$covered" in
        ""|*NOMEMBERS*) echo "FAILREASON:could not read the layer plan from /v1/cluster/status"; return 1 ;;
        *GAP*)          echo "FAILREASON:layer plan has a gap ($covered)"; return 1 ;;
    esac

    local reply; reply=$(node_post_chat "$coord" "$model" 600)
    if printf '%s' "$reply" | grep -qi paris; then
        echo "OKDETAIL:layers $covered"
        return 0
    fi
    echo "FAILREASON:wrong reply: $(printf '%s' "$reply" | head -c 120)"
    return 1
}

# --- real subsets -----------------------------------------------------------
# Sample every machine's free memory once, while they are all idle. probe_mem
# caches to build/.probe-<node>.txt; sampling later (mid-sweep) would freeze a
# baseline taken while another cell held 80 GB.
rm -f build/.probe-*.txt 2>/dev/null
cleanup_all
sleep 3
declare -a OFFLINE=()
for _n in "${NODES[@]}"; do
    _b=$(probe_mem "$_n" || echo unreachable)
    printf 'baseline %-11s %s\n' "$_n" "$_b"
    [ "$_b" = unreachable ] && OFFLINE+=("$_n")
done
[ ${#OFFLINE[@]} -gt 0 ] && echo "note: offline this run: ${OFFLINE[*]} — their cells are SKIP, not FAIL"

# A powered-off machine is not a product failure. Every subset containing one
# used to burn the full ready-timeout and then record FAIL "never reached
# ready" — 44 such rows sat in the matrix for days, drowning the real
# failures and keeping the G_TOPO audit gate red for a reason no product
# change could fix.
subset_offline() {   # subset_offline <node...>  -> prints the offline member
    local _m _o
    for _m in "$@"; do
        for _o in ${OFFLINE[@]+"${OFFLINE[@]}"}; do
            [ "$_m" = "$_o" ] && { echo "$_m"; return 0; }
        done
    done
    return 1
}

# A subset spanning OS families is not a product configuration (CLAUDE.md hard
# constraint #2, 2026-08-12): the coordinator refuses it at HELLO. Recorded as
# SKIP with the reason rather than dropped, so the matrix cannot read as full
# coverage of something the product no longer claims.
subset_mixed_os() {  # subset_mixed_os <node...> -> prints "linux+windows" if mixed
    local _m fams="" f
    for _m in "$@"; do
        node_os "$_m"; f="$NODE_OS"
        case " $fams " in *" $f "*) ;; *) fams="$fams $f" ;; esac
    done
    set -- $fams
    [ $# -le 1 ] && return 1
    echo "$*" | tr ' ' '+'
    return 0
}

fails=0
for model in $MODELS; do
    for mask in $(seq 1 15); do
        subset=()
        for i in 0 1 2 3; do
            [ $(( (mask >> i) & 1 )) = 1 ] && subset+=("${NODES[$i]}")
        done
        [ ${#subset[@]} -le "$MAX_NODES" ] || continue
        # Coordinator: DGX when present (it holds every model), else the first
        # Windows machine in the subset — that is the all-Windows household.
        coord="${subset[0]}"
        for m in "${subset[@]}"; do [ "$m" = "$COORD_NODE" ] && coord="$COORD_NODE"; done
        local_key=$(IFS=+; echo "${subset[*]}")

        if done_already "$model" "$local_key" "$coord"; then
            echo "skip (already passed): $model on $local_key"
            continue
        fi
        if off=$(subset_offline "${subset[@]}"); then
            record SKIP "$model" "$local_key" "$coord" real "$off is powered off / unreachable this run"
            echo "   SKIP $off offline"
            continue
        fi
        if mixed=$(subset_mixed_os "${subset[@]}"); then
            record SKIP "$model" "$local_key" "$coord" real \
                "mixed-OS subset ($mixed): clusters must be homogeneous (CLAUDE.md #2) — refused at HELLO by design"
            echo "   SKIP mixed-OS ($mixed)"
            continue
        fi
        echo "== $model on $local_key (coord $coord) =="
        # Idle the machines BEFORE asking the advisor: it reports what is free
        # right now, so a leftover cluster from the previous cell makes every
        # verdict read "would not fit" (this produced a table claiming DGX alone
        # cannot hold DSv4, which the very first cell had already disproved).
        cleanup_all
        sleep 3
        verdict=$(advisor_verdict "$model" "$coord" "${subset[@]}")
        [ -n "${IDLETOKEN_TOPO_DEBUG:-}" ] && echo "   [debug] advisor verdict: '$verdict'"
        case "$verdict" in
            no\ *)
                record SKIP "$model" "$local_key" "$coord" real "advisor: would not fit, needs ${verdict#no } GB more"
                echo "   SKIP advisor says it would not fit (needs ${verdict#no } GB more)"
                continue ;;
            unavailable*)
                record SKIP "$model" "$local_key" "$coord" real "backend not implemented in this build"
                echo "   SKIP backend not implemented"; continue ;;
        esac
        out=$(run_cell "$model" "$coord" "${subset[@]}"); rc=$?
        case "$rc" in
            0) record PASS "$model" "$local_key" "$coord" real "${out#OKDETAIL:}"; echo "   PASS ${out#OKDETAIL:}" ;;
            2) record SKIP "$model" "$local_key" "$coord" real "${out#SKIPREASON:}"; echo "   SKIP ${out#SKIPREASON:}" ;;
            *) record FAIL "$model" "$local_key" "$coord" real "${out#FAILREASON:}"; echo "   FAIL ${out#FAILREASON:}"; fails=$((fails + 1)) ;;
        esac
    done
done

# --- simulated 5..10 --------------------------------------------------------
# Multiple worker processes on the coordinator machine. Real protocol, real
# split, real load — simulated only in that the GPUs are the same one.
if [ "$DO_SIM" = 1 ]; then
    sim_model="${IDLETOKEN_TOPO_SIM_MODEL:-qwen3.5-0.8b}"
    file=$(gguf_name "$sim_model")
    cg=$(node_gguf_for "$COORD_NODE" "$file") || cg=""
    for n in 5 6 7 8 9 10; do
        key="sim-${n}x"
        if done_already "$sim_model" "$key" "$COORD_NODE"; then
            echo "skip (already passed): $sim_model on $key"; continue
        fi
        echo "== $sim_model on $key (simulated) =="
        cleanup_all; sleep 2
        code=$(mint_code)
        ssh -f -o BatchMode=yes "$COORD_NODE" "cd $COORD_HOME && exec ./idletoken-coord --pair-code $code --num-workers $n --http --model-id $sim_model --model-path '$cg/$file' --gguf-dir '$cg' --n-predict 0 --ctx-size 8192 > /tmp/tm-sim-coord.log 2>&1" >/dev/null 2>&1
        sleep 3
        for w in $(seq 1 "$n"); do
            ssh -f -o BatchMode=yes "$COORD_NODE" "cd $COORD_HOME && exec ./idletoken-worker --pair-code $code --model '$cg/$file' --gguf-dir '$cg' --bind 0.0.0.0:$((14200 + w)) > /tmp/tm-sim-w$w.log 2>&1" >/dev/null 2>&1
            sleep 1
        done
        ready=""
        for i in $(seq 1 40); do
            st=$(node_get "$COORD_NODE" /v1/cluster/status)
            case "$st" in *'"phase":"ready"'*) ready="$st"; break ;; esac
            sleep 10
        done
        if [ -z "$ready" ]; then
            record FAIL "$sim_model" "$key" "$COORD_NODE" simulated "never reached ready"
            echo "   FAIL never reached ready"; fails=$((fails + 1)); continue
        fi
        reply=$(node_post_chat "$COORD_NODE" "$sim_model" 300)
        if printf '%s' "$reply" | grep -qi paris; then
            record PASS "$sim_model" "$key" "$COORD_NODE" simulated "$n workers, one machine"
            echo "   PASS ($n simulated workers)"
        else
            record FAIL "$sim_model" "$key" "$COORD_NODE" simulated "wrong reply"
            echo "   FAIL wrong reply"; fails=$((fails + 1))
        fi
    done
fi

cleanup_all
report
[ "$fails" = 0 ] && echo TOPO_MATRIX_OK || echo "TOPO_MATRIX_FAIL: $fails failing cells"
