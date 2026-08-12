#!/usr/bin/env bash
# IdleToken end-to-end acceptance ladder.
#
# Single source of truth for "are we done yet". Runs a staged ladder of gates
# from "infra works" up to "the real cluster serves correct DSv4-Flash inference
# over the OpenAI/Anthropic API". Each gate prints PASS/FAIL/SKIP; the first
# failing gate is the FRONTIER — the one thing to work on next. Exit code is 0
# only when G_FINAL (the cluster product goal, acceptance-criteria §8) passes;
# G_PLAT / G_SCHED (platform business layer) report in the ladder but are
# independent of the exit criterion (project spec decision 11: the platform layer
# is independent of the cluster's G-FINAL).
#
# Designed to be the oracle for an autonomous agent loop: run it, read FRONTIER,
# make that gate pass, re-run. No human babysitting.
#
# Usage:  scripts/acceptance.sh            # run all gates, report + frontier
#         scripts/acceptance.sh -v         # verbose (show check output)
#         scripts/acceptance.sh --gate G4  # run ONE gate (no earlier-gate skip)
#
# Node aliases resolve via ~/.ssh/config (see memory: real-cluster-topology).
# Off-LAN (Tailscale) runs can override the route without touching checks:
#   IDLETOKEN_COORD_NODE=DGX IDLETOKEN_API_HOST=100.97.254.63 scripts/acceptance.sh --gate G4
set -u

VERBOSE=0
ONLY_GATE=""
while [ $# -gt 0 ]; do
    case "$1" in
        -v) VERBOSE=1 ;;
        --gate) shift; ONLY_GATE="${1:-}" ;;
        *) echo "acceptance.sh: unknown arg $1" >&2; exit 2 ;;
    esac
    shift
done

# --- Testbed configuration ---------------------------------------------------
# Machine-specific facts (each Windows box's user directory) live in
# scripts/testbed.env, which is **not committed**: those are the account names of
# whoever lent us the machines -- third-party personal data that must not ship
# with an open-source repository. Template: testbed.env.example.
# shellcheck disable=SC1091
. "$(dirname "$0")/testbed-lib.sh"

# --- cluster topology ------------------------------------------------------
# Node aliases come **entirely from configuration**; this script has no defaults.
#
# 2026-08-08: this used to hardcode `${IDLETOKEN_COORD_NODE:-linux-coord}` and
# `${IDLETOKEN_WORKER_NODES:-win-a win-b}` -- the aliases of one maintainer's own
# machines. Two problems, the second the more serious:
#   1. The product's premise is "any number of machines", yet the names of one
#      specific set were pinned all over the repo.
#   2. The spec has long said that missing configuration must name the machine it
#      is missing and exit, **never falling back to another machine** -- which is
#      exactly how a gate ends up green having tested the wrong box. A hardcoded
#      default is that fallback: a stranger who runs this without a testbed.env
#      has the script ssh to aliases they do not have, and the failure points at
#      ssh rather than at "you have not configured any nodes".
COORD_NODE="${IDLETOKEN_COORD_NODE:-}"
read -r -a WORKER_NODES <<< "${IDLETOKEN_WORKER_NODES:-}"
# On bash 3.2 (what macOS ships) **expanding an empty array also trips `set -u`**
# (fixed only in 4.4), and `arr=()` does not help -- hence the `${arr[@]+...}`
# idiom at these three top-level sites. The other uses inside gate functions need
# no change: those gates only run when nodes are configured.
# Gates that need no machine at all: running only those must not force anyone to
# configure a testbed.
# 2026-08-08: the "exit when unconfigured" block above used to run
# unconditionally, so `--gate G_MODEL` -- the one the README explicitly calls
# purely local and hardware-free -- exited 2 without a testbed.env. CI could not
# run it, a stranger cloning the repo could not run it, and that README sentence
# was false. Requiring something presupposes actually using it.
LOCAL_ONLY_GATES=" G_MODEL G_SCHED "
needs_nodes=1
if [ -n "$ONLY_GATE" ] && [[ "$LOCAL_ONLY_GATES" == *" $ONLY_GATE "* ]]; then needs_nodes=0; fi
if [ "$needs_nodes" = 1 ] && { [ -z "$COORD_NODE" ] || [ ${#WORKER_NODES[@]} -eq 0 ]; }; then
    cat >&2 <<'NOCFG'
acceptance.sh: no nodes are configured.

  The ladder ssh's into your own machines, so it has to know which ones you have
  and what they are called. Create a local configuration from the template (the
  file is not committed):

      cp scripts/testbed.env.example scripts/testbed.env
      # Edit it and set at least IDLETOKEN_COORD_NODE and
      # IDLETOKEN_WORKER_NODES, whose values are aliases from your ~/.ssh/config.

  Or configure just this one run:

      IDLETOKEN_COORD_NODE=<alias> IDLETOKEN_WORKER_NODES="<alias> <alias>" scripts/acceptance.sh
NOCFG
    exit 2
fi
ALL_NODES=("$COORD_NODE" ${WORKER_NODES[@]+"${WORKER_NODES[@]}"})
# An override has to be stated loudly (principle 13, honest reporting): a run on
# fewer or different machines must never read like a complete one.
if [ -n "${IDLETOKEN_WORKERS_OVERRIDDEN:-}" ]; then
    echo "note: the worker node set was overridden to [${WORKER_NODES[*]+${WORKER_NODES[*]}}] (testbed.env names a different set)" >&2
fi

# There used to be a patch here that replaced the hardcoded coordinator alias in
# ALL_NODES with $COORD_NODE. Once the alias stopped being hardcoded it became
# unnecessary: ALL_NODES[0] simply is $COORD_NODE.
# That patch was treating the consequences of hardcoding -- a multi-node gate
# would iterate to the hardcoded alias, notice it was not COORD_NODE, and end up
# running Windows cmd syntax against a Linux box, reporting a
# "healthy-probe-nonzero" far removed from the cause (it fooled us once into
# believing G_HW had genuinely gone red on the DGX).
# Removing the hardcoding removes that entire class of symptom.

# Must MATCH the version ds4cuda.dll/ds4xcuda.dll were built against — v12.8
# (build_dist_win.bat says why: a 12.8 build runs on any r525+ driver, and
# 13.x runtime DLLs sitting beside a 12.8-built kernel DLL are a *silent* CPU
# fallback). This default used to be v13.3, i.e. the acceptance runner put on
# PATH exactly the combination our own packaging script warns against.
# Note the layout difference: 12.x keeps runtime DLLs in bin\, 13.x in bin\x64.
# Only the build node has the toolkit; deploy-only boxes load the DLLs bundled
# beside the exe.
WIN_CUDA_BIN="${IDLETOKEN_WIN_CUDA_BIN:-C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin}"

# The one Windows box with a build toolchain (MinGW + Rust + NSIS). The others
# are pure deploy machines. G1's rebuild subcheck and G_RELEASE both run here.
WIN_BUILD_NODE="${IDLETOKEN_WIN_BUILD_NODE:-}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# The repository checkout on each Windows box. The user directory comes from
# IDLETOKEN_PROFILE_<alias> in testbed.env, and the checkout is by convention
# <user directory>/IdleToken.
# This used to be a hardcoded case statement (one copy in each of three scripts),
# which both drifted and pinned the machine owners' account names into the repo.
# Falling back to another machine's path is exactly how a gate goes green having
# tested the wrong box -- better to leave it empty and let the startup check
# below say so plainly.
win_home() { testbed_repo_home "$1"; }

# Validate at startup: missing configuration must be reported **before any gate
# runs**, not surface inside some gate as a baffling ssh error.
for _n in ${WORKER_NODES[@]+"${WORKER_NODES[@]}"}; do
    if [ -z "$(win_home "$_n")" ]; then
        echo "acceptance.sh: a machine in the node set has no user directory configured." >&2
        testbed_hint "$_n"
        exit 2
    fi
done
WIN_HOME="$(win_home "$WIN_BUILD_NODE")"   # the build node's repository checkout
DGX_HOME='~/work/IdleToken'
# Coord API host: derive from ssh config so the ladder follows whatever address
# actually reaches the coordinator (LAN at home, Tailscale off-LAN). Hardcoding
# an IP here rotted once already (2026-07-22 DHCP reshuffle made .101 = win-a).
API_HOST="${IDLETOKEN_API_HOST:-$(ssh -G "$COORD_NODE" 2>/dev/null | awk '/^hostname /{print $2; exit}')}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${IDLETOKEN_API_PORT:-8000}"
# Real DSv4-Flash Q2 GGUF on the coordinator node (P6 real-reply gate). Path is
# resolved on the remote (DGX) shell, so $HOME expands there.
GGUF_DGX="${IDLETOKEN_GGUF:-\$HOME/work/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf}"

SSH="ssh -o BatchMode=yes -o ConnectTimeout=8"
# Fire-and-forget variant. Backgrounding *inside* the remote shell does not
# work here: the remote bash sits in do_wait for the job it just spawned and the
# ssh channel never closes, so the caller hangs forever (`nohup`, `setsid`,
# `</dev/null`, `disown` — all tried, all hang). `ssh -f` backgrounds the local
# client after auth instead, which does return.
SSHF="ssh -f -o BatchMode=yes -o ConnectTimeout=8"

# --- reporting -------------------------------------------------------------
FRONTIER=""
declare -a RESULTS
pass() { RESULTS+=("PASS $1"); printf '  \033[32m[PASS]\033[0m %s\n' "$1"; }
fail() { RESULTS+=("FAIL $1"); printf '  \033[31m[FAIL]\033[0m %s — %s\n' "$1" "$2";
         [ -z "$FRONTIER" ] && FRONTIER="$1"; }
# skip [reason] — default reason is "blocked by the frontier"; gates whose
# DEPENDENCIES are missing (not failing) pass an explicit reason instead of
# faking a pass (design principle 13: report honestly).
skip() { RESULTS+=("SKIP $1"); printf '  \033[33m[SKIP]\033[0m %s (%s)\n' "$1" "${2:-blocked by $FRONTIER}"; }
vlog() { [ "$VERBOSE" = 1 ] && printf '      | %s\n' "$1"; }

# Run a gate only if no earlier gate has failed; else auto-SKIP.
# With --gate G<n>, run exactly that gate (prefix match) and nothing else.
gate() {  # gate <name> <function>
    local name="$1" fn="$2"
    if [ -n "$ONLY_GATE" ]; then
        case "$name" in
            "$ONLY_GATE"|"$ONLY_GATE"_*) "$fn" "$name" ;;
        esac
        return
    fi
    if [ -n "$FRONTIER" ]; then skip "$name"; return; fi
    "$fn" "$name"
}

# Gates that do not depend on the cluster: they **run regardless of the FRONTIER**
# and **do not set one**.
#
# Why this exists: the ladder's "first failure becomes the FRONTIER, everything
# after is skipped" rule is for the product track, where later gates genuinely
# depend on earlier machines. Purely platform-side gates (which need only
# node_modules on the control machine and touch no cluster node) should not be
# governed by it. Observed: an unrelated laptop was powered off -> G0 red ->
# G_SCHED skipped as "blocked by G0_ssh_mesh". **That gate might as well not
# exist**: one machine fewer at home and nothing watches the scheduler platform.
#
# It also sets no FRONTIER: the platform layer is independent of the cluster's
# G-FINAL (spec decision 11), and its failure must not point the product track's
# "what to do next" at itself.
gate_local() {  # gate_local <name> <function>
    local name="$1" fn="$2"
    if [ -n "$ONLY_GATE" ]; then
        case "$name" in
            "$ONLY_GATE"|"$ONLY_GATE"_*) "$fn" "$name" ;;
        esac
        return
    fi
    local saved="$FRONTIER"
    "$fn" "$name"
    FRONTIER="$saved"
}

# Needs no cluster node, but IS a product invariant. Two differences from
# gate_local: it runs regardless of the FRONTIER (an offline laptop must not be
# able to hide it), and it DOES set one on failure — a violated hard constraint
# may never be written off as "remaining red is platform-layer only" by the §8
# exit contract. fail() only claims an empty FRONTIER, so an earlier product
# failure still keeps its place at the head of the ladder.
#
# Register these BEFORE G_FINAL: after it, a failure here would set the FRONTIER
# too late to stop `PASS G_FINAL` from exiting 0.
gate_always() {  # gate_always <name> <function>
    local name="$1" fn="$2"
    if [ -n "$ONLY_GATE" ]; then
        case "$name" in
            "$ONLY_GATE"|"$ONLY_GATE"_*) "$fn" "$name" ;;
        esac
        return
    fi
    "$fn" "$name"
}

# =====================================================================
# G0 — SSH mesh: control node can reach every cluster node
# =====================================================================
g0_ssh_mesh() {
    local name="$1" bad=""
    # Three retries per machine. **A single ssh is not a reliable liveness test**:
    # a busy Windows box (one that has just finished a load, or is being used by
    # another gate) stalls at "Connection timed out during banner exchange" -- TCP
    # connects but sshd never completes the handshake -- and recovers once it is
    # idle. Measured 2026-08-09: the same machine timed out on the first attempt
    # and returned its hostname immediately on the second.
    # Without retries the whole ladder dies at gate zero, and that looks like "the
    # machine went offline".
    # The backoff comes from **measured recovery times**, not from a guess: three
    # transient dropouts in one day on 2026-08-09 (the same set of Windows nodes,
    # two of them affected once each), each recovering within a minute or two. The
    # first version allowed 3x10s -- not enough, and the eighth round died here
    # anyway. The total window is about 110s.
    local n i h waits="5 15 30 60"
    for n in "${ALL_NODES[@]}"; do
        h=$($SSH "$n" hostname 2>/dev/null | tr -d '\r')
        if [ -z "$h" ]; then
            for i in $waits; do
                vlog "$n: no response, retrying in ${i}s"
                sleep "$i"
                h=$($SSH "$n" hostname 2>/dev/null | tr -d '\r')
                [ -n "$h" ] && break
            done
        fi
        if [ -z "$h" ]; then bad="$bad $n"; else vlog "$n -> $h"; fi
    done
    [ -z "$bad" ] && pass "$name" || fail "$name" "unreachable (after 3 retries):$bad"
}

# =====================================================================
# G1 — Build: worker + CUDA lib present and the exe runs (--help)
# =====================================================================
g1_build() {
    local name="$1"
    # --- rebuild from the CURRENT REPO TREE first -------------------------
    # Without this the gate only proves "a runnable file sits on the machine".
    # It stayed green for days while the Windows worker could not be built at
    # all (ds4_cuda.cu missing from the sync list + two exports missing from the
    # .def table): the checks below happily ran a three-week-old exe.
    # The repo is the source of truth, so push sources first — building whatever
    # the machine happens to hold is the same false green one level down.
    # Measured 22s on win-a.
    local bn="$WIN_BUILD_NODE" bhome before after errs
    if printf '%s\n' "${WORKER_NODES[@]}" | grep -qx "$bn"; then
        bhome=$(win_home "$bn")
        if ! bash "$REPO_ROOT/scripts/sync-to-win.sh" "$bn" >/dev/null 2>&1; then
            fail "$name" "sync-to-win.sh $bn failed (repo -> build node)"; return
        fi
        # A leftover engine process HOLDS the exe open, and Windows then fails
        # the link with `cannot open output file ... Permission denied` — which
        # leaves the OLD exe on disk, i.e. exactly the "it looks built but you
        # are running last week's binary" state this subcheck exists to catch.
        # The ladder's pre-run scrub only covers the coord node; the build node
        # needs its own. Safe here: G1 runs long before any cluster is started.
        # PowerShell, not `taskkill ... 2>NUL` — redirections in cmd over ssh
        # fail on these boxes (R-03).
        $SSH "$bn" "powershell -NoProfile -Command \"Get-Process idletoken-worker,idletoken-coord,idletoken-platform-agent -ErrorAction SilentlyContinue | Stop-Process -Force\"" >/dev/null 2>&1
        sleep 2
        before=$($SSH "$bn" "powershell -NoProfile -Command \"cd $bhome; if(Test-Path idletoken-worker.exe){(Get-Item idletoken-worker.exe).LastWriteTime.Ticks}else{0}\"" 2>/dev/null | tr -d '\r ')
        $SSH "$bn" "cd /d ${bhome//\//\\} && build_ds4x_win.bat" >/dev/null 2>&1
        after=$($SSH "$bn" "powershell -NoProfile -Command \"cd $bhome; if(Test-Path idletoken-worker.exe){(Get-Item idletoken-worker.exe).LastWriteTime.Ticks}else{0}\"" 2>/dev/null | tr -d '\r ')
        # `LINK_DONE` in the log is NOT a success marker — the script echoes it
        # unconditionally, so a failed link still writes it. A log that reads
        # like success is exactly how the broken build hid. Judge by the two
        # things that cannot lie: a newer exe, and no compiler errors.
        errs=$($SSH "$bn" "powershell -NoProfile -Command \"cd $bhome; (Select-String -Path ds4x_build.log -Pattern 'undefined reference','error:' -ErrorAction SilentlyContinue | Measure-Object).Count\"" 2>/dev/null | tr -d '\r ')
        # A failed query is not the same as no artifact. `${after:-0}` turns one
        # ssh hiccup into a 0 and reports "rebuild produced no newer exe", pointing
        # at a build log that is perfectly clean. A round was wasted this way on
        # 2026-08-09: the machine was dropping out intermittently and the exe was
        # in fact new. The two also call for different responses -- a broken build
        # needs a code change, a failed query needs a retry.
        if [ -z "$before" ] || [ -z "$after" ]; then
            fail "$name" "could not read the timestamp of idletoken-worker.exe on $bn (an ssh query failure, not a build failure) -- retry"; return
        fi
        if [ "$after" -le "$before" ]; then
            fail "$name" "$bn rebuild produced no newer idletoken-worker.exe (see $bhome/ds4x_build.log)"; return
        fi
        if [ "${errs:-1}" != "0" ]; then
            fail "$name" "$bn rebuild log has $errs error/undefined-reference line(s) (see $bhome/ds4x_build.log)"; return
        fi
        vlog "$bn rebuilt the worker from the synced tree (no link errors)"
    else
        vlog "rebuild subcheck skipped: build node $bn not in this run's node set"
    fi

    # Every Windows worker: native exe + CUDA dll + runnable. This used to check
    # win-a only (hardcoded), so a second Windows worker's binaries were never
    # build-checked at all — and the product's whole point is any N machines.
    local n home w help
    for n in "${WORKER_NODES[@]}"; do
        home=$(win_home "$n")
        w=$($SSH "$n" "powershell -NoProfile -Command \"cd $home; ''+(Test-Path idletoken-worker.exe)+(Test-Path ds4cuda.dll)\"" 2>/dev/null | tr -d '\r ')
        if [ "$w" != "TrueTrue" ]; then fail "$name" "$n missing idletoken-worker.exe/ds4cuda.dll (got '$w')"; return; fi
        help=$($SSH "$n" "powershell -NoProfile -Command \"cd $home; \$env:PATH='$WIN_CUDA_BIN;'+\$env:PATH; (& .\\idletoken-worker.exe --help 2>&1 | Select-Object -First 1)\"" 2>/dev/null | tr -d '\r')
        case "$help" in *idletoken-worker*) vlog "$n --help ok" ;; *) fail "$name" "$n idletoken-worker.exe --help did not run (got '$help')"; return ;; esac
    done
    # DGX: coord + worker binaries
    local d
    d=$($SSH "$COORD_NODE" "cd $DGX_HOME 2>/dev/null && ls idletoken-coord idletoken-worker 2>/dev/null | wc -l" 2>/dev/null | tr -d '\r ')
    [ "$d" = "2" ] || { fail "$name" "DGX missing idletoken-coord/idletoken-worker (found $d/2)"; return; }
    pass "$name"
}

# =====================================================================
# G2 — Probe: every node reports a real GPU (name, cc>0, vram>0) + ram + disk
# =====================================================================
g2_probe() {
    local name="$1" bad=""
    # DGX (Linux, unified memory: vram aliases ram)
    local dg; dg=$($SSH "$COORD_NODE" "cd $DGX_HOME && ./idletoken-worker --probe-only 2>&1" 2>/dev/null)
    echo "$dg" | grep -q "vram usable:" && echo "$dg" | grep -Eq "gpu:.*[A-Za-z]" || bad="$bad DGX"
    vlog "DGX probe: $(echo "$dg" | grep -E 'gpu:|vram usable' | tr '\n' ';')"
    # Windows workers: need GPU name + cc>0 + vram_total>0 (the current gap)
    for n in "${WORKER_NODES[@]}"; do
        local home; home=$(win_home "$n")
        local p; p=$($SSH "$n" "powershell -NoProfile -Command \"cd $home; \$env:PATH='$WIN_CUDA_BIN;'+\$env:PATH; (& .\\idletoken-worker.exe --probe-only 2>&1)\"" 2>/dev/null)
        # accept when gpu name non-empty AND cc not 0.0 AND vram total not 0 B
        if echo "$p" | grep -Eq "gpu:\s+\S+.*cc [1-9]" && ! echo "$p" | grep -q "vram total : 0 B"; then
            vlog "$n GPU probe ok"
        else
            bad="$bad $n"
            vlog "$n probe gpu line: $(echo "$p" | grep -E 'gpu:|vram total' | tr '\n' ';')"
        fi
    done
    [ -z "$bad" ] && pass "$name" || fail "$name" "GPU not reported by:$bad"
}

# =====================================================================
# G_TOPO — any-combination coverage. The sweep itself runs for hours
#          (scripts/topology_matrix.sh, resumable), so the ladder does not
#          re-run it: this gate AUDITS the recorded matrix. No matrix on disk
#          is an honest SKIP, not a pass.
# =====================================================================
TOPO_RESULTS="${IDLETOKEN_TOPO_RESULTS:-$PWD/build/topology-matrix.tsv}"

g_topo() {
    local name="$1"
    if [ ! -s "$TOPO_RESULTS" ] || [ "$(tail -n +2 "$TOPO_RESULTS" | wc -l | tr -d ' ')" = 0 ]; then
        skip "$name" "no topology matrix recorded — run scripts/topology_matrix.sh (hours; resumable)"
        return
    fi
    local nfail; nfail=$(awk -F'\t' '$1=="FAIL"' "$TOPO_RESULTS" | wc -l | tr -d ' ')
    if [ "$nfail" != 0 ]; then
        fail "$name" "$nfail failing cell(s): $(awk -F'\t' '$1=="FAIL" {print $2"@"$3}' "$TOPO_RESULTS" | tr '\n' ' ')"
        return
    fi
    # Coverage the matrix must actually demonstrate, not just "nothing failed":
    #   - every model the registry calls runnable passed somewhere
    #   - at least one multi-machine cell (single machines prove no topology)
    #   - at least one WINDOWS coordinator (the all-Windows household)
    local missing=""
    for m in $(python3 -c '
import glob, json
print(" ".join(json.load(open(f))["id"] for f in sorted(glob.glob("models/*.json")) if json.load(open(f)).get("available")))'); do
        awk -F'\t' -v m="$m" '$1=="PASS" && $2==m {f=1} END{exit !f}' "$TOPO_RESULTS" || missing="$missing $m"
    done
    [ -z "$missing" ] || { fail "$name" "no passing cell for:$missing"; return; }
    awk -F'\t' '$1=="PASS" && $5=="real" && $3 ~ /\+/ {f=1} END{exit !f}' "$TOPO_RESULTS" \
        || { fail "$name" "no multi-machine cell passed (only single-machine clusters proven)"; return; }
    awk -F'\t' '$1=="PASS" && $4 ~ /^win/ {f=1} END{exit !f}' "$TOPO_RESULTS" \
        || { fail "$name" "no Windows machine has served as coordinator"; return; }
    local npass nskip
    npass=$(awk -F'\t' '$1=="PASS"' "$TOPO_RESULTS" | wc -l | tr -d ' ')
    nskip=$(awk -F'\t' '$1=="SKIP"' "$TOPO_RESULTS" | wc -l | tr -d ' ')
    vlog "matrix: $npass pass, $nskip skip, 0 fail"
    vlog "simulated cells: $(awk -F'\t' '$1=="PASS" && $5=="simulated" {c++} END{print c+0}' "$TOPO_RESULTS") (labelled, not real hardware)"
    pass "$name"
}

# =====================================================================
# G_FETCH — weight acquisition on a machine with no HuggingFace route:
#           mirror fallback, resume, verification, actionable failures.
#           Uses the smallest model so the gate stays cheap; the cached copy
#           makes reruns instant, and the resume check re-fetches ~1 MiB.
# =====================================================================
FETCH_DIR="${IDLETOKEN_FETCH_DIR:-/tmp/idletoken-fetchgate}"
FETCH_MODEL="${IDLETOKEN_FETCH_MODEL:-qwen3.5-0.8b}"
FETCH_QUANT="${IDLETOKEN_FETCH_QUANT:-Q4_K_M}"

g_fetch() {
    local name="$1"
    local out
    out=$($SSH "$COORD_NODE" "cd $DGX_HOME && bash scripts/model_fetch.sh $FETCH_MODEL $FETCH_QUANT $FETCH_DIR 2>&1" 2>/dev/null | tr -d '\r')
    case "$out" in
        *MODEL_FETCH_OK*) ;;
        *) fail "$name" "fetch failed: $(echo "$out" | tail -1)"; return ;;
    esac
    # Which route did it take? On a machine that cannot reach HF the mirror
    # path is the ONLY way this passes, which is exactly the case under test.
    vlog "$(echo "$out" | grep -E '^endpoint:' | head -1)"
    echo "$out" | grep -q "GGUF verified" || { fail "$name" "downloaded file was not verified as GGUF"; return; }

    # Resume: lop off the last MiB and confirm it continues instead of
    # restarting (these files run to 80 GB — restarting is not an option).
    local f; f=$(echo "$out" | grep -E '^saved:' | awk '{print $2}')
    [ -n "$f" ] || { fail "$name" "fetcher did not report the saved path"; return; }
    local r
    r=$($SSH "$COORD_NODE" "sz=\$(stat -c %s '$f'); truncate -s \$((sz - 1048576)) '$f'; cd $DGX_HOME && bash scripts/model_fetch.sh $FETCH_MODEL $FETCH_QUANT $FETCH_DIR 2>&1" 2>/dev/null | tr -d '\r')
    echo "$r" | grep -q "resuming at" || { fail "$name" "did not resume a partial download"; return; }
    echo "$r" | grep -q "MODEL_FETCH_OK" || { fail "$name" "resume did not complete"; return; }
    vlog "$(echo "$r" | grep -E 'resuming at' | head -1)"

    # Failures must be actionable, not a stack trace or a silent zero-byte file.
    local bad
    bad=$($SSH "$COORD_NODE" "cd $DGX_HOME && bash scripts/model_fetch.sh definitely-not-a-model 2>&1 | tail -1" 2>/dev/null | tr -d '\r')
    case "$bad" in
        *MODEL_FETCH_FAIL*definitely-not-a-model*) ;;
        *) fail "$name" "unknown-model failure was not explained (got '$bad')"; return ;;
    esac
    local unreach
    unreach=$($SSH "$COORD_NODE" "cd $DGX_HOME && HF_ENDPOINT=https://idletoken-no-such-host.invalid bash scripts/model_fetch.sh $FETCH_MODEL $FETCH_QUANT $FETCH_DIR 2>&1 | tail -1" 2>/dev/null | tr -d '\r')
    case "$unreach" in
        *"no reachable endpoint"*) ;;
        *) fail "$name" "unreachable-endpoint failure was not explained (got '$unreach')"; return ;;
    esac
    pass "$name"
}

# =====================================================================
# G_ADVISE — capability table: every machine can answer "what can I run?"
#            in terms a non-expert can act on, and the answer is the
#            PLANNER's (drift between the two would be a table that lies).
# =====================================================================
g_advise() {
    local name="$1" bad=""
    for n in "${ALL_NODES[@]}"; do
        local txt js
        if [ "$n" = "$COORD_NODE" ]; then
            txt=$($SSH "$n" "cd $DGX_HOME && ./idletoken-worker --advise 2>/dev/null" 2>/dev/null | tr -d '\r')
            js=$($SSH "$n" "cd $DGX_HOME && ./idletoken-worker --advise-json 2>/dev/null | tail -1" 2>/dev/null | tr -d '\r')
        else
            local home; home=$(win_home "$n")
            txt=$($SSH "$n" "cd /d ${home//\//\\} && idletoken-worker.exe --advise 2>NUL" 2>/dev/null | tr -d '\r')
            js=$($SSH "$n" "cd /d ${home//\//\\} && idletoken-worker.exe --advise-json 2>NUL" 2>/dev/null | tr -d '\r' | tail -1)
        fi
        # The table must exist, name real models, and mark at least one runnable.
        echo "$txt" | grep -q "capability report" || { bad="$bad $n(no-table)"; continue; }
        echo "$txt" | grep -q "yes (GPU"           || { bad="$bad $n(nothing-runnable)"; continue; }
        # Un-runnable rows must say how much memory is missing, not just "no".
        if echo "$txt" | grep -qE "^  \S+ +\S+ +\S+ +no  "; then
            echo "$txt" | grep -q "more memory" || bad="$bad $n(no-shortfall-hint)"
        fi
        # JSON for the client: parseable, non-empty models[], modes from the
        # known set (a typo here silently greys out the whole model picker).
        local jv
        jv=$(printf '%s' "$js" | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception as e: print("bad-json"); raise SystemExit
ms=d.get("models") or []
if not ms: print("empty-models"); raise SystemExit
allowed={"gpu_only","hybrid","no","unavailable"}
bad=[m for m in ms if m.get("mode") not in allowed]
print("ok" if not bad else "bad-mode:"+str(bad[:1]))' 2>/dev/null)
        [ "$jv" = "ok" ] || bad="$bad $n(json:${jv:-none})"
        vlog "$n: $(echo "$txt" | grep -c 'yes (GPU') runnable rows"
    done
    [ -z "$bad" ] && pass "$name" || fail "$name" "capability table wrong on:$bad"
}

# =====================================================================
# G_HW — hardware floor: a machine below it must be REFUSED with a sentence
#        that says what is required, not limp along and emit garbage.
#        Exercises the three refusal paths through the probe's test hooks so
#        the gate does not need a pre-Turing card lying around.
# =====================================================================

# Run `idletoken-worker --probe-only` on a node with extra env, echo "rc=<n>" plus
# the output. Windows nodes are cmd, Linux is sh — same contract either way.
hw_probe() {  # hw_probe <node> [VAR=VAL ...]
    local node="$1"; shift
    if [ "$node" = "$COORD_NODE" ]; then
        $SSH "$node" "cd $DGX_HOME && env $* ./idletoken-worker --probe-only 2>&1; echo rc=\$?" 2>/dev/null | tr -d '\r'
    else
        local home; home=$(win_home "$node")
        local sets=""
        for kv in "$@"; do sets="set $kv & $sets"; done
        # cmd expands %errorlevel% at PARSE time, so a plain `& echo rc=%errorlevel%`
        # reports the value from before the exe ran (it read rc=0 for a refusal
        # and quietly turned this gate green). `cmd /v:on` + !errorlevel! reads it
        # at execution time.
        $SSH "$node" "cmd /v:on /c \"cd /d ${home//\//\\} & ${sets}idletoken-worker.exe --probe-only 2>&1 & echo rc=!errorlevel!\"" 2>/dev/null | tr -d '\r'
    fi
}

g_hw() {
    local name="$1" bad=""
    for n in "${ALL_NODES[@]}"; do
        # 1) healthy machine: must pass AND report a driver version (the field
        #    that tells us whether the shipped CUDA can load at all).
        local ok; ok=$(hw_probe "$n")
        case "$ok" in
            *"rc=0"*) ;;
            *) bad="$bad $n(healthy-probe-nonzero)"; continue ;;
        esac
        echo "$ok" | grep -Eq "driver [0-9]+\.[0-9]+" || { bad="$bad $n(no-driver-version)"; continue; }
        vlog "$n: $(echo "$ok" | grep -E '^  gpu:' | head -1 | sed 's/^ *//')"

        # 2) no driver at all -> refuse, and say a driver is needed.
        local nd; nd=$(hw_probe "$n" "IDLETOKEN_FORCE_NO_NVML=1")
        case "$nd" in *"rc=2"*) ;; *) bad="$bad $n(no-driver-not-refused)"; continue ;; esac
        echo "$nd" | grep -qi "driver" || bad="$bad $n(no-driver-msg-unclear)"

        # 3) pre-Turing card -> refuse, and name the requirement.
        local lc; lc=$(hw_probe "$n" "IDLETOKEN_FAKE_CC=6.1")
        case "$lc" in *"rc=2"*) ;; *) bad="$bad $n(old-card-not-refused)"; continue ;; esac
        echo "$lc" | grep -q "7.5" || bad="$bad $n(old-card-msg-unclear)"

        # 4) driver older than the shipped CUDA -> refuse.
        local od; od=$(hw_probe "$n" "IDLETOKEN_FAKE_DRIVER=470.00")
        case "$od" in *"rc=2"*) ;; *) bad="$bad $n(old-driver-not-refused)" ;; esac
    done
    [ -z "$bad" ] && pass "$name" || fail "$name" "hardware floor not enforced:$bad"
}

# =====================================================================
# G3 — Packaging (E3): a self-contained dist/ runs with ONLY the driver.
#      Two halves, both required:
#      (1) coord node: scripts/package_dist.sh rebuilds dist/ (engine binaries
#          that exist + run scripts + MANIFEST.txt) and self-checks each
#          packaged binary via --help — last line DIST_OK. DIST_SKIP there
#          (binaries not built) is a FAIL here: G1 already proved they exist,
#          so a skip on the coord node means something is actually broken.
#      (2) win-a: the driver-only bundle (exe + ds4cuda/cudart/cublas DLLs,
#          built by scripts/build_link.bat) runs --probe-only WITHOUT the CUDA
#          toolkit on PATH.
# =====================================================================
g3_package() {
    local name="$1"
    # (1) coord-node dist: build + manifest + --help self-check, honest verdict.
    local pk
    pk=$($SSH "$COORD_NODE" "cd $DGX_HOME && bash scripts/package_dist.sh 2>&1 | tail -1" 2>/dev/null | tr -d '\r')
    case "$pk" in
        DIST_OK)    vlog "coord-node dist/ rebuilt, self-checked, manifest written" ;;
        DIST_SKIP*) fail "$name" "package_dist.sh skipped on $COORD_NODE (engine binaries missing there?)"; return ;;
        *)          fail "$name" "coord-node packaging failed (got '$pk'; run scripts/package_dist.sh on $COORD_NODE)"; return ;;
    esac
    # Ground truth double-check: the manifest file really exists and is non-empty.
    local mf
    mf=$($SSH "$COORD_NODE" "test -s $DGX_HOME/dist/MANIFEST.txt && echo yes || echo no" 2>/dev/null | tr -d '\r')
    [ "$mf" = "yes" ] || { fail "$name" "dist/MANIFEST.txt missing/empty on $COORD_NODE despite DIST_OK"; return; }

    # (2) The Windows half: **repackage first, then assert** -- symmetrical with
    # the Linux half, where package_dist.sh rebuilds. This half used to inspect
    # merely "whatever happened to be sitting in dist\", which is exactly the
    # "certifies the wrong artifact" its own comment warned about. On top of that,
    # G1 now rebuilds the engine every round, so without repackaging the freshness
    # assertion below is guaranteed to fail and the ladder could never be green in
    # a single run.
    local pkw
    pkw=$($SSH "$WIN_BUILD_NODE" "cd /d ${WIN_HOME//\//\\} && scripts\\build_dist_win.bat" 2>/dev/null | tr -d '\r' | tail -1)
    case "$pkw" in
        *DIST_FAIL*) fail "$name" "repackaging failed on $WIN_BUILD_NODE: $pkw"; return ;;
        *) vlog "$WIN_BUILD_NODE dist\ rebuilt ($pkw)" ;;
    esac

    # (3) Windows driver-only probe (no CUDA bin on PATH).
    local out; out=$($SSH "$WIN_BUILD_NODE" "powershell -NoProfile -Command \"cd $WIN_HOME/dist 2>&1; if(Test-Path idletoken-worker.exe){ (& .\\idletoken-worker.exe --probe-only 2>&1 | Select-Object -First 1) } else { 'NO_DIST' }\"" 2>/dev/null | tr -d '\r')
    case "$out" in
        *IdleToken*|*resource*) vlog "$WIN_BUILD_NODE dist ran --probe-only driver-only" ;;
        NO_DIST) fail "$name" "no dist/ bundle on $WIN_BUILD_NODE (need self-contained folder + DLLs)"; return ;;
        *) fail "$name" "$WIN_BUILD_NODE dist bundle did not run driver-only (got '$out')"; return ;;
    esac

    # (3) The bundle must be THE CURRENT BUILD, not whatever was copied in once.
    # On 2026-08-04 dist\idletoken-worker.exe turned out to be from 07-16: no ds4x
    # at all, so the shipped bundle could not run one small model — and this
    # gate was green throughout, because a three-week-old binary answers
    # "--probe-only" just as well as today's. A packaging gate that only asks
    # "does it start" certifies the wrong artifact.
    local fresh
    fresh=$($SSH "$WIN_BUILD_NODE" "powershell -NoProfile -Command \"cd $WIN_HOME; \
        \$r=(Get-Item idletoken-worker.exe).LastWriteTime; \
        \$d=(Get-Item dist/idletoken-worker.exe).LastWriteTime; \
        if(\$d -ge \$r){'FRESH'}else{'STALE ' + \$d.ToString('yyyy-MM-dd') + ' vs repo ' + \$r.ToString('yyyy-MM-dd')}\"" 2>/dev/null | tr -d '\r')
    case "$fresh" in
        FRESH) vlog "dist/ binary is not older than the built one" ;;
        *) fail "$name" "dist/ is stale — rerun scripts/build_dist_win.bat ($fresh)"; return ;;
    esac

    # (4) …and it must carry the CURRENT feature set. Freshness by timestamp
    # alone would still pass a bundle built from a stale build script, so ask
    # the binary a question only a current build can answer.
    local caps
    caps=$($SSH "$WIN_BUILD_NODE" "powershell -NoProfile -Command \"cd $WIN_HOME/dist; \
        (& .\\idletoken-worker.exe --advise-json 2>&1 | Select-Object -First 1)\"" 2>/dev/null | tr -d '\r')
    case "$caps" in
        *'"models"'*|*'"hw_status"'*) vlog "dist binary answers --advise-json (multi-model build)" ; pass "$name" ;;
        *) fail "$name" "dist binary has no --advise-json — it predates multi-model (got '${caps:0:80}')" ;;
    esac
}

# =====================================================================
# E3b: the current tree can still PRODUCE the installer.
#
# G3 checks the dist\ FOLDER; install_walkthrough.sh checks an installer that
# already exists. Nobody checked the step between them, and on 2026-08-04 that
# step was broken while every gate stayed green: the installer on the build node
# predated the IdleToken rename by two hours, and rebuilding it died on a path
# that no longer existed (Cargo/Tauri bake absolute paths into target\). E1 only
# asks whether an exe exists; the P gates drive a debug client on the coord node.
# Neither builds what we ship.
g_release() {
    local name="$1"
    local bn="$WIN_BUILD_NODE" bhome nsis
    if ! printf '%s\n' "${WORKER_NODES[@]}" | grep -qx "$bn"; then
        skip "$name" "build node $bn is not in this run's node set"; return
    fi
    bhome=$(win_home "$bn")
    nsis="$bhome/client/src-tauri/target/release/bundle/nsis"
    local vite="$REPO_ROOT/client/node_modules/.bin/vite"
    [ -x "$vite" ] || { skip "$name" "no frontend toolchain on the control machine (client/node_modules is missing)"; return; }

    # The build node has no node/pnpm, so the frontend has to arrive prebuilt.
    # Release mode = the platform address real users get.
    if ! ( cd "$REPO_ROOT/client" && ./node_modules/.bin/tsc && "$vite" build --mode release ) >/dev/null 2>&1; then
        fail "$name" "frontend release build failed (client/ on the control machine)"; return
    fi
    local tgz=/tmp/idletoken-dist-sync.tar.gz
    if ! ( cd "$REPO_ROOT/client" && COPYFILE_DISABLE=1 tar czf "$tgz" --exclude='._*' dist ); then
        fail "$name" "failed to package client/dist"; return
    fi
    scp -q "$tgz" "$bn:$bhome/client/dist-sync.tar.gz" 2>/dev/null || { rm -f "$tgz"; fail "$name" "failed to push client/dist to $bn"; return; }
    $SSH "$bn" "cd /d ${bhome//\//\\}\\client && tar xzf dist-sync.tar.gz && del /q dist-sync.tar.gz" >/dev/null 2>&1
    rm -f "$tgz"
    # Put the control machine's dist back to a plain build immediately. A left
    # -behind release dist points at the production platform, and P2_auth then
    # silently stops testing the offline local identity it is supposed to test
    # — with nothing in the diff to show why (2026-07-29).
    ( cd "$REPO_ROOT/client" && "$vite" build ) >/dev/null 2>&1

    local before after out
    before=$($SSH "$bn" "powershell -NoProfile -Command \"if(Test-Path '$nsis'){(Get-ChildItem '$nsis' -Filter *.exe | Sort-Object LastWriteTime | Select-Object -Last 1).LastWriteTime.Ticks}else{0}\"" 2>/dev/null | tr -d '\r ')
    out=$($SSH "$bn" "cd /d ${bhome//\//\\} && scripts\\build_client_release.bat" 2>/dev/null | tr -d '\r' | tail -1)
    case "$out" in
        CLIENT_RELEASE_OK) ;;
        *) fail "$name" "$bn could not produce an installer (last line '$out'; see the tauri output under client\\src-tauri)"; return ;;
    esac
    after=$($SSH "$bn" "powershell -NoProfile -Command \"if(Test-Path '$nsis'){(Get-ChildItem '$nsis' -Filter *.exe | Sort-Object LastWriteTime | Select-Object -Last 1).LastWriteTime.Ticks}else{0}\"" 2>/dev/null | tr -d '\r ')
    # A green last line alone would let the PREVIOUS installer stand in for this
    # one — the same way dist\ certified a three-week-old binary (see G3 (3)).
    if [ "${after:-0}" -le "${before:-0}" ]; then
        fail "$name" "CLIENT_RELEASE_OK but the installer timestamp did not advance -- this certifies the previous artifact"; return
    fi
    vlog "$bn rebuilt the installer from the current tree (artifact timestamp refreshed)"
    pass "$name"
}

# =====================================================================
# G4 — Single-node inference: DGX coord+worker load GGUF, API yields a token
# =====================================================================
g4_single_infer() {
    local name="$1"
    # Placeholder oracle: a helper on DGX brings up a 1-node cluster and does a
    # real decode step, printing SINGLE_INFER_OK on success. Not yet wired.
    local out; out=$($SSH "$COORD_NODE" "cd $DGX_HOME && test -x scripts/run_single_infer.sh && ./scripts/run_single_infer.sh 2>&1 | tail -1 || echo NO_HELPER" 2>/dev/null | tr -d '\r')
    [ "$out" = "SINGLE_INFER_OK" ] && pass "$name" || fail "$name" "single-node inference not proven (got '$out')"
}

# =====================================================================
# G5 — Cluster ready: coord + >=1 real worker handshake -> cluster_ready
# =====================================================================
g5_cluster_ready() {
    local name="$1"
    local out; out=$($SSH "$COORD_NODE" "cd $DGX_HOME && test -x scripts/run_cluster.sh && ./scripts/run_cluster.sh --check-ready 2>&1 | tail -1 || echo NO_HELPER" 2>/dev/null | tr -d '\r')
    # Tear down what this gate started. A readiness check has no reason to keep
    # 80 GB of weights resident, and leaving them there is how G6 failed with an
    # empty error: it went on to start a SECOND cluster, and two copies of DSv4
    # do not fit in 119 GB — something got OOM-killed and the gate reported
    # `could not start cluster ()` with nothing to go on.
    $SSH "$COORD_NODE" "cd $DGX_HOME && ./scripts/run_cluster.sh --stop" >/dev/null 2>&1
    [ "$out" = "CLUSTER_READY" ] && pass "$name" || fail "$name" "cluster did not reach ready (got '$out')"
}

# =====================================================================
# G6 — End to end: real API call returns a coherent DSv4 response (GOAL)
# =====================================================================
g6_e2e() {
    local name="$1" started=0
    # Self-sufficient: if no API is live, bring the cluster up on the coord
    # node via run_cluster.sh --serve (and tear it down after the checks).
    if ! curl -s -m 3 "http://$API_HOST:$API_PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
        vlog "no live API; starting cluster via run_cluster.sh --serve"
        # Free the memory first. run_cluster.sh's own leftover-cleanup only
        # knows processes it has pid files for; anything another gate left
        # behind still holds its 80 GB, and the start then dies on memory with
        # no message at all.
        $SSH "$COORD_NODE" "cd $DGX_HOME && ./scripts/run_cluster.sh --stop; pkill -x idletoken-coord; pkill -x idletoken-worker; true" >/dev/null 2>&1
        local up
        up=$($SSH "$COORD_NODE" "cd $DGX_HOME && bash scripts/run_cluster.sh --serve 2>&1 | tail -1" 2>/dev/null | tr -d '\r')
        case "$up" in
            API_READY*) started=1; vlog "$up" ;;
            # Empty means the remote command produced nothing — usually the box
            # ran out of memory mid-load. Say so instead of printing "()".
            "") fail "$name" "run_cluster.sh --serve produced NO output (OOM during load? check free memory on $COORD_NODE)"; return ;;
            *) fail "$name" "could not start cluster ($up)"; return ;;
        esac
    fi

    # Anthropic shape — 200 + non-empty coherent text + non-zero usage.
    local a; a=$(curl -s -m 120 "http://$API_HOST:$API_PORT/v1/messages" \
        -H 'content-type: application/json' \
        -d '{"model":"deepseek-v4-flash","max_tokens":24,"messages":[{"role":"user","content":"Reply with the single word: pong"}]}' 2>/dev/null)
    local av; av=$(printf '%s' "$a" | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit("not json")
t = "".join(b.get("text","") for b in d.get("content",[]) if b.get("type")=="text")
u = d.get("usage",{})
if not t.strip(): sys.exit("empty text")
if not (u.get("input_tokens",0)>0 and u.get("output_tokens",0)>0): sys.exit("zero usage")
# On-topic, not merely non-empty. The standard requires an answer on topic for a
# decidable prompt; a
# non-empty check alone goes green on a garbage-token stream, which is exactly
# the failure that bit three times on 2026-07-29 (stale DLL / truncated GGUF /
# missing shard flag all produced ready clusters answering BOS noise).
if "pong" not in t.lower(): sys.exit("off-topic reply: " + t.strip()[:80])
print("OK "+t.strip()[:60])' 2>&1)
    case "$av" in OK*) vlog "anthropic: $av" ;; *) [ "$started" = 1 ] && $SSH "$COORD_NODE" "cd $DGX_HOME && bash scripts/run_cluster.sh --stop" >/dev/null 2>&1; fail "$name" "Anthropic /v1/messages: $av"; return ;; esac

    # OpenAI shape.
    local o; o=$(curl -s -m 120 "http://$API_HOST:$API_PORT/v1/chat/completions" \
        -H 'content-type: application/json' \
        -d '{"model":"deepseek-v4-flash","max_tokens":24,"messages":[{"role":"user","content":"Reply with the single word: pong"}]}' 2>/dev/null)
    local ov; ov=$(printf '%s' "$o" | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit("not json")
ch = d.get("choices",[])
t = ch[0]["message"]["content"] if ch else ""
u = d.get("usage",{})
if not t.strip(): sys.exit("empty content")
if not (u.get("prompt_tokens",0)>0 and u.get("completion_tokens",0)>0): sys.exit("zero usage")
if "pong" not in t.lower(): sys.exit("off-topic reply: " + t.strip()[:80])
print("OK "+t.strip()[:60])' 2>&1)
    [ "$started" = 1 ] && $SSH "$COORD_NODE" "cd $DGX_HOME && bash scripts/run_cluster.sh --stop" >/dev/null 2>&1
    case "$ov" in OK*) vlog "openai: $ov"; pass "$name" ;; *) fail "$name" "OpenAI /v1/chat/completions: $ov" ;; esac
}

# =====================================================================
# G-PRIV — Privacy / token encryption: envelope crypto invariants hold
#          (docs/privacy-design.md). Depends on E6 for the live-cluster
#          log/pcap checks; the crypto pipeline itself is proven headlessly
#          by the self-test oracle (portable C, no GPU needed).
# =====================================================================
g_priv() {
    local name="$1"
    # Two layers of oracle on the coord node, both hardware-free:
    #  (1) headless self-tests — crypto invariants + envelope codec: no
    #      plaintext on the wire, workers hold no key/vocab/text, wrong/tampered
    #      key rejected, encrypt-at-rest, pluggable Obfuscator, zeroize,
    #      Tier-1 hardening. Prints G_PRIV_SELFTEST_OK + G_PRIV_HTTP_SELFTEST_OK.
    #  (2) real-socket e2e — the sealed-envelope proxy in front of a mock coord:
    #      consumer talks only ciphertext, proxy decrypts + forwards plaintext
    #      over loopback + seals the reply, forged request rejected. Prints
    #      PRIVACY_PROXY_E2E_OK.
    local st
    st=$($SSH "$COORD_NODE" "cd $DGX_HOME && make -f Makefile.privacy selftest 2>&1" 2>/dev/null | tr -d '\r')
    echo "$st" | grep -q "G_PRIV_SELFTEST_OK"      || { fail "$name" "crypto self-test failed"; return; }
    echo "$st" | grep -q "G_PRIV_HTTP_SELFTEST_OK" || { fail "$name" "envelope codec self-test failed"; return; }
    vlog "headless self-tests passed on $COORD_NODE"

    local e2e
    e2e=$($SSH "$COORD_NODE" "cd $DGX_HOME && bash scripts/privacy_proxy_e2e.sh 2>&1 | tail -1" 2>/dev/null | tr -d '\r')
    if [ "$e2e" = "PRIVACY_PROXY_E2E_OK" ]; then
        vlog "real-socket proxy e2e passed on $COORD_NODE"
        pass "$name"
    else
        fail "$name" "sealed-proxy e2e did not pass (got '$e2e')"
    fi
}

# =====================================================================
# G-PAIR — engine-native LAN discovery + verification-code pairing. Two engine
#          processes (idletoken-coord + idletoken-worker) self-assemble on the coord
#          node using ONLY a shared code — no manual --coordinator — via the
#          UDP beacon + mutual-auth preamble, reaching cluster_ready. This is the
#          engine counterpart of the client-driven P3 gate (which drives the same
#          join through the Tauri UI). Model is mock: G_PAIR proves discovery +
#          code auth + join; inference correctness is G6/P6.
# =====================================================================
g_pair() {
    local name="$1"
    # Portable-C unit tests first (crypto vectors, code helpers, auth preamble,
    # broadcast + mock providers) — fast, hardware-free.
    local ut
    ut=$($SSH "$COORD_NODE" "cd $DGX_HOME && export PATH=/usr/local/cuda-13.0/bin:\$PATH && make disctest 2>&1 | tail -1" 2>/dev/null | tr -d '\r')
    echo "$ut" | grep -q "DISCOVERY_TEST_OK" || { fail "$name" "discovery unit tests failed ($ut)"; return; }
    vlog "discovery unit tests passed on $COORD_NODE"
    # Real two-process code pairing to cluster_ready (loopback beacon).
    local out
    out=$($SSH "$COORD_NODE" "cd $DGX_HOME && timeout 160 bash scripts/pair_selftest.sh 2>&1 | tail -3" 2>/dev/null | tr -d '\r')
    case "$out" in
        *G_PAIR_OK*) vlog "two engine processes joined by code -> cluster_ready" ;;
        *) fail "$name" "engine code-pairing did not reach cluster_ready (got '$(echo "$out" | tail -1)')"; return ;;
    esac
    # CROSS-MACHINE, not just loopback. Everything above runs two processes on
    # ONE box, so it cannot see anything that only breaks over the physical LAN
    # (a firewall rule, a broadcast that does not leave the host, an address the
    # joiner resolves to itself). The product's claim is "same code on several
    # machines" — with the loopback check alone, a fleet-wide join failure went
    # unnoticed until the topology matrix hit it days later.
    local xm
    xm=$($SSH "$COORD_NODE" "cd $DGX_HOME && WORKER_NODE=${WORKER_NODES[0]} timeout 200 bash scripts/pair_xmachine_check.sh 2>&1 | tail -3" 2>/dev/null | tr -d '\r')
    case "$xm" in
        *XMACHINE_PAIR_OK*) vlog "${WORKER_NODES[0]} joined the coord over the LAN by code"; pass "$name" ;;
        *XMACHINE_PAIR_SKIP*)
            # Missing small-model weights on the coordinator is a provisioning
            # gap, not a pairing failure — say which, do not fake either.
            vlog "cross-machine half skipped: $(echo "$xm" | head -1)"; pass "$name" ;;
        *) fail "$name" "cross-machine code pairing failed (got '$(echo "$xm" | tail -1)')" ;;
    esac
}

# =====================================================================
# G-MODEL — pluggable model interface (multi-model design §8, Phase A/B out
#           gate). LOCAL + hardware-free: the orchestration layer must carry no
#           model-specific constants, the wire must carry a model id, and the
#           ds4x generic backend must align numerically with its reference.
#           1. registry + planner unit tests (plantest): model lookup, refusal
#              of unavailable models, MLA overhead, boundary-multiple split.
#           2. ds4x unit tests (ds4xtest): GGUF-metadata config dispatch +
#              CPU MLA-MoE forward vs numpy oracle (bundle / gguf-load /
#              pipeline / incremental-decode / PP-split — the last two bit-exact).
#           3. no-hardcode scan: orchestration source (NOT vendor/ds4, that is
#              the ds4 backend body) is free of bare layer-count 43 / the
#              literal model name / the old DS4_N_LAYER macro.
#           4. protocol v2: ASSIGN_PLAN carries model identity.
#           Runs on the control machine — pure C, no GPU, no model file.
# =====================================================================
g_model() {
    local name="$1"
    local repo; repo=$(cd "$(dirname "$0")/.." && pwd)
    command -v cc >/dev/null 2>&1 || { skip "$name" "no C compiler on the control machine"; return; }
    command -v python3 >/dev/null 2>&1 || { skip "$name" "python3 needed for ds4x fixtures"; return; }
    mkdir -p "$repo/build/fixtures"

    # 1. registry + planner
    local pt
    pt=$(cd "$repo" && cc -Wall -Wextra -std=c99 -Iinclude src/common/plan.c \
            src/common/model.c src/common/advise.c src/tools/plan_test.c -o build/plan_test 2>&1 \
            && ./build/plan_test 2>&1 | tail -1)
    echo "$pt" | grep -q "PLAN_TEST_OK" || { fail "$name" "planner/registry tests failed ($pt)"; return; }
    vlog "registry + planner unit tests passed"

    # 1b. models/<id>.json vs the COMPILED registry. Two hand-maintained copies
    # of the same numbers (manifest feeds the platform catalogue + client, the
    # registry feeds the in-engine resource planner). Drift does not crash
    # anything — it just sizes clusters from numbers the engine disagrees with.
    local mc
    mc=$(cd "$repo" && cc -Wall -Wextra -std=c99 -Iinclude src/common/model.c \
            src/tools/model_registry_dump.c -o build/model_registry_dump 2>&1 \
            && python3 scripts/model_manifest_check.py build/model_registry_dump 2>&1 | tail -1)
    echo "$mc" | grep -q "MODEL_MANIFEST_CHECK_OK" || { fail "$name" "model manifest/registry mismatch ($mc)"; return; }
    vlog "models/*.json agree with the compiled registry"

    # 2. ds4x config + forward (generates fixtures, incl. a tiny real GGUF)
    local cf ff
    cf=$(cd "$repo" && python3 scripts/make_test_gguf.py build/fixtures >/dev/null 2>&1 \
            && cc -Wall -Wextra -std=c99 -Iinclude src/common/gguf.c src/common/model.c \
                 src/ds4x/ds4x_config.c src/tools/ds4x_config_test.c -o build/ds4x_config_test 2>&1 \
            && ./build/ds4x_config_test build/fixtures 2>&1 | tail -1)
    echo "$cf" | grep -q "DS4X_CONFIG_TEST_OK" || { fail "$name" "ds4x config tests failed ($cf)"; return; }
    ff=$(cd "$repo" && python3 scripts/ds4x_ref.py build/fixtures/ds4x_vectors.bin >/dev/null 2>&1 \
            && cc -Wall -Wextra -std=c99 -Iinclude src/common/model.c src/ds4x/ds4x_config.c \
                 src/ds4x/ds4x_forward.c src/ds4x/ds4x_model.c src/ds4x/ds4x_runner.c \
                 src/ds4x/ds4x_quant.c src/common/gguf.c src/tools/ds4x_forward_test.c \
                 -o build/ds4x_forward_test -lm 2>&1 \
            && ./build/ds4x_forward_test build/fixtures/ds4x_vectors.bin \
                 build/fixtures/ds4x_vectors.bin.gguf 2>&1 | tail -1)
    echo "$ff" | grep -q "DS4X_FORWARD_TEST_OK" || { fail "$name" "ds4x forward alignment failed ($ff)"; return; }
    vlog "ds4x config + forward (5 channels incl. bit-exact decode/PP-split) passed"

    # 2b. GGUF byte-level BPE tokenizer (vocab load / decode / special / encode).
    local tt
    tt=$(cd "$repo" && cc -Wall -Wextra -std=c99 -Iinclude src/common/gguf.c \
            src/ds4x/ds4x_tokenizer.c src/tools/ds4x_tok_test.c -o build/ds4x_tok_test 2>&1 \
            && ./build/ds4x_tok_test build/fixtures 2>&1 | tail -1)
    echo "$tt" | grep -q "DS4X_TOK_TEST_OK" || { fail "$name" "tokenizer tests failed ($tt)"; return; }
    vlog "GGUF tokenizer (decode/special/BPE round-trip) passed"

    # 2c. CPU dequant formulas (F16/Q8_0/Q4_0) + unsupported-type refusal.
    local qt
    qt=$(cd "$repo" && cc -Wall -Wextra -std=c99 -Iinclude src/ds4x/ds4x_quant.c \
            src/tools/ds4x_quant_test.c -o build/ds4x_quant_test -lm 2>&1 \
            && ./build/ds4x_quant_test 2>&1 | tail -1)
    echo "$qt" | grep -q "DS4X_QUANT_TEST_OK" || { fail "$name" "dequant tests failed ($qt)"; return; }
    vlog "CPU dequant (F16/Q8_0/Q4_0) formulas verified"

    # 2c-bis. MoE with NO shared expert. The forward used to call the shared
    #     expert unconditionally while the loader only filled it when
    #     n_expert_shared > 0 — a plain NULL deref (verified: removing the guard
    #     gives SIGSEGV on this fixture). DSv4 has a shared expert, so every
    #     existing channel missed it; Qwen3-MoE has none.
    local ns
    ns=$(cd "$repo" && python3 scripts/ds4x_ref.py build/fixtures/ds4x_noshared.bin \
            --n-shared 0 >/dev/null 2>&1 \
            && ./build/ds4x_forward_test build/fixtures/ds4x_noshared.bin \
                 build/fixtures/ds4x_noshared.bin.gguf 2>&1 | tail -1)
    echo "$ns" | grep -q "DS4X_FORWARD_TEST_OK" || { fail "$name" "MoE without shared expert failed ($ns)"; return; }
    vlog "MoE without a shared expert (NULL-deref regression) passed"

    # 2d. non-MLA attention families. The ds4x backend is tri-state
    #     (MLA | GQA | LINEAR) with PER-LAYER dispatch for hybrids, so 2's MLA
    #     channel alone no longer covers it. Both of these prove a bit-exact
    #     PP 2-stage split and (for the hybrid) that carrying the recurrent
    #     state + conv window makes incremental decode identical to one-shot
    #     prefill — the property the whole linear-attention line rests on.
    local gq gd
    gq=$(cd "$repo" && python3 scripts/ds4x_ref.py build/fixtures/ds4x_qwen3.bin --arch qwen3 >/dev/null 2>&1 \
            && cc -Wall -Wextra -std=c99 -Iinclude src/common/model.c src/ds4x/ds4x_config.c \
                 src/ds4x/ds4x_forward.c src/ds4x/ds4x_model.c src/ds4x/ds4x_runner.c \
                 src/ds4x/ds4x_quant.c src/common/gguf.c src/tools/ds4x_gqa_test.c \
                 -o build/ds4x_gqa_test -lm 2>&1 \
            && ./build/ds4x_gqa_test build/fixtures/ds4x_qwen3.bin \
                 build/fixtures/ds4x_qwen3.bin.gguf 2>&1 | tail -1)
    echo "$gq" | grep -q "DS4X_GQA_TEST_OK" || { fail "$name" "ds4x GQA alignment failed ($gq)"; return; }
    gd=$(cd "$repo" && python3 scripts/ds4x_ref.py build/fixtures/ds4x_gdn.bin --arch qwen3next >/dev/null 2>&1 \
            && cc -Wall -Wextra -std=c99 -Iinclude src/common/model.c src/ds4x/ds4x_config.c \
                 src/ds4x/ds4x_forward.c src/ds4x/ds4x_model.c src/ds4x/ds4x_runner.c \
                 src/ds4x/ds4x_quant.c src/common/gguf.c src/tools/ds4x_gdn_test.c \
                 -o build/ds4x_gdn_test -lm 2>&1 \
            && ./build/ds4x_gdn_test build/fixtures/ds4x_gdn.bin 2>&1 | tail -1)
    echo "$gd" | grep -q "DS4X_GDN_TEST_OK" || { fail "$name" "ds4x linear-attention (GDN) alignment failed ($gd)"; return; }
    vlog "ds4x GQA + hybrid linear-attention (state-carry + PP-split bit-exact) passed"

    # 3. no-hardcode scan over the orchestration layer (vendor/ds4 excluded).
    #    The smell is a model name baked in as a VALUE (a quoted string literal
    #    "deepseek-v4-flash" in a response/assignment) or the old fixed
    #    layer-count macro. Help/usage text mentioning the default by name is
    #    fine — that is documentation, not a hardcoded code path.
    local bad=""
    grep -RnE '"deepseek-v4-flash"' "$repo/src/coord" "$repo/src/worker" 2>/dev/null >/dev/null \
        && bad="$bad model-name-value-literal"
    grep -Rn "IDLETOKEN_DS4_N_LAYER" "$repo/src" 2>/dev/null >/dev/null && bad="$bad DS4_N_LAYER-macro"
    grep -RnE '\b43\b' "$repo/src/common/plan.c" 2>/dev/null >/dev/null && bad="$bad plan.c-literal-43"
    [ -n "$bad" ] && { fail "$name" "orchestration still hardcodes model:$bad"; return; }
    vlog "orchestration source free of model-specific constants"

    # 4. protocol v2 carries model identity
    # >= 2: v2 introduced the model id in ASSIGN_PLAN, v3 added the quant.
    # Pinning the literal version made this gate fail on the v3 bump even
    # though the property it guards (wire carries model identity) still held.
    # Match the #define, not any mention — the header's comment names the macro
    # too, and a greedy match on that yielded an empty version.
    local pv
    pv=$(grep -E '^#define +IDLETOKEN_PROTO_VERSION' "$repo/include/idletoken_proto.h" \
         | grep -oE '[0-9]+' | head -1)
    [ -n "$pv" ] && [ "$pv" -ge 2 ] \
        || { fail "$name" "protocol version '$pv' < 2 (model id in ASSIGN_PLAN)"; return; }
    grep -q "model_id" "$repo/src/worker/worker_main.c" \
        || { fail "$name" "worker does not parse ASSIGN_PLAN model_id"; return; }

    pass "$name"
}

# =====================================================================
# Product gates (P1-P3) — the Tauri client on the coordinator node, driven
# through the IDLETOKEN_UI_TEST channel (directives execute via the same provider
# paths user clicks take; results come back as `UI_TEST_REPORT <tag> <json>`
# on the client's stderr). Requires the debug shell built on the node:
#   cd client/src-tauri && cargo build   (binaries staged in target/debug/)
# =====================================================================
CLIENT_DIR="$DGX_HOME/client/src-tauri"
CLIENT_BIN='./target/debug/idletoken-client'
# WEBKIT var: NVIDIA white-screen fix; no_proxy: Clash on the coord node must
# not intercept localhost RPC.
CLIENT_ENV_BASE='WEBKIT_DISABLE_DMABUF_RENDERER=1 no_proxy=localhost,127.0.0.1'
# The display is resolved by client_display() when the run starts, see below.
CLIENT_ENV="$CLIENT_ENV_BASE DISPLAY=${IDLETOKEN_DISPLAY:-:1}"

# The product gates need a display that can actually draw. This used to hardcode
# `DISPLAY=:1` -- **the desktop session a human had logged into on the
# coordinator**.
#
# 2026-08-08, a whole round was blocked by it: Xorg on the DGX had died at some
# point, leaving orphaned `gdm-x-session` / `gnome-session` processes pointing at
# a `:1` that no longer existed and an empty `/tmp/.X11-unix/`. P1 reported
# "client did not exit cleanly" while the real cause, `Failed to initialize GTK`,
# sat deep in the log -- far from the symptom. Restarting gdm brought Xorg back,
# but on `:0` at the **login screen** (its auth belongs to the gdm user), and `:1`
# still did not exist because nobody had logged in.
#
# In other words: **the entire product track depended on someone being logged into
# that machine's desktop**. That is not a premise automated acceptance may rest
# on; one reboot or one crashed session and the ladder cannot finish, while the
# symptom it reports points somewhere else.
#
# The search order is now: 1. an explicitly configured environment variable,
# 2. a real session that already works, 3. an Xvfb we start ourselves.
# Using a virtual display is **stated loudly** (principle 13): a round on a
# virtual display must not read like a round on a real desktop -- it is enough to
# verify rendering and the programmatic assertions, but it does not replace a
# human looking at the screen.
client_display() {
    if [ -n "${IDLETOKEN_DISPLAY:-}" ]; then echo "$IDLETOKEN_DISPLAY"; return 0; fi
    if $SSH "$COORD_NODE" "DISPLAY=:1 timeout 5 xdpyinfo >/dev/null 2>&1"; then
        echo ":1"; return 0
    fi
    # No usable real session -- start a virtual one (idempotent: reused if already running).
    if ! $SSH "$COORD_NODE" "command -v Xvfb >/dev/null 2>&1"; then
        echo ":1"; return 0   # no Xvfb installed: keep the old behaviour so the P gates go red with the GTK cause
    fi
    $SSH "$COORD_NODE" "DISPLAY=:77 timeout 5 xdpyinfo >/dev/null 2>&1" || \
        $SSHF "$COORD_NODE" "exec Xvfb :77 -screen 0 1600x1000x24 -nolisten tcp < /dev/null > /tmp/idletoken-xvfb.log 2>&1"
    local i
    for i in 1 2 3 4 5 6 7 8; do
        if $SSH "$COORD_NODE" "DISPLAY=:77 timeout 5 xdpyinfo >/dev/null 2>&1"; then
            # A virtual display **must have a window manager**, or Tauri's window
            # is created but never mapped or sized: the client runs, the UI-test
            # report is produced as usual, but **it does not exit when the window
            # closes** (timeout kills it -> the gate reports "client did not exit
            # cleanly" while the cause is three layers away). On a bare Xvfb this
            # took several rounds to pin down, hence the note.
            $SSH "$COORD_NODE" "pgrep -f '[o]penbox' >/dev/null" || \
                $SSHF "$COORD_NODE" "exec env DISPLAY=:77 openbox < /dev/null > /tmp/idletoken-openbox.log 2>&1"
            sleep 2
            echo ":77"; return 0
        fi
        sleep 1
    done
    echo ":1"
}

client_cleanup() {
    $SSH "$COORD_NODE" "pkill -f 'debug/idletoken-clien[t]'; pkill -f 'debug/idletoken-coor[d]'; pkill -f 'debug/idletoken-worke[r]'; true" >/dev/null 2>&1
}

# The debug shell loads its frontend from devUrl (:1420). Make the gates
# self-contained: if nothing serves :1420 on the node, serve the prebuilt
# static dist/ (synced there) with python http.server. Idempotent.
#
# Started through $SSHF (see above): backgrounding inside the remote shell hangs
# the ssh, so the local client backgrounds itself instead. Only bites on the
# first run after a reboot (when :1420 isn't already served), which is why it
# survived earlier green runs.
ensure_frontend() {
    $SSH "$COORD_NODE" "curl -s -m 2 -o /dev/null http://127.0.0.1:1420/" >/dev/null 2>&1 && return 0
    $SSHF "$COORD_NODE" "cd $DGX_HOME/client/dist && exec python3 -m http.server 1420 --bind 127.0.0.1 < /dev/null > /tmp/idletoken-frontend.log 2>&1" >/dev/null 2>&1
    local i
    for i in 1 2 3 4 5 6 7 8 9 10; do
        $SSH "$COORD_NODE" "curl -s -m 2 -o /dev/null http://127.0.0.1:1420/" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

# Run one client instance in the foreground with the given directives; prints
# its log afterwards. $1=directives $2=logfile $3=timeout-guard(s)
client_run() {
    # Resolve the display once per run, and say so loudly when it is virtual.
    if [ -z "${CLIENT_DISPLAY_RESOLVED:-}" ]; then
        CLIENT_DISPLAY_RESOLVED="$(client_display)"
        CLIENT_ENV="$CLIENT_ENV_BASE DISPLAY=$CLIENT_DISPLAY_RESOLVED"
        if [ "$CLIENT_DISPLAY_RESOLVED" = ":77" ]; then
            echo "note: no usable desktop session on the coordinator; the product gates are running on an **Xvfb virtual display** (:77)." >&2
            echo "      The programmatic assertions still hold; a human visual walkthrough is not covered by this round." >&2
        fi
    fi
    ensure_frontend
    # WebKit caches the frontend bundle, **index.html included**. A rebuilt dist
    # can therefore have no effect at all: on 2026-08-05 a newly added UI-test
    # directive produced no output no matter how it was run, while older
    # directives worked in the same session -- because the webview was loading the
    # cached older bundle. This is the same class of structural false green as
    # "the ladder is running a three-week-old exe", moved to the UI side: **a
    # product gate can certify a stale frontend**.
    # Clearing it before each run costs one cold load.
    $SSH "$COORD_NODE" "rm -rf ~/.local/share/ai.idletoken.client/WebKitCache" >/dev/null 2>&1
    $SSH "$COORD_NODE" "cd $CLIENT_DIR && rm -f $2 && timeout $3 env $CLIENT_ENV IDLETOKEN_UI_TEST='$1' $CLIENT_BIN > $2 2>&1; echo CLIENT_EXIT=\$?; cat $2" 2>/dev/null | tr -d '\r'
}

p1_client() {
    local name="$1"
    # The client spawns the engine as SIDECARS from client/src-tauri/binaries/.
    # That directory is staged by hand (release script / stage_sidecars.sh), so
    # it drifts — and when it does, every P gate and G-FINAL certify an engine
    # nobody built today. On 2026-08-04 it was six days stale, old enough to
    # predate protocol v5: a freshly installed worker joined and the coordinator
    # answered `recv HELLO: Protocol error` in a loop. Same shape as the stale
    # dist\ bundle. Assert freshness before any product gate runs.
    # Check target/debug/, NOT src-tauri/binaries/. A debug Tauri build resolves
    # sidecars next to itself; binaries/ is what the BUNDLER reads. Aiming this
    # assertion at binaries/ (the first attempt did) reports FRESH while the
    # client keeps running a months-old engine.
    local sc
    sc=$($SSH "$COORD_NODE" "cd $DGX_HOME && \
        w=client/src-tauri/target/debug/idletoken-worker; \
        if [ ! -e \$w ]; then echo NO_SIDECAR; \
        elif [ idletoken-worker -nt \$w ]; then echo STALE; \
        else echo FRESH; fi" 2>/dev/null | tr -d '\r')
    case "$sc" in
        FRESH) vlog "client sidecars are not older than the built engine" ;;
        NO_SIDECAR) fail "$name" "no staged sidecars on $COORD_NODE (run scripts/stage_sidecars.sh)"; return ;;
        *) fail "$name" "client sidecars are STALE on $COORD_NODE — run scripts/stage_sidecars.sh"; return ;;
    esac
    local base
    base=$($SSH "$COORD_NODE" "cd $DGX_HOME && ./idletoken-worker --probe-json 2>/dev/null | tail -1" | tr -d '\r')
    local base_host base_gpu
    base_host=$(echo "$base" | python3 -c 'import json,sys;print(json.load(sys.stdin)["hostname"])' 2>/dev/null)
    base_gpu=$(echo "$base" | python3 -c 'import json,sys;print(json.load(sys.stdin)["gpu_name"])' 2>/dev/null)
    if [ -z "$base_host" ] || [ -z "$base_gpu" ]; then fail "$name" "probe baseline failed on $COORD_NODE"; return; fi
    vlog "baseline: $base_host / $base_gpu"

    client_cleanup
    local out
    # One extra directive in the same launch: the support-facing "one-click
    # export" **must contain no credentials**. The sentinel token is written into
    # settings before the report is produced, so leaksApiToken=false is a verified
    # statement rather than an empty one (tokenChecked reports honestly whether
    # this cell was tested at all).
    # Do not put the timeout budget back to 50: that value was always **marginal**.
    # A 25s quit timer plus a cold WebKit start (the cache is cleared every round,
    # see above) plus probe plus the advise sidecar inside diagnostics plus
    # shutdown was measured at 59.6s. The gate was therefore flaky, and what it
    # reported was "client did not exit cleanly" -- which sounds like a client bug
    # and was really too short a stopwatch. Several rounds went into this on
    # 2026-08-09: first report-diagnostics was suspected, then the missing window
    # manager (that one was real and fixed separately), and only then was the
    # actual cause measured.
    out=$(client_run "report-probe,report-diagnostics:token=SENTINEL-acc,quit:25000" /tmp/idletoken-p1.log 90)
    client_cleanup
    echo "$out" | grep -q "CLIENT_EXIT=0" || { fail "$name" "client did not exit cleanly"; return; }
    local rep
    rep=$(echo "$out" | grep "UI_TEST_REPORT probe" | head -1 | sed 's/.*UI_TEST_REPORT probe //')
    [ -n "$rep" ] || { fail "$name" "no probe report from the UI channel"; return; }
    local ok
    ok=$(echo "$rep" | python3 -c "
import json,sys
s=json.load(sys.stdin)
print('ok' if s.get('hostname')=='$base_host' and s.get('gpu_name')=='$base_gpu' and s.get('source')=='engine' else 'mismatch: '+repr((s.get('hostname'),s.get('gpu_name'),s.get('source'))))" 2>/dev/null)
    if [ "$ok" != "ok" ]; then fail "$name" "client probe != engine probe ($ok)"; return; fi
    vlog "client rendered real engine probe (source=engine, matches --probe-json)"

    local drep dok
    drep=$(echo "$out" | grep "UI_TEST_REPORT diagnostics" | head -1 | sed 's/.*UI_TEST_REPORT diagnostics //')
    [ -n "$drep" ] || { fail "$name" "no diagnostics report from the UI channel"; return; }
    dok=$(echo "$drep" | python3 -c "
import json,sys
d=json.load(sys.stdin)
if not d.get('tokenChecked'): print('redaction check never exercised (no sentinel token in settings)')
elif d.get('leaksApiToken'): print('the diagnostics bundle CONTAINS the access token')
elif not d.get('probeGpu'):  print('diagnostics has no GPU fact: '+repr(d.get('probeError')))
else: print('ok')" 2>/dev/null)
    if [ "$dok" = "ok" ]; then
        vlog "diagnostics bundle carries real hardware facts and no access token"
        pass "$name"
    else
        fail "$name" "diagnostics bundle: $dok"
    fi
}

p2_auth() {
    local name="$1"
    client_cleanup
    local out
    out=$(client_run "auth-flow,quit:20000" /tmp/idletoken-p2.log 40)
    client_cleanup
    local rep
    rep=$(echo "$out" | grep "UI_TEST_REPORT auth" | head -1 | sed 's/.*UI_TEST_REPORT auth //')
    [ -n "$rep" ] || { fail "$name" "no auth report from the UI channel"; return; }
    local ok
    ok=$(echo "$rep" | python3 -c "
import json,sys
r=json.load(sys.stdin)
bad=[k for k,v in [('signup',r.get('signup')=='ok'),('signedOutNull',r.get('signedOutNull') is True),('wrongPwRejected',r.get('wrongPwRejected') is True),('reSignin',r.get('reSignin')=='ok')] if not v]
print('ok' if not bad else 'failed: '+','.join(bad)+' in '+json.dumps(r))" 2>/dev/null)
    if [ "$ok" = "ok" ]; then
        vlog "signup / signout / wrong-password rejected / re-signin all correct"
        pass "$name"
    else
        fail "$name" "$ok"
    fi
}

p3_pairing() {
    local name="$1"
    client_cleanup
    ensure_frontend
    # Two instances on the coordinator node: A creates + auto-starts, B joins
    # by code (loopback beacon). Same flow two real machines take on the LAN.
    # Hand both instances a REAL model (`:model=<abs>`, the same token P6 uses).
    # Without it the client falls back to the registry's `default_gguf`, which is
    # a BARE FILENAME — the worker cannot stat it, and until 2026-07-29 it simply
    # fell back to mock, so the cluster reported `ready` and this gate passed on a
    # MOCK cluster. Its layer-coverage assertion is satisfied by mock just as well
    # as by real weights, so nothing here could ever have noticed.
    local gguf_abs; gguf_abs=$($SSH "$COORD_NODE" "echo $GGUF_DGX" 2>/dev/null | tr -d '\r')
    case "$gguf_abs" in
        ""|*'$'*) fail "$name" "could not expand the GGUF path on $COORD_NODE (got '$gguf_abs')"; return ;;
    esac
    $SSH "$COORD_NODE" "rm -f /tmp/idletoken-p3a.log /tmp/idletoken-p3b.log" >/dev/null 2>&1
    $SSHF "$COORD_NODE" "cd $CLIENT_DIR && exec env $CLIENT_ENV IDLETOKEN_UI_TEST='pairing-create:ACCEPT:model=$gguf_abs,pairing-auto-start,quit:600000' $CLIENT_BIN < /dev/null > /tmp/idletoken-p3a.log 2>&1" >/dev/null 2>&1
    sleep 4
    $SSHF "$COORD_NODE" "cd $CLIENT_DIR && exec env $CLIENT_ENV IDLETOKEN_UI_TEST='pairing-join:ACCEPT:as=accept-node-b:model=$gguf_abs,quit:600000' $CLIENT_BIN < /dev/null > /tmp/idletoken-p3b.log 2>&1" >/dev/null 2>&1

    # A REAL 80 GiB load takes minutes, not the 90s that sufficed for mock.
    local st="" i
    for i in $(seq 1 100); do
        st=$($SSH "$COORD_NODE" "curl -s -m 2 http://127.0.0.1:$API_PORT/v1/cluster/status; true" 2>/dev/null | tr -d '\r')
        case "$st" in *'"phase":"ready"'*) break ;; esac
        st=""
        sleep 6
    done
    if [ -z "$st" ]; then
        client_cleanup
        fail "$name" "cluster never reached ready (see /tmp/idletoken-p3{a,b}.log on $COORD_NODE)"
        return
    fi
    vlog "cluster status: $st"
    local ok
    ok=$(echo "$st" | python3 -c "
import json,sys
s=json.load(sys.stdin)
m=sorted(s.get('members',[]),key=lambda x:x['stage'])
cover=(m and m[0]['layer_lo']==0 and m[-1]['layer_hi']==43
       and all(m[i]['layer_hi']==m[i+1]['layer_lo'] for i in range(len(m)-1)))
print('ok' if s.get('phase')=='ready' and cover else 'bad: '+json.dumps(s))" 2>/dev/null)
    client_cleanup
    if [ "$ok" = "ok" ]; then
        vlog "UI-initiated pairing produced a real engine cluster (43 layers covered)"
        pass "$name"
    else
        fail "$name" "$ok"
    fi
}

# Launch a creator+joiner pair on the coord node (loopback beacon) and echo the
# creator's UI_TEST pairing-phases report once the cluster reaches ready.
# $1=code $2=logtag $3=extra-create-directives $4=extra-join-directives
# $5=client-life-ms $6=poll-tries $7=poll-sleep-s
pairing_pair_report() {
    local code="$1" tag="$2" cx="$3" jx="$4" life="$5" tries="$6" slp="$7"
    client_cleanup
    ensure_frontend
    $SSH "$COORD_NODE" "rm -f /tmp/idletoken-$tag-a.log /tmp/idletoken-$tag-b.log" >/dev/null 2>&1
    $SSHF "$COORD_NODE" "cd $CLIENT_DIR && exec env $CLIENT_ENV IDLETOKEN_UI_TEST='pairing-create:$code$cx,pairing-auto-start,report-pairing-phases,quit:$life' $CLIENT_BIN < /dev/null > /tmp/idletoken-$tag-a.log 2>&1" >/dev/null 2>&1
    sleep 4
    $SSHF "$COORD_NODE" "cd $CLIENT_DIR && exec env $CLIENT_ENV IDLETOKEN_UI_TEST='pairing-join:$code:as=$tag-node-b$jx,quit:$life' $CLIENT_BIN < /dev/null > /tmp/idletoken-$tag-b.log 2>&1" >/dev/null 2>&1
    local rep="" i
    for i in $(seq 1 "$tries"); do
        rep=$($SSH "$COORD_NODE" "grep 'UI_TEST_REPORT pairing-phases' /tmp/idletoken-$tag-a.log 2>/dev/null | head -1 | sed 's/.*UI_TEST_REPORT pairing-phases //'; true" 2>/dev/null | tr -d '\r')
        [ -n "$rep" ] && break
        sleep "$slp"
    done
    printf '%s' "$rep"
}

# =====================================================================
# P4 — Auto-orchestration: after pairing + one start, the cluster auto-probes,
#      splits, loads and reaches ready with the API address online in the UI
#      snapshot — no CLI step. (Per-node layer display in the snapshot needs a
#      real 2-machine LAN with matched hostnames; the engine layer plan itself
#      is asserted by P3. Model is mock here — P4 tests orchestration, not
#      inference.)
# =====================================================================
p4_orchestration() {
    local name="$1"
    # Mock is DELIBERATE here (see the header) — but it now has to be SAID.
    # Until 2026-07-29 a worker silently fell back to mock whenever the model
    # would not load, so this gate got one for free; since then the worker
    # refuses unless asked. Declaring it is strictly better: the gate states
    # which of its assertions are about orchestration rather than inference,
    # exactly as P5 already does.
    local saved_env="$CLIENT_ENV"
    CLIENT_ENV="$CLIENT_ENV IDLETOKEN_MOCK_OK=1 IDLETOKEN_ALLOW_MOCK=1"
    local rep; rep=$(pairing_pair_report ORCHES p4 "" "" 150000 40 3)
    CLIENT_ENV="$saved_env"
    client_cleanup
    [ -n "$rep" ] || { fail "$name" "no pairing-phases report (see /tmp/idletoken-p4-{a,b}.log on $COORD_NODE)"; return; }
    vlog "pairing-phases: $rep"
    local ok
    ok=$(echo "$rep" | python3 -c "
import json,sys
r=json.load(sys.stdin)
bad=[k for k,v in [
  ('sawStarting', r.get('sawStarting') is True),
  ('endedReady', r.get('endedReady') is True),
  ('apiOnline', r.get('apiOnline') is True),
  ('apiBaseUrl', bool(r.get('apiBaseUrl'))),
  ('coordinatorReady', r.get('coordinatorReady') is True),
  ('peersReady', r.get('peersReady') is True),
] if not v]
print('ok' if not bad else 'failed: '+','.join(bad)+' | '+json.dumps(r))" 2>/dev/null)
    if [ "$ok" = "ok" ]; then
        vlog "auto-orchestration idle->starting->ready; API online in the UI snapshot"
        pass "$name"
    else
        fail "$name" "$ok"
    fi
}

# =====================================================================
# P5 — Settings actually take effect (acceptance-criteria P5). This oracle
#      covers the settings→engine slice that is scriptable today: a cluster
#      formed with a CUSTOM apiPort + apiToken must serve on that port (and
#      NOT on the default), and must enforce the token on the API.
#
#      CONTRACT (integration-plan task 1.2, implemented in parallel):
#        - coord grows `--api-token <tok>`; `--api-bind <host>:<port>` is
#          driven by AppSettings instead of the hardcoded 8000;
#        - pairing_create/join accept a tuning object carrying these, and the
#          UI-test/headless channel forwards extra `:apiPort=N:apiToken=S`
#          spec tokens into it (same style as the existing `:model=` token).
#      Until that pipeline lands this gate honestly fails (it IS the oracle
#      for it). The fuller P5 items in the md (model select via the pluggable
#      interface, VRAM/RAM caps into the split, KV-cache policy, i18n/theme
#      persistence) remain PASS requirements at the md level; extending this
#      oracle to them is tracked in the md §P5 note.
# =====================================================================
p5_settings() {
    local name="$1"
    local port=18111 tok="p5-secret-token"
    local tune=":apiPort=$port:apiToken=$tok"
    # Mock model: the coord has no engine, and a production coord must 503 on
    # chat rather than fake a reply. The with-token 200 below therefore needs
    # the LABELED mock-completion path — opt in via the test-only env, which
    # the client passes down to its coord sidecar (env is inherited).
    local saved_env="$CLIENT_ENV"
    # Both switches are required: MOCK_OK is the coordinator's (return a mock
    # completion) and ALLOW_MOCK is the worker's (permit a mock load). Similar
    # names, different layers, and omitting either shows up as a cluster that will
    # not start rather than as wrong content.
    CLIENT_ENV="$CLIENT_ENV IDLETOKEN_MOCK_OK=1 IDLETOKEN_ALLOW_MOCK=1"
    local rep; rep=$(pairing_pair_report P5SETT p5 "$tune" "$tune" 150000 40 3)
    CLIENT_ENV="$saved_env"
    if [ -z "$rep" ]; then
        client_cleanup
        fail "$name" "cluster with custom apiPort/apiToken never reached ready — settings→engine pipeline missing? (integration-plan 1.2; see /tmp/idletoken-p5-{a,b}.log on $COORD_NODE)"
        return
    fi
    vlog "pairing-phases: $rep"

    # (a) custom port serves /health; the default port must NOT be bound.
    local h; h=$($SSH "$COORD_NODE" "curl -s -m 3 http://127.0.0.1:$port/health; true" 2>/dev/null | tr -d '\r')
    if ! echo "$h" | grep -q '"status":"ok"'; then
        client_cleanup
        fail "$name" "custom apiPort $port not serving /health (apiPort setting not applied to the engine)"
        return
    fi
    local hd; hd=$($SSH "$COORD_NODE" "curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$API_PORT/health; true" 2>/dev/null | tr -d '\r')
    if [ "$hd" != "000" ]; then
        client_cleanup
        fail "$name" "default port $API_PORT still serving (http $hd) — custom apiPort did not replace it"
        return
    fi
    vlog "custom port $port healthy; default $API_PORT closed"

    # (b) token enforced: with Bearer -> 200, without -> 401. (Mock model: only
    # the status code is asserted here; reply coherence is G6/P6's job.)
    local req='{"model":"deepseek-v4-flash","max_tokens":4,"messages":[{"role":"user","content":"ping"}]}'
    local wresp; wresp=$($SSH "$COORD_NODE" "curl -s -m 60 -w '\n%{http_code}' http://127.0.0.1:$port/v1/chat/completions -H 'content-type: application/json' -H 'Authorization: Bearer $tok' -d '$req'; true" 2>/dev/null | tr -d '\r')
    local with; with=$(printf '%s' "$wresp" | tail -1)
    local wbody; wbody=$(printf '%s' "$wresp" | sed '$d')
    local without; without=$($SSH "$COORD_NODE" "curl -s -m 10 -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/v1/chat/completions -H 'content-type: application/json' -d '$req'; true" 2>/dev/null | tr -d '\r')
    client_cleanup
    if [ "$with" != "200" ] || [ "$without" != "401" ]; then
        fail "$name" "apiToken not enforced (with token: http $with, expected 200; without: http $without, expected 401)"
        return
    fi
    # Origin proof: the 200 body must be the coord's LABELED mock completion,
    # not anything a middle layer could have fabricated.
    case "$wbody" in
        *"HOMEAI MOCK ENGINE"*)
            vlog "apiToken enforced: with=200 (coord mock marker), without=401"
            pass "$name" ;;
        *)  fail "$name" "with-token 200 body lacks coord mock marker (got: ${wbody:0:80})" ;;
    esac
}

# =====================================================================
# P6 — API exposure: a UI-formed cluster with the REAL model auto-listens; the
#      UI advertises the address; an external client connects to THAT address
#      and gets an E6-level DSv4 reply. Needs the real GGUF (honest fail if
#      absent — P6 cannot pass on a mock model).
# =====================================================================
p6_api_exposure() {
    local name="$1"
    local has; has=$($SSH "$COORD_NODE" "test -f $GGUF_DGX && echo yes || echo no" 2>/dev/null | tr -d '\r')
    [ "$has" = "yes" ] || { fail "$name" "GGUF not found on $COORD_NODE ($GGUF_DGX) — P6 needs the real model"; return; }
    # The path travels into the client as a UI directive and from there straight
    # into sidecar argv — no shell anywhere on that route, so a `$HOME` in it
    # would reach the engine literally (the weights sidecar reports
    # "cannot stat $HOME/..." and the cluster silently falls back to mock).
    # Expand it once, on the node.
    local gguf_abs; gguf_abs=$($SSH "$COORD_NODE" "echo $GGUF_DGX" 2>/dev/null | tr -d '\r')
    case "$gguf_abs" in
        ""|*'$'*) fail "$name" "could not expand the GGUF path on $COORD_NODE (got '$gguf_abs')"; return ;;
    esac

    # Model load across coord + 2 workers is slow; allow up to ~10 min.
    local rep; rep=$(pairing_pair_report APIEXP p6 ":model=$gguf_abs" ":model=$gguf_abs" 600000 100 6)
    if [ -z "$rep" ]; then
        client_cleanup
        fail "$name" "real-model cluster never reached ready (see /tmp/idletoken-p6-{a,b}.log on $COORD_NODE)"
        return
    fi
    local base; base=$(echo "$rep" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("apiBaseUrl") or "")' 2>/dev/null)
    [ -n "$base" ] || { client_cleanup; fail "$name" "UI advertised no API baseUrl ($rep)"; return; }
    vlog "UI-advertised API address: $base"

    # External client hits the UI-advertised address (curl runs on the coord
    # node so the advertised LAN ip resolves) and must get a real reply.
    local reply
    reply=$($SSH "$COORD_NODE" "curl -s -m 180 $base/v1/messages -H 'content-type: application/json' -d '{\"model\":\"deepseek-v4-flash\",\"max_tokens\":24,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: pong\"}]}'" 2>/dev/null)
    client_cleanup
    local ok
    ok=$(printf '%s' "$reply" | python3 -c '
import json,sys
raw=sys.stdin.read()
try: d=json.loads(raw)
except Exception: sys.exit("not json: "+raw[:80])
t="".join(b.get("text","") for b in d.get("content",[]) if b.get("type")=="text")
u=d.get("usage",{})
if not t.strip(): sys.exit("empty text")
if not (u.get("input_tokens",0)>0 and u.get("output_tokens",0)>0): sys.exit("zero usage")
print("ok "+t.strip()[:40])' 2>&1)
    case "$ok" in
        ok*) vlog "external client reply via UI address: $ok"; pass "$name" ;;
        *) fail "$name" "external client got no real reply from $base ($ok)" ;;
    esac
}

# =====================================================================
# G_HOMO — a cluster is homogeneous. The coordinator must REFUSE a worker whose
#      OS family differs from the first worker that joined (CLAUDE.md hard
#      constraint #2, 2026-08-12) — a mixed cluster has no oracle, so letting
#      one form manufactures a green nobody can falsify (docs/macos-node.md §5).
#
# Runs entirely on the control machine: one coordinator + stub workers on
# loopback. The mixed case cannot be staged with real binaries on one box —
# every build reports the OS it was compiled for — so the stub takes
# IDLETOKEN_MOCK_OS_FAMILY. That forced byte is the ONLY thing the gate feeds
# the coordinator; accept-vs-refuse is entirely the coordinator's decision.
#
# Both halves are checked. The refusal alone would also pass if the coordinator
# rejected *everything*, so a same-OS worker must still form the cluster.
# =====================================================================
g_homo() {
    local name="$1" port="${IDLETOKEN_HOMO_PORT:-14311}"
    local repo; repo=$(cd "$(dirname "$0")/.." && pwd)
    command -v cc  >/dev/null 2>&1 || { skip "$name" "no C compiler on the control machine"; return; }
    command -v python3 >/dev/null 2>&1 || { skip "$name" "python3 needed for the stub worker"; return; }
    (cd "$repo" && make coord >/dev/null 2>&1) || { fail "$name" "make coord failed on the control machine"; return; }

    # This host's family, and any *other* legal one to claim.
    local self other
    case "$(uname -s)" in
        Linux)                self=1 ;;
        Darwin)               self=3 ;;
        MINGW*|MSYS*|CYGWIN*) self=2 ;;
        *)                    self=0 ;;
    esac
    other=2; [ "$self" = 2 ] && other=1

    local tmp; tmp=$(mktemp -d)
    local cpid="" m1="" m2="" m3=""
    _homo_cleanup() {
        for p in $m1 $m2 $m3 $cpid; do kill "$p" 2>/dev/null; done
        wait $m1 $m2 $m3 $cpid 2>/dev/null
        rm -rf "$tmp"
    }
    # Wait up to 10 s for a pid to exit; 1 if it is still alive (the stub blocks
    # forever on recv once accepted, so "still alive" IS the failure signal).
    _homo_wait() {
        local p="$1" i=0
        while kill -0 "$p" 2>/dev/null && [ $i -lt 100 ]; do sleep 0.1; i=$((i + 1)); done
        kill -0 "$p" 2>/dev/null && return 1 || return 0
    }

    "$repo/idletoken-coord" --bind "127.0.0.1:$port" --num-workers 2 --n-predict 0 \
        >"$tmp/coord.log" 2>&1 &
    cpid=$!
    sleep 1

    # 1. native-OS worker joins first and sets the cluster's family
    python3 "$repo/scripts/mock_worker.py" "127.0.0.1:$port" >"$tmp/m1.log" 2>&1 &
    m1=$!
    sleep 1
    if ! grep -q "^coord: worker 0 is " "$tmp/coord.log"; then
        fail "$name" "the first stub worker never joined (see $tmp/coord.log)"; _homo_cleanup; return
    fi

    # 2. a worker claiming another OS must be refused, and be TOLD why
    IDLETOKEN_MOCK_OS_FAMILY="$other" python3 "$repo/scripts/mock_worker.py" \
        "127.0.0.1:$port" >"$tmp/m2.log" 2>&1 &
    m2=$!
    # Two ways a broken check shows up: the stub blocks forever on recv (still
    # alive), or it gets an ASSIGN_PLAN. Name both — "was accepted" is a far more
    # useful red than "no refusal in the log".
    if ! _homo_wait "$m2"; then
        fail "$name" "a $other-family worker was ACCEPTED into a $self-family cluster (still connected)"
        _homo_cleanup; return
    fi
    if grep -q "plan received" "$tmp/m2.log"; then
        fail "$name" "a $other-family worker was ACCEPTED into a $self-family cluster (got ASSIGN_PLAN)"
        _homo_cleanup; return
    fi
    if ! grep -q "mixed-OS clusters" "$tmp/coord.log"; then
        fail "$name" "coordinator logged no mixed-OS refusal (see $tmp/coord.log)"; _homo_cleanup; return
    fi
    if ! grep -q "REFUSED: cluster is" "$tmp/m2.log"; then
        fail "$name" "the refused worker got no reason, only a dead socket (see $tmp/m2.log)"
        _homo_cleanup; return
    fi
    vlog "mixed-OS join refused, and the refusal reached the worker"

    # 3. positive control: same-OS worker still forms the cluster
    python3 "$repo/scripts/mock_worker.py" "127.0.0.1:$port" >"$tmp/m3.log" 2>&1 &
    m3=$!
    local i=0
    while [ $i -lt 100 ] && ! grep -q "cluster ready" "$tmp/coord.log"; do sleep 0.1; i=$((i + 1)); done
    if ! grep -q "cluster ready" "$tmp/coord.log"; then
        fail "$name" "same-OS worker did not complete the cluster — the check refuses everything"
        _homo_cleanup; return
    fi
    vlog "same-OS worker joined; cluster ready"

    # 4. The REAL worker binary, refused by the REAL coordinator. Everything
    #    above drove stubs; this is the only step that proves the shipped binary
    #    tells its user why it did not join, in the form the client's supervisor
    #    parses (JOIN_REFUSED marker + exit 2 => show the reason, do not retry;
    #    contract in include/idletoken_resource.h).
    #
    #    Staged by having the stub claim the OTHER family FIRST, so this machine's
    #    own worker becomes the odd one out. A machine that cannot pass its own
    #    hardware floor prints the same marker from the earlier gate, which is
    #    equally valid evidence -- that is why the assertion is on the marker, not
    #    on which of the two refusals produced it.
    _homo_cleanup
    tmp=$(mktemp -d); cpid=""; m1=""; m2=""; m3=""
    if [ ! -x "$repo/idletoken-worker" ]; then
        (cd "$repo" && make worker >/dev/null 2>&1) || true
    fi
    if [ ! -x "$repo/idletoken-worker" ]; then
        vlog "no idletoken-worker on this machine -- real-binary half not run"
    else
        "$repo/idletoken-coord" --bind "127.0.0.1:$port" --num-workers 2 --n-predict 0 \
            >"$tmp/coord.log" 2>&1 &
        cpid=$!
        sleep 1
        IDLETOKEN_MOCK_OS_FAMILY="$other" python3 "$repo/scripts/mock_worker.py" \
            "127.0.0.1:$port" >"$tmp/m1.log" 2>&1 &
        m1=$!
        sleep 1
        "$repo/idletoken-worker" --coordinator "127.0.0.1:$port" \
            --bind "127.0.0.1:$((port + 90))" >"$tmp/real.log" 2>&1
        local rc=$?
        if ! grep -q "JOIN_REFUSED: " "$tmp/real.log"; then
            fail "$name" "the real worker was refused but printed no JOIN_REFUSED marker — the client cannot show a reason (see $tmp/real.log)"
            _homo_cleanup; return
        fi
        if [ "$rc" != 2 ]; then
            fail "$name" "refused worker exited $rc, expected 2 (IDLETOKEN_EXIT_JOIN_REFUSED)"
            _homo_cleanup; return
        fi
        vlog "real worker refused: $(grep -o 'JOIN_REFUSED: .*' "$tmp/real.log" | head -1)"
    fi

    _homo_cleanup
    pass "$name"
}

# =====================================================================
# G_FINAL — the whole product (acceptance-criteria §128 / §5 G-FINAL): the
#      GUI-driven end-to-end flow works AND the API serves real dual-protocol
#      DSv4 replies. Composite gate: it re-checks the ladder's own RECORDS
#      (P1–P6 all PASS + G6 dual-protocol real reply) instead of re-running
#      the checks — no duplicated oracle logic, and no way to go green unless
#      every component gate went green in THIS run. The literal clean-machine
#      install walkthrough on top of this is integration-plan 4.3.
# =====================================================================
g_final() {
    local name="$1"
    if [ -n "$ONLY_GATE" ]; then
        fail "$name" "G_FINAL is composite over this run's gate records — run the full ladder (no --gate)"
        return
    fi
    local req="G6_e2e_api P1_client P2_auth P3_pairing P4_orchestration P5_settings P6_api_exposure"
    local missing="" g
    for g in $req; do
        printf '%s\n' ${RESULTS[@]+"${RESULTS[@]}"} | grep -qx "PASS $g" || missing="$missing $g"
    done
    if [ -z "$missing" ]; then
        vlog "all component gates green in this run: $req"
        pass "$name"
    else
        fail "$name" "component gates not green in this run:$missing"
    fi
}

# =====================================================================
# G_PLAT — three-way integration (integration-plan 2.2): platform gateway
#      + build/idletoken-platform-agent + a REAL idletoken-coord (mock model, --http)
#      on the coord node; one consumer message crosses the full sealed-envelope
#      chain and the gateway unseals a well-formed reply. Oracle:
#      scripts/platform_e2e_real_coord.sh (last line G_PLAT_OK). Missing deps
#      (binaries/pnpm/gateway install) surface as an explicit SKIP with the
#      reason — never a fake pass. Platform layer is INDEPENDENT of the cluster
#      goal (spec decision 11), so this gate runs after G_FINAL and does
#      not affect the exit code.
# =====================================================================
g_dspark() {
    local name="$1"
    # G-DSPARK-LOAD (docs/dspark-design.md §9): the DSpark draft module that
    # ships with the official 0731 release loads and accounts for every byte.
    #
    # Runs on the coord node because that is where the 5.6 GiB module lives.
    # An absent module is a SKIP with the reason, not a FAIL: DSpark is an
    # OPTIONAL accelerator, and a cluster without it is fully conformant.
    # This gate sits after G_FINAL for the same reason G_PLAT does — a red
    # optional accelerator must not set the FRONTIER and skip the ladder.
    # PARKED (2026-08-03, user's call): the DSpark work stays in the tree but is
    # not in use. Its sub-checks load the 80 GiB model three times, which is a
    # lot of ladder time for a feature nothing calls — so they only run when
    # asked for. Set IDLETOKEN_DSPARK_GATE=1 to exercise the parked work (do that
    # before touching it again, so bit-rot surfaces as a gate failure rather
    # than a surprise months later).
    if [ "${IDLETOKEN_DSPARK_GATE:-0}" != "1" ]; then
        skip "$name" "DSpark is parked and unused; set IDLETOKEN_DSPARK_GATE=1 to check it"
        return
    fi
    local gguf="$(dirname "$GGUF_DGX")/DeepSeek-V4-Flash-DSpark-support.gguf"
    local out
    # `cd` gets its own failure path: folding it into the -f test would report
    # a missing checkout as "module not present" and quietly SKIP a real break.
    out=$($SSH "$COORD_NODE" "cd $DGX_HOME || { echo NO_REPO; exit 0; }; \
            if [ ! -f $gguf ]; then echo NO_MODULE; exit 0; fi; \
            mkdir -p build && cc -Wall -Wextra -std=c99 -Iinclude \
              src/common/gguf.c src/common/model.c src/tools/dspark_check.c \
              -o build/dspark_check 2>&1 \
            && ./build/dspark_check $gguf 2>&1 | tail -3" 2>/dev/null | tr -d '\r')
    case "$out" in
        *NO_REPO*)
            fail "$name" "no IdleToken checkout at $DGX_HOME on $COORD_NODE" ;;
        *NO_MODULE*)
            skip "$name" "DSpark module not present on $COORD_NODE (optional accelerator)" ;;
        *DSPARK_LOAD_OK*)
            vlog "DSpark draft module: 81 tensors, 3 stages, byte accounting self-checks"
            # Second sub-check: the drafter's block attends non-causally. A
            # causal block still runs and still returns plausible numbers — it
            # just drafts badly — so this needs its own assertion rather than
            # being left for the accept rate to reveal much later.
            local nc
            nc=$($SSH "$COORD_NODE" "cd $DGX_HOME && \
                    /usr/local/cuda/bin/nvcc -O2 -Ivendor/ds4 \
                      -o build/dspark_noncausal_test src/tools/dspark_noncausal_test.c \
                      build/worker/vendor/ds4_cuda.o -lcudart -lcublas -lnvidia-ml 2>&1 \
                    && ./build/dspark_noncausal_test 2>&1 | tail -1" 2>/dev/null | tr -d '\r')
            case "$nc" in
                *DSPARK_NONCAUSAL_OK*)
                    vlog "non-causal block attention verified (rows identical / causal differs)"
                    # Third sub-check: the aux-hidden capture. Cross-checked
                    # against an independent host-side mean of cur_hc, so this
                    # catches a wrong reduction, not just a missing one.
                    local ax
                    ax=$($SSH "$COORD_NODE" "cd $DGX_HOME && \
                            /usr/local/cuda/bin/nvcc -O2 -Ivendor/ds4 -Iinclude \
                              -o build/dspark_aux_test src/tools/dspark_aux_test.c \
                              src/common/gguf.c \
                              build/worker/vendor/ds4.o build/worker/vendor/ds4_cuda.o \
                              build/worker/vendor/rax.o -lcudart -lcublas -lnvidia-ml -lm 2>&1 \
                            && ./build/dspark_aux_test $GGUF_DGX $gguf 2>&1 | tail -1" 2>/dev/null | tr -d '\r')
                    case "$ax" in
                        *DSPARK_AUX_OK*)
                            vlog "aux-hidden capture matches an independent host mean"
                            # Fourth sub-check: the markov bias. Cross-checked
                            # against a host dequant that uses a DIFFERENT GGUF
                            # parser, so a shared misreading of Q8_0 cannot make
                            # both sides agree.
                            local mk
                            mk=$($SSH "$COORD_NODE" "cd $DGX_HOME && \
                                    /usr/local/cuda/bin/nvcc -O2 -Ivendor/ds4 -Iinclude \
                                      -o build/dspark_markov_test src/tools/dspark_markov_test.c \
                                      src/common/gguf.c build/worker/vendor/ds4.o \
                                      build/worker/vendor/ds4_cuda.o build/worker/vendor/rax.o \
                                      -lcudart -lcublas -lnvidia-ml -lm 2>&1 \
                                    && ./build/dspark_markov_test $GGUF_DGX $gguf 2>&1 | tail -1" 2>/dev/null | tr -d '\r')
                            case "$mk" in
                                *DSPARK_MARKOV_OK*)
                                    vlog "markov bias matches an independent host dequant + matmul"
                                    # Fifth sub-check: the drafter actually
                                    # produces a block. Deliberately a LOW bar
                                    # (runs, ids valid, not degenerate) — draft
                                    # QUALITY is G-DSPARK-VERIFY's job, and
                                    # claiming more here would be dishonest.
                                    local dr
                                    dr=$($SSH "$COORD_NODE" "cd $DGX_HOME && \
                                            /usr/local/cuda/bin/nvcc -O2 -Ivendor/ds4 \
                                              -o build/dspark_draft_test src/tools/dspark_draft_test.c \
                                              build/worker/vendor/ds4.o build/worker/vendor/ds4_cuda.o \
                                              build/worker/vendor/rax.o -lcudart -lcublas -lnvidia-ml -lm 2>&1 \
                                            && ./build/dspark_draft_test $GGUF_DGX $gguf 64 2>&1 | tail -1" 2>/dev/null | tr -d '\r')
                                    case "$dr" in
                                        *DSPARK_DRAFT_OK*)
                                            vlog "drafter produced a non-degenerate block"
                                            pass "$name" ;;
                                        "")
                                            fail "$name" "could not build or run dspark_draft_test on $COORD_NODE" ;;
                                        *)
                                            fail "$name" "draft block check failed ($dr)" ;;
                                    esac ;;
                                "")
                                    fail "$name" "could not build or run dspark_markov_test on $COORD_NODE" ;;
                                *)
                                    fail "$name" "markov bias check failed ($mk)" ;;
                            esac ;;
                        "")
                            fail "$name" "could not build or run dspark_aux_test on $COORD_NODE" ;;
                        *)
                            fail "$name" "aux-hidden capture check failed ($ax)" ;;
                    esac ;;
                "")
                    fail "$name" "could not build or run dspark_noncausal_test on $COORD_NODE" ;;
                *)
                    fail "$name" "non-causal block attention check failed ($nc)" ;;
            esac ;;
        "")
            fail "$name" "could not run dspark_check on $COORD_NODE (ssh/build?)" ;;
        *)
            fail "$name" "DSpark module failed to validate ($out)" ;;
    esac
}

# =====================================================================
# G_SCHED -- the scheduler platform (the seven assertions of
# docs/scheduler-design.md §9 plus the two added after E3).
#
# These assertions had always passed, but until now they **lived only in jest** --
# the ladder could not see the scheduler platform, so no gate watched over
# "scheduling works". Like G_PLAT it belongs to the platform business layer: it
# reports in the ladder but **takes no part in the G_FINAL exit decision** (spec
# decision 11: the platform layer is independent of the cluster).
#
# It runs **locally on the control machine** (unlike G_PLAT, which ssh's to the
# DGX): it needs only the gateway's node_modules, no real cluster and no GPU.
# Missing dependencies -> SKIP.
# =====================================================================
g_sched() {
    local name="$1"
    local out
    out=$(bash scripts/scheduler_gate.sh 2>&1 | grep -E "^G_SCHED_(OK|FAIL|SKIP)" | tail -1)
    case "$out" in
        G_SCHED_OK)
            vlog "scheduler gates green (breaker/blacklist/failure classification/no stampede/affinity/anti-starvation/honest congestion + agent concurrency + cross-instance slots)"
            pass "$name" ;;
        G_SCHED_SKIP*)
            skip "$name" "${out#G_SCHED_SKIP: }" ;;
        "")
            fail "$name" "scripts/scheduler_gate.sh reached no conclusion (timeout? see \$TMPDIR/idletoken-sched-gate.log)" ;;
        *)
            fail "$name" "a scheduler assertion failed (details in the output of scripts/scheduler_gate.sh)" ;;
    esac
}

g_plat() {
    local name="$1"
    local out
    # Grep the explicit result markers, NOT the last combined line: the
    # script's EXIT trap prints a "logs kept in ..." note to stderr AFTER
    # G_PLAT_OK, so `tail -1` of 2>&1 reads the cleanup note instead.
    out=$($SSH "$COORD_NODE" "cd $DGX_HOME && timeout 600 bash scripts/platform_e2e_real_coord.sh 2>&1 | grep -E '^G_PLAT_(OK|FAIL|SKIP)' | tail -1" 2>/dev/null | tr -d '\r')
    case "$out" in
        G_PLAT_OK)
            vlog "sealed chain platform->agent->real coord->platform green on $COORD_NODE"
            pass "$name" ;;
        G_PLAT_SKIP*)
            skip "$name" "dependency missing on $COORD_NODE: ${out#G_PLAT_SKIP: }" ;;
        "")
            fail "$name" "could not run scripts/platform_e2e_real_coord.sh on $COORD_NODE (ssh/timeout?)" ;;
        *)
            fail "$name" "three-way sealed e2e did not pass (got '$out'; logs kept on $COORD_NODE, see script output)" ;;
    esac
}

# =====================================================================
echo "======================================================"
echo " IdleToken acceptance ladder   $(date '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo)"
echo "======================================================"
# Pre-run scrub: a leaked engine/cluster from an earlier run (or a live demo)
# holds tens of GB of model memory and poisons this run's model loads — seen
# as a flaky G4. Kill leftovers on the coord node and let memory settle.
# ([b]racket patterns keep pkill from matching this ssh command itself.)
#
# 2026-08-08: this line used to hardcode `homeai-*`, while after the 08-04 rename
# the binaries are called `idletoken-*` -- so **from the day of the rename it
# matched no process at all**, "succeeding" every round while doing nothing.
# Four days later a 7-hour experiment leftover holding 80 GB of unified memory
# surfaced it: G4 went red, which is exactly what the scrub was supposed to
# prevent. `pkill` returns 1 when it matches nothing, that was swallowed by
# `; true`, and so it could not fail -- **another check that cannot fail and
# therefore does not exist**.
#
# The names now come from $ENGINE_BINS in one place, so a rename touches one line;
# and the scrub **asserts that it worked**: anything left over exits on the spot
# rather than carrying a poisoned machine through a dozen more gates.
# With no coordinator configured (running local gates only) there is nothing to
# scrub, so it is skipped -- otherwise it would print the alarming false warning
# "unkillable engine processes on the coordinator" when the truth is that there is
# no node at all.
[ -z "$COORD_NODE" ] || $SSH "$COORD_NODE" '
    pat="[i]dletoken-(coord|worker|platform-agent|client)"
    pkill -f "$pat" 2>/dev/null; sleep 2
    pkill -9 -f "$pat" 2>/dev/null; sleep 1
    left=$(pgrep -f "$pat" 2>/dev/null | wc -l)
    [ "$left" -eq 0 ] || { echo "SCRUB_FAIL: $left" >&2; exit 1; }
' >/dev/null 2>&1 \
    || echo "  !! pre-run scrub failed: unkillable engine processes on the coordinator. G4/G5 will likely go red this round because the memory is occupied."
gate G0_ssh_mesh       g0_ssh_mesh
gate G1_build          g1_build
gate G2_probe          g2_probe
gate G_HW              g_hw
gate G_ADVISE          g_advise
gate G_FETCH           g_fetch
gate G_TOPO            g_topo
gate G3_package        g3_package
gate G_RELEASE         g_release
gate G4_single_infer   g4_single_infer
gate G5_cluster_ready  g5_cluster_ready
gate G6_e2e_api        g6_e2e
gate G_PRIV            g_priv
gate G_PAIR            g_pair
gate G_MODEL           g_model
gate P1_client         p1_client
gate P2_auth           p2_auth
gate P3_pairing        p3_pairing
gate P4_orchestration  p4_orchestration
gate P5_settings       p5_settings
gate P6_api_exposure   p6_api_exposure
# Cluster homogeneity: a hard-constraint invariant that needs no cluster node.
# gate_always, and before G_FINAL — see the helper's comment.
gate_always G_HOMO     g_homo
gate G_FINAL           g_final
# Both of these run AFTER G_FINAL and are independent of it: G_PLAT is the
# platform business layer (spec decision 11), G_DSPARK is an optional
# accelerator. Neither may set the FRONTIER for the product ladder.
gate G_DSPARK          g_dspark
gate G_PLAT            g_plat
# G_SCHED needs only the gateway dependencies on the control machine and touches
# no cluster node -> gate_local (see above).
gate_local G_SCHED     g_sched
echo "------------------------------------------------------"
if [ -n "$ONLY_GATE" ]; then
    # Single-gate mode: report just that gate; no ladder-wide claims.
    if [ -z "$FRONTIER" ]; then echo " GATE $ONLY_GATE: PASS"; exit 0
    else echo " GATE $ONLY_GATE: FAIL"; exit 1; fi
fi
if [ -z "$FRONTIER" ]; then
    echo -e " \033[32mALL GATES PASS — project goal met.\033[0m"
    echo "FRONTIER: none"
    exit 0
fi
echo " FRONTIER: $FRONTIER   <- make this gate pass next"
# §8 exit contract: 0 iff G_FINAL (the cluster product goal) passed. G_PLAT is
# the platform business layer — reported above, but independent of this exit
# (spec decision 11). A red G_PLAT with a green G_FINAL still exits 0.
if printf '%s\n' ${RESULTS[@]+"${RESULTS[@]}"} | grep -qx "PASS G_FINAL"; then
    echo " note: G_FINAL passed — remaining red is platform-layer only (exit 0 per §8)"
    exit 0
fi
exit 1
