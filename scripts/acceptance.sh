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
# Node aliases resolve via ~/.ssh/config (configured in scripts/testbed.env;
# template: testbed.env.example). Off-LAN runs can override the route without
# touching checks:
#   IDLETOKEN_COORD_NODE=<coord-alias> IDLETOKEN_API_HOST=192.168.1.x scripts/acceptance.sh --gate G4
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
LOCAL_ONLY_GATES=" G_MODEL G_SCHED G_VERSION G_MAC_SMOKE G_PRIV7 G_PPL G_INTEGRITY "
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
# --- temporarily unavailable nodes -----------------------------------------
# IDLETOKEN_SKIP_NODES names machines that are configured but OUT OF SERVICE
# for this run (powered off, lent out, or being driven by another agent whose
# work an ssh-in would corrupt). They are removed from the node set LOUDLY:
# gates that iterate the worker set see the reduced set, and gates whose whole
# subject is a skipped node SKIP with the reason instead of failing after a
# two-minute retry loop. This is the honest middle ground between "fail the
# whole ladder at G0 because one laptop is off" and "quietly test fewer
# machines while reporting a complete run" (2026-08-14, WS-F).
read -r -a SKIP_NODES <<< "${IDLETOKEN_SKIP_NODES:-}"
node_skipped() {
    local s
    for s in ${SKIP_NODES[@]+"${SKIP_NODES[@]}"}; do
        [ "$s" = "$1" ] && return 0
    done
    return 1
}
if [ -n "$COORD_NODE" ] && node_skipped "$COORD_NODE"; then
    echo "acceptance.sh: IDLETOKEN_SKIP_NODES contains the coordinator ($COORD_NODE) — nothing can run without it." >&2
    exit 2
fi
if [ ${#SKIP_NODES[@]} -gt 0 ]; then
    _kept=()
    for _n in ${WORKER_NODES[@]+"${WORKER_NODES[@]}"}; do
        if node_skipped "$_n"; then
            echo "note: node $_n is OUT OF SERVICE this run (IDLETOKEN_SKIP_NODES) — gates that need it will SKIP, not fail" >&2
        else
            _kept+=("$_n")
        fi
    done
    WORKER_NODES=(${_kept[@]+"${_kept[@]}"})
    unset _kept
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
# The repository checkout on the coordinator node. Comes from testbed.env
# (IDLETOKEN_COORD_HOME) like the other machine facts; the tilde stays literal
# here and expands on the remote shell.
DGX_HOME="${IDLETOKEN_COORD_HOME:-~/work/IdleToken}"
# Coord API host: derive from ssh config so the ladder follows whatever address
# actually reaches the coordinator (LAN at home, Tailscale off-LAN). Hardcoding
# an IP here rotted once already (2026-07-22 DHCP reshuffle made .101 = win-a).
API_HOST="${IDLETOKEN_API_HOST:-$(ssh -G "$COORD_NODE" 2>/dev/null | awk '/^hostname /{print $2; exit}')}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${IDLETOKEN_API_PORT:-8000}"
# Real large-model GGUF on the coordinator node (P6 real-reply gate). Where the
# weights live is a property of YOUR machine, so it comes from testbed.env
# (IDLETOKEN_GGUF; template: testbed.env.example). The path is resolved on the
# remote shell, so $HOME expands there. The neutral default below only makes an
# unconfigured run fail with "GGUF not found" at a visible path instead of
# pointing at one maintainer's directory layout.
GGUF_DGX="${IDLETOKEN_GGUF:-\$HOME/models/your-model.gguf}"

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
    # The gate's subject is the WINDOWS build (the coord-node half is a two-file
    # ls). With every Windows worker out of service, running the remainder and
    # reporting PASS would certify a build nobody checked.
    if [ ${#WORKER_NODES[@]} -eq 0 ]; then
        skip "$name" "all Windows worker nodes are out of service this run (IDLETOKEN_SKIP_NODES) — the Windows build was not checked"
        return
    fi
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
        # build_worker_win.bat, NOT build_ds4x_win.bat. The ds4 line was shelved
        # on 2026-08-16 and its build scripts were taken down, but this gate went
        # on calling one -- a spot that shelving missed. It mattered: the ds4x
        # script compiles src/ds4x + vendor/ds4 and links -lds4cuda -lds4xcuda,
        # so the gate was building a worker the product no longer ships, and
        # requiring a CUDA Toolkit on the build node that a real user does not
        # need. It also carries the hand-maintained object list whose twin was
        # fixed in build_worker_win.bat (be895a9): it never compiles modelsize.c
        # at all, so it has failed on `undefined reference to
        # idletoken_model_size_resolve` ever since advise.c started calling it.
        # Repairing that script would have kept a retired path alive; the path
        # the product ships is the one worth gating.
        #
        # This script also prints a REAL success marker. The ds4x one echoed
        # LINK_DONE unconditionally -- the false green the checks below exist to
        # work around. They stay as belt and braces, but WORKER_BUILD_OK is the
        # primary judgement now, and the script runs `--help` before claiming it.
        local blog
        blog=$($SSH "$bn" "cd /d ${bhome//\//\\} && call build_worker_win.bat" 2>&1 | tr -d '\r')
        if ! printf '%s\n' "$blog" | grep -q "WORKER_BUILD_OK"; then
            fail "$name" "$bn worker rebuild failed: $(printf '%s\n' "$blog" | grep -E 'WORKER_BUILD_FAIL' | head -1) (see $bhome/worker_build.log)"; return
        fi
        after=$($SSH "$bn" "powershell -NoProfile -Command \"cd $bhome; if(Test-Path idletoken-worker.exe){(Get-Item idletoken-worker.exe).LastWriteTime.Ticks}else{0}\"" 2>/dev/null | tr -d '\r ')
        # Belt and braces behind WORKER_BUILD_OK: a newer exe, and no compiler
        # errors in the log. These were the ONLY judgement while the gate drove
        # the ds4x script, which echoed LINK_DONE whether or not the link
        # succeeded -- a log that reads like success is exactly how a broken
        # build hid for days. Keeping them costs one ssh round trip.
        errs=$($SSH "$bn" "powershell -NoProfile -Command \"cd $bhome; (Select-String -Path worker_build.log -Pattern 'undefined reference','error:' -ErrorAction SilentlyContinue | Measure-Object).Count\"" 2>/dev/null | tr -d '\r ')
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
            fail "$name" "$bn rebuild produced no newer idletoken-worker.exe (see $bhome/worker_build.log)"; return
        fi
        if [ "${errs:-1}" != "0" ]; then
            fail "$name" "$bn rebuild log has $errs error/undefined-reference line(s) (see $bhome/worker_build.log)"; return
        fi
        vlog "$bn rebuilt the worker from the synced tree (no link errors)"
    else
        vlog "rebuild subcheck skipped: build node $bn not in this run's node set"
    fi

    # Every Windows worker: native exe + runnable. This used to check win-a only
    # (hardcoded), so a second Windows worker's binaries were never build-checked
    # at all — and the product's whole point is any N machines.
    #
    # ds4cuda.dll is NOT required any more (2026-08-16 shelving). The worker's
    # link line is `-lwinpthread -lws2_32 -lbcrypt` and nothing else; ds4x is not
    # compiled in, ds4_stub.c stands in for it. The DLL still happens to sit on
    # both boxes from old builds, so this check was passing on a leftover file
    # rather than on a real dependency — and would have gone red on a clean
    # machine, which is what a new user's node is.
    local n home w help
    for n in "${WORKER_NODES[@]}"; do
        home=$(win_home "$n")
        w=$($SSH "$n" "powershell -NoProfile -Command \"cd $home; ''+(Test-Path idletoken-worker.exe)\"" 2>/dev/null | tr -d '\r ')
        if [ "$w" != "True" ]; then fail "$name" "$n missing idletoken-worker.exe (got '$w')"; return; fi
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
    [ ${#WORKER_NODES[@]} -eq 0 ] && vlog "no Windows workers in this run's node set — only the coordinator was probed"
    # Windows workers: need GPU name + cc>0 + vram_total>0 (the current gap)
    for n in ${WORKER_NODES[@]+"${WORKER_NODES[@]}"}; do
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
# G_TOPO — any-combination coverage. The sweep itself runs for hours, so the
#          ladder does not re-run it: this gate AUDITS the recorded matrix.
#          No matrix on disk is an honest SKIP, not a pass.
#
# 2026-08-15 (llama.cpp pivot, WS-F migration this gate was missing from):
# the ds4-line sweep (scripts/topology_matrix.sh -> build/topology-matrix.tsv)
# is PARKED. Its oracle was per-cell token-id equality (`ids_sha`), retired with
# decision #10, and its rules encode two invariants that no longer hold: "a
# cluster must be homogeneous" and "DSv4 is the only runnable cluster model".
# Running it would audit a product we no longer ship. `IDLETOKEN_DS4_TOPO_GATE=1`
# still runs the parked assertions so the frozen line cannot rot unnoticed.
#
# The live gate audits the llama.cpp matrix (results/matrix-llamacpp-*.jsonl,
# newest wins). What it demands is the v2 product claim, not a cell count:
#   - no FAIL cell;
#   - a REAL single-machine cell on EACH supported compute platform
#     (linux / macos / windows) — "one binary, three platforms";
#   - a REAL multi-machine cell — a single machine proves no topology;
#   - a REAL cross-OS cell — heterogeneous clusters are the v2 claim (§1.4),
#     and an unproven claim is the thing this gate exists to catch;
#   - every PASS cell carries coherent output AND a decode number: a cell with
#     no measurement is a note, not evidence.
# Cells recorded PENDING are named in the failure so the frontier says which
# machine to go and measure, rather than "coverage insufficient".
# =====================================================================
TOPO_RESULTS="${IDLETOKEN_TOPO_RESULTS:-$PWD/build/topology-matrix.tsv}"
TOPO_MATRIX="${IDLETOKEN_TOPO_MATRIX:-}"

g_topo() {
    local name="$1"
    if [ "${IDLETOKEN_DS4_TOPO_GATE:-0}" != "1" ]; then
        g_topo_llamacpp "$name"
        return
    fi
    vlog "IDLETOKEN_DS4_TOPO_GATE=1 — running the parked ds4-line matrix audit"
    if [ ! -s "$TOPO_RESULTS" ] || [ "$(tail -n +2 "$TOPO_RESULTS" | wc -l | tr -d ' ')" = 0 ]; then
        skip "$name" "no ds4-line topology matrix recorded — run scripts/topology_matrix.sh (hours; resumable)"
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

# The live (llama.cpp) half of G_TOPO — see the block comment above for what it
# demands and why. Pure audit of a recorded file: it starts no cluster, so it
# stays cheap enough to run every round.
g_topo_llamacpp() {
    local name="$1" files
    if [ -n "$TOPO_MATRIX" ]; then
        files="$TOPO_MATRIX"
    else
        files=$(ls -1 "$REPO_ROOT"/results/matrix-llamacpp-*.jsonl 2>/dev/null | sort)
    fi
    if [ -z "$files" ]; then
        skip "$name" "no llama.cpp topology matrix recorded (results/matrix-llamacpp-*.jsonl) — measure the cells and record them"
        return
    fi
    local out
    # shellcheck disable=SC2086
    out=$(python3 - $files <<'PY' 2>&1
import json, sys

# EVERY recorded matrix file, oldest first, later record for a cell winning.
# Reading only the newest file would silently drop every cell measured before
# today the moment a new dated file appears -- the audit would then "not cover"
# platforms that were in fact proven months ago.
paths = sys.argv[1:]
by_cell = {}
for path in paths:
    for n, line in enumerate(open(path), 1):
        line = line.strip()
        if not line:
            continue
        try:
            c = json.loads(line)
        except Exception as e:
            print("BAD line %d of %s: %s" % (n, path, e)); raise SystemExit(1)
        by_cell[c.get("cell")] = c
cells = list(by_cell.values())
if not cells:
    print("BAD no cells in: " + " ".join(paths)); raise SystemExit(1)

def is_real(c):   return c.get("kind") == "real"
def passed(c):    return c.get("result") == "PASS"
def oses(c):      return [c.get("coord_os", "")] + list(c.get("worker_os", []))

bad = [c["cell"] for c in cells if c.get("result") == "FAIL"]
if bad:
    print("FAIL failing cell(s): " + " ".join(bad)); raise SystemExit(0)

# A PASS with no measurement is a claim, not evidence.
thin = [c["cell"] for c in cells
        if passed(c) and is_real(c)
        and not (c.get("coherent") is True and (c.get("decode_tps") or 0) > 0)]
if thin:
    print("FAIL PASS cell(s) with no coherent output or no decode number: "
          + " ".join(thin)); raise SystemExit(0)

pending = [c["cell"] for c in cells if c.get("result") == "PENDING"]
missing = []

for want in ("linux", "macos", "windows"):
    if not any(passed(c) and is_real(c) and c.get("topology") == "single"
               and c.get("coord_os") == want for c in cells):
        missing.append("single-machine/" + want)
if not any(passed(c) and is_real(c) and c.get("topology") == "cluster" for c in cells):
    missing.append("multi-machine")
if not any(passed(c) and is_real(c) and c.get("topology") == "cluster"
           and len({o for o in oses(c) if o}) > 1 for c in cells):
    missing.append("cross-OS cluster")
# Inherited from the parked ds4 gate, and still a product claim: the
# all-Windows household needs a Windows machine that can BE the coordinator.
if not any(passed(c) and is_real(c) and c.get("topology") == "cluster"
           and c.get("coord_os") == "windows" for c in cells):
    missing.append("Windows-coordinated cluster")

if missing:
    msg = "FAIL matrix does not cover: " + " ".join(missing)
    if pending:
        msg += " — recorded PENDING: " + " ".join(pending)
    print(msg); raise SystemExit(0)

npass = sum(1 for c in cells if passed(c))
nreal = sum(1 for c in cells if passed(c) and is_real(c))
print("OK %d passing cell(s), %d on real hardware, %d pending; %d matrix file(s)"
      % (npass, nreal, len(pending), len(paths)))
PY
)
    case "$out" in
        OK*)   vlog "${out#OK }"; pass "$name" ;;
        FAIL*) fail "$name" "${out#FAIL }" ;;
        *)     fail "$name" "could not audit $f: $out" ;;
    esac
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
    # Both Windows halves run on the build node; with it out of service the
    # coord half alone cannot honestly certify E3 (the decision under test —
    # "no cublas in the bundle, driver-only probe" — is a Windows fact).
    if node_skipped "$WIN_BUILD_NODE" || [ ${#WORKER_NODES[@]} -eq 0 ]; then
        skip "$name" "Windows build node $WIN_BUILD_NODE is out of service this run — the driver-only bundle was not checked"
        return
    fi
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

    # The bundle ships binaries containing vendored ds4 (MIT) and rax (BSD-3);
    # both licences require the notice to travel with a binary distribution, as
    # does Apache-2.0 4(d) for our NOTICE. Asserted HERE and not only inside
    # package_dist.sh, because what ships is the folder, not the script: this
    # gate is the thing that looks at the artifact.
    local lic
    lic=$($SSH "$COORD_NODE" "cd $DGX_HOME/dist && for f in LICENSE NOTICE licenses/ds4-MIT.txt licenses/rax-BSD-3-Clause.txt; do test -s \"\$f\" || echo MISSING:\$f; done" 2>/dev/null | tr -d '\r')
    [ -z "$lic" ] || { fail "$name" "dist/ ships binaries without their licence texts ($lic)"; return; }
    vlog "coord-node dist/ carries LICENSE, NOTICE and the vendored licences"

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

    # Same licence assertion on the Windows bundle (see the Linux half above).
    local licw
    licw=$($SSH "$WIN_BUILD_NODE" "powershell -NoProfile -Command \"cd $WIN_HOME/dist; @('LICENSE','NOTICE','licenses/ds4-MIT.txt','licenses/rax-BSD-3-Clause.txt') | ForEach-Object { if (-not (Test-Path \$_) -or (Get-Item \$_).Length -eq 0) { 'MISSING:' + \$_ } }\"" 2>/dev/null | tr -d '\r')
    [ -z "$licw" ] || { fail "$name" "$WIN_BUILD_NODE dist\ ships binaries without their licence texts ($licw)"; return; }
    vlog "$WIN_BUILD_NODE dist\ carries LICENSE, NOTICE and the vendored licences"

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
    if ! printf '%s\n' ${WORKER_NODES[@]+"${WORKER_NODES[@]}"} | grep -qx "$bn"; then
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

    local before after out key_path sign_bin sign_tmp updater_zip updater_sig zip_count
    before=$($SSH "$bn" "powershell -NoProfile -Command \"if(Test-Path '$nsis'){(Get-ChildItem '$nsis' -Filter *.exe | Sort-Object LastWriteTime | Select-Object -Last 1).LastWriteTime.Ticks}else{0}\"" 2>/dev/null | tr -d '\r ')
    # The only trusted updater private key lives on the control machine. Build
    # the installer and updater zip on Windows, then sign the zip here. The key
    # never reaches the build node, its filesystem, environment, or process
    # list. This also supports the existing updater key's intentionally empty
    # password without depending on Windows' treatment of empty env variables.
    key_path="${IDLETOKEN_UPDATER_KEY:-$HOME/.idletoken/updater.key}"
    [ -r "$key_path" ] || { fail "$name" "updater signing key missing at $key_path — restore the existing key; do not generate a replacement"; return; }
    sign_bin="$REPO_ROOT/client/node_modules/.bin/tauri"
    [ -x "$sign_bin" ] || { fail "$name" "local Tauri signer missing at $sign_bin"; return; }
    out=$($SSH "$bn" "cd /d ${bhome//\//\\} && set \"IDLETOKEN_DEFER_UPDATER_SIGNING=1\" && scripts\\build_client_release.bat" \
      2>/dev/null | tr -d '\r' | tail -1)
    case "$out" in
        CLIENT_RELEASE_DEFERRED_SIGNING) ;;
        *) fail "$name" "$bn could not produce an installer (last line '$out'; see the tauri output under client\\src-tauri)"; return ;;
    esac
    after=$($SSH "$bn" "powershell -NoProfile -Command \"if(Test-Path '$nsis'){(Get-ChildItem '$nsis' -Filter *.exe | Sort-Object LastWriteTime | Select-Object -Last 1).LastWriteTime.Ticks}else{0}\"" 2>/dev/null | tr -d '\r ')
    # A green last line alone would let the PREVIOUS installer stand in for this
    # one — the same way dist\ certified a three-week-old binary (see G3 (3)).
    if [ "${after:-0}" -le "${before:-0}" ]; then
        fail "$name" "CLIENT_RELEASE_DEFERRED_SIGNING but the installer timestamp did not advance -- this certifies the previous artifact"; return
    fi

    # --- what is actually INSIDE the installer ----------------------------
    # A fresh timestamp says the build ran, not what it packed. Two ways that
    # goes wrong, both seen on 2026-08-14: the v2 engine (idletoken-server +
    # idletoken-rpc-server) is what makes an installed client able to infer at all,
    # and tauri.windows.conf.json REPLACES `bundle.resources` from
    # tauri.conf.json rather than adding to it — so the Windows installer
    # shipped with no LICENSE/NOTICE while every other platform had them.
    # NSIS archives list with 7-Zip, so ask the artifact itself.
    local listing missing f
    listing=$($SSH "$bn" "powershell -NoProfile -Command \"\$z='C:\\Program Files\\7-Zip\\7z.exe'; if(-not (Test-Path \$z)){exit 3}; \$i=Get-ChildItem '$nsis' -Filter *.exe | Sort-Object LastWriteTime -Descending | Select-Object -First 1; & \$z l \$i.FullName\"" 2>/dev/null | tr -d '\r')
    if [ -z "$listing" ]; then
        fail "$name" "could not list the installer on $bn (is 7-Zip installed at C:\\Program Files\\7-Zip\\7z.exe? the gate refuses to certify a package it cannot open)"; return
    fi
    missing=""
    # The engine halves first: idletoken-server serves, idletoken-rpc-server is what a
    # worker node runs. Shipping one without the other gives an installed
    # client a cluster mode it cannot actually join.
    for f in idletoken-client.exe idletoken-coord.exe idletoken-worker.exe \
             idletoken-server.exe idletoken-rpc-server.exe \
             LICENSE.txt NOTICE.txt ds4-MIT.txt rax-BSD-3-Clause.txt llamacpp-MIT.txt; do
        printf '%s\n' "$listing" | grep -qF "$f" || missing="$missing $f"
    done
    [ -z "$missing" ] || { fail "$name" "the installer $bn just built is missing:$missing"; return; }
    vlog "installer carries both engine binaries and every licence text"

    sign_tmp=$(mktemp -d "${TMPDIR:-/tmp}/idletoken-updater-sign.XXXXXX") || {
        fail "$name" "could not create a local updater signing directory"; return;
    }
    if ! scp -q "${bn}:${nsis}/*.nsis.zip" "$sign_tmp/" 2>/dev/null; then
        rm -rf "$sign_tmp"
        fail "$name" "failed to fetch the updater archive from $bn"; return
    fi
    zip_count=$(find "$sign_tmp" -maxdepth 1 -type f -name '*.nsis.zip' | wc -l | tr -d ' ')
    if [ "$zip_count" != "1" ]; then
        rm -rf "$sign_tmp"
        fail "$name" "expected exactly one updater archive from $bn, found $zip_count"; return
    fi
    updater_zip=$(find "$sign_tmp" -maxdepth 1 -type f -name '*.nsis.zip' -print)
    if ! "$sign_bin" signer sign -f "$key_path" -p "${TAURI_SIGNING_PRIVATE_KEY_PASSWORD:-}" "$updater_zip" >/dev/null 2>&1; then
        rm -rf "$sign_tmp"
        fail "$name" "local updater signing failed with the trusted key"; return
    fi
    updater_sig="$updater_zip.sig"
    if [ ! -s "$updater_sig" ] || ! scp -q "$updater_sig" "${bn}:${nsis}/$(basename "$updater_sig")" 2>/dev/null; then
        rm -rf "$sign_tmp"
        fail "$name" "failed to return the updater signature to $bn"; return
    fi
    if ! $SSH "$bn" "powershell -NoProfile -Command \"if(-not (Test-Path '$nsis/$(basename "$updater_sig")' -PathType Leaf)){exit 1}; if((Get-Item '$nsis/$(basename "$updater_sig")').Length -le 0){exit 2}\"" >/dev/null 2>&1; then
        rm -rf "$sign_tmp"
        fail "$name" "the updater signature is missing or empty on $bn"; return
    fi
    rm -rf "$sign_tmp"
    vlog "$bn rebuilt the installer from the current tree; updater archive signed on the control machine"
    pass "$name"
}

# =====================================================================
# G4 — Single-node inference: DGX coord+worker load GGUF, API yields a token
# =====================================================================
g4_single_infer() {
    local name="$1"
    # RETIRED (2026-08-14, llama.cpp pivot, decision #10 in
    # docs/v2-rebuild-plan-2026-08.md): this gate's oracle
    # (scripts/run_single_infer.sh) judges by EXACT greedy token-id equality
    # against the official single-node ds4 run. Measured the same day: two
    # correct engines on the same GGUF, same template, greedy, agree for 3
    # tokens and diverge at the 4th — token-for-token equality is not a valid
    # correctness oracle across engine changes, and the engine it compares
    # against (ds4) is itself frozen. Distribution-level replacement: G_PPL
    # (perplexity band, scripts/ppl_gate.sh); single-machine serving is
    # covered live by G_MAC_SMOKE and the llamacpp matrix
    # (results/matrix-llamacpp-*.jsonl).
    # The gate code stays runnable for the parked ds4 line — same contract as
    # G_DSPARK: set IDLETOKEN_DS4_TOKEN_GATE=1 to exercise it.
    if [ "${IDLETOKEN_DS4_TOKEN_GATE:-0}" != "1" ]; then
        skip "$name" "retired 2026-08-14 (exact-token oracle, decision #10; see G_PPL) — set IDLETOKEN_DS4_TOKEN_GATE=1 to run the parked ds4 check"
        return
    fi
    local out; out=$($SSH "$COORD_NODE" "cd $DGX_HOME && test -x scripts/run_single_infer.sh && ./scripts/run_single_infer.sh 2>&1 | tail -1 || echo NO_HELPER" 2>/dev/null | tr -d '\r')
    [ "$out" = "SINGLE_INFER_OK" ] && pass "$name" || fail "$name" "single-node inference not proven (got '$out')"
}

# =====================================================================
# G5 — Cluster ready: coord + >=1 real worker handshake -> cluster_ready
# =====================================================================
g5_cluster_ready() {
    local name="$1"
    # PORTED 2026-08-20 (T15). The old oracle was scripts/run_cluster.sh
    # --check-ready: coord + N workers over the INFER_* wire, judged on 43
    # contiguous ds4 layer ranges. Both the wire and the engine are retired, so
    # NO build of the current tree could pass this — and 15 gates skip behind
    # it, which is how the multi-machine half of the ladder lost every gate it
    # had (the oracle-single-machine-bias review's main finding).
    #
    # The replacement runs on the CONTROL machine and drives real nodes over
    # ssh, rather than running on the coordinator: the llama.cpp cluster is
    # coord + rpc-supervisor workers on separate boxes, so the thing under test
    # no longer fits inside one host.
    #
    # SMALL model on purpose. This gate asks "does a real cluster FORM" —
    # pairing, PSK, engine ready, a tensor split across ≥2 nodes. Whether the
    # answer is any good is G6's question, and G6 pays the 80 GiB for it. A
    # readiness check has no business holding that memory: leaving it resident
    # is how G6 once failed with an empty error, because a SECOND cluster then
    # would not fit in 119 GB and something got OOM-killed silently.
    if [ ${#WORKER_NODES[@]} -eq 0 ]; then
        skip "$name" "no worker node in service this run — 'a cluster forms' has no single-machine form"
        return
    fi
    local w="${IDLETOKEN_G5_WORKER:-${WORKER_NODES[0]}}"
    local model="${IDLETOKEN_G5_MODEL:-${IDLETOKEN_SMOKE_MODEL_ID:-qwen3.5-0.8b}}"
    local gg="${IDLETOKEN_G5_GGUF:-}"
    # Reuse the smoke model's path on the coord node when one is configured.
    [ -n "$gg" ] || gg=$(coord_small_gguf 2>/dev/null) || gg=""
    local out
    out=$(bash "$REPO_ROOT/scripts/run_cluster_llamacpp.sh" --check-ready \
            --tag g5 --api-port "${IDLETOKEN_G5_PORT:-18531}" \
            --coord "$COORD_NODE" --worker "$w" --model "$model" \
            ${gg:+--gguf "$gg"} --ready-wait "${IDLETOKEN_G5_READY_S:-420}" 2>&1 | tail -1)
    bash "$REPO_ROOT/scripts/run_cluster_llamacpp.sh" --stop --tag g5 >/dev/null 2>&1
    case "$out" in
        CLUSTER_READY) vlog "coord $COORD_NODE + worker $w formed a real llama.cpp cluster ($model)"; pass "$name" ;;
        *) fail "$name" "cluster did not reach ready: ${out#CLUSTER_FAIL: }" ;;
    esac
}

# =====================================================================
# G6 — End to end: real API call returns a coherent DSv4 response (GOAL)
# =====================================================================
g6_e2e() {
    local name="$1" started=0
    # PORTED 2026-08-20 (T15) from scripts/run_cluster.sh, whose INFER_* wire and
    # ds4 engine are retired -- no current build could reach this gate at all.
    # The assertions are unchanged (200 + on-topic reply + non-zero usage, both
    # protocol faces); only the vehicle moved to the llama.cpp cluster path.
    #
    # MODEL: DeepSeek-V4-Flash, the one model on this testbed that genuinely
    # needs more than one machine. The small curated models are the vehicle for
    # P3-P6 (orchestration and API exposure, where load time is dead weight);
    # G6 is the end-to-end product claim, so it drives the big one. Weights are
    # already staged on both testbed nodes from T14.
    #
    # TIMEOUTS: the ready window is 900 s, not the old 240 s. Basis:
    # matrix_cell_cluster_llamacpp.sh measured 306 s to ready for this model at
    # a 68/32 split and notes it grows as the coordinator's share does; T14's
    # regression ran at 88.8/11.2, a much larger local share, so its load was
    # longer still. 900 s is ~3x the measured datum. Requests get 300 s.
    # These are TIMEOUTS, not sleeps: a generous value costs nothing when the
    # model loads quickly and only ever bites on a real failure. The opposite
    # mistake is on record here -- P1's budget was called flaky and grown twice
    # before anyone measured that the cause was a dead desktop session.
    # LOOPBACK, not $API_HOST. The requests below execute ON the coordinator
    # (hard constraint #5 forces --api-bind to 127.0.0.1), and $API_HOST
    # resolves to this testbed's ssh hostname for that node -- a Tailscale
    # address. Aiming there produced an empty body and the gate reported
    # "not json", which reads as a broken engine rather than a wrong address.
    local api_base="http://127.0.0.1:$API_PORT"
    local g6_model="${IDLETOKEN_G6_MODEL:-deepseek-v4-flash}"
    if ! curl -s -m 3 "$api_base/health" 2>/dev/null | grep -q '"status":"ok"'; then
        if [ ${#WORKER_NODES[@]} -eq 0 ]; then
            skip "$name" "no worker node in service this run — G6 is the CLUSTER end-to-end claim and has no single-machine form"
            return
        fi
        vlog "no live API; starting a real cluster via run_cluster_llamacpp.sh"
        # Free the memory first: anything an earlier gate left behind still
        # holds its 80 GB, and the load then dies with nothing to go on.
        $SSH "$COORD_NODE" "pkill -x idletoken-coord; pkill -x llama-server; true" >/dev/null 2>&1
        local g6_worker="${IDLETOKEN_G6_WORKER:-${WORKER_NODES[0]}}"
        local up
        up=$(IDLETOKEN_CLUSTER_READY_S=900 bash "$REPO_ROOT/scripts/run_cluster_llamacpp.sh" --serve \
                --tag g6 --api-port "$API_PORT" \
                --coord "$COORD_NODE" --worker "$g6_worker" \
                --model "$g6_model" ${GGUF_DGX:+--gguf "$GGUF_DGX"} \
                --ready-wait 900 2>&1 | tail -1)
        case "$up" in
            CLUSTER_LLAMACPP_READY*) started=1; vlog "cluster up: $(printf '%s' "$up" | tail -c 200)" ;;
            "") fail "$name" "run_cluster_llamacpp.sh produced NO output (OOM during load? check free memory on $COORD_NODE)"; return ;;
            *) fail "$name" "could not start the cluster (${up#CLUSTER_FAIL: })"; return ;;
        esac
        # The API is on the coordinator's loopback (hard constraint #5), so the
        # requests below run THERE, not here.
    fi
    _g6_stop() { [ "$started" = 1 ] && bash "$REPO_ROOT/scripts/run_cluster_llamacpp.sh" --stop --tag g6 >/dev/null 2>&1; }
    # Requests execute on the coordinator: --api-bind is forced to 127.0.0.1.
    _g6_curl() {  # _g6_curl <path> <json>
        $SSH "$COORD_NODE" "cat > /tmp/g6req.json" <<< "$2"
        $SSH "$COORD_NODE" "curl -s -m 300 $api_base$1 -H content-type:application/json -d @/tmp/g6req.json" 2>/dev/null
    }

    # Anthropic shape — 200 + non-empty coherent text + non-zero usage.
    local a; a=$(_g6_curl /v1/messages "{\"model\":\"$g6_model\",\"max_tokens\":24,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: pong\"}]}")
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
    case "$av" in OK*) vlog "anthropic: $av" ;; *) _g6_stop; fail "$name" "Anthropic /v1/messages: $av"; return ;; esac

    # OpenAI shape.
    local o; o=$(_g6_curl /v1/chat/completions "{\"model\":\"$g6_model\",\"max_tokens\":24,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: pong\"}]}")
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
    case "$ov" in OK*) vlog "openai: $ov" ;; *) _g6_stop; fail "$name" "OpenAI /v1/chat/completions: $ov"; return ;; esac

    # SSE: the stream must actually stream AND terminate. A client that never
    # sees [DONE] hangs forever, which is the exact shape of the Clash-proxy
    # trap in CLAUDE.md -- so "it returned tokens" is not enough on its own.
    local ss; ss=$($SSH "$COORD_NODE" "cat > /tmp/g6sse.json" <<< "{\"model\":\"$g6_model\",\"max_tokens\":24,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: pong\"}]}" \
        && $SSH "$COORD_NODE" "curl -s -N -m 300 $api_base/v1/chat/completions -H content-type:application/json -d @/tmp/g6sse.json" 2>/dev/null \
        | python3 -c '
import sys
chunks = done = 0; text = ""
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data:"): continue
    body = line[5:].strip()
    if body == "[DONE]": done = 1; continue
    chunks += 1
    try:
        import json; d = json.loads(body)
        text += ((d.get("choices") or [{}])[0].get("delta") or {}).get("content","") or ""
    except Exception: pass
if not chunks: sys.exit("no data: chunks at all — the response did not stream")
if not done:   sys.exit("stream never sent [DONE] (%d chunks) — a client would hang" % chunks)
if "pong" not in text.lower(): sys.exit("streamed text off-topic: " + text.strip()[:60])
print("OK %d chunks, [DONE] seen" % chunks)' 2>&1)
    case "$ss" in OK*) vlog "sse: $ss" ;; *) _g6_stop; fail "$name" "SSE stream: $ss"; return ;; esac

    # cache_hit / cached_tokens honesty (CLAUDE.md decision 11). The contract is
    # NOT "the number is high" -- prefix caching is the engine's business. It is
    # that the field is REPORTED FROM THE ENGINE and never guessed: coord reads
    # usage.prompt_tokens_details.cached_tokens, falls back to timings.cache_n,
    # and on a missing field warns and reports an honest 0. So: send the same
    # long prefix twice and require the second reply to carry the field at all,
    # with a value that is a number and not larger than the prompt.
    local pfx; pfx=$(python3 -c 'print("The quick brown fox jumps over the lazy dog. " * 40)')
    local body2="{\"model\":\"$g6_model\",\"max_tokens\":8,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"$pfx Reply with the single word: pong\"}]}"
    _g6_curl /v1/chat/completions "$body2" >/dev/null 2>&1
    local c2; c2=$(_g6_curl /v1/chat/completions "$body2")
    # MEASURED 2026-08-20: the coordinator emits `cache_hit` and `cached_tokens`
    # at the TOP LEVEL of both faces -- not under usage.prompt_tokens_details,
    # which is where it READS them from the engine. The platform agent probes
    # for the byte-literal shape `"cache_hit":true`, so top-level is the
    # contract; asserting the engine-side spelling would have failed a correct
    # coordinator.
    local cv; cv=$(printf '%s' "$c2" | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit("not json")
if "cached_tokens" not in d or "cache_hit" not in d:
    sys.exit("reply carries no top-level cache_hit/cached_tokens — the fields the "
             "coordinator promises to report from the engine are absent "
             "(present keys: %s)" % ",".join(sorted(d))[:120])
ct = d["cached_tokens"]; pt = (d.get("usage") or {}).get("prompt_tokens", 0)
if not isinstance(ct, int): sys.exit("cached_tokens is %r, not an integer" % (ct,))
if ct < 0 or ct > pt: sys.exit("cached_tokens=%s is outside [0, prompt_tokens=%s]" % (ct, pt))
if d["cache_hit"] and ct == 0: sys.exit("cache_hit=true with cached_tokens=0 — the two disagree")
print("OK cache_hit=%s cached_tokens=%s of prompt_tokens=%s" % (d["cache_hit"], ct, pt))' 2>&1)
    _g6_stop
    case "$cv" in OK*) vlog "cache: $cv"; pass "$name" ;; *) fail "$name" "cached_tokens contract: $cv" ;; esac
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
    if [ "$e2e" != "PRIVACY_PROXY_E2E_OK" ]; then
        fail "$name" "sealed-proxy e2e did not pass (got '$e2e')"; return
    fi
    vlog "real-socket proxy e2e passed on $COORD_NODE"

    # (3) CLUSTER half (T15, 2026-08-20). The two oracles above are hardware-free
    #     and cover md items 4/5/6 (bad key refused, sealed round-trip, pluggable
    #     layers). Items 1-3 -- no plaintext on the wire, no prompt plaintext in
    #     the cluster's logs, and the placement that keeps layer 0 home -- can
    #     only be shown on a REAL multi-machine cluster, and under llama.cpp they
    #     changed shape entirely: an rpc-server holds tensors, not tokens, and
    #     the wire is ggml-RPC-over-TLS rather than our retired INFER_*.
    #     Until this, those three were claimed in prose and asserted nowhere.
    #     The reader controls run even with no testbed, so a laptop still checks
    #     that the argv reader can go red.
    local rc
    rc=$(bash "$REPO_ROOT/scripts/gpriv_cluster_gate.sh" --selftest 2>&1 | tail -1)
    case "$rc" in
        G_PRIV_CLUSTER_OK*) vlog "${rc#G_PRIV_CLUSTER_OK }" ;;
        *) fail "$name" "the layer-0 argv reader failed its own controls (${rc#G_PRIV_CLUSTER_FAIL: }) — a reader that cannot go red proves nothing about a green cluster"; return ;;
    esac
    if [ ${#WORKER_NODES[@]} -eq 0 ]; then
        vlog "cluster half skipped: no worker node in service this run (reader controls only)"
        pass "$name"; return
    fi
    local cl
    # Pass the GGUF explicitly. Without it the cluster script resolves the path
    # from models/<id>.json, whose default_quant for the smoke model is IQ2_XXS
    # -- a file that is not on this testbed -- and the gate then failed with
    # "an overlay --rpc-host did NOT refuse", blaming the privacy control for a
    # missing weights file.
    local priv_gguf; priv_gguf=$(coord_small_gguf 2>/dev/null) || priv_gguf=""
    cl=$(IDLETOKEN_GPRIV_COORD="$COORD_NODE" \
         IDLETOKEN_GPRIV_WORKER="${IDLETOKEN_GPRIV_WORKER:-${WORKER_NODES[0]}}" \
         IDLETOKEN_GPRIV_MODEL="${IDLETOKEN_GPRIV_MODEL:-${IDLETOKEN_SMOKE_MODEL_ID:-qwen3.5-0.8b}}" \
         IDLETOKEN_GPRIV_GGUF="$priv_gguf" \
         bash "$REPO_ROOT/scripts/gpriv_cluster_gate.sh" 2>&1 | tail -1)
    case "$cl" in
        G_PRIV_CLUSTER_OK*) vlog "${cl#G_PRIV_CLUSTER_OK }"; pass "$name" ;;
        G_PRIV_CLUSTER_SKIP*) vlog "cluster half skipped: ${cl#G_PRIV_CLUSTER_SKIP: }"; pass "$name" ;;
        *) fail "$name" "cluster privacy half: ${cl#G_PRIV_CLUSTER_FAIL: }" ;;
    esac
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
    # With no worker node available this run, the cross-machine half has no
    # machine to run on. Say so and pass on the loopback evidence alone — the
    # same honesty contract as the XMACHINE_PAIR_SKIP branch below.
    if [ ${#WORKER_NODES[@]} -eq 0 ]; then
        vlog "cross-machine half skipped: no worker node in service this run (loopback pairing only)"
        pass "$name"
        return
    fi
    # CROSS-MACHINE, not just loopback. Everything above runs two processes on
    # ONE box, so it cannot see anything that only breaks over the physical LAN
    # (a firewall rule, a broadcast that does not leave the host, an address the
    # joiner resolves to itself). The product's claim is "same code on several
    # machines" — with the loopback check alone, a fleet-wide join failure went
    # unnoticed until the topology matrix hit it days later.
    #
    # PORTED 2026-08-20 (T15). The old oracle was scripts/pair_xmachine_check.sh,
    # which drives the worker with --model-path + --shard-repo: the ds4
    # layer-sharding path, retired with the engine. Same claim, current vehicle
    # -- a worker on ANOTHER MACHINE finds the coordinator from the code alone
    # and gets as far as serving its tensor share.
    #
    # NOT re-checked here: mixed engine versions refused at HELLO, the refusal
    # naming the machine to upgrade, and JOIN_REFUSED/exit 2. G_VERSION owns all
    # of that and proves it WITH a positive control on a real loopback llama.cpp
    # cluster; duplicating it here would be a second copy to keep honest.
    local xw="${IDLETOKEN_PAIR_WORKER:-${WORKER_NODES[0]}}"
    local xg; xg=$(coord_small_gguf 2>/dev/null) || xg=""
    if [ -z "$xg" ]; then
        # A provisioning gap is not a pairing failure. Say which -- do not fake
        # either verdict.
        vlog "cross-machine half skipped: no smoke GGUF on $COORD_NODE"
        pass "$name"; return
    fi
    local xm
    xm=$(bash "$REPO_ROOT/scripts/run_cluster_llamacpp.sh" --check-ready \
            --tag pair --api-port "${IDLETOKEN_PAIR_PORT:-18534}" \
            --coord "$COORD_NODE" --worker "$xw" \
            --model "${IDLETOKEN_SMOKE_MODEL_ID:-qwen3.5-0.8b}" --gguf "$xg" \
            --ready-wait "${IDLETOKEN_PAIR_READY_S:-420}" 2>&1 | tail -1)
    bash "$REPO_ROOT/scripts/run_cluster_llamacpp.sh" --stop --tag pair >/dev/null 2>&1
    case "$xm" in
        CLUSTER_READY) vlog "$xw found the coordinator over the LAN by code alone and served its share"; pass "$name" ;;
        *) fail "$name" "cross-machine code pairing failed: ${xm#CLUSTER_FAIL: }" ;;
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
            src/common/model.c src/common/modelsize.c src/common/advise.c \
            src/tools/plan_test.c -o build/plan_test 2>&1 \
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
    #
    # PARKED (2026-08-14, llama.cpp pivot — docs/v2-rebuild-plan-2026-08.md §1.1
    # and pivot doc §6): the ds4x generic-kernel line is FROZEN; its bit-exact
    # alignment channels (config/forward/tokenizer/dequant/GQA/GDN, sub-checks
    # 2–2d below) are retired from the default ladder along with it, in the
    # same style as G_DSPARK. The code stays runnable so bit-rot in the frozen
    # line surfaces as a gate failure rather than a surprise: set
    # IDLETOKEN_DS4X_GATE=1 before touching ds4x again. Sub-checks 1/1b
    # (registry+planner), 3 (no-hardcode scan) and 4 (protocol carries model
    # identity) are engine-agnostic orchestration properties and keep running.
    if [ "${IDLETOKEN_DS4X_GATE:-0}" = "1" ]; then
    # The frozen ds4x line is not published: the public mirror strips
    # scripts/ds4x_ref.py and src/ds4x/. Asking for the gate there must fail
    # loudly, not stumble over a missing file three commands in.
    if [ ! -f "$repo/scripts/ds4x_ref.py" ]; then
        fail "$name" "IDLETOKEN_DS4X_GATE=1, but scripts/ds4x_ref.py is not in this distribution (the ds4x line is frozen and unpublished)"; return
    fi
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
    else
        vlog "ds4x bit-exact channels parked (frozen line, 2026-08-14) — set IDLETOKEN_DS4X_GATE=1 to run them"
    fi

    # 3. no-hardcode scan over the orchestration layer (vendor/ds4 excluded).
    #    The smell is a model name baked in as a VALUE (a quoted string literal
    #    "deepseek-v4-flash" in a response/assignment) or the old fixed
    #    layer-count macro. Help/usage text mentioning the default by name is
    #    fine — that is documentation, not a hardcoded code path.
    #
    #    Comments are documentation too. The scan strips C comment lines before
    #    matching: a block-comment body line (`* ...`), a `//` line, or a
    #    single-line `/* ... */` mentioning the default (e.g. WS-B4's comment
    #    "reported \"deepseek-v4-flash\" while serving Qwen") is prose, not a
    #    baked-in code path. A real value literal in a statement is not stripped
    #    by any of these filters, so it still trips the gate.
    local bad=""
    grep -RnE '"deepseek-v4-flash"' "$repo/src/coord" "$repo/src/worker" 2>/dev/null \
        | grep -vE ':[[:space:]]*\*' \
        | grep -vE ':[[:space:]]*//' \
        | grep -vE ':[[:space:]]*/\*' \
        | grep -q . \
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
# IDLETOKEN_ALLOW_SMALL_CLUSTER: P3/P6 drive a curated model that FITS one
# machine, and the scheduler would correctly release the second node. The
# client spawns the coordinator, so the hatch has to travel in ITS environment.
CLIENT_ENV_BASE='WEBKIT_DISABLE_DMABUF_RENDERER=1 no_proxy=localhost,127.0.0.1 IDLETOKEN_ALLOW_SMALL_CLUSTER=1'
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

# Resolve the display in the CALLER shell before any client launch. In
# particular, client_run is normally invoked inside `out=$(client_run ...)`;
# resolving it only inside that command substitution loses CLIENT_ENV when the
# subshell exits. P1/P2 then pass while P3's direct background launches fall
# back to dead DISPLAY=:1 and wait ten minutes before reporting a pairing
# failure, with the real GTK error hidden in the remote logs.
ensure_client_display() {
    if [ -z "${CLIENT_DISPLAY_RESOLVED:-}" ]; then
        CLIENT_DISPLAY_RESOLVED="$(client_display)"
        CLIENT_ENV="$CLIENT_ENV_BASE DISPLAY=$CLIENT_DISPLAY_RESOLVED"
        if [ "$CLIENT_DISPLAY_RESOLVED" = ":77" ]; then
            echo "note: no usable desktop session on the coordinator; the product gates are running on an **Xvfb virtual display** (:77)." >&2
            echo "      The programmatic assertions still hold; a human visual walkthrough is not covered by this round." >&2
        fi
    fi
}

# Run one client instance in the foreground with the given directives; prints
# its log afterwards. $1=directives $2=logfile $3=timeout-guard(s)
client_run() {
    ensure_client_display
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
    ensure_client_display
    # The shell itself must come from the current Rust sources too. The first
    # WS-F full-suite run only compared the staged worker with the root worker;
    # it therefore called the sidecars FRESH while running an Aug-08 client
    # binary against Aug-14 sources. That stale shell produced both UI reports
    # but never honoured the current graceful-quit path, blocking every product
    # gate with a misleading teardown failure.
    local shell_fresh
    shell_fresh=$($SSH "$COORD_NODE" "cd $DGX_HOME && \
        b=client/src-tauri/target/debug/idletoken-client; \
        if [ ! -x \$b ]; then echo NO_CLIENT; \
        elif find client/src-tauri/src client/src-tauri/Cargo.toml client/src-tauri/Cargo.lock \
             -type f -newer \$b -print -quit 2>/dev/null | grep -q .; then echo STALE_CLIENT; \
        else echo FRESH_CLIENT; fi" 2>/dev/null | tr -d '\r')
    case "$shell_fresh" in
        FRESH_CLIENT) vlog "client shell is not older than its Rust sources" ;;
        NO_CLIENT) fail "$name" "no debug client on $COORD_NODE (cd client/src-tauri && cargo build)"; return ;;
        *) fail "$name" "debug client is STALE on $COORD_NODE — run cd client/src-tauri && cargo build"; return ;;
    esac
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
    # 2026-08-14, MEASURED AND RULED OUT: on the Xvfb fallback display this gate
    # fails "client did not exit cleanly" **and the budget is not the cause** --
    # raising it to 150s changed nothing while BOTH UI_TEST_REPORTs were produced
    # correctly in the same run. Nor is it the tray (`quit:<ms>` goes through
    # app_quit/app.exit(0), deliberately bypassing close-to-tray). What is left:
    # the coordinator has **no logged-in desktop session** (:1 is dead), so the
    # client renders through software WebKit (`libEGL: DRI3 error: Could not get
    # DRI3 device`) and its teardown never completes. Environment, not stopwatch:
    # do not "fix" this by growing the number again.
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
    ensure_client_display
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
    ensure_client_display
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
    local gguf_abs
    gguf_abs=$(coord_p36_gguf) || { fail "$name" "no curated large GGUF on $COORD_NODE (searched IDLETOKEN_GGUF_DIRS_${COORD_NODE} for $(basename "${IDLETOKEN_P36_GGUF:-Qwen3.8-27B-UD-Q4_K_M.gguf}"))"; return; }
    $SSH "$COORD_NODE" "rm -f /tmp/idletoken-p3a.log /tmp/idletoken-p3b.log" >/dev/null 2>&1
    $SSHF "$COORD_NODE" "cd $CLIENT_DIR && exec env $CLIENT_ENV IDLETOKEN_UI_TEST='pairing-create:ACCEPT:model=$gguf_abs,pairing-auto-start,quit:600000' $CLIENT_BIN < /dev/null > /tmp/idletoken-p3a.log 2>&1" >/dev/null 2>&1
    # Launch the joiner only once the creator's directive loop has actually run
    # (the directives echo prints right before pairing_create fires, and the
    # beacon follows within a second). The joiner listens for the beacon for
    # only 8s after ITS webview loads, and webview startup skew on Xvfb
    # (10-30s) dwarfs any fixed sleep — a missed window reads as "no cluster
    # found" and then 600 s of nothing (seen 2026-08-20).
    local w
    for w in $(seq 1 60); do
        $SSH "$COORD_NODE" "grep -aq 'UI_TEST_REPORT directives' /tmp/idletoken-p3a.log 2>/dev/null" && break
        sleep 2
    done
    sleep 3
    # dbus-run-session + private XDG_RUNTIME_DIR: the client's single-instance
    # lock (added 2026-08-16) is a D-Bus name on the session bus, and both
    # instances on one DISPLAY share the X11-autolaunched bus — instance B
    # exits before printing a single line (0-byte log, seen 2026-08-20). A
    # private bus scopes the lock per instance. The runtime dir must move with
    # it: the private bus's xdg-document-portal otherwise tries to mount
    # /run/user/<uid>/doc, which the real session's portal already holds, and
    # WebKit never comes up (fuse init failed, also 2026-08-20). Product
    # behavior on a real desktop is untouched — this is test-harness plumbing.
    $SSHF "$COORD_NODE" "cd $CLIENT_DIR && rm -rf /tmp/idletoken-p3b-xdg && mkdir -m 700 /tmp/idletoken-p3b-xdg && exec dbus-run-session -- env $CLIENT_ENV XDG_RUNTIME_DIR=/tmp/idletoken-p3b-xdg IDLETOKEN_UI_TEST='pairing-join:ACCEPT:as=accept-node-b:model=$gguf_abs,quit:600000' $CLIENT_BIN < /dev/null > /tmp/idletoken-p3b.log 2>&1" >/dev/null 2>&1

    # A REAL 80 GiB load takes minutes, not the 90s that sufficed for mock.
    local st="" i
    for i in $(seq 1 100); do
        st=$($SSH "$COORD_NODE" "curl -s -m 2 http://127.0.0.1:$API_PORT/idletoken/v1/cluster/status; true" 2>/dev/null | tr -d '\r')
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
    # Two status schemas, one verdict each. The llamacpp line has no per-member
    # layer ranges (the split is a tensor-split share, not a layer table); its
    # "this is a real cluster" facts are engine_state=ready + a worker with an
    # rpc endpoint. The legacy branch keeps the 43-layer coverage check.
    # 2026-08-15: the old single-schema assertion KeyError'd on the llamacpp
    # payload and the swallowed stderr made the gate fail with an EMPTY reason
    # — so this version never dies silently.
    ok=$(echo "$st" | python3 -c "
import json,sys
try:
    s=json.load(sys.stdin)
    if s.get('engine')=='llamacpp':
        m=s.get('members',[])
        coords=[x for x in m if x.get('role')=='coordinator']
        workers=[x for x in m if x.get('role')=='worker']
        good=(s.get('phase')=='ready' and s.get('engine_state')=='ready'
              and s.get('cluster_size',0)>=2 and len(coords)==1
              and len(workers)>=1
              and all(x.get('state')=='ready' for x in m)
              and all(w.get('rpc_endpoint') for w in workers))
        print('ok' if good else 'bad: '+json.dumps(s))
    else:
        m=sorted(s.get('members',[]),key=lambda x:x['stage'])
        cover=(m and m[0]['layer_lo']==0 and m[-1]['layer_hi']==43
               and all(m[i]['layer_hi']==m[i+1]['layer_lo'] for i in range(len(m)-1)))
        print('ok' if s.get('phase')=='ready' and cover else 'bad: '+json.dumps(s))
except Exception as e:
    print('bad: assertion error %s' % e)")
    client_cleanup
    if [ "$ok" = "ok" ]; then
        vlog "UI-initiated pairing produced a real engine cluster (schema-checked for the running engine line)"
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
    ensure_client_display
    client_cleanup
    ensure_frontend
    $SSH "$COORD_NODE" "rm -f /tmp/idletoken-$tag-a.log /tmp/idletoken-$tag-b.log" >/dev/null 2>&1
    $SSHF "$COORD_NODE" "cd $CLIENT_DIR && exec env $CLIENT_ENV IDLETOKEN_UI_TEST='pairing-create:$code$cx,pairing-auto-start,report-pairing-phases,quit:$life' $CLIENT_BIN < /dev/null > /tmp/idletoken-$tag-a.log 2>&1" >/dev/null 2>&1
    # Wait for the creator's directive loop before launching the joiner — same
    # beacon-window race as p3_pairing above.
    local w
    for w in $(seq 1 60); do
        $SSH "$COORD_NODE" "grep -aq 'UI_TEST_REPORT directives' /tmp/idletoken-$tag-a.log 2>/dev/null" && break
        sleep 2
    done
    sleep 3
    # dbus-run-session + private XDG_RUNTIME_DIR: same single-instance-lock and
    # document-portal scoping as p3_pairing above.
    $SSHF "$COORD_NODE" "cd $CLIENT_DIR && rm -rf /tmp/idletoken-$tag-b-xdg && mkdir -m 700 /tmp/idletoken-$tag-b-xdg && exec dbus-run-session -- env $CLIENT_ENV XDG_RUNTIME_DIR=/tmp/idletoken-$tag-b-xdg IDLETOKEN_UI_TEST='pairing-join:$code:as=$tag-node-b$jx,quit:$life' $CLIENT_BIN < /dev/null > /tmp/idletoken-$tag-b.log 2>&1" >/dev/null 2>&1
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
# The SMALL model on the coordinator node, resolved by the testbed convention
# (basename of the control machine's smoke model, searched in that node's
# configured gguf dirs). P4/P5 must NOT use $GGUF_DGX: that is the 80 GiB DSv4,
# whose load time blows their 120 s ready windows — P3 survives it only because
# its window is 600 s.
# The curated LARGE llama.cpp model for P3/P6 (T15, 2026-08-20). Decision: P3
# and P6 test UI-driven orchestration and API exposure, where an 80 GiB load is
# dead weight; G6 is the end-to-end product claim and drives DeepSeek-V4-Flash
# instead. Qwen3.8-27B is the curated SKU T14 calibrated, and it fits one
# machine -- so these two gates run under IDLETOKEN_ALLOW_SMALL_CLUSTER=1, the
# same documented test vehicle every cross-machine gate uses.
coord_p36_gguf() {
    local fname dirs d
    fname=$(basename "${IDLETOKEN_P36_GGUF:-Qwen3.8-27B-UD-Q4_K_M.gguf}")
    eval "dirs=\${IDLETOKEN_GGUF_DIRS_${COORD_NODE}:-}"
    for d in $dirs; do
        if $SSH "$COORD_NODE" "test -r '$d/$fname'" >/dev/null 2>&1; then
            echo "$d/$fname"; return 0
        fi
    done
    return 1
}
P36_MODEL_ID="${IDLETOKEN_P36_MODEL_ID:-qwen3.8-27b}"

coord_small_gguf() {
    local fname dirs d
    fname=$(basename "${IDLETOKEN_SMOKE_GGUF:-Qwen3.5-0.8B-Q4_K_M.gguf}")
    eval "dirs=\${IDLETOKEN_GGUF_DIRS_${COORD_NODE}:-}"
    for d in $dirs; do
        if $SSH "$COORD_NODE" "test -r '$d/$fname'" >/dev/null 2>&1; then
            echo "$d/$fname"; return 0
        fi
    done
    return 1
}

p4_orchestration() {
    local name="$1"
    # REAL model since 2026-08-15. Mock existed because the only model was an
    # 80 GiB DSv4 (minutes to load); the smoke model loads in ~5 s, and the v2
    # client deliberately has NO mock start path (no-silent-fallback) — it
    # refuses "no GGUF file selected", which is exactly what killed this gate
    # when it still declared IDLETOKEN_MOCK_OK.
    local gguf_abs
    gguf_abs=$(coord_small_gguf) || { fail "$name" "no small smoke GGUF on $COORD_NODE (searched IDLETOKEN_GGUF_DIRS_${COORD_NODE} for $(basename "${IDLETOKEN_SMOKE_GGUF:-Qwen3.5-0.8B-Q4_K_M.gguf}"))"; return; }
    # XRCHES, not ORCHES: join codes use the no-O/0/I/1 alphabet and the v2
    # engine VALIDATES it — 'O' was never noticed while the mock era stopped
    # short of the engine (`invalid join code 'ORCHES'`, measured round 5).
    local rep; rep=$(pairing_pair_report XRCHES p4 ":model=$gguf_abs" "" 150000 40 3)
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
    # REAL model since 2026-08-15 (see p4_orchestration — the v2 client has no
    # mock start path, by design). The smoke model loads in seconds, so the
    # settings pipeline is proven against the same engine users run.
    local gguf_abs
    gguf_abs=$(coord_small_gguf) || { fail "$name" "no small smoke GGUF on $COORD_NODE (searched IDLETOKEN_GGUF_DIRS_${COORD_NODE})"; return; }
    local rep; rep=$(pairing_pair_report P5SETT p5 "$tune:model=$gguf_abs" "$tune" 150000 40 3)
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

    # (b) token enforced: with Bearer -> 200, without -> 401. (Reply coherence
    # at length is G6/P6's job; here the body only has to prove the ENGINE
    # answered through the token-gated port.)
    local req="{\"model\":\"${IDLETOKEN_SMOKE_MODEL_ID:-qwen3.5-0.8b}\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}"
    local wresp; wresp=$($SSH "$COORD_NODE" "curl -s -m 60 -w '\n%{http_code}' http://127.0.0.1:$port/v1/chat/completions -H 'content-type: application/json' -H 'Authorization: Bearer $tok' -d '$req'; true" 2>/dev/null | tr -d '\r')
    local with; with=$(printf '%s' "$wresp" | tail -1)
    local wbody; wbody=$(printf '%s' "$wresp" | sed '$d')
    local without; without=$($SSH "$COORD_NODE" "curl -s -m 10 -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/v1/chat/completions -H 'content-type: application/json' -d '$req'; true" 2>/dev/null | tr -d '\r')
    client_cleanup
    if [ "$with" != "200" ] || [ "$without" != "401" ]; then
        fail "$name" "apiToken not enforced (with token: http $with, expected 200; without: http $without, expected 401)"
        return
    fi
    # Origin proof, real-engine era: the 200 body must be a completion the
    # ENGINE produced — a chat.completion object with non-empty content (the
    # old check demanded the coord's mock marker, which no longer exists).
    local origin
    origin=$(printf '%s' "$wbody" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    c=((d.get('choices') or [{}])[0].get('message') or {}).get('content','')
    print('ok' if d.get('object')=='chat.completion' and c.strip() else 'bad: '+json.dumps(d)[:120])
except Exception as e:
    print('bad: %s' % e)")
    if [ "$origin" = "ok" ]; then
        vlog "apiToken enforced: with=200 (real engine completion), without=401"
        pass "$name"
    else
        fail "$name" "with-token 200 body is not a real engine completion ($origin)"
    fi
}

# =====================================================================
# P6 — API exposure: a UI-formed cluster with the REAL model auto-listens; the
#      UI advertises the address; an external client connects to THAT address
#      and gets an E6-level DSv4 reply. Needs the real GGUF (honest fail if
#      absent — P6 cannot pass on a mock model).
#
#      2026-08-14: the external client now presents the API token, because the
#      product changed under this gate. A fresh install mints a random token and
#      the API is CLOSED by default (the decision G_LOCAL_TOKEN asserts), so an
#      unauthenticated request has been getting
#      `{"error":{"type":"authentication_error"}}` — valid JSON with no content,
#      which read as "empty text" and looked like a broken engine. What a real
#      user does is copy the base URL AND the token out of the UI, so that is
#      what the gate does; the no-token 401 is asserted here too, so "advertised
#      address is reachable" can never quietly mean "advertised address is open".
# =====================================================================
p6_api_exposure() {
    local name="$1"
    local tok="p6-external-client-token"
    # The path travels into the client as a UI directive and from there straight
    # into sidecar argv — no shell anywhere on that route, so a `$HOME` in it
    # would reach the engine literally (the weights sidecar reports
    # "cannot stat $HOME/..." and the cluster silently falls back to mock).
    # Expand it once, on the node.
    local gguf_abs
    gguf_abs=$(coord_p36_gguf) || { fail "$name" "no curated large GGUF on $COORD_NODE (searched IDLETOKEN_GGUF_DIRS_${COORD_NODE} for $(basename "${IDLETOKEN_P36_GGUF:-Qwen3.8-27B-UD-Q4_K_M.gguf}"))"; return; }

    # Model load across coord + 2 workers is slow; allow up to ~10 min.
    # ORDER MATTERS: the UI-test pairing regex (client/src/App.tsx) ends with
    # `(?::model=(\S+))?` — `\S+` swallows everything after it, so `:model=`
    # must be LAST. Putting it first silently turned the token into part of the
    # GGUF path and the worker refused to serve ("No such file or directory").
    local spec=":apiToken=$tok:model=$gguf_abs"
    local rep; rep=$(pairing_pair_report APEXPT p6 "$spec" "$spec" 600000 100 6)
    if [ -z "$rep" ]; then
        client_cleanup
        fail "$name" "real-model cluster never reached ready (see /tmp/idletoken-p6-{a,b}.log on $COORD_NODE)"
        return
    fi
    local base; base=$(echo "$rep" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("apiBaseUrl") or "")' 2>/dev/null)
    [ -n "$base" ] || { client_cleanup; fail "$name" "UI advertised no API baseUrl ($rep)"; return; }
    vlog "UI-advertised API address: $base"

    # External client hits the UI-advertised address (curl runs on the coord
    # node so the advertised LAN ip resolves) and must get a real reply — with
    # the token the UI hands the user, since the API is closed by default.
    local req="{\"model\":\"$P36_MODEL_ID\",\"max_tokens\":24,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: pong\"}]}"
    local reply
    reply=$($SSH "$COORD_NODE" "curl -s -m 180 $base/v1/messages -H 'content-type: application/json' -H 'Authorization: Bearer $tok' -d '$req'" 2>/dev/null)
    # Same address, no credentials: must be refused. Asserted BEFORE cleanup,
    # while the cluster is still up.
    local anon; anon=$($SSH "$COORD_NODE" "curl -s -m 30 -o /dev/null -w '%{http_code}' $base/v1/messages -H 'content-type: application/json' -d '$req'; true" 2>/dev/null | tr -d '\r')
    client_cleanup
    if [ "$anon" != "401" ]; then
        fail "$name" "the advertised address answered an UNAUTHENTICATED request with http $anon (expected 401) — a LAN-reachable API that anyone can drive"
        return
    fi
    vlog "unauthenticated request to the advertised address: 401"
    local ok
    ok=$(printf '%s' "$reply" | python3 -c '
import json,sys
raw=sys.stdin.read()
try: d=json.loads(raw)
except Exception: sys.exit("not json: "+raw[:80])
if "error" in d: sys.exit("api error: "+json.dumps(d["error"])[:120])
t="".join(b.get("text","") for b in d.get("content",[]) if b.get("type")=="text")
u=d.get("usage",{})
if not t.strip(): sys.exit("empty text")
if not (u.get("input_tokens",0)>0 and u.get("output_tokens",0)>0): sys.exit("zero usage")
print("ok "+t.strip()[:40])' 2>&1)
    case "$ok" in
        ok*) vlog "external client reply via UI address: $ok"; pass "$name" ;;
        # Print the body, not just the verdict: the cluster is torn down by
        # client_cleanup one line above, so "empty text" with no evidence costs
        # a whole 5-minute rebuild to see what the API actually said.
        *) fail "$name" "external client got no real reply from $base ($ok) — body: $(printf '%s' "$reply" | tr -d '\n' | cut -c1-300)" ;;
    esac
}

# =====================================================================
# G_HOMO — a cluster is homogeneous. The coordinator must REFUSE a worker whose
#      OS family differs from the first worker that joined (CLAUDE.md hard
#      constraint #2, 2026-08-12) — a mixed cluster has no oracle, so letting
#      one form manufactures a green nobody can falsify (docs/archive/macos-node.md §5).
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
    # RETIRED (2026-08-14, llama.cpp pivot — v2 plan §1.4): heterogeneous
    # clusters are ALLOWED now. The "no oracle for a mixed cluster" argument
    # died with the exact-token oracle itself (decision #10: two correct
    # engines diverge at token 4 even on one machine); the real cluster
    # invariant is "every node runs the SAME llama.cpp build", enforced at
    # HELLO and gated by G_VERSION below. The os_family refusal still exists
    # in the legacy ds4 INFER path, so the parked assertions stay runnable:
    # set IDLETOKEN_HOMO_GATE=1 to exercise them (G_DSPARK-style parking).
    if [ "${IDLETOKEN_HOMO_GATE:-0}" != "1" ]; then
        skip "$name" "retired 2026-08-14 (heterogeneous clusters allowed; the invariant is engine-version equality — see G_VERSION) — set IDLETOKEN_HOMO_GATE=1 to run the parked ds4-line check"
        return
    fi
    local repo; repo=$(cd "$(dirname "$0")/.." && pwd)
    command -v cc  >/dev/null 2>&1 || { skip "$name" "no C compiler on the control machine"; return; }
    command -v python3 >/dev/null 2>&1 || { skip "$name" "python3 needed for the stub worker"; return; }
    (cd "$repo" && make coord >/dev/null 2>&1) || { fail "$name" "make coord failed on the control machine"; return; }

    # Steps 1-3 stage Linux(1) against Windows(2) — never this host's family,
    # and never macOS(3). macOS is sealed as a compute node (G_MACSEAL), so on
    # a Mac control machine a stub claiming the native family gets refused by
    # the seal and this gate would go red for a reason it is not testing.
    local fam_a=1 fam_b=2
    # Step 4 drives the REAL worker binary, which reports the family it was
    # built for; the stub must therefore claim something else.
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

    # 1. the first worker joins and sets the cluster's family
    IDLETOKEN_MOCK_OS_FAMILY="$fam_a" python3 "$repo/scripts/mock_worker.py" \
        "127.0.0.1:$port" >"$tmp/m1.log" 2>&1 &
    m1=$!
    sleep 1
    if ! grep -q "^coord: worker 0 is " "$tmp/coord.log"; then
        fail "$name" "the first stub worker never joined (see $tmp/coord.log)"; _homo_cleanup; return
    fi

    # 2. a worker claiming another OS must be refused, and be TOLD why
    IDLETOKEN_MOCK_OS_FAMILY="$fam_b" python3 "$repo/scripts/mock_worker.py" \
        "127.0.0.1:$port" >"$tmp/m2.log" 2>&1 &
    m2=$!
    # Two ways a broken check shows up: the stub blocks forever on recv (still
    # alive), or it gets an ASSIGN_PLAN. Name both — "was accepted" is a far more
    # useful red than "no refusal in the log".
    if ! _homo_wait "$m2"; then
        fail "$name" "a $fam_b-family worker was ACCEPTED into a $fam_a-family cluster (still connected)"
        _homo_cleanup; return
    fi
    if grep -q "plan received" "$tmp/m2.log"; then
        fail "$name" "a $fam_b-family worker was ACCEPTED into a $fam_a-family cluster (got ASSIGN_PLAN)"
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
    IDLETOKEN_MOCK_OS_FAMILY="$fam_a" python3 "$repo/scripts/mock_worker.py" \
        "127.0.0.1:$port" >"$tmp/m3.log" 2>&1 &
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
# G_VERSION — the ONE llama.cpp-cluster invariant (v2 plan §5.2, replaces
#      G_HOMO's os_family rule): every node runs the SAME llama.cpp build.
#      A version mismatch is refused at HELLO with a sentence that names the
#      machine to upgrade; matching versions form a working cluster.
#
# Runs entirely on the control machine: the REAL coordinator (llamacpp cluster
# mode) + the REAL idletoken-worker --rpc-supervisor on loopback, with the
# pinned engine build in vendor/llama.cpp/build/bin. The mismatch cannot be
# staged with two real builds on one box, so the worker's claimed version is
# forced through IDLETOKEN_TEST_ENGINE_VERSION (src/common/enginever.c) — a
# TEST-ONLY override that prints a loud banner; the gate asserts the banner is
# present, so a silent removal of the override would surface here rather than
# make the mismatch case vacuously green.
#
# Both halves are checked (same reasoning as the old G_HOMO): the refusal
# alone would also pass if the coordinator refused everyone, so a matching
# worker must still join and the cluster must actually answer.
# =====================================================================
g_version() {
    local name="$1" repo; repo=$(cd "$(dirname "$0")/.." && pwd)
    local base="${IDLETOKEN_VERSION_PORT:-14351}"
    local wire_port="$base" api_port=$((base + 4000)) llama_port=$((base + 4200))
    local rpc_port=$((base + 4400)) disc_port=$((base + 700))
    local engine_dir="${IDLETOKEN_ENGINE_DIR:-$repo/vendor/llama.cpp/build/bin}"
    local code="GVERSN"
    command -v cc >/dev/null 2>&1 || { skip "$name" "no C compiler on the control machine"; return; }
    if [ ! -x "$engine_dir/llama-server" ] || [ ! -x "$engine_dir/ggml-rpc-server" ]; then
        skip "$name" "no pinned llama.cpp build in $engine_dir (scripts/build_llamacpp.sh) — the version invariant needs a real engine to version-probe"
        return
    fi
    if [ -z "${IDLETOKEN_SMOKE_GGUF:-}" ] || [ -z "${IDLETOKEN_SMOKE_MODEL_ID:-}" ]; then
        skip "$name" "set IDLETOKEN_SMOKE_GGUF + IDLETOKEN_SMOKE_MODEL_ID to a small local model — the positive half must load real weights across the loopback cluster"
        return
    fi
    [ -r "$IDLETOKEN_SMOKE_GGUF" ] || { fail "$name" "IDLETOKEN_SMOKE_GGUF is set but unreadable: $IDLETOKEN_SMOKE_GGUF"; return; }
    (cd "$repo" && make coord >/dev/null 2>&1)  || { fail "$name" "make coord failed on the control machine"; return; }
    (cd "$repo" && make worker >/dev/null 2>&1) || { fail "$name" "make worker failed on the control machine"; return; }

    local tmp; tmp=$(mktemp -d)
    local cpid="" wpid=""
    # $1=keep -> leave the logs on disk (failures must stay diagnosable; a
    # message naming a path the gate then deletes is worse than no message).
    _ver_cleanup() {
        [ -n "$wpid" ] && kill "$wpid" 2>/dev/null
        [ -n "$cpid" ] && kill "$cpid" 2>/dev/null
        # wait before the backstop pkill so bash's job monitor does not print
        # "Terminated" lines for the children as they die.
        [ -n "$wpid" ] && wait "$wpid" 2>/dev/null
        [ -n "$cpid" ] && wait "$cpid" 2>/dev/null
        # Backstop for supervised grandchildren, scoped to this gate's ports.
        pkill -f "ggml-rpc-serve[r].* -p $rpc_port" 2>/dev/null
        pkill -f "llama-serve[r].*--port $llama_port" 2>/dev/null
        [ "${1:-}" = keep ] || rm -rf "$tmp"
    }
    # Wait up to $2 x 0.2s for pid $1 to exit; return 1 if it is still alive.
    _ver_wait_exit() {
        local p="$1" i=0 lim="${2:-150}"
        while kill -0 "$p" 2>/dev/null && [ "$i" -lt "$lim" ]; do sleep 0.2; i=$((i + 1)); done
        kill -0 "$p" 2>/dev/null && return 1 || return 0
    }

    # Coordinator: llamacpp CLUSTER mode, one rpc worker, pairing by code.
    # IDLETOKEN_ALLOW_SMALL_CLUSTER=1 is the documented test vehicle (G_SIZE
    # claim 4): without it the scheduler would — correctly — release the
    # worker because a 0.8B model fits one machine, and the positive half
    # would never exercise the joined cluster.
    (cd "$repo" && exec env IDLETOKEN_RPC_PSK_FILE="$tmp/psk_coord" \
        IDLETOKEN_ALLOW_SMALL_CLUSTER=1 IDLETOKEN_LLAMA_LOG="$tmp/llama.log" \
        ./idletoken-coord --bind "127.0.0.1:$wire_port" --num-workers 1 \
        --pair-code "$code" --discovery-port "$disc_port" \
        --model-id "$IDLETOKEN_SMOKE_MODEL_ID" \
        --llama-server-bin "$engine_dir/llama-server" \
        --llama-gguf "$IDLETOKEN_SMOKE_GGUF" --llama-port "$llama_port" \
        --ctx-size 1024 \
        --http --api-bind "127.0.0.1:$api_port" >"$tmp/coord.log" 2>&1) &
    cpid=$!
    sleep 2
    if ! kill -0 "$cpid" 2>/dev/null; then
        fail "$name" "the coordinator did not start in llamacpp cluster mode (see $tmp/coord.log)"
        cpid=""; _ver_cleanup keep; return
    fi

    # --- 1. a worker claiming a DIFFERENT engine build must be refused ------
    (cd "$repo" && exec env IDLETOKEN_RPC_PSK_FILE="$tmp/psk_bad" \
        IDLETOKEN_TEST_ENGINE_VERSION='999 (fakesha0)' \
        ./idletoken-worker --rpc-supervisor --engine-dir "$engine_dir" \
        --pair-code "$code" --coordinator "127.0.0.1:$wire_port" \
        --discovery-port "$disc_port" \
        --rpc-host 127.0.0.1 --rpc-port "$rpc_port" >"$tmp/w_bad.log" 2>&1) &
    wpid=$!
    if ! _ver_wait_exit "$wpid" 150; then
        fail "$name" "a worker with a mismatched engine version was ACCEPTED (still connected after 30s; see $tmp/w_bad.log)"
        _ver_cleanup keep; return
    fi
    wait "$wpid" 2>/dev/null; local rc=$?
    wpid=""
    if ! grep -q "TEST OVERRIDE" "$tmp/w_bad.log"; then
        fail "$name" "the version override banner is missing — the mismatch case never actually faked a version (see $tmp/w_bad.log)"
        _ver_cleanup keep; return
    fi
    if ! grep -q "JOIN_REFUSED: " "$tmp/w_bad.log"; then
        fail "$name" "the mismatched worker got no JOIN_REFUSED marker — the client cannot show the reason (see $tmp/w_bad.log)"
        _ver_cleanup keep; return
    fi
    if ! grep -qi "upgrade" "$tmp/w_bad.log"; then
        fail "$name" "the refusal does not tell the user to UPGRADE anything (see $tmp/w_bad.log)"
        _ver_cleanup keep; return
    fi
    if [ "$rc" != 2 ]; then
        fail "$name" "refused worker exited $rc, expected 2 (IDLETOKEN_EXIT_JOIN_REFUSED)"
        _ver_cleanup keep; return
    fi
    if ! grep -qE "refused .*upgrade" "$tmp/coord.log"; then
        fail "$name" "the coordinator log does not name the machine to upgrade (see $tmp/coord.log)"
        _ver_cleanup keep; return
    fi
    vlog "mismatched engine version refused at HELLO; reason names the machine and says upgrade"

    # --- 2. positive control: a matching worker joins and the cluster serves -
    (cd "$repo" && exec env IDLETOKEN_RPC_PSK_FILE="$tmp/psk_good" \
        ./idletoken-worker --rpc-supervisor --engine-dir "$engine_dir" \
        --pair-code "$code" --coordinator "127.0.0.1:$wire_port" \
        --discovery-port "$disc_port" \
        --rpc-host 127.0.0.1 --rpc-port "$rpc_port" >"$tmp/w_good.log" 2>&1) &
    wpid=$!
    local i=0
    while [ $i -lt 150 ] && ! grep -q "rpc worker 0 ready" "$tmp/coord.log"; do sleep 0.2; i=$((i + 1)); done
    if ! grep -q "rpc worker 0 ready" "$tmp/coord.log"; then
        fail "$name" "a matching-version worker did not join — the check refuses everyone (see $tmp/coord.log, $tmp/w_good.log)"
        _ver_cleanup keep; return
    fi
    # /health is the coordinator's; readiness is its engine_state field (the
    # raw idletoken-server 503-while-loading trap does not apply to this surface,
    # but "ready" must still be asserted, not assumed).
    local h="" ok=""
    i=0
    while [ $i -lt 240 ]; do
        h=$(curl -s -m 3 "http://127.0.0.1:$api_port/health" 2>/dev/null)
        case "$h" in *'"engine_state":"ready"'*) break ;; esac
        # Resource refusal and sidecar startup errors terminate the
        # coordinator. Do not spend four minutes polling a port whose owner is
        # already dead; keep the evidence and report the scheduler/engine cause.
        if ! kill -0 "$cpid" 2>/dev/null; then
            fail "$name" "the joined cluster coordinator exited before engine ready: $(tail -2 "$tmp/coord.log" 2>/dev/null | tr '\n' ' ' | tail -c 240) (logs kept in $tmp)"
            cpid=""; _ver_cleanup keep; return
        fi
        h=""
        sleep 1; i=$((i + 1))
    done
    if [ -z "$h" ]; then
        fail "$name" "the joined cluster's engine never reached ready (see $tmp/coord.log, $tmp/llama.log)"
        _ver_cleanup keep; return
    fi
    ok=$(curl -s -m 120 "http://127.0.0.1:$api_port/v1/chat/completions" \
        -H 'content-type: application/json' \
        -d '{"model":"x","max_tokens":512,"messages":[{"role":"user","content":"Reply with the single word: pong"}]}' 2>/dev/null \
        | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit("not json")
ch=d.get("choices",[])
t=ch[0]["message"]["content"] if ch else ""
sys.exit(0) if t.strip() else sys.exit("empty content")' 2>&1) || {
        fail "$name" "the matched-version cluster did not answer a chat request ($ok)"
        _ver_cleanup keep; return; }
    vlog "matching versions joined; the loopback rpc cluster served a real completion"
    _ver_cleanup
    pass "$name"
}

# =====================================================================
# G_MACSEAL — macOS is sealed as a COMPUTE node (2026-08-13, user's call), and
#      still first-class as a CONTROL machine. The Metal code stays in the tree
#      and keeps compiling; what is refused is a Mac serving layers.
#
# Why a gate at all: a seal that only exists in prose decays into a half-truth
# the moment someone touches the handshake, and the README will keep saying
# "Windows and Linux" while a Mac quietly joins. Four assertions, and the last
# two are the ones that keep this honest:
#
#   1. the coordinator refuses a macOS worker AS THE FIRST WORKER — a cluster
#      of nothing but Macs is homogeneous, so G_HOMO would wave it through;
#   2. it still accepts a non-macOS worker (else "refuse everything" passes);
#   3. the hardware floor refuses Apple Silicon on the node's own side, so a
#      Mac says no before it ever dials a coordinator;
#   4. the escape hatch still lifts the seal. That hatch is how the parked
#      path gets revived and measured later; if it silently stopped working,
#      nothing else in the ladder would notice, and unsealing would start with
#      an archaeology session instead of an experiment.
#
# Runs entirely on the control machine: stub workers on loopback plus the local
# worker binary with its vendor byte forced (IDLETOKEN_FAKE_VENDOR=apple), so
# the refusal can be exercised on the CUDA machines that run everything else.
# =====================================================================
g_macseal() {
    local name="$1" repo; repo=$(cd "$(dirname "$0")/.." && pwd)
    local port_a="${IDLETOKEN_MACSEAL_PORT:-14331}"
    # RETIRED (2026-08-14, llama.cpp pivot — v2 plan §1.3): macOS is UNSEALED
    # as a compute node. The compute layer is llama.cpp's own Metal backend
    # now (not the ds4 line this seal guarded), and the mac compute path is
    # gated positively by G_MAC_SMOKE below plus the cross-OS cluster cells in
    # results/matrix-llamacpp-*.jsonl. The seal itself still exists in the
    # legacy ds4 INFER path (deliberately: that line has no Metal kernels and
    # no oracle), so the parked assertions stay runnable, G_DSPARK-style: set
    # IDLETOKEN_MACSEAL_GATE=1 to exercise them.
    if [ "${IDLETOKEN_MACSEAL_GATE:-0}" != "1" ]; then
        skip "$name" "retired 2026-08-14 (macOS compute unsealed on the llama.cpp line — see G_MAC_SMOKE) — set IDLETOKEN_MACSEAL_GATE=1 to run the parked ds4-line seal check"
        return
    fi
    command -v cc      >/dev/null 2>&1 || { skip "$name" "no C compiler on the control machine"; return; }
    command -v python3 >/dev/null 2>&1 || { skip "$name" "python3 needed for the stub worker"; return; }
    (cd "$repo" && make coord >/dev/null 2>&1) || { fail "$name" "make coord failed on the control machine"; return; }

    # The enum value, read from the header instead of hardcoded: this number
    # crosses into the client (client/src/types.ts) and onto users' screens, so
    # the gate must break if the enum is ever renumbered.
    local sealed_code
    sealed_code=$(awk '/IDLETOKEN_HW_OK = 0/ {n = 0; seen = 1; next}
                       seen && /^[[:space:]]+IDLETOKEN_HW_[A-Z_]+/ {
                           n++
                           if ($0 ~ /IDLETOKEN_HW_MACOS_SEALED/) { print n; exit }
                       }' "$repo/include/idletoken_resource.h")
    if [ -z "$sealed_code" ]; then
        fail "$name" "IDLETOKEN_HW_MACOS_SEALED is not in include/idletoken_resource.h — the seal was removed without removing this gate"
        return
    fi

    local tmp; tmp=$(mktemp -d)
    local cpid="" s1="" s2="" s3=""
    _mac_cleanup() {
        for p in $s1 $s2 $s3 $cpid; do kill "$p" 2>/dev/null; done
        wait $s1 $s2 $s3 $cpid 2>/dev/null
        rm -rf "$tmp"
    }
    _mac_wait() {   # 1 if the pid is still alive after 10 s (= it was accepted)
        local p="$1" i=0
        while kill -0 "$p" 2>/dev/null && [ $i -lt 100 ]; do sleep 0.1; i=$((i + 1)); done
        kill -0 "$p" 2>/dev/null && return 1 || return 0
    }

    # --- 1. a macOS worker is refused even with an empty cluster behind it ---
    "$repo/idletoken-coord" --bind "127.0.0.1:$port_a" --num-workers 2 --n-predict 0 \
        >"$tmp/coord_a.log" 2>&1 &
    cpid=$!
    sleep 1
    IDLETOKEN_MOCK_OS_FAMILY=3 python3 "$repo/scripts/mock_worker.py" \
        "127.0.0.1:$port_a" >"$tmp/mac.log" 2>&1 &
    s1=$!
    if ! _mac_wait "$s1"; then
        fail "$name" "a macOS worker was ACCEPTED as the first node of a cluster (still connected)"
        _mac_cleanup; return
    fi
    if grep -q "plan received" "$tmp/mac.log"; then
        fail "$name" "a macOS worker was ACCEPTED as the first node of a cluster (got ASSIGN_PLAN)"
        _mac_cleanup; return
    fi
    if ! grep -q "sealed" "$tmp/coord_a.log"; then
        fail "$name" "coordinator logged no seal refusal (see $tmp/coord_a.log)"
        _mac_cleanup; return
    fi
    if ! grep -q "REFUSED: macOS compute nodes are sealed" "$tmp/mac.log"; then
        fail "$name" "the refused Mac got no reason, only a dead socket (see $tmp/mac.log)"
        _mac_cleanup; return
    fi
    vlog "macOS join refused as first worker, and the refusal reached the node"

    # --- 2. positive control: Linux workers still form a cluster ------------
    IDLETOKEN_MOCK_OS_FAMILY=1 python3 "$repo/scripts/mock_worker.py" \
        "127.0.0.1:$port_a" >"$tmp/s2.log" 2>&1 &
    s2=$!
    IDLETOKEN_MOCK_OS_FAMILY=1 python3 "$repo/scripts/mock_worker.py" \
        "127.0.0.1:$port_a" >"$tmp/s3.log" 2>&1 &
    s3=$!
    local i=0
    while [ $i -lt 100 ] && ! grep -q "cluster ready" "$tmp/coord_a.log"; do sleep 0.1; i=$((i + 1)); done
    if ! grep -q "cluster ready" "$tmp/coord_a.log"; then
        fail "$name" "non-macOS workers did not form a cluster — the seal refuses everything"
        _mac_cleanup; return
    fi
    vlog "non-macOS workers still form a cluster"
    _mac_cleanup

    # --- 3/4. the node's own hardware floor, and the escape hatch ----------
    if [ ! -x "$repo/idletoken-worker" ]; then
        (cd "$repo" && make idletoken-worker >/dev/null 2>&1) || true
    fi
    if [ ! -x "$repo/idletoken-worker" ]; then
        fail "$name" "no idletoken-worker on the control machine — the hardware-floor half of the seal cannot be checked, and a gate that skips its own subject is not a gate"
        return
    fi
    local hw_sealed hw_lifted
    hw_sealed=$(IDLETOKEN_FAKE_VENDOR=apple "$repo/idletoken-worker" --probe-json 2>/dev/null \
                | sed -n 's/.*"hw_status":\([0-9]*\).*/\1/p')
    if [ "$hw_sealed" != "$sealed_code" ]; then
        fail "$name" "an Apple-vendor probe reported hw_status=$hw_sealed, expected $sealed_code (MACOS_SEALED) — a Mac would pass its own hardware floor"
        return
    fi
    hw_lifted=$(IDLETOKEN_FAKE_VENDOR=apple IDLETOKEN_ALLOW_MACOS_NODE=1 \
                "$repo/idletoken-worker" --probe-json 2>/dev/null \
                | sed -n 's/.*"hw_status":\([0-9]*\).*/\1/p')
    # Not "== OK": on a machine with other problems the lifted path may still
    # refuse for a different reason, and that is a correct answer. What must
    # change is that the SEAL is no longer the thing saying no.
    if [ "$hw_lifted" = "$sealed_code" ]; then
        fail "$name" "IDLETOKEN_ALLOW_MACOS_NODE did not lift the seal — the parked macOS path can no longer be revived or measured"
        return
    fi
    vlog "hardware floor: sealed=$hw_sealed, with the hatch=$hw_lifted"

    # --- 5. the client mirrors the same number ----------------------------
    local ts_code
    ts_code=$(sed -n 's/^export const HW_MACOS_SEALED = \([0-9]*\).*/\1/p' \
              "$repo/client/src/types.ts" 2>/dev/null | head -1)
    if [ "$ts_code" != "$sealed_code" ]; then
        fail "$name" "client/src/types.ts says HW_MACOS_SEALED=${ts_code:-missing}, engine says $sealed_code — the UI would show the wrong reason"
        return
    fi

    pass "$name"
}

# =====================================================================
# G_MAC_SMOKE — macOS as a COMPUTE platform (2026-08-14 unsealing, v2 plan
#      §1.3): the pinned llama.cpp Metal build plus the coordinator's llamacpp
#      single-machine mode really serve inference on a Mac. This is the
#      positive gate that replaces G_MACSEAL's refusal assertions — unsealing
#      a platform without a gate would repeat the exact mistake the seal was
#      created to prevent (claiming "supported" with nothing watching it).
#
# Runs on the control machine when it IS a Mac (the testbed's M4 is both the
# control machine and the third compute platform). On a non-mac control
# machine it SKIPs with the reason: the mac cell then lives in the cross-OS
# matrix instead.
# =====================================================================
g_mac_smoke() {
    local name="$1" repo; repo=$(cd "$(dirname "$0")/.." && pwd)
    local base="${IDLETOKEN_MACSMOKE_PORT:-14371}"
    local api_port=$((base + 4000)) llama_port=$((base + 4200))
    local engine_dir="${IDLETOKEN_ENGINE_DIR:-$repo/vendor/llama.cpp/build/bin}"
    if [ "$(uname -s)" != "Darwin" ]; then
        skip "$name" "the control machine is not a Mac — mac compute is covered by the cross-OS matrix cells instead"
        return
    fi
    command -v cc >/dev/null 2>&1 || { skip "$name" "no C compiler on the control machine"; return; }
    if [ ! -x "$engine_dir/llama-server" ]; then
        skip "$name" "no pinned llama.cpp Metal build in $engine_dir (scripts/build_llamacpp.sh)"
        return
    fi
    if [ -z "${IDLETOKEN_SMOKE_GGUF:-}" ] || [ -z "${IDLETOKEN_SMOKE_MODEL_ID:-}" ]; then
        skip "$name" "set IDLETOKEN_SMOKE_GGUF + IDLETOKEN_SMOKE_MODEL_ID to a small local model — mac compute cannot be smoked without weights"
        return
    fi
    [ -r "$IDLETOKEN_SMOKE_GGUF" ] || { fail "$name" "IDLETOKEN_SMOKE_GGUF is set but unreadable: $IDLETOKEN_SMOKE_GGUF"; return; }
    (cd "$repo" && make coord >/dev/null 2>&1) || { fail "$name" "make coord failed on the control machine"; return; }

    local tmp; tmp=$(mktemp -d)
    local cpid=""
    # $1=keep -> leave the logs on disk. A gate that names a log path in its
    # failure message and then deletes that path is not diagnosable: the first
    # full-suite run of this gate failed with "see .../coord.log" pointing at an
    # already-removed directory. Failures keep their evidence; successes clean up.
    _msm_cleanup() {
        [ -n "$cpid" ] && kill "$cpid" 2>/dev/null
        [ -n "$cpid" ] && wait "$cpid" 2>/dev/null
        pkill -f "llama-serve[r].*--port $llama_port" 2>/dev/null
        [ "${1:-}" = keep ] || rm -rf "$tmp"
    }
    (cd "$repo" && exec env IDLETOKEN_LLAMA_LOG="$tmp/llama.log" \
        ./idletoken-coord --model-id "$IDLETOKEN_SMOKE_MODEL_ID" \
        --llama-server-bin "$engine_dir/llama-server" \
        --llama-gguf "$IDLETOKEN_SMOKE_GGUF" --llama-port "$llama_port" \
        --ctx-size 4096 \
        --http --api-bind "127.0.0.1:$api_port" >"$tmp/coord.log" 2>&1) &
    cpid=$!
    local h="" i=0
    while [ $i -lt 120 ]; do
        h=$(curl -s -m 3 "http://127.0.0.1:$api_port/health" 2>/dev/null)
        case "$h" in *'"engine_state":"ready"'*) break ;; esac
        h=""
        sleep 1; i=$((i + 1))
    done
    if [ -z "$h" ]; then
        # Say WHY, not just "never reached ready". Three distinguishable causes,
        # and the first full-suite run of this gate produced none of this detail:
        #   - the coordinator died (its log has the reason);
        #   - a FOREIGN process holds the engine port, so our child dies with
        #     "couldn't bind" while the health probe may transiently see the
        #     stale owner as ready (the hazard recorded in
        #     results/llamacpp-b1-sidecar-20260814.md deviation #5);
        #   - the engine is simply still loading.
        local why="" owner
        if ! kill -0 "$cpid" 2>/dev/null; then
            why="the coordinator process exited"
        fi
        owner=$(lsof -nP -iTCP:"$llama_port" -sTCP:LISTEN 2>/dev/null | awk 'NR>1{print $2}' | sort -u | tr '\n' ' ')
        [ -n "$owner" ] && why="${why:+$why; }engine port $llama_port is held by pid(s) $owner"
        why="${why:-the engine never became ready in 120s}"
        fail "$name" "$why — last engine state: $(grep -o '\"engine_state\":\"[a-z]*\"' <<<"$(curl -s -m 3 "http://127.0.0.1:$api_port/health" 2>/dev/null)" | tail -1), coord log: $(tail -2 "$tmp/coord.log" 2>/dev/null | tr '\n' ' ' | tail -c 200) (logs kept in $tmp)"
        _msm_cleanup keep; return
    fi
    # Both API faces, decidable prompt — same bar as G6, small model. The
    # generous max_tokens is deliberate: this model family thinks before it
    # answers, and a tight budget yields an empty content with
    # finish_reason=length (that is the model, not the serving path).
    local o a
    o=$(curl -s -m 120 "http://127.0.0.1:$api_port/v1/chat/completions" \
        -H 'content-type: application/json' \
        -d '{"model":"x","max_tokens":512,"messages":[{"role":"user","content":"Reply with the single word: pong"}]}' 2>/dev/null \
        | python3 -c '
import json,sys
d=json.load(sys.stdin)
ch=d.get("choices",[])
t=ch[0]["message"]["content"] if ch else ""
u=d.get("usage",{})
if not t.strip(): sys.exit("empty content")
if not (u.get("prompt_tokens",0)>0 and u.get("completion_tokens",0)>0): sys.exit("zero usage")
if "pong" not in t.lower(): sys.exit("off-topic reply: "+t.strip()[:80])
print("OK "+t.strip()[:40])' 2>&1)
    case "$o" in OK*) vlog "openai face on Metal: $o" ;; *)
        fail "$name" "OpenAI face on the mac engine: $o (logs kept in $tmp)"; _msm_cleanup keep; return ;; esac
    a=$(curl -s -m 120 "http://127.0.0.1:$api_port/v1/messages" \
        -H 'content-type: application/json' \
        -d '{"model":"x","max_tokens":512,"messages":[{"role":"user","content":"Reply with the single word: pong"}]}' 2>/dev/null \
        | python3 -c '
import json,sys
d=json.load(sys.stdin)
t="".join(b.get("text","") for b in d.get("content",[]) if b.get("type")=="text")
if not t.strip(): sys.exit("empty text")
if "pong" not in t.lower(): sys.exit("off-topic reply: "+t.strip()[:80])
print("OK "+t.strip()[:40])' 2>&1)
    case "$a" in OK*) vlog "anthropic face on Metal: $a" ;; *)
        fail "$name" "Anthropic face on the mac engine: $a (logs kept in $tmp)"; _msm_cleanup keep; return ;; esac
    _msm_cleanup
    pass "$name"
}

# =====================================================================
# G_API_NS — the route namespace split (docs/api-surface.md §4).
#
#   /v1            carries SOMEBODY ELSE'S protocol (OpenAI, Anthropic) only
#   /idletoken/v1  carries ours, spelled the same on coordinator and gateway
#
# Two halves, and BOTH are needed:
#
#   1. Live routing. New paths answer, the pre-migration spellings are gone, and
#      the vendor routes are untouched. The oracle is **404 vs anything else**,
#      not 200: with a mock cluster the vendor routes legitimately answer 503
#      (routed, no real model). 404 means the route is absent; 503 means it is
#      present and cannot serve. Asserting 200 here would force the gate to
#      carry real weights, and a gate that needs 80 GiB gets skipped forever.
#
#   2. A source sweep. This is the half that actually earns its keep. The paths
#      cross five artifacts (engine C, client Rust, client TS, gateway TS, shell
#      scripts) and only the C is type-checked against the header — a leftover
#      "/v1/stats" in a .sh or .rs file compiles, runs, and quietly 404s. The
#      client dashboard would simply stop showing numbers; nothing turns red.
#      So the sweep fails on ANY pre-migration spelling outside the legacy
#      allowlist, which is short and deliberate (§4.3): the gateway and nginx
#      answer the old rendezvous path for one release because already-installed
#      engines hardcode it, and the metering client falls back to the old
#      tokenize path because that is a billing count, not a dashboard number.
# =====================================================================
g_api_ns() {
    local name="$1" repo; repo=$(cd "$(dirname "$0")/.." && pwd)
    local wire_port="${IDLETOKEN_NS_PORT:-14411}" api_port=$((${IDLETOKEN_NS_PORT:-14411} + 4000))
    command -v cc >/dev/null 2>&1 || { skip "$name" "no C compiler on the control machine"; return; }
    (cd "$repo" && make coord >/dev/null 2>&1) || { fail "$name" "make coord failed on the control machine"; return; }

    # ---- half 2 first: it needs no processes, so a stale reference is reported
    #      even on a machine where the mock cluster cannot come up. -----------
    local stale
    # -I skips binaries. Without it the built coordinator matches: the new
    # spelling /idletoken/v1/stats CONTAINS the old one as a substring, and
    # grep's "Binary file ... matches" line survives the /idletoken/v1 filter
    # below (the line has no path in it), so every green build reported itself.
    stale=$(cd "$repo" && grep -rnI \
        -e '/v1/stats' -e '/v1/capability' -e '/v1/cluster/status' \
        -e '/v1/tokenize' -e '/v1/rendezvous' -e '/v1/privacy' \
        --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=build \
        --exclude-dir=vendor --exclude-dir=results --exclude-dir=baseline \
        --exclude-dir=docs --exclude-dir=dist --exclude-dir=target \
        . 2>/dev/null \
        | grep -v '/idletoken/v1' \
        | grep -v 'platform/packages/gateway/src/metering/tokenizer.service.ts' \
        | grep -v 'platform/packages/gateway/src/rendezvous/rendezvous.controller.ts' \
        | grep -v 'platform/packages/gateway/test/rendezvous.e2e.spec.ts' \
        | grep -v 'platform/ops/nginx-idletoken.conf' \
        | grep -v 'scripts/acceptance.sh')
    if [ -n "$stale" ]; then
        fail "$name" "pre-migration route spelling still referenced (these 404 silently at runtime): $(printf '%s' "$stale" | head -3 | tr '\n' ' ')"
        return
    fi
    vlog "no pre-migration route spellings outside the documented legacy allowlist"

    # ---- half 1: live routing against a mock cluster -----------------------
    local tmp; tmp=$(mktemp -d)
    local cpid="" wpid=""
    _ns_cleanup() {
        [ -n "$wpid" ] && { kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null; }
        [ -n "$cpid" ] && { kill "$cpid" 2>/dev/null; wait "$cpid" 2>/dev/null; }
        rm -rf "$tmp"
    }
    (cd "$repo" && exec ./idletoken-coord --bind "127.0.0.1:$wire_port" --num-workers 1 \
        --n-predict 0 --http --api-bind "127.0.0.1:$api_port" >"$tmp/coord.log" 2>&1) &
    cpid=$!
    sleep 1
    # os_family=1 (Linux): on a Mac control machine the macOS seal (G_MACSEAL)
    # would refuse the mock worker and the cluster would never form -- a red for
    # an unrelated reason.
    (cd "$repo" && IDLETOKEN_MOCK_OS_FAMILY=1 exec python3 scripts/mock_worker.py \
        "127.0.0.1:$wire_port" >"$tmp/mock.log" 2>&1) &
    wpid=$!
    local i=0
    while [ $i -lt 100 ] && ! grep -q "cluster ready" "$tmp/coord.log"; do sleep 0.2; i=$((i + 1)); done
    if ! grep -q "cluster ready" "$tmp/coord.log"; then
        fail "$name" "the mock cluster never came up — cannot check routing (see $tmp/coord.log)"
        _ns_cleanup; return
    fi

    local code
    _ns_code() {  # _ns_code <method> <path> [body]
        if [ "$1" = POST ]; then
            curl -s -o /dev/null -w '%{http_code}' -m 10 -X POST \
                 "http://127.0.0.1:$api_port$2" -H 'content-type: application/json' -d "$3"
        else
            curl -s -o /dev/null -w '%{http_code}' -m 5 "http://127.0.0.1:$api_port$2"
        fi
    }

    # new control-plane paths must answer
    for p in /idletoken/v1/stats /idletoken/v1/capability /idletoken/v1/cluster/status; do
        code=$(_ns_code GET "$p")
        if [ "$code" != 200 ]; then
            fail "$name" "GET $p returned $code, expected 200 — the control plane did not move"
            _ns_cleanup; return
        fi
    done
    code=$(_ns_code POST /idletoken/v1/tokenize '{"text":"hello"}')
    if [ "$code" = 404 ]; then
        fail "$name" "POST /idletoken/v1/tokenize is 404 — the metering route did not move"
        _ns_cleanup; return
    fi
    vlog "control plane answers on /idletoken/v1 (tokenize: $code, 503 = routed but no vocab)"

    # pre-migration spellings must be GONE (the decision was delete, not alias)
    for p in /v1/stats /v1/capability /v1/cluster/status; do
        code=$(_ns_code GET "$p")
        if [ "$code" != 404 ]; then
            fail "$name" "GET $p still answers ($code) — the old control-plane path was not removed"
            _ns_cleanup; return
        fi
    done
    code=$(_ns_code POST /v1/tokenize '{"text":"hello"}')
    if [ "$code" != 404 ]; then
        fail "$name" "POST /v1/tokenize still answers ($code) — the old metering path was not removed"
        _ns_cleanup; return
    fi
    vlog "pre-migration control-plane paths all 404"

    # vendor-compatible routes must be untouched. 503 here is correct (mock
    # cluster, no weights); 404 would mean the migration ate somebody else's
    # protocol, which is the one thing this split must never do.
    for p in /v1/chat/completions /v1/messages; do
        code=$(_ns_code POST "$p" '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":1}')
        if [ "$code" = 404 ]; then
            fail "$name" "POST $p is 404 — the namespace split removed a vendor-compatible route"
            _ns_cleanup; return
        fi
    done
    code=$(_ns_code GET /health)
    if [ "$code" != 200 ]; then
        fail "$name" "GET /health returned $code — the liveness probe moved or broke"
        _ns_cleanup; return
    fi
    vlog "vendor routes (/v1/chat/completions, /v1/messages) and /health unaffected"

    _ns_cleanup
    pass "$name"
}

# =====================================================================
# G_NO_PROMPT_LOG — a shared machine does not write other people's prompts
#      to its own disk (docs/privacy-design.md; audit of 2026-08-13).
#
# The coordinator is the one plaintext window in the sealed path: the agent
# opens the envelope and hands it over loopback. That window is by design. What
# was NOT by design is that the chat handler printed the first 40 characters of
# every prompt to stderr -- which the client captures and the scripts redirect
# to files. The envelope kept the prompt off the wire and the log copied it
# straight back out, onto the disk of a stranger who is renting out compute.
#
# Three assertions, and the middle one is what makes the other two mean
# anything:
#
#   1. by default, a request's text does NOT appear in the log;
#   2. with IDLETOKEN_LOG_PROMPTS=1 and a LOCAL request, it DOES -- the
#      positive control. Without it, a build that logged nothing at all (or
#      one whose log went somewhere else entirely) would pass assertion 1 and
#      the gate would be worthless;
#   3. with IDLETOKEN_LOG_PROMPTS=1 and a PLATFORM-origin request, it does NOT.
#      The operator's debug switch may expose their own prompts; it may never
#      expose a consumer's.
#
# Uses a mock cluster with IDLETOKEN_MOCK_OK: the assertion is about what the
# handler prints before inference, so no weights are needed.
# =====================================================================
g_no_prompt_log() {
    local name="$1" repo; repo=$(cd "$(dirname "$0")/.." && pwd)
    local wire_port="${IDLETOKEN_PROMPTLOG_PORT:-14431}" api_port=$((${IDLETOKEN_PROMPTLOG_PORT:-14431} + 4000))
    command -v cc >/dev/null 2>&1 || { skip "$name" "no C compiler on the control machine"; return; }
    (cd "$repo" && make coord >/dev/null 2>&1) || { fail "$name" "make coord failed on the control machine"; return; }

    # Needs a REAL vocabulary. Under IDLETOKEN_MOCK_OK the request short-circuits
    # into the mock-completion branch and never reaches the tokenize step that
    # owns the log line -- the first version of this gate did exactly that, and
    # its own "was anything logged at all?" sentinel caught it. Rather than
    # assert against a path that cannot leak, skip loudly and name what is
    # unchecked (same contract as G_API_MODELS/count_tokens).
    if [ -z "${IDLETOKEN_SMOKE_GGUF:-}" ] || [ -z "${IDLETOKEN_SMOKE_MODEL_ID:-}" ]; then
        skip "$name" "set IDLETOKEN_SMOKE_GGUF + IDLETOKEN_SMOKE_MODEL_ID to a small local model — without a real vocabulary the request never reaches the line under test"
        return
    fi
    if [ ! -r "$IDLETOKEN_SMOKE_GGUF" ]; then
        fail "$name" "IDLETOKEN_SMOKE_GGUF is set but unreadable: $IDLETOKEN_SMOKE_GGUF"; return
    fi
    # llamacpp single-machine path (ported 2026-08-20): the line under test is
    # llama_chat_route's tokenize log — the product path. The previous
    # incarnation drove the retired ds4 wire (mock worker + built-in ds4x
    # tokenizer, shelved 2026-08-16), whose chat endpoints now refuse, so the
    # sentinel red-lined on "no request was logged at all".
    local engine_dir="${IDLETOKEN_ENGINE_DIR:-$repo/vendor/llama.cpp/build/bin}"
    if [ ! -x "$engine_dir/llama-server" ]; then
        skip "$name" "no pinned llama.cpp build in $engine_dir (scripts/build_llamacpp.sh)"
        return
    fi
    local llama_port=$((wire_port + 4300))

    # A phrase that could not plausibly appear in the engine's own output.
    local secret="zzq-canary-prompt-9f3a1c"
    local tmp cpid wpid=""
    _pl_up() {   # _pl_up <logfile> [extra env]
        tmp=$(mktemp -d)
        # `exec env ...`, not `env ... exec ...`: the latter hands the literal
        # word "exec" to env as the program to run, which fails before the
        # coordinator is ever started -- and the symptom is an unhelpful
        # "cluster did not come up".
        (cd "$repo" && exec env $2 IDLETOKEN_LLAMA_LOG="$tmp/llama.log" \
            ./idletoken-coord --model-id "$IDLETOKEN_SMOKE_MODEL_ID" \
            --llama-server-bin "$engine_dir/llama-server" \
            --llama-gguf "$IDLETOKEN_SMOKE_GGUF" --llama-port "$llama_port" \
            --ctx-size 4096 \
            --http --api-bind "127.0.0.1:$api_port" >"$1" 2>&1) &
        cpid=$!
        local h="" i=0
        while [ $i -lt 120 ]; do
            h=$(curl -s -m 3 "http://127.0.0.1:$api_port/health" 2>/dev/null)
            case "$h" in *'"engine_state":"ready"'*) return 0 ;; esac
            sleep 1; i=$((i + 1))
        done
        return 1
    }
    # Kill by PID *and* by the port in the command line. `( ... ) &` does not
    # always leave $! pointing at the process that ends up holding the socket,
    # and a leaked coordinator makes the NEXT sub-run fail with "Address already
    # in use" -- which reads exactly like the gate's own failure message. The
    # pattern is scoped to this gate's ports so it cannot disturb other gates.
    _pl_kill_stray() {
        pkill -f "idletoken-coord.*--api-bind 127.0.0.1:$api_port" 2>/dev/null
        pkill -f "llama-serve[r].*--port $llama_port" 2>/dev/null
        sleep 0.3
    }
    _pl_down() {
        [ -n "$wpid" ] && { kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null; }
        [ -n "$cpid" ] && { kill "$cpid" 2>/dev/null; wait "$cpid" 2>/dev/null; }
        _pl_kill_stray
        rm -rf "$tmp"
    }
    _pl_ask() {  # _pl_ask [origin-header]
        # `${hdr[@]+...}`: under `set -u` on macOS's bash 3.2 a bare "${hdr[@]}"
        # on an EMPTY array is an unbound-variable error, not an empty list.
        # Same idiom the ladder already uses for RESULTS.
        local hdr=()
        [ -n "$1" ] && hdr=(-H "X-IdleToken-Origin: $1")
        curl -s -o /dev/null -m 30 -X POST "http://127.0.0.1:$api_port/v1/chat/completions" \
             -H 'content-type: application/json' ${hdr[@]+"${hdr[@]}"} \
             -d "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"$secret\"}],\"max_tokens\":4}"
    }

    _pl_kill_stray   # a leftover from an interrupted earlier run would look like a gate failure

    local log
    # --- 1. default: the text must not be in the log ------------------------
    log=$(mktemp)
    if ! _pl_up "$log" ""; then fail "$name" "mock cluster did not come up (see $log)"; _pl_down; return; fi
    _pl_ask ""
    sleep 0.3
    if grep -q "$secret" "$log"; then
        fail "$name" "the prompt text was written to the coordinator log by default (see $log)"
        _pl_down; rm -f "$log"; return
    fi
    if ! grep -q "tokenized .*-tok prompt" "$log"; then
        fail "$name" "no request was logged at all — assertion 1 would pass for the wrong reason (see $log)"
        _pl_down; rm -f "$log"; return
    fi
    _pl_down; rm -f "$log"
    vlog "default: request logged, text absent"

    # --- 2. positive control: opt-in + LOCAL origin must quote it -----------
    log=$(mktemp)
    if ! _pl_up "$log" "IDLETOKEN_LOG_PROMPTS=1"; then fail "$name" "mock cluster did not come up (see $log)"; _pl_down; return; fi
    _pl_ask ""
    sleep 0.3
    if ! grep -q "$secret" "$log"; then
        fail "$name" "IDLETOKEN_LOG_PROMPTS=1 did not quote a LOCAL prompt — assertion 1 proves nothing if the gate cannot see the text even when it is meant to be there (see $log)"
        _pl_down; rm -f "$log"; return
    fi
    _pl_down; rm -f "$log"
    vlog "opt-in + local origin: text present (the gate can see it when it is there)"

    # --- 3. opt-in must NOT quote a platform-dispatched prompt --------------
    log=$(mktemp)
    if ! _pl_up "$log" "IDLETOKEN_LOG_PROMPTS=1"; then fail "$name" "mock cluster did not come up (see $log)"; _pl_down; return; fi
    _pl_ask "platform"
    sleep 0.3
    if grep -q "$secret" "$log"; then
        fail "$name" "a PLATFORM-dispatched prompt was quoted in the log — the operator's debug switch exposed a consumer's content (see $log)"
        _pl_down; rm -f "$log"; return
    fi
    if ! grep -q "origin=platform" "$log"; then
        fail "$name" "the request was not recognised as platform-origin, so assertion 3 tested nothing (see $log)"
        _pl_down; rm -f "$log"; return
    fi
    _pl_down; rm -f "$log"
    vlog "opt-in + platform origin: text absent, and the origin really was recognised"

    pass "$name"
}

# =====================================================================
# G_LOCAL_TOKEN — the local API is closed on a fresh install
#      (docs/api-surface.md §3 decision 3, §5.3).
#
# The engine has always accepted `--api-token` and always defaulted to open;
# what changed is that the client now generates one for a new install. That is a
# behaviour of `loadSettings()`, so the gate runs the real function rather than
# grepping for it: bundle settings.ts with the vite that already builds the
# client, then drive it against a fake localStorage.
#
# Four claims, and the last two matter as much as the first:
#
#   1. a fresh install gets a token, 32 hex chars;
#   2. two fresh installs get DIFFERENT tokens (a constant baked into the
#      defaults object would satisfy claim 1 and be worthless — same token on
#      every machine we ship);
#   3. an UPGRADED install with an empty token keeps it empty. Filling one in
#      would 401 every curl, script and Claude Code config that machine already
#      had working, with no visible cause;
#   4. a token the user actually set is preserved.
#
# Claims 3 and 4 are not hypothetical: while writing this gate the harness used
# the wrong storage key, every case took the fresh-install branch, and the run
# said a user's own token had been overwritten. The check caught its own setup
# — which is the point of asserting behaviour instead of matching source text.
# =====================================================================
g_local_token() {
    local name="$1" repo; repo=$(cd "$(dirname "$0")/.." && pwd)
    [ -d "$repo/client/node_modules/vite" ] || { skip "$name" "client deps not installed (npm i in client/)"; return; }
    local out; out=$(mktemp -d)
    if ! (cd "$repo/client" && node -e "
const {build}=require('vite');
build({configFile:false,logLevel:'error',build:{lib:{entry:'src/settings.ts',formats:['cjs'],fileName:()=>'settings.cjs'},outDir:'$out',emptyOutDir:true,minify:false}})
  .then(()=>process.exit(0)).catch(e=>{console.error(e.message);process.exit(1)});
" >"$out/build.log" 2>&1); then
        fail "$name" "could not bundle client/src/settings.ts (see $out/build.log)"
        return
    fi
    local verdict
    verdict=$(node -e "
const store = new Map();
globalThis.localStorage = {
  getItem: k => store.has(k) ? store.get(k) : null,
  setItem: (k,v) => store.set(k,v),
  removeItem: k => store.delete(k),
};
const S = require('$out/settings.cjs');
// The real storage key, read from the module's own behaviour: save a settings
// object and see which key it lands under. Hardcoding a guess here is exactly
// how this gate first fooled itself.
S.saveSettings({ probe: 1 });
const K = [...store.keys()][0];
store.clear();
const fail = m => { console.log('FAIL ' + m); process.exit(0); };
const a = S.loadSettings();
if (!/^[0-9a-f]{32}\$/.test(a.apiToken || '')) fail('a fresh install got apiToken=' + JSON.stringify(a.apiToken) + ', expected 32 hex chars');
const b = S.loadSettings();
if (a.apiToken === b.apiToken) fail('two fresh installs got the SAME token — it is a constant, not a secret');
store.set(K, JSON.stringify({ schemaVersion: 99, apiToken: '' }));
if (S.loadSettings().apiToken !== '') fail('an upgraded install with no token had one filled in — every existing client config on that machine starts 401ing');
store.set(K, JSON.stringify({ schemaVersion: 99, apiToken: 'user-chose-this' }));
if (S.loadSettings().apiToken !== 'user-chose-this') fail('a user-set token was not preserved');
console.log('OK ' + K);
" 2>&1)
    case "$verdict" in
        OK\ *) vlog "fresh install closed by default; upgrades and user tokens untouched (key ${verdict#OK })" ;;
        FAIL\ *) fail "$name" "${verdict#FAIL }"; rm -rf "$out"; return ;;
        *) fail "$name" "could not run the settings check: $(printf '%s' "$verdict" | head -3 | tr '\n' ' ')"; rm -rf "$out"; return ;;
    esac
    rm -rf "$out"
    pass "$name"
}

# =====================================================================
# G_API_MODELS — the two vendor routes a "use it like any other API" client
#      needs (docs/api-surface.md §6): GET /v1/models and
#      POST /v1/messages/count_tokens.
#
# The coordinator half. The platform half lives in the gateway's own suite
# (test/models-endpoint.e2e.spec.ts) because it needs a database.
#
# Two claims, and the second one is the reason this gate exists at all:
#
#   1. /v1/models lists EXACTLY the loaded model. Not the registry — the chat
#      handler never reads body.model, so any extra id we listed would accept a
#      request and answer with a different model's output.
#
#   2. count_tokens equals what a real request actually prefills. Its only use
#      is context budgeting; a number that disagrees with the prefill is worse
#      than no number. Both paths call coord_encode_request_prompt, so today
#      they agree by construction — this assertion is what stops someone from
#      later "simplifying" count_tokens into a second implementation.
#
# Claim 2 needs a real vocabulary, which the mock cluster has not got. Rather
# than let it pass quietly on a machine that never checked it, it is reported as
# its own SKIP line naming the two variables that enable it. A gate that silently
# drops half its coverage is how this repo gets green ladders that mean nothing.
# =====================================================================
g_api_models() {
    local name="$1" repo; repo=$(cd "$(dirname "$0")/.." && pwd)
    local wire_port="${IDLETOKEN_MODELS_PORT:-14421}" api_port=$((${IDLETOKEN_MODELS_PORT:-14421} + 4000))
    command -v cc >/dev/null 2>&1 || { skip "$name" "no C compiler on the control machine"; return; }
    (cd "$repo" && make coord >/dev/null 2>&1) || { fail "$name" "make coord failed on the control machine"; return; }

    local tmp; tmp=$(mktemp -d)
    local cpid="" wpid=""
    _gm_cleanup() {
        [ -n "$wpid" ] && { kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null; }
        [ -n "$cpid" ] && { kill "$cpid" 2>/dev/null; wait "$cpid" 2>/dev/null; }
        # claim 2's engine sidecar; the port guard keeps claim 1 (no engine)
        # from pkilling every llama-server on the machine.
        [ -n "${llama_port:-}" ] && pkill -f "llama-serve[r].*--port $llama_port" 2>/dev/null
        rm -rf "$tmp"
    }

    # ---- claim 1: /v1/models on a mock cluster (no weights needed) ---------
    (cd "$repo" && exec ./idletoken-coord --bind "127.0.0.1:$wire_port" --num-workers 1 \
        --n-predict 0 --http --api-bind "127.0.0.1:$api_port" >"$tmp/coord.log" 2>&1) &
    cpid=$!
    sleep 1
    (cd "$repo" && IDLETOKEN_MOCK_OS_FAMILY=1 exec python3 scripts/mock_worker.py \
        "127.0.0.1:$wire_port" >"$tmp/mock.log" 2>&1) &
    wpid=$!
    local i=0
    while [ $i -lt 100 ] && ! grep -q "cluster ready" "$tmp/coord.log"; do sleep 0.2; i=$((i + 1)); done
    if ! grep -q "cluster ready" "$tmp/coord.log"; then
        fail "$name" "the mock cluster never came up (see $tmp/coord.log)"
        _gm_cleanup; return
    fi

    local body served n_ids
    body=$(curl -s -m 5 "http://127.0.0.1:$api_port/v1/models")
    # The id the coordinator says it is serving, from its own stats endpoint —
    # not a literal here, or the gate would pin one model forever.
    served=$(curl -s -m 5 "http://127.0.0.1:$api_port/idletoken/v1/stats" \
             | sed -n 's/.*"model":"\([^"]*\)".*/\1/p')
    n_ids=$(printf '%s' "$body" | grep -o '"object":"model"' | wc -l | tr -d ' ')
    if [ "$n_ids" != 1 ]; then
        fail "$name" "GET /v1/models listed $n_ids models, expected exactly 1 (the loaded one); body: $(printf '%s' "$body" | cut -c1-160)"
        _gm_cleanup; return
    fi
    case "$body" in
        *'"object":"list"'*) : ;;
        *) fail "$name" "GET /v1/models is not an OpenAI list envelope: $(printf '%s' "$body" | cut -c1-160)"; _gm_cleanup; return ;;
    esac
    if [ -z "$served" ] || ! printf '%s' "$body" | grep -q "\"id\":\"$served\""; then
        fail "$name" "GET /v1/models does not name the loaded model ('$served'): $(printf '%s' "$body" | cut -c1-160)"
        _gm_cleanup; return
    fi
    vlog "/v1/models lists exactly the loaded model ($served)"
    _gm_cleanup

    # ---- claim 2: count_tokens == the real prefill (needs a vocabulary) ----
    if [ -z "${IDLETOKEN_SMOKE_GGUF:-}" ] || [ -z "${IDLETOKEN_SMOKE_MODEL_ID:-}" ]; then
        pass "$name"
        skip "$name/count_tokens" "set IDLETOKEN_SMOKE_GGUF + IDLETOKEN_SMOKE_MODEL_ID to a small local model to check count_tokens against the real prefill"
        return
    fi
    if [ ! -r "$IDLETOKEN_SMOKE_GGUF" ]; then
        fail "$name" "IDLETOKEN_SMOKE_GGUF is set but unreadable: $IDLETOKEN_SMOKE_GGUF"
        return
    fi
    # Claim 2 runs the llamacpp single-machine path: count_tokens and the chat
    # relay both call llama_prompt_token_count() (engine template + tokenizer),
    # so this is the pairing the assertion exists to keep welded. The previous
    # incarnation drove the retired ds4 wire (mock worker + built-in ds4x
    # tokenizer) — shelved 2026-08-16, so its chat endpoints refuse and both
    # numbers came back empty. Same disease as pre-T15 G5, same cure: port the
    # oracle to the line the product actually runs.
    local engine_dir="${IDLETOKEN_ENGINE_DIR:-$repo/vendor/llama.cpp/build/bin}"
    if [ ! -x "$engine_dir/llama-server" ]; then
        skip "$name/count_tokens" "no pinned llama.cpp build in $engine_dir (scripts/build_llamacpp.sh)"
        pass "$name"
        return
    fi
    local a2 llama_port
    a2=$((api_port + 2)); llama_port=$((wire_port + 4300))
    tmp=$(mktemp -d); cpid=""; wpid=""
    (cd "$repo" && exec env IDLETOKEN_LLAMA_LOG="$tmp/llama.log" \
        ./idletoken-coord --model-id "$IDLETOKEN_SMOKE_MODEL_ID" \
        --llama-server-bin "$engine_dir/llama-server" \
        --llama-gguf "$IDLETOKEN_SMOKE_GGUF" --llama-port "$llama_port" \
        --ctx-size 4096 \
        --http --api-bind "127.0.0.1:$a2" >"$tmp/coord.log" 2>&1) &
    cpid=$!
    local h=""
    i=0
    while [ $i -lt 120 ]; do
        h=$(curl -s -m 3 "http://127.0.0.1:$a2/health" 2>/dev/null)
        case "$h" in *'"engine_state":"ready"'*) break ;; esac
        h=""
        sleep 1; i=$((i + 1))
    done
    if [ -z "$h" ]; then
        fail "$name" "the $IDLETOKEN_SMOKE_MODEL_ID engine never reached ready (see $tmp/coord.log and $tmp/llama.log)"
        pkill -f "llama-serve[r].*--port $llama_port" 2>/dev/null
        _gm_cleanup; return
    fi

    # A multi-turn body with a system prompt: the chat template's framing is the
    # whole point, so a single bare message would not distinguish the two
    # implementations this assertion exists to keep welded together.
    local cbody='{"model":"x","system":"You are terse.","messages":[{"role":"user","content":"hello world"},{"role":"assistant","content":"hi"},{"role":"user","content":"and now a longer follow-up question"}]}'
    local counted prefilled
    counted=$(curl -s -m 20 -X POST "http://127.0.0.1:$a2/v1/messages/count_tokens" \
              -H 'content-type: application/json' -d "$cbody" \
              | sed -n 's/.*"input_tokens":\([0-9]*\).*/\1/p')
    local before; before=$(wc -l < "$tmp/coord.log")
    curl -s -m 120 -o /dev/null -X POST "http://127.0.0.1:$a2/v1/messages" \
         -H 'content-type: application/json' -d "$cbody"
    prefilled=$(tail -n +$((before + 1)) "$tmp/coord.log" \
                | sed -n 's/.*tokenized \([0-9]*\)-tok prompt.*/\1/p' | head -1)
    if [ -z "$counted" ] || [ -z "$prefilled" ]; then
        fail "$name" "could not read both numbers (count_tokens='$counted', prefill='$prefilled'; see $tmp/coord.log)"
        _gm_cleanup; return
    fi
    if [ "$counted" != "$prefilled" ]; then
        fail "$name" "count_tokens says $counted but the request prefilled $prefilled — the two encoders have drifted apart"
        _gm_cleanup; return
    fi
    vlog "count_tokens == real prefill ($counted tokens, $IDLETOKEN_SMOKE_MODEL_ID)"
    _gm_cleanup
    pass "$name"
}

# =====================================================================
# G_SIZE — large models cluster, small models do not. The coordinator must
#      REFUSE to serve a single-node model on more than one machine (CLAUDE.md
#      hard constraint, 2026-08-12), and must still serve a cluster model
#      across many.
#
# Runs on the control machine with no cluster and no weights: every assertion
# is about a startup decision the coordinator makes before it opens a socket,
# so `--num-workers 2` never has to be satisfied.
#
# BOTH halves are checked, for the same reason G_HOMO checks both: a build that
# refused every multi-node start would pass the negative half on its own. And
# the escape hatch is checked too — the cross-machine gates depend on it, so a
# silent change to its name would disable them without turning anything red.
# =====================================================================
g_size() {
    local name="$1" repo; repo=$(cd "$(dirname "$0")/.." && pwd)
    # One port per step. Reusing one made step 3 fail with "Address already in
    # use" from step 2's just-killed listener, which reads exactly like the
    # policy check refusing the model -- a red for the wrong reason.
    local base="${IDLETOKEN_SIZE_PORT:-14321}"
    local port_a=$base port_b=$((base + 1)) port_c=$((base + 2)) port_d=$((base + 3))
    command -v cc >/dev/null 2>&1 || { skip "$name" "no C compiler on the control machine"; return; }
    (cd "$repo" && make coord >/dev/null 2>&1) || { fail "$name" "make coord failed on the control machine"; return; }

    # Pick the models from the manifests rather than naming them here: the
    # registry decides which are small, and a hardcoded id here would keep
    # passing after that model was retired.
    local small large
    small=$(python3 - <<'PY'
import glob, json
for f in sorted(glob.glob("models/*.json")):
    m = json.load(open(f))
    if m.get("available") and m.get("deployment") == "single-node":
        print(m["id"]); break
PY
)
    large=$(python3 - <<'PY'
import glob, json
for f in sorted(glob.glob("models/*.json")):
    m = json.load(open(f))
    if m.get("available") and m.get("deployment") == "cluster":
        print(m["id"]); break
PY
)
    [ -n "$small" ] && [ -n "$large" ] || {
        fail "$name" "need one available single-node and one available cluster model in models/*.json (got '$small' / '$large')"
        return
    }

    # 1. small model + 2 nodes => refused, with a reason the client can show.
    #
    # Backgrounded with a log rather than `out=$(...)`: a coordinator that does
    # NOT refuse goes on to wait for workers forever, and the command
    # substitution would block on its stdout — the gate would HANG instead of
    # failing. A check that cannot report the very thing it is looking for is
    # not a check. (Found by running this gate with the override exported.)
    local log; log=$(mktemp)
    (cd "$repo" && exec ./idletoken-coord --model-id "$small" --num-workers 2 --n-predict 0 \
        --bind "127.0.0.1:$port_a" >"$log" 2>&1) &
    local pid=$!
    local i=0
    while [ $i -lt 100 ] && ! grep -qE "JOIN_REFUSED: |waiting for worker" "$log"; do sleep 0.1; i=$((i + 1)); done
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    if grep -q "waiting for worker" "$log"; then
        fail "$name" "coordinator accepted a 2-node start for the single-node model $small (see $log)"
        rm -f "$log"; return
    fi
    if ! grep -q "JOIN_REFUSED: " "$log"; then
        fail "$name" "$small was refused without the JOIN_REFUSED marker — the client cannot show a reason (see $log)"
        rm -f "$log"; return
    fi
    vlog "$small refused on 2 nodes: $(grep -o 'JOIN_REFUSED: .*' "$log" | head -1 | cut -c1-90)"
    rm -f "$log"

    # 2. ...but one node is fine. Reaching the "waiting for worker" line means
    #    the deployment check let it through; we kill it there rather than
    #    supply a worker, because a worker would need real weights.
    log=$(mktemp)
    (cd "$repo" && exec ./idletoken-coord --model-id "$small" --num-workers 1 --n-predict 0 \
        --bind "127.0.0.1:$port_b" >"$log" 2>&1) &
    pid=$!
    i=0
    while [ $i -lt 60 ] && ! grep -q "waiting for worker" "$log"; do sleep 0.1; i=$((i + 1)); done
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    if ! grep -q "waiting for worker" "$log"; then
        fail "$name" "$small was blocked on ONE machine too — the check refuses everything (see $log)"
        rm -f "$log"; return
    fi
    rm -f "$log"
    vlog "$small starts fine on one machine"

    # 3. a cluster model on 2 nodes must NOT be refused by this check.
    log=$(mktemp)
    (cd "$repo" && exec ./idletoken-coord --model-id "$large" --num-workers 2 --n-predict 0 \
        --bind "127.0.0.1:$port_c" >"$log" 2>&1) &
    pid=$!
    i=0
    while [ $i -lt 60 ] && ! grep -q "waiting for worker" "$log"; do sleep 0.1; i=$((i + 1)); done
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    if grep -q "JOIN_REFUSED" "$log"; then
        fail "$name" "the cluster model $large was refused a 2-node start (see $log)"; rm -f "$log"; return
    fi
    if ! grep -q "waiting for worker" "$log"; then
        fail "$name" "$large never reached the worker wait on 2 nodes (see $log)"; rm -f "$log"; return
    fi
    rm -f "$log"
    vlog "$large accepted on 2 nodes"

    # 4. the escape hatch the cross-machine gates rely on still opens. Same
    #    background-and-kill shape as above rather than `timeout`, which the
    #    control machine may not have (macOS ships none).
    log=$(mktemp)
    (cd "$repo" && export IDLETOKEN_ALLOW_SMALL_CLUSTER=1 && exec ./idletoken-coord \
        --model-id "$small" --num-workers 2 --n-predict 0 \
        --bind "127.0.0.1:$port_d" >"$log" 2>&1) &
    pid=$!
    i=0
    while [ $i -lt 60 ] && ! grep -q "waiting for worker" "$log"; do sleep 0.1; i=$((i + 1)); done
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    if ! grep -q "IDLETOKEN_ALLOW_SMALL_CLUSTER=1" "$log" || ! grep -q "waiting for worker" "$log"; then
        fail "$name" "IDLETOKEN_ALLOW_SMALL_CLUSTER=1 did not let $small start on 2 nodes — the cross-machine gates have lost their vehicle (see $log)"
        rm -f "$log"; return
    fi
    rm -f "$log"
    vlog "override still works, and says so"

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
    # PARKED (2026-08-03, user's call): the DSpark work stayed in the tree but
    # was not in use; its sub-checks load the 80 GiB model three times, which is
    # a lot of ladder time for a feature nothing calls.
    #
    # RETIRED (2026-08-20, T15): parking implied "coming back". It is not. The
    # subject is speculative decoding for DSv4 ON THE ds4 BACKEND, and all three
    # legs of that are gone: ds4 was shelved from the test and public surface on
    # 2026-08-16 (CLAUDE.md hard constraint #1), speculative decoding is on v2's
    # out-of-scope list, and the sub-checks below compile against
    # build/worker/vendor/ds4_cuda.o with nvcc — which no current build
    # produces. Sub-gates 2-4 of docs/acceptance-criteria.md §G-DSPARK
    # (DRAFT/VERIFY/PERF) were never implemented and now never will be; the md
    # entry keeps the history.
    # The code stays runnable behind the env, same contract as G4 / G_HOMO /
    # G_MACSEAL: an archaeologist gets a gate, not a surprise.
    if [ "${IDLETOKEN_DSPARK_GATE:-0}" != "1" ]; then
        skip "$name" "retired 2026-08-20 (ds4 shelved 2026-08-16; speculative decoding is out of v2 scope) — set IDLETOKEN_DSPARK_GATE=1 to run the parked ds4 check"
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
# G_UPDATE / G_TRAY -- the two shell-level promises of the desktop client:
# it can update itself safely, and it keeps serving when its window is closed.
#
# Both delegate to their own scripts (scripts/client_update_gate.sh /
# client_tray_gate.sh) for the same reason G_SCHED does: the assertions are long
# enough to deserve a file, and that file is runnable by hand while working on
# the feature.
#
# WHICH MACHINE. A GUI client is being driven, so it needs a machine with a
# desktop. `IDLETOKEN_CLIENT_NODE` names it; when it is unset, the first Windows
# worker node in the testbed is used, because Windows is where these two
# features actually mean something (the notification area, and an NSIS
# installer handover). With no Windows node configured the gate falls back to
# the control machine and SAYS SO -- a macOS pass is a smoke test, not the
# promise.
# =====================================================================

# Echo the node these two gates should drive, or nothing for "this machine".
client_gui_node() {
    if [ -n "${IDLETOKEN_CLIENT_NODE:-}" ]; then printf '%s' "$IDLETOKEN_CLIENT_NODE"; return; fi
    local n prof
    for n in ${WORKER_NODES[@]+"${WORKER_NODES[@]}"}; do
        prof="$(testbed_profile "$n")"
        case "$prof" in [A-Za-z]:/*) printf '%s' "$n"; return ;; esac
    done
}

g_update() {
    local name="$1" node out
    node="$(client_gui_node)"
    [ -n "$node" ] || vlog "no Windows client node configured -- running the update gate on this machine (smoke test only)"
    out=$(IDLETOKEN_CLIENT_NODE="$node" bash scripts/client_update_gate.sh 2>&1 | grep -E "^UPDATE_GATE_(OK|FAIL|SKIP)" | tail -1)
    case "$out" in
        UPDATE_GATE_OK)
            vlog "update check/download/signature verified on ${node:-this machine} (5 cases incl. tampered artifact refused)"
            pass "$name" ;;
        UPDATE_GATE_SKIP*) skip "$name" "${out#UPDATE_GATE_SKIP: }" ;;
        "") fail "$name" "scripts/client_update_gate.sh reached no conclusion" ;;
        *) fail "$name" "${out#UPDATE_GATE_FAIL: }" ;;
    esac
}

g_tray() {
    local name="$1" node out
    node="$(client_gui_node)"
    [ -n "$node" ] || vlog "no Windows client node configured -- running the tray gate on this machine (smoke test only)"
    out=$(IDLETOKEN_CLIENT_NODE="$node" bash scripts/client_tray_gate.sh 2>&1 | grep -E "^TRAY_GATE_(OK|FAIL|SKIP)" | tail -1)
    case "$out" in
        TRAY_GATE_OK)
            vlog "close-to-tray, geometry and start-in-tray verified on ${node:-this machine}"
            pass "$name" ;;
        TRAY_GATE_SKIP*) skip "$name" "${out#TRAY_GATE_SKIP: }" ;;
        "") fail "$name" "scripts/client_tray_gate.sh reached no conclusion" ;;
        *) fail "$name" "${out#TRAY_GATE_FAIL: }" ;;
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
# Run a command with a wall-clock deadline, collecting output in a FILE.
#
# Two separate hazards, both observed on 2026-08-14 running the full ladder:
#   1. `cmd | grep | tail` blocks until the write end of the pipe is closed by
#      EVERY holder -- so a daemonized descendant that outlives the script (jest
#      "Force exiting" leaves such handles) hangs the whole ladder forever, long
#      after the gate itself has finished. Writing to a file removes the pipe.
#   2. macOS ships no coreutils `timeout`, so the ladder had no local deadline
#      primitive at all.
# The killer targets the process GROUP so leaked descendants go too.
run_deadline() {  # run_deadline <seconds> <outfile> <command...>
    local secs="$1" outf="$2"; shift 2
    ( "$@" >"$outf" 2>&1 ) &
    local pid=$! i=0
    while kill -0 "$pid" 2>/dev/null && [ "$i" -lt "$secs" ]; do sleep 1; i=$((i + 1)); done
    if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null; sleep 2; kill -KILL "$pid" 2>/dev/null
        return 124
    fi
    wait "$pid" 2>/dev/null
    return 0
}

g_sched() {
    local name="$1"
    # The scheduler gate script belongs to the platform business layer and is
    # deliberately not shipped in the public mirror. Its absence is a fact
    # about this checkout, not a failure of anything under test.
    if [ ! -f "$REPO_ROOT/scripts/scheduler_gate.sh" ]; then
        skip "$name" "scripts/scheduler_gate.sh is not present in this checkout (platform business layer, not shipped publicly)"
        return
    fi
    local out log; log=$(mktemp)
    if ! run_deadline "${IDLETOKEN_SCHED_TIMEOUT:-900}" "$log" bash scripts/scheduler_gate.sh; then
        out=$(grep -E "^G_SCHED_(OK|FAIL|SKIP)" "$log" | tail -1)
        if [ -n "$out" ]; then
            # It DID reach a verdict; only the cleanup overstayed (a leaked
            # descendant kept running). Honour the verdict and say so.
            vlog "scheduler_gate.sh reached a verdict but did not exit within the deadline (leaked background handle); using the verdict"
        else
            fail "$name" "scripts/scheduler_gate.sh did not finish within ${IDLETOKEN_SCHED_TIMEOUT:-900}s (see $log)"
            return
        fi
    else
        out=$(grep -E "^G_SCHED_(OK|FAIL|SKIP)" "$log" | tail -1)
    fi
    rm -f "$log"
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
    # Same rule as G_SCHED: the platform e2e script is business-layer and not
    # shipped in the public mirror, so a checkout without it SKIPs honestly.
    if [ ! -f "$REPO_ROOT/scripts/platform_e2e_real_coord.sh" ]; then
        skip "$name" "scripts/platform_e2e_real_coord.sh is not present in this checkout (platform business layer, not shipped publicly)"
        return
    fi
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
# G_PRIV7 — raw embeddings never leave the coordinator (v2 plan §5.1, the
#      layer-0 privacy invariant). Delegates to
#      scripts/gpriv7_embedding_check.sh, which taps the RPC byte stream and
#      FIRST proves it can recover embedding rows from a deliberately bad
#      configuration (all layers remote, plaintext) before certifying that
#      the product configuration leaks none. A checker that has not shown it
#      can go red proves nothing — that lesson is the whole design.
# =====================================================================
g_priv7() {
    local name="$1" out
    out=$(bash scripts/gpriv7_embedding_check.sh 2>&1 | grep -E "^G_PRIV7_(OK|FAIL|SKIP)" | tail -1)
    case "$out" in
        G_PRIV7_OK*)
            vlog "${out#G_PRIV7_OK }"
            pass "$name" ;;
        G_PRIV7_SKIP*) skip "$name" "${out#G_PRIV7_SKIP: }" ;;
        "") fail "$name" "scripts/gpriv7_embedding_check.sh reached no conclusion" ;;
        *) fail "$name" "${out#G_PRIV7_FAIL: }" ;;
    esac
}

# =====================================================================
# G_PPL — distribution-level numeric gate (v2 plan §1.10 / F3): the engine's
#      perplexity over a fixed committed corpus stays inside a recorded band.
#      Replaces the exact-token-match oracles (G4's token-id equality, the
#      ds4x bit-exact channels) that decision #10 retired. Delegates to
#      scripts/ppl_gate.sh; the reference band lives in
#      test-assets/ppl/<model>.<quant>.json.
# =====================================================================
# G-INTEGRITY (threat-model.md): the curated-model hash gate and canary
# honesty. Local — no cluster node, no GPU, no weights — and every check in the
# script carries its own positive control (see the script's header).
g_integrity() {
    local name="$1" out
    out=$(bash scripts/integrity_gate.sh 2>&1 | grep -E "^G_INTEGRITY_(OK|FAIL|SKIP)" | tail -1)
    case "$out" in
        G_INTEGRITY_OK*)
            vlog "${out#G_INTEGRITY_OK: }"
            pass "$name" ;;
        G_INTEGRITY_SKIP*) skip "$name" "${out#G_INTEGRITY_SKIP: }" ;;
        "") fail "$name" "scripts/integrity_gate.sh reached no conclusion" ;;
        *) fail "$name" "${out#G_INTEGRITY_FAIL:}" ;;
    esac
}

# G-SHARED (shared-mode-plan-2026-08.md): when this machine executes somebody
# else's request, that person's prompt is not readable here by ordinary means.
# Local — one small model, no cluster — and, like G-INTEGRITY, every claim in
# the script is paired with a positive control (usually the same code run in
# LOCAL mode, which must come out the other way).
g_shared() {
    local name="$1" out
    out=$(bash scripts/shared_mode_gate.sh 2>&1 | grep -E "^G_SHARED_(OK|FAIL|SKIP)" | tail -1)
    case "$out" in
        G_SHARED_OK*)
            vlog "${out#G_SHARED_OK: }"
            pass "$name" ;;
        G_SHARED_SKIP*) skip "$name" "${out#G_SHARED_SKIP: }" ;;
        "") fail "$name" "scripts/shared_mode_gate.sh reached no conclusion" ;;
        *) fail "$name" "${out#G_SHARED_FAIL:}" ;;
    esac
}

# G-WEDGE (results/coord-wedge-20260817.md): a dead engine must not take the
# coordinator down with it, and with several slots one dark relay must not take
# the other slots with it. Local — the fixture is a stub engine, no GPU and no
# cluster node — and the script arms its own positive control (it refuses to
# conclude anything unless the stub actually went dark).
g_wedge() {
    local name="$1" out
    out=$(bash scripts/coord_wedge_gate.sh 2>&1 | grep -E "^WEDGE_GATE_(OK|FAIL|SKIP)" | tail -1)
    case "$out" in
        WEDGE_GATE_OK*) pass "$name" ;;
        WEDGE_GATE_SKIP*) skip "$name" "${out#WEDGE_GATE_SKIP: }" ;;
        "") fail "$name" "scripts/coord_wedge_gate.sh reached no conclusion" ;;
        *) fail "$name" "${out#WEDGE_GATE_FAIL:}" ;;
    esac
}

# G-FIT-FAIL (evidence: results/multislot-vram-fix-win-20260818.md): an engine
# that reports it cannot fit the model into free device memory must stop the
# start (refuse, exit 3) instead of being served into WDDM paging that can
# freeze the whole machine. Local — the fixture is the same stub engine the
# wedge gate uses; the script carries its own control run (no warning -> must
# serve normally), so a refuse-everything implementation cannot pass it.
g_fit_fail() {
    local name="$1" out
    out=$(bash scripts/fit_fail_gate.sh 2>&1 | grep -E "^FIT_GATE_(OK|FAIL|SKIP)" | tail -1)
    case "$out" in
        FIT_GATE_OK*) pass "$name" ;;
        FIT_GATE_SKIP*) skip "$name" "${out#FIT_GATE_SKIP: }" ;;
        "") fail "$name" "scripts/fit_fail_gate.sh reached no conclusion" ;;
        *) fail "$name" "${out#FIT_GATE_FAIL:}" ;;
    esac
}

# G-BUDGET-SRC (docs/t8-quant-budget-fix-2026-08.md): the memory budget must be
# computed from the GGUF the engine will really open, not from the manifest's
# default quantization, and the coordinator must log which it used. Local — the
# fixtures are sparse files of the right size plus the same stub engine — and
# the script carries its own control (the same command against a differently
# sized file must move the budget), so a coordinator that ignores the path
# cannot pass it. G-FIT-FAIL is the backup for when this estimate is wrong
# anyway; this is the estimate itself.
g_budget_source() {
    local name="$1" out
    out=$(bash scripts/budget_source_gate.sh 2>&1 | grep -E "^BUDGET_GATE_(OK|FAIL|SKIP)" | tail -1)
    case "$out" in
        BUDGET_GATE_OK*) pass "$name" ;;
        BUDGET_GATE_SKIP*) skip "$name" "${out#BUDGET_GATE_SKIP: }" ;;
        "") fail "$name" "scripts/budget_source_gate.sh reached no conclusion" ;;
        *) fail "$name" "${out#BUDGET_GATE_FAIL:}" ;;
    esac
}

# G_OVERFLOW (docs/overflow-b2b-plan-2026-08.md §2 O5): overflow routing's six
# claims, the first of which is the rule that may not break — a job the platform
# dispatched is finished here or refused here, never forwarded on. That one is
# asserted through the REAL agent binary, because the whole rule rests on the
# agent setting X-IdleToken-Origin and a dropped header would break it in
# silence. The script carries a control for every claim (it refuses to conclude
# anything unless the machine really filled up, the connection log really
# records connections, and the plaintext search really finds a planted marker).
g_overflow() {
    local name="$1" out
    out=$(bash scripts/overflow_gate.sh 2>&1 | grep -E "^OVERFLOW_GATE_(OK|FAIL|SKIP)" | tail -1)
    case "$out" in
        OVERFLOW_GATE_OK*)
            vlog "${out#OVERFLOW_GATE_OK: }"
            pass "$name" ;;
        OVERFLOW_GATE_SKIP*) skip "$name" "${out#OVERFLOW_GATE_SKIP: }" ;;
        "") fail "$name" "scripts/overflow_gate.sh reached no conclusion" ;;
        *) fail "$name" "${out#OVERFLOW_GATE_FAIL: }" ;;
    esac
}

g_ppl() {
    local name="$1" out
    out=$(bash scripts/ppl_gate.sh 2>&1 | grep -E "^G_PPL_(OK|FAIL|SKIP)" | tail -1)
    case "$out" in
        G_PPL_OK*)
            vlog "${out#G_PPL_OK }"
            pass "$name" ;;
        G_PPL_SKIP*) skip "$name" "${out#G_PPL_SKIP: }" ;;
        "") fail "$name" "scripts/ppl_gate.sh reached no conclusion" ;;
        *) fail "$name" "${out#G_PPL_FAIL: }" ;;
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
# 2026-08-08: this line used to hardcode the pre-08-04-rename binary names,
# while the binaries are now called `idletoken-*` -- so **from the day of the
# rename it matched no process at all**, "succeeding" every round while doing
# nothing.
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
# The client's own two shell promises. Registered after P6 and deliberately NOT
# part of G_FINAL's composition: G_FINAL is defined in
# docs/acceptance-criteria.md as P1-P6 + G6, and quietly widening a gate that
# owns the exit code would change what a green ladder claims. They are reported
# on their own line, where a red one is visible.
gate G_UPDATE          g_update
gate G_TRAY            g_tray
# Cluster invariants that need no cluster node. gate_always, and before
# G_FINAL — see the helper's comment.
# 2026-08-14 migration (WS-F1): G_HOMO and G_MACSEAL are retired-in-place
# (each prints a dated SKIP and keeps its parked assertions behind an env,
# G_DSPARK-style); their successors are G_VERSION (engine-version equality is
# the one llama.cpp-cluster invariant) and G_MAC_SMOKE (macOS compute,
# positively gated instead of sealed).
gate_always G_HOMO     g_homo
gate_always G_VERSION  g_version
gate_always G_SIZE     g_size
gate_always G_MACSEAL  g_macseal
gate_always G_MAC_SMOKE g_mac_smoke
# The API route namespace (docs/api-surface.md). Needs no cluster node, and its
# source-sweep half is the only thing standing between a rename and a silent 404
# in the client dashboard -- so it must not be hideable by an offline laptop.
gate_always G_API_NS   g_api_ns
# The two vendor routes a third-party client needs (docs/api-surface.md §6).
# Also gate_always: it needs no cluster node, and "/v1/models lists something we
# cannot actually serve" is a product-level lie, not a platform-layer nicety.
gate_always G_API_MODELS g_api_models
# "The local API is closed on a fresh install" is a security posture, not a
# platform nicety — gate_always so an offline laptop cannot hide it.
gate_always G_LOCAL_TOKEN g_local_token
# "a shared machine does not write other people's prompts to its own disk" is a
# privacy invariant, not a platform-layer nicety -- gate_always.
gate_always G_NO_PROMPT_LOG g_no_prompt_log
# Raw embeddings stay on the coordinator (privacy invariant, WS-F2) and the
# distribution-level numeric band (WS-F3) — both local, both product-level.
gate_always G_PRIV7    g_priv7
gate_always G_PPL      g_ppl
# Model integrity + canary honesty (threat-model.md). gate_always: "the file we
# load is the curated model" and "a provider cannot recognise the probe" are
# product-level invariants, and an offline laptop must not be able to hide them.
gate_always G_INTEGRITY g_integrity
# Shared-mode plaintext (shared-mode-plan-2026-08.md). gate_always for the same
# reason as G_INTEGRITY: "the provider cannot read what a buyer sent" is a
# product-level promise that appears on the purchase page, and a machine that
# cannot run the ladder must not be able to skip past it quietly.
gate_always G_SHARED    g_shared
# Coordinator liveness under a dead engine (results/coord-wedge-20260817.md).
# gate_always for the same reason as the two above: "one stuck request does not
# take the machine with it" is what separates a home server from a toy, and it
# needs no cluster node to check — so no machine gets to skip past it quietly.
gate_always G_WEDGE     g_wedge
# Engine-reported "cannot fit" must refuse the start (multislot-vram-budget-fix).
# gate_always for the same reason as G_WEDGE: the consequence it guards against
# is a frozen machine, and it needs no cluster node to check.
gate_always G_FIT_FAIL  g_fit_fail
# The budget that G_FIT_FAIL backs up: it must be computed from the file the
# engine will open. gate_always for the same reason — a wrong budget is how a
# machine gets asked for KV it does not have, and no cluster node is needed to
# check it.
gate_always G_BUDGET_SRC g_budget_source
gate G_FINAL           g_final
# Both of these run AFTER G_FINAL and are independent of it: G_PLAT is the
# platform business layer (spec decision 11), G_DSPARK is an optional
# accelerator. Neither may set the FRONTIER for the product ladder.
gate G_DSPARK          g_dspark
gate G_PLAT            g_plat
# G_SCHED needs only the gateway dependencies on the control machine and touches
# no cluster node -> gate_local (see above).
gate_local G_SCHED     g_sched
# G_OVERFLOW is the platform track's (plan §2 O5) and touches no cluster node —
# its fixtures are a stub engine and a stub platform on this machine — so
# gate_local, alongside G_SCHED. Registered after G_PLAT for the same reason
# that one is: the platform business layer is independent of the cluster's
# G_FINAL (spec decision 11) and must not point the product track's "what to do
# next" at itself.
gate_local G_OVERFLOW  g_overflow
echo "------------------------------------------------------"
if [ -n "$ONLY_GATE" ]; then
    # Single-gate mode: report just that gate; no ladder-wide claims.
    #
    # A SKIP is NOT a PASS. This printed "GATE X: PASS" for a gate that had
    # skipped for want of its inputs (2026-08-15: G_PPL skipped because the
    # smoke GGUF never reached the child process, and the summary line called it
    # PASS) -- the exact false green the ladder exists to prevent. Exit 2 keeps
    # it distinguishable from both a pass and a real failure.
    _only_verdict=""
    for _r in ${RESULTS[@]+"${RESULTS[@]}"}; do
        case "$_r" in
            "PASS $ONLY_GATE"|"PASS $ONLY_GATE"_*) _only_verdict="PASS" ;;
            "FAIL $ONLY_GATE"|"FAIL $ONLY_GATE"_*) _only_verdict="FAIL" ;;
            "SKIP $ONLY_GATE"|"SKIP $ONLY_GATE"_*) [ -n "$_only_verdict" ] || _only_verdict="SKIP" ;;
        esac
    done
    case "$_only_verdict" in
        PASS) echo " GATE $ONLY_GATE: PASS"; exit 0 ;;
        SKIP) echo " GATE $ONLY_GATE: SKIP (not run — this is not a pass)"; exit 2 ;;
        FAIL) echo " GATE $ONLY_GATE: FAIL"; exit 1 ;;
        *)    echo " GATE $ONLY_GATE: NO SUCH GATE (nothing ran)"; exit 2 ;;
    esac
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
