#!/usr/bin/env bash
# ds4x GPU benchmark: split the time of every GPU call into **kernel** and
# **non-kernel** (H2D/D2H copies + launch + sync), and run the same model on
# **several machines** side by side.
#
# Why this exists instead of "run ds4x_infer by hand when you need it":
#
# docs/linear-attention-design.md §4m records two rounds of kernel optimisation.
# Both were implemented, both were measured, both were reverted for zero gain,
# and "keep activations resident in VRAM" was written off as "large change,
# unclear benefit". **All three of those judgements were made on a DGX Spark
# only** — and GB10 has unified memory, where a cudaMemcpy is essentially a
# local memory copy. So "a blocking H2D/D2H pair on every matvec" was almost
# free on that one machine.
#
# Measured on a discrete card on 2026-08-11 (RTX 5060 Ti, PCIe) — same code,
# same model, same prompt:
#
#     Qwen3.5-0.8B   DGX GB10   kernel 305 ms  non-kernel  73 ms  (10.8 us/call)
#                    5060 Ti    kernel 484 ms  non-kernel 626 ms  (92.5 us/call)
#
# **On the discrete card 56% of the GPU-path time is not spent computing**, and
# the fixed per-call cost differs by 8.6x. This is the CLAUDE.md known risk
# "DGX Spark numbers do not transfer to home hardware (different memory
# architecture)" coming true — and this time it bit an **optimisation decision**,
# not just a performance number.
#
# So the point of this script is to **make "evaluated on the DGX only" hard to
# do**. It runs every machine in the testbed by default, and says plainly when
# only one machine answered that the result is not a conclusion about the fleet.
#
# Usage:
#   scripts/ds4x_gpu_bench.sh                                   # default model, all nodes
#   scripts/ds4x_gpu_bench.sh --gguf Qwen3-8B-Q4_K_M.gguf       # another model
#   scripts/ds4x_gpu_bench.sh --nodes "win_PC DGX_Spark"        # specific nodes
#   scripts/ds4x_gpu_bench.sh --n-predict 64                    # longer decode
#
# Reading the output: **the per-call overhead (us) is the invariant**. It barely
# moves with model size, so the smaller the model — the less each call does —
# the larger a share it takes. That, not "the small model did not fit in VRAM",
# is why small models suffer most (the weights are all resident; the tool prints
# `N on CPU` when they are not).
set -u
cd "$(dirname "$0")/.." || exit 1
. "$(dirname "$0")/testbed-lib.sh"

SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
GGUF_NAME="Qwen3.5-0.8B-Q4_K_M.gguf"
NPRED=32
PROMPT="The capital of France is"
NODES=""

while [ $# -gt 0 ]; do
    case "$1" in
        --gguf)       shift; GGUF_NAME="${1:-}" ;;
        --nodes)      shift; NODES="${1:-}" ;;
        --n-predict)  shift; NPRED="${1:-32}" ;;
        --prompt)     shift; PROMPT="${1:-}" ;;
        -h|--help)    sed -n "2,$(($(grep -n '^set -u' "$0" | head -1 | cut -d: -f1) - 1))p" "$0"; exit 0 ;;
        *) echo "ds4x_gpu_bench.sh: unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done

[ -n "$NODES" ] || NODES="${IDLETOKEN_COORD_NODE:-} ${IDLETOKEN_WORKER_NODES:-}"
NODES="$(printf '%s' "$NODES" | tr -s ' ')"
[ -n "$(printf '%s' "$NODES" | tr -d ' ')" ] || {
    echo "No nodes to measure. Either pass --nodes, or set" >&2
    echo "IDLETOKEN_COORD_NODE / IDLETOKEN_WORKER_NODES in scripts/testbed.env" >&2
    echo "(template: testbed.env.example)." >&2
    exit 2
}

# `uname -s` succeeds on Linux/macOS and fails under Windows OpenSSH (whose
# default shell is cmd). Same test topology_matrix.sh uses: **ask the machine,
# never guess from its name**.
is_win() { ! $SSH "$1" "uname -s" >/dev/null 2>&1; }

# Rows accumulate and are formatted at the end: printing the table as we go
# would be interleaved with ssh's own output.
RESULTS=""
MEASURED=0
SKIPPED=""

for node in $NODES; do
    [ -n "$node" ] || continue
    printf '\n=== %s ===\n' "$node" >&2

    if ! $SSH "$node" "echo ok" >/dev/null 2>&1; then
        echo "  unreachable, skipping" >&2
        SKIPPED="$SKIPPED $node(unreachable)"
        continue
    fi

    if is_win "$node"; then
        repo="$(testbed_repo_home "$node")"
        if [ -z "$repo" ]; then testbed_hint "$node"; SKIPPED="$SKIPPED $node(unconfigured)"; continue; fi
        # On Windows this tool comes from build_xi_win.bat, not the normal
        # build. If it is missing, name the command to run rather than skipping
        # silently -- a silent skip means measuring only the Linux box again.
        if ! $SSH "$node" "cd /d \"$repo\" && if exist ds4x_infer.exe (echo FOUND)" 2>/dev/null | grep -q FOUND; then
            echo "  no ds4x_infer.exe. On that machine run: build_xi_win.bat" >&2
            SKIPPED="$SKIPPED $node(no-tool)"
            continue
        fi
        gguf=""
        for d in $(testbed_gguf_dirs "$node"); do
            if $SSH "$node" "if exist \"$d\\\\$GGUF_NAME\" (echo FOUND)" 2>/dev/null | grep -q FOUND; then
                gguf="$d/$GGUF_NAME"; break
            fi
        done
        if [ -z "$gguf" ]; then
            echo "  $GGUF_NAME not found in $(testbed_gguf_dirs "$node")" >&2
            SKIPPED="$SKIPPED $node(no-weights)"
            continue
        fi
        # Throw the first run away. Coming off idle the GPU clock is at its
        # lowest step (measured on the DGX: 208 MHz / 4.4 W), so the first run's
        # kernel time is an order of magnitude too high. This fooled us twice on
        # 2026-08-11: once read as "92.5 us/call" (steady state 57), once as a
        # 1408 ms matmul kernel (steady state 115) -- the latter was nearly
        # filed as a performance regression.
        $SSH "$node" "cd /d \"$repo\" && set IDLETOKEN_DS4X_PROF=1 && ds4x_infer.exe \"$gguf\" --text \"$PROMPT\" --n-predict $NPRED --quiet" >/dev/null 2>&1
        out=$($SSH "$node" "cd /d \"$repo\" && set IDLETOKEN_DS4X_PROF=1 && ds4x_infer.exe \"$gguf\" --text \"$PROMPT\" --n-predict $NPRED --quiet" 2>&1 | tr -d '\r')
    else
        repo="${IDLETOKEN_COORD_HOME:-~/work/IdleToken}"
        # Two similarly named tools exist on Linux: build/ds4x_infer is the
        # **CPU-only** reference build, only build/ds4x_infer_cuda has the GPU
        # path. Picking the wrong one yields a healthy-looking "benchmark" in
        # which no GPU call ever happened -- precisely the accident this script
        # exists to prevent. So accept _cuda only, and name the make target.
        if ! $SSH "$node" "test -x $repo/build/ds4x_infer_cuda" 2>/dev/null; then
            echo "  no build/ds4x_infer_cuda (build/ds4x_infer is the CPU-only build; it does not count)." >&2
            echo "  On that machine run: make ds4xinfer-cuda" >&2
            SKIPPED="$SKIPPED $node(no-tool)"
            continue
        fi
        gguf=""
        for d in $(testbed_gguf_dirs "$node"); do
            if $SSH "$node" "test -f $d/$GGUF_NAME" 2>/dev/null; then gguf="$d/$GGUF_NAME"; break; fi
        done
        if [ -z "$gguf" ]; then
            echo "  $GGUF_NAME not found in $(testbed_gguf_dirs "$node")" >&2
            SKIPPED="$SKIPPED $node(no-weights)"
            continue
        fi
        # As above: discard the first run so the clock ramps up first.
        $SSH "$node" "cd $repo && IDLETOKEN_DS4X_PROF=1 ./build/ds4x_infer_cuda '$gguf' --text '$PROMPT' --n-predict $NPRED --quiet" >/dev/null 2>&1
        out=$($SSH "$node" "cd $repo && IDLETOKEN_DS4X_PROF=1 ./build/ds4x_infer_cuda '$gguf' --text '$PROMPT' --n-predict $NPRED --quiet" 2>&1)
    fi

    dev=$(printf '%s' "$out" | sed -n 's/.*CUDA on \([^—]*\)—.*/\1/p' | head -1 | sed 's/ *$//')
    if [ -z "$dev" ]; then
        # Without that line this run never touched the GPU (CPU build, no
        # usable device, or IDLETOKEN_DS4X_CPU still set). Treat it as a
        # failure; never mix CPU numbers into a GPU benchmark.
        echo "  this run did not use the GPU -- not counted. First lines of the tool's output:" >&2
        printf '%s\n' "$out" | head -4 | sed 's/^/    /' >&2
        SKIPPED="$SKIPPED $node(no-gpu)"
        continue
    fi
    echo "  $dev" >&2
    MEASURED=$((MEASURED + 1))

    # ds4x cuda: 2337 matvecs  kernel 207 ms (0.089 ms/call)  total 423 ms (0.181 ms/call)  overhead 51%
    # The bucket name (matvecs / matmuls / gdn chunks) sits before field 4 and
    # varies in length, so match on keywords rather than column numbers.
    printf '%s\n' "$out" | grep '^ds4x cuda:' | while IFS= read -r line; do
        calls=$(printf '%s' "$line"  | sed -n 's/^ds4x cuda: \([0-9]*\) .*/\1/p')
        # Buckets are matvecs / matmuls / "gdn chunks". A greedy [a-z ]* will
        # not do: on the matvecs line " kernel" follows immediately, is all
        # lowercase and spaces, and gets swallowed too.
        kind=$(printf '%s' "$line"   | sed -n 's/^ds4x cuda: [0-9]* \([a-z]*\( chunks\)\{0,1\}\).*/\1/p')
        kernel=$(printf '%s' "$line" | sed -n 's/.*kernel \([0-9]*\) ms.*/\1/p')
        total=$(printf '%s' "$line"  | sed -n 's/.*total \([0-9]*\) ms.*/\1/p')
        [ -n "$calls" ] && [ -n "$kernel" ] && [ -n "$total" ] || continue
        # Non-kernel cost per call (us) -- **this is the invariant**, and the
        # only number that compares directly across models and machines.
        per=$(awk -v t="$total" -v k="$kernel" -v c="$calls" 'BEGIN{ if(c>0) printf "%.1f", (t-k)*1000.0/c; else print "-" }')
        pct=$(awk -v t="$total" -v k="$kernel" 'BEGIN{ if(t>0) printf "%.0f", (t-k)*100.0/t; else print "-" }')
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$node" "$kind" "$calls" "$kernel" "$total" "$per" "$pct" >> /tmp/ds4x_bench_rows
    done
done

echo
echo "model $GGUF_NAME | decode $NPRED tokens | prompt \"$PROMPT\""
echo
printf '%-14s %-12s %8s %10s %10s %14s %10s\n' node bucket calls kernel_ms total_ms per_call_us overhead
printf '%-14s %-12s %8s %10s %10s %14s %10s\n' -------------- ------------ -------- ---------- ---------- -------------- ----------
if [ -f /tmp/ds4x_bench_rows ]; then
    while IFS=$'\t' read -r n k c ke t p pc; do
        printf '%-14s %-12s %8s %10s %10s %14s %9s%%\n' "$n" "$k" "$c" "$ke" "$t" "$p" "$pc"
    done < /tmp/ds4x_bench_rows
    rm -f /tmp/ds4x_bench_rows
fi

echo
[ -n "$SKIPPED" ] && echo "skipped: $SKIPPED"
if [ "$MEASURED" -lt 2 ]; then
    echo
    echo "WARNING: only $MEASURED machine(s) answered -- **this is not a conclusion about the fleet**."
    echo "  Unified-memory machines (DGX Spark / Apple Silicon) drive the per-call"
    echo "  copy cost to near zero, while the product targets home **discrete**"
    echo "  NVIDIA cards. Evaluating this class of optimisation on the former alone"
    echo "  has already produced one wrong 'no gain, revert' verdict"
    echo "  (linear-attention-design.md §4m). Measure at least one discrete-card"
    echo "  machine before concluding."
fi
