#!/usr/bin/env bash
# ds4x GPU 基准：把每次 GPU 调用的时间拆成 **kernel** 与 **非 kernel**（H2D/D2H
# 拷贝 + launch + 同步），在**多台机器上**跑同一个模型并排出来。
#
# 为什么要有这个脚本，而不是"需要时手跑一下 ds4x_infer"：
#
# `docs/linear-attention-design.md` §4m 记录了两轮 kernel 优化，都实现了、都实测了、
# 都因为收益为零而回退，并把"让激活常驻显存"判为"大改动、收益不明"。**那三次评估
# 全部只在 DGX Spark 上做的**，而 GB10 是统一内存——一次 `cudaMemcpy` 基本就是本地
# 内存拷贝。于是"每次 matvec 都做一对阻塞 H2D/D2H"这笔账在那台机器上几乎免费。
#
# 2026-08-11 在离散卡上一测（RTX 5060 Ti，PCIe），同一份代码、同一个模型、同一段
# prompt：
#
#     Qwen3.5-0.8B   DGX GB10   kernel 305 ms  非 kernel  73 ms  (每次调用 10.8 µs)
#                    5060 Ti    kernel 484 ms  非 kernel 626 ms  (每次调用 92.5 µs)
#
# **离散卡上 56% 的 GPU 路径时间不在算东西**，每次调用的固定开销差 8.6 倍。
# 这正是 CLAUDE.md 已知风险"DGX Spark 数据不能直接用于家用（统一内存架构不同）"
# 的兑现——而且这次它咬的是**优化决策**，不只是性能数字。
#
# 所以这个脚本的存在意义是：**让"只在 DGX 上评估"这件事变得困难**。它默认跑测试床
# 里的每一台机器，并在只测到一台时明说这不构成整个机群的结论。
#
# 用法：
#   scripts/ds4x_gpu_bench.sh                                   # 默认模型，所有节点
#   scripts/ds4x_gpu_bench.sh --gguf Qwen3-8B-Q4_K_M.gguf       # 换模型
#   scripts/ds4x_gpu_bench.sh --nodes "win_PC DGX_Spark"        # 指定节点
#   scripts/ds4x_gpu_bench.sh --n-predict 64                    # 更长的解码
#
# 读数怎么看：**每次调用开销（µs）是那个不变量**。它基本不随模型大小变化，所以
# 模型越小、每次调用干的活越少，它占的比例就越高——这就是小模型受害更重的原因，
# 不是"小模型没塞进显存"（权重本来就全在显存里，工具会打印 `N on CPU`）。
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
        -h|--help)    sed -n '1,40p' "$0"; exit 0 ;;
        *) echo "ds4x_gpu_bench.sh: 不认识的参数 $1" >&2; exit 2 ;;
    esac
    shift
done

[ -n "$NODES" ] || NODES="${IDLETOKEN_COORD_NODE:-} ${IDLETOKEN_WORKER_NODES:-}"
NODES="$(printf '%s' "$NODES" | tr -s ' ')"
[ -n "$(printf '%s' "$NODES" | tr -d ' ')" ] || {
    echo "没有节点可测。要么用 --nodes 指定，要么在 scripts/testbed.env 里配" >&2
    echo "IDLETOKEN_COORD_NODE / IDLETOKEN_WORKER_NODES（模板见 testbed.env.example）。" >&2
    exit 2
}

# uname -s 在 Linux/macOS 上成功，在 Windows OpenSSH（默认 shell 是 cmd）下失败。
# 与 topology_matrix.sh 用的是同一个判据：**问机器，不猜名字**。
is_win() { ! $SSH "$1" "uname -s" >/dev/null 2>&1; }

# 结果按行累积，最后统一排版——边跑边打表格会被 ssh 的输出打断。
RESULTS=""
MEASURED=0
SKIPPED=""

for node in $NODES; do
    [ -n "$node" ] || continue
    printf '\n=== %s ===\n' "$node" >&2

    if ! $SSH "$node" "echo ok" >/dev/null 2>&1; then
        echo "  连不上，跳过" >&2
        SKIPPED="$SKIPPED $node(连不上)"
        continue
    fi

    if is_win "$node"; then
        repo="$(testbed_repo_home "$node")"
        if [ -z "$repo" ]; then testbed_hint "$node"; SKIPPED="$SKIPPED $node(未配置)"; continue; fi
        # Windows 上这个工具由 build_xi_win.bat 产出，不是常规构建的一部分——
        # 没有就明说该跑哪条命令，不要静默跳过（静默跳过 = 又只测了 Linux 那台）。
        if ! $SSH "$node" "cd /d \"$repo\" && if exist ds4x_infer.exe (echo FOUND)" 2>/dev/null | grep -q FOUND; then
            echo "  缺 ds4x_infer.exe。在该机上跑：build_xi_win.bat" >&2
            SKIPPED="$SKIPPED $node(缺工具)"
            continue
        fi
        gguf=""
        for d in $(testbed_gguf_dirs "$node"); do
            if $SSH "$node" "if exist \"$d\\\\$GGUF_NAME\" (echo FOUND)" 2>/dev/null | grep -q FOUND; then
                gguf="$d/$GGUF_NAME"; break
            fi
        done
        if [ -z "$gguf" ]; then
            echo "  在 $(testbed_gguf_dirs "$node") 里找不到 $GGUF_NAME" >&2
            SKIPPED="$SKIPPED $node(缺权重)"
            continue
        fi
        # 先跑一趟丢掉。GPU 从空闲态起来时时钟还在最低档（实测 DGX 空闲 208 MHz /
        # 4.4 W），第一趟的 kernel 时间会虚高一个数量级——2026-08-11 两次都被这个
        # 骗到，一次读成"每次调用 92.5 µs"（稳态 57），一次读成 matmul kernel
        # 1408 ms（稳态 115），后者差点被当成性能回归。
        $SSH "$node" "cd /d \"$repo\" && set IDLETOKEN_DS4X_PROF=1 && ds4x_infer.exe \"$gguf\" --text \"$PROMPT\" --n-predict $NPRED --quiet" >/dev/null 2>&1
        out=$($SSH "$node" "cd /d \"$repo\" && set IDLETOKEN_DS4X_PROF=1 && ds4x_infer.exe \"$gguf\" --text \"$PROMPT\" --n-predict $NPRED --quiet" 2>&1 | tr -d '\r')
    else
        repo="${IDLETOKEN_COORD_HOME:-~/work/IdleToken}"
        # Linux 侧有两个同名工具：build/ds4x_infer 是**纯 CPU** 参考构建，
        # build/ds4x_infer_cuda 才带 GPU 路径。拿错了会得到一份看着正常、
        # 其实一次 GPU 调用都没发生的"基准"——那正是这个脚本要防的事故类型，
        # 所以这里只认 _cuda，并且找不到就明说该跑 make 什么。
        if ! $SSH "$node" "test -x $repo/build/ds4x_infer_cuda" 2>/dev/null; then
            echo "  缺 build/ds4x_infer_cuda（build/ds4x_infer 是纯 CPU 构建，不算数）。" >&2
            echo "  在该机上跑：make ds4xinfer-cuda" >&2
            SKIPPED="$SKIPPED $node(缺工具)"
            continue
        fi
        gguf=""
        for d in $(testbed_gguf_dirs "$node"); do
            if $SSH "$node" "test -f $d/$GGUF_NAME" 2>/dev/null; then gguf="$d/$GGUF_NAME"; break; fi
        done
        if [ -z "$gguf" ]; then
            echo "  在 $(testbed_gguf_dirs "$node") 里找不到 $GGUF_NAME" >&2
            SKIPPED="$SKIPPED $node(缺权重)"
            continue
        fi
        # 同上：丢掉第一趟，让时钟先起来。
        $SSH "$node" "cd $repo && IDLETOKEN_DS4X_PROF=1 ./build/ds4x_infer_cuda '$gguf' --text '$PROMPT' --n-predict $NPRED --quiet" >/dev/null 2>&1
        out=$($SSH "$node" "cd $repo && IDLETOKEN_DS4X_PROF=1 ./build/ds4x_infer_cuda '$gguf' --text '$PROMPT' --n-predict $NPRED --quiet" 2>&1)
    fi

    dev=$(printf '%s' "$out" | sed -n 's/.*CUDA on \([^—]*\)—.*/\1/p' | head -1 | sed 's/ *$//')
    if [ -z "$dev" ]; then
        # 没有这一行就意味着这一趟根本没走 GPU（CPU 构建 / 没有可用设备 /
        # IDLETOKEN_DS4X_CPU 还开着）。把它当成失败，不要把 CPU 的数字混进 GPU 基准。
        echo "  这一趟没有走 GPU——不计入。工具输出的前几行：" >&2
        printf '%s\n' "$out" | head -4 | sed 's/^/    /' >&2
        SKIPPED="$SKIPPED $node(未走GPU)"
        continue
    fi
    echo "  $dev" >&2
    MEASURED=$((MEASURED + 1))

    # ds4x cuda: 2337 matvecs  kernel 207 ms (0.089 ms/call)  total 423 ms (0.181 ms/call)  overhead 51%
    # 桶名（matvecs / matmuls / gdn chunks）在第 4 个字段之前，长度不一，所以按
    # 关键字取而不是按列号取。
    printf '%s\n' "$out" | grep '^ds4x cuda:' | while IFS= read -r line; do
        calls=$(printf '%s' "$line"  | sed -n 's/^ds4x cuda: \([0-9]*\) .*/\1/p')
        # 桶名是 matvecs / matmuls / "gdn chunks"。不能用 [a-z ]* 贪婪匹配：matvecs
        # 那一行紧跟着的就是 " kernel"，全是小写和空格，会被一起吞进来。
        kind=$(printf '%s' "$line"   | sed -n 's/^ds4x cuda: [0-9]* \([a-z]*\( chunks\)\{0,1\}\).*/\1/p')
        kernel=$(printf '%s' "$line" | sed -n 's/.*kernel \([0-9]*\) ms.*/\1/p')
        total=$(printf '%s' "$line"  | sed -n 's/.*total \([0-9]*\) ms.*/\1/p')
        [ -n "$calls" ] && [ -n "$kernel" ] && [ -n "$total" ] || continue
        # 每次调用的非 kernel 开销（µs）——**这是那个不变量**，也是唯一一个
        # 能跨模型、跨机器直接比较的数。
        per=$(awk -v t="$total" -v k="$kernel" -v c="$calls" 'BEGIN{ if(c>0) printf "%.1f", (t-k)*1000.0/c; else print "-" }')
        pct=$(awk -v t="$total" -v k="$kernel" 'BEGIN{ if(t>0) printf "%.0f", (t-k)*100.0/t; else print "-" }')
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$node" "$kind" "$calls" "$kernel" "$total" "$per" "$pct" >> /tmp/ds4x_bench_rows
    done
done

echo
echo "模型 $GGUF_NAME ｜ 解码 $NPRED token ｜ prompt \"$PROMPT\""
echo
printf '%-14s %-12s %8s %10s %10s %14s %10s\n' 节点 桶 调用数 kernel_ms 总计_ms 每次开销_µs 开销占比
printf '%-14s %-12s %8s %10s %10s %14s %10s\n' -------------- ------------ -------- ---------- ---------- -------------- ----------
if [ -f /tmp/ds4x_bench_rows ]; then
    while IFS=$'\t' read -r n k c ke t p pc; do
        printf '%-14s %-12s %8s %10s %10s %14s %9s%%\n' "$n" "$k" "$c" "$ke" "$t" "$p" "$pc"
    done < /tmp/ds4x_bench_rows
    rm -f /tmp/ds4x_bench_rows
fi

echo
[ -n "$SKIPPED" ] && echo "跳过：$SKIPPED"
if [ "$MEASURED" -lt 2 ]; then
    echo
    echo "⚠ 只测到 $MEASURED 台机器——**这不构成整个机群的结论**。"
    echo "  统一内存的机器（DGX Spark / Apple Silicon）会把每次调用的拷贝开销压到"
    echo "  接近于零，而产品的目标硬件是家用**离散** N 卡。只在前者上评估这类优化"
    echo "  已经导致过一次错误的「收益为零、回退」判断（linear-attention-design.md §4m）。"
    echo "  至少再测一台离散卡机器再下结论。"
fi
