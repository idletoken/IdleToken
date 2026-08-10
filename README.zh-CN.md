<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-dark.svg">
    <img src="docs/images/logo.svg" alt="IdleToken" width="380">
  </picture>
</p>

<p align="center">
  <b>把家里闲置的 NVIDIA 显卡，组成一个推理集群。</b><br>
  跑单机装不下的模型，对外是 OpenAI / Anthropic 兼容 API。
</p>

<p align="center">
  <a href="https://github.com/idletoken/IdleToken/issues">问题反馈</a> ·
  <a href="README.md">English</a>
</p>

<p align="center">
  <img src="docs/images/screenshot.png" alt="IdleToken" width="820">
</p>

---

一张 16 GB 的卡装不下 300B 的模型，四张加起来可以。IdleToken 实测每台机器还剩多少
资源，按容量把模型切给各节点，在局域网上做流水线并行，然后把结果开成一个 API ——
全程在图形界面里，不用写配置文件。

**一台机器就够开始。** 之后加机器进来，切分会自己重算。

## 快速开始

> **安装包尚未发布。** 首个版本发出来之前，请按下面「硬件要求」一节从源码构建。

1. 安装并启动客户端
2. 登录，选一个你的卡放得下的模型
3. 点开始 —— API 在 `:8000` 上线

```sh
export ANTHROPIC_BASE_URL=http://<你的机器>:8000
export ANTHROPIC_API_KEY=whatever
claude
```

要加机器：在一台上生成验证码，其它机器输码。发现、容量探测、切分、权重下载都是自动的。

## 能跑哪些模型

| 模型 | 权重 | 需要 |
| --- | ---: | --- |
| Qwen3.5-0.8B | 0.49 GiB | 任何达标的卡 |
| Qwen3.5-4B | 2.54 GiB | 8 GB |
| Qwen3-8B | 4.68 GiB | 8 GB |
| Qwen3.5-9B | 5.28 GiB | 12 GB |
| Qwen3.5-27B | 15.58 GiB | 24 GB |
| Qwen3.5-35B-A3B | 20.49 GiB | 24 GB，或两台 |
| **DeepSeek-V4-Flash-0731** | **80.76 GiB** | **一个集群** —— 304B / 激活 13B |

除 DeepSeek-V4-Flash 是 Q2，其余均为 Q4_K_M。客户端会告诉你哪些放得下、能开多长
上下文，放不下的**还差多少 GB**。

## 硬件要求

- **每台计算节点都要有 NVIDIA 卡**：算力 ≥ 7.5（RTX 20 系及以后），显存 ≥ 4 GB。
  纯 CPU 节点会被拒绝，不会静默降级。
- 驱动：Windows ≥ 527.41 · Linux ≥ 580.65
- Windows / Linux 参与计算；macOS 与手机可组网和监控
- 千兆局域网够用；数据不出你的网络
- CUDA Toolkit 可选 —— 装了长提示词处理明显更快

首个版本发布之前只能从源码构建（之后会有 Windows 安装包与 Linux 包）：

```sh
make                                        # 引擎（需要 CUDA toolkit + nvcc）
cd client && pnpm install && pnpm build
cd src-tauri && cargo build --release       # 桌面客户端
```

## 怎么工作的

线上只传隐状态 —— DeepSeek-V4-Flash 每 token 每个 stage 边界 64 KB，所以千兆不是
瓶颈。新加入的机器**只拉分给它的那几层**（HTTP Range）：一台拿到 43 层里 4 层的
机器只下了 9.54 GB，而不是 86.72 GB。切分会扣掉系统已占用的部分，而不是拿总显存
去除。

千兆网 + 32K 上下文在纯流水线并行下达不到我们定的时延目标 —— 需要 2.5G 以上
或序列并行，后者不在 v0.1 里。快慢完全取决于你的卡和你的网络，所以别信别人机器
上的数字，用 `scripts/bench.py` 量你自己的。

exo 和 llama.cpp 的 RPC 后端支持的硬件比我们广得多。IdleToken 拿这个换了图形界面、
按容量自动切分和原生 Anthropic 支持。

## 状态

Beta。发版由一条跑在真机上的验收阶梯把关（`scripts/acceptance.sh`），其中一条
走查跑在**发出去的安装包**上（干净机器）。

已知缺口：安装包尚未发布；发出来的那版不会有代码签名，Windows 会拦一次；权重
自动下载是新加的，还没端到端走查过（可以退回在设置里手填 GGUF 路径）；没有自动更新。

提问与报错请走 [Issues](https://github.com/idletoken/IdleToken/issues)。违反上面
那些硬性要求的改动——退回 CPU、拿总显存拍脑袋而不实测——即使能跑也不会被合入。

## 许可

Apache-2.0。"IdleToken" 这个名字与标识不随代码授权 —— 请用你自己的名字发布衍生版本。
vendor：[ds4](https://github.com/antirez/ds4)（MIT）、TweetNaCl、BLAKE2b。
模型权重不随本软件分发。
