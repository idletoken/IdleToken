<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-dark.svg">
    <img src="docs/images/logo.svg" alt="IdleToken" width="380">
  </picture>
</p>

<p align="center">
  <b>在你自己的机器上跑模型。</b><br>
  用不完的 token 分享出去，不够用的时候用别人的。
</p>

<p align="center">
  <a href="https://github.com/idletoken/IdleToken/issues">问题反馈</a> ·
  <a href="README.md">English</a>
</p>

<p align="center">
  <img src="docs/images/screenshot.png" alt="IdleToken" width="820">
</p>

---

你的显卡一天里大半时间是闲着的。IdleToken 把模型放上去——一台机器，或者几台
合成一个集群——把它产出的 token 变成 OpenAI / Anthropic 兼容 API。让 Claude Code
指向它，token 就来自你自己的硬件。

用不完的产能可以分享出去，不够用的时候可以用别人闲着的机器。分享赚积分，借用花
积分。

**客户端开源，是为了让你能自己查。** 它跑在你的机器上、看得到你的 prompt、驱动
你的显卡——这种软件本来就该让人读得到源码。撮合分享方与借用方的市场是一个托管
服务，不在本仓库里。

## 快速开始

> **安装包尚未发布。** 首个版本发出来之前，请按下面「硬件要求」一节从源码构建。

1. 安装并启动客户端
2. 登录，选一个你的硬件放得下的模型
3. 点开始 —— API 在 `:8000` 上线

```sh
export ANTHROPIC_BASE_URL=http://<你的机器>:8000
export ANTHROPIC_API_KEY=whatever
claude
```

一台机器就够。要加机器：在一台上生成验证码，其它机器输码；发现、容量探测、切分、
权重下载全自动，机器进进出出时切分会自己重算。

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
上下文，放不下的**还差多少 GB**。把机器合起来才够得着最后一行——单张消费级卡装不
下 80 GiB。

## 分享与借用

默认关闭。不打开，什么都不出你的网络；关掉就结束。

- **别人的请求到你这儿是什么样子。** 它是信封加密（X25519 sealed box）来的，只在
  你的协调节点内存里解开一瞬，绝不写进日志。集群里的其它机器自始至终只看得到
  hidden states——没有文字、没有密钥、没有词表。
- **平台看得到什么。** 明文，因为它要做内容审核与计量。这是共享市场的代价，写在
  明面上，不塞进脚注。自己给自己跑的集群不涉及加密，也不涉及任何第三方。
- **是积分，不是钱。** 分享赚、借用花，新账户有初始额度。不提现。

## 硬件要求

- **每台计算节点都要有 NVIDIA 卡**：算力 ≥ 7.5（RTX 20 系及以后），显存 ≥ 4 GB。
  纯 CPU 节点会被拒绝，不会静默降级。
- 驱动：Windows ≥ 527.41 · Linux ≥ 580.65
- Windows / Linux 参与计算；macOS 与手机可组网和监控
- 机器之间千兆局域网；共享出去的集群走**出站**长连接连平台，不需要端口转发
- CUDA Toolkit 可选 —— 装了长提示词处理明显更快

首个版本发布之前只能从源码构建（之后会有 Windows 安装包与 Linux 包）：

```sh
make                                        # 引擎（需要 CUDA toolkit + nvcc）
cd client && pnpm install && pnpm build
cd src-tauri && cargo build --release       # 桌面客户端
```

## 怎么工作的

线上只传隐状态 —— DeepSeek-V4-Flash 每 token 每个 stage 边界 64 KB，所以千兆不是
瓶颈。新加入集群的机器**只拉分给它的那几层**（HTTP Range）：一台拿到 43 层里 4 层
的机器只下了 9.54 GB，而不是 86.72 GB。切分会扣掉系统已占用的部分，而不是拿总显存
去除——所以客户端能在你花一小时下载之前就告诉你放不放得下。

千兆网 + 32K 上下文在纯流水线并行下达不到我们定的时延目标 —— 需要 2.5G 以上或
序列并行，后者不在 v0.1 里。快慢完全取决于你的硬件和你的网络，所以别信别人机器上
的数字，用 `scripts/bench.py` 量你自己的。

## 状态

Beta。发版由一条跑在真机上的验收阶梯把关（`scripts/acceptance.sh`），其中一条走查
跑在**发出去的安装包**上（干净机器）。

已知缺口：安装包尚未发布；发出来的那版不会有代码签名，Windows 会拦一次；权重自动
下载是新加的，还没端到端走查过（可以退回在设置里手填 GGUF 路径）；没有自动更新。
计划中：macOS 作为计算节点——目前它只能组网和监控，不参与计算。

提问与报错请走 [Issues](https://github.com/idletoken/IdleToken/issues)。违反上面
那些硬性要求的改动——退回 CPU、拿总显存拍脑袋而不实测——即使能跑也不会被合入。

## 许可

Apache-2.0。"IdleToken" 这个名字与标识不随代码授权 —— 请用你自己的名字发布衍生版本。
vendor：[ds4](https://github.com/antirez/ds4)（MIT）、TweetNaCl、BLAKE2b。
模型权重不随本软件分发。
