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

## 能做什么

- **本地跑模型**，在 `:8000` 上提供 OpenAI / Anthropic 兼容 API。
- **单机装不下就多机合跑**：一个验证码把机器组成集群，层切分与权重下载全自动。
- **提前告诉你放不放得下**：哪些模型这台机器能跑、能开多长上下文、放不下的还差
  多少 GB——不用先下载再试。
- **每台只下自己那部分权重**：分到 43 层里 4 层的机器就只下这 4 层，不是整个文件。
- **富余产能可以分享**，不够用时可以用别人的。默认关闭，用积分结算，不涉及真钱。

## 怎么用

**1. 安装并启动。** 安装包尚未发布——首个版本发出来之前请按下面从源码构建。

**2. 登录，选模型。** 能力面板会标出这台机器放得下哪些。点开始，缺权重会自动下载，
然后 API 在 `:8000` 上线。

**3. 让客户端指向它。**

```sh
export ANTHROPIC_BASE_URL=http://<你的机器>:8000
export ANTHROPIC_API_KEY=whatever
claude
```

或者用 OpenAI 的调法：

```sh
curl http://<你的机器>:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"你好"}]}'
```

**4. 加机器（可选）。** 一台点「创建集群」拿到验证码，其它机器点「加入集群」输码。
它们在局域网里自动找到彼此，按各自的余量切分模型，全部就绪后 API 重新上线。

**5. 分享或借用（可选）。** 打开分享就把富余产能放出去，关掉就停。别人的请求是
加密送到你机器上的，不写进日志；市场本身能看到明文，因为它要做审核与计量。

## 模型

| 模型 | 权重 | 需要 |
| --- | ---: | --- |
| Qwen3.5-0.8B | 0.49 GiB | 任何达标的卡 |
| Qwen3.5-4B | 2.54 GiB | 8 GB |
| Qwen3-8B | 4.68 GiB | 8 GB |
| Qwen3.5-9B | 5.28 GiB | 12 GB |
| Qwen3.5-27B | 15.58 GiB | 24 GB |
| Qwen3.5-35B-A3B | 20.49 GiB | 24 GB，或两台 |
| **DeepSeek-V4-Flash-0731** | **80.76 GiB** | **一个集群** —— 304B / 激活 13B |

除 DeepSeek-V4-Flash 是 Q2，其余均为 Q4_K_M。

## 硬件要求

- **每台计算节点都要有 NVIDIA 卡**：算力 ≥ 7.5（RTX 20 系及以后），显存 ≥ 4 GB。
  纯 CPU 节点会被拒绝，不会静默降级。
- 驱动：Windows ≥ 527.41 · Linux ≥ 580.65
- Windows / Linux 参与计算；macOS 与手机可组网和监控
- 机器之间千兆局域网。分享走出站连接，不需要端口转发。
- CUDA Toolkit 可选 —— 装了长提示词处理更快

## 从源码构建

```sh
make                                        # 引擎（需要 CUDA toolkit + nvcc）
cd client && pnpm install && pnpm build
cd src-tauri && cargo build --release       # 桌面客户端
```

## 状态

Beta。安装包尚未发布；发出来的那版不会有代码签名，也没有自动更新。macOS 作为计算
节点在计划中——目前它只能组网和监控。

报错与提问走 [Issues](https://github.com/idletoken/IdleToken/issues)。

## 许可

Apache-2.0。"IdleToken" 这个名字与标识不随代码授权。
vendor：[ds4](https://github.com/antirez/ds4)（MIT）、TweetNaCl、BLAKE2b。
模型权重不随本软件分发。
