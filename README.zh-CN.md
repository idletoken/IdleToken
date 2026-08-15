<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-dark.svg">
    <img src="docs/images/logo.svg" alt="IdleToken" width="380">
  </picture>
</p>

**闲时分享，忙时扩容。**

让闲置的算力动起来，需求高峰时再取用更多 Token。

[English](README.md)

<p align="center">
  <img src="docs/images/screenshot.png" alt="IdleToken" width="820">
</p>

---

## 为什么做 IdleToken

智能体的工作负载天然是有峰谷的。大多数时候，可能只有一两个推理任务在运行；但遇到复杂任务时，多个智能体会同时展开工作，短时间内产生大量并行请求，算力需求会迅速上升。

而本地部署的机器，恰好也是有时很忙、有时很闲。人不会时时刻刻都在跑模型，所以一台机器可能在大部分时间里都有富余算力，却偏偏在真正需要大量推理时显得不够用。

IdleToken 想做的，就是把不同人的这些闲时和忙时连接起来：**不用的时候，把算力分享出去赚取火花；需要更多算力的时候，再用火花使用别人此刻闲置的资源。** 让平时闲着的算力，在需要的时候流动到真正有需求的地方。

## 快速上手

IdleToken 有两种用法，取决于你有没有一台带受支持 GPU 的机器。

### 没有机器：直接用平台

在 [idletoken.ai](https://idletoken.ai) 注册并创建 API key，然后像使用其它第三方模型服务一样接入：

```sh
export ANTHROPIC_BASE_URL=https://api.idletoken.ai
export ANTHROPIC_API_KEY='<你的平台 API key>'
claude
```

OpenAI 兼容接口使用同一地址。请求运行在他人分享的集群上，按火花计费；新账户赠 100 火花。

### 有机器：自己部署

IdleToken 是一个桌面客户端，日常使用不需要命令行。Windows、Linux 和 macOS 安装包见 [Releases](https://github.com/idletoken/IdleToken/releases) 页面。

1. **选一个模型**——从内置列表中选择，或打开本地 GGUF 文件、粘贴 Hugging Face 链接。客户端会根据本机可用显存和内存判断能否运行。
2. **启动服务**——缺少的权重自动下载。API 默认监听 `:8000`，客户端会显示地址和生成的 API key。

Claude Code 接入：

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_API_KEY='<IdleToken 中显示的 key>'
claude
```

或使用 OpenAI 兼容接口：

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'authorization: Bearer <IdleToken 中显示的 key>' \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"你好"}]}'
```

单机装不下的模型，可由同一局域网内的多台机器共同运行。在一台机器上创建集群，其余机器用六位验证码或同一账号加入；IdleToken 会探测各机可用内存并切分模型。Windows、Linux、macOS 节点可混合组网，所有节点须运行相同版本的 IdleToken。工作节点经加密连接从 API 所在机器获取模型分片，prompt 不会以明文跨节点传输。

集群闲置时可打开共享赚取火花；共享默认关闭。

## 模型

IdleToken 可运行其固定版本 llama.cpp 支持的文本生成 GGUF 模型——内置列表、本地文件或 Hugging Face 链接均可，模型信息直接读取自 GGUF 文件头。暂不支持多模态输入。

内置模型：

| 模型 | 默认权重大小 | 默认量化 | 说明 |
| --- | --- | --- | --- |
| Qwen3.5-0.8B | 0.49 GiB | Q4_K_M |  |
| Qwen3.5-4B | 2.54 GiB | Q4_K_M |  |
| Qwen3-8B | 4.68 GiB | Q4_K_M |  |
| Qwen3.5-9B | 5.28 GiB | Q4_K_M |  |
| Qwen3.5-27B | 15.58 GiB | Q4_K_M |  |
| Qwen3.5-35B-A3B | 20.49 GiB | Q4_K_M | 3B 激活参数 |
| DeepSeek-V4-Flash-0731 | 80.76 GiB | IQ2_XXS + Q2_K | 304B 总参数、13B 激活参数 |

多款 Qwen 模型另有 Q5、Q6、Q8 和 BF16 版本。DeepSeek-V4-Flash 支持集群运行，其余内置模型为单机运行。

## 硬件要求

以下要求针对运行推理的机器；通过平台调用无需显卡、无需安装。

| 平台 | 计算硬件 | 另需 |
| --- | --- | --- |
| Windows 10/11 | NVIDIA，计算能力 ≥ 7.5（RTX 20 系及以后），显存 ≥ 4 GB | 驱动 ≥ 527.41，CUDA Toolkit 12.x |
| Linux | NVIDIA，计算能力 ≥ 7.5（RTX 20 系及以后），显存 ≥ 4 GB | 驱动 ≥ 580.65，CUDA Toolkit 13.0 |
| macOS | 支持 Metal 的 Apple Silicon，统一内存足以容纳所选模型 | 不需要安装 CUDA |

- 操作系统可混合组网，各节点的 IdleToken 版本必须一致。
- 集群机器之间需要可直连的局域网；张量流量不经过 VPN / 覆盖网络（如 Tailscale）。建议千兆有线或更快。
- 纯 CPU 机器、AMD 显卡、Intel Mac 与手机不能参与计算，但可运行客户端登录、聊天和控制集群。
- API 所在机器需保存完整 GGUF；工作节点只需容纳分配给它的分片。

## 从源码构建

Linux / macOS：

```sh
./scripts/build_llamacpp.sh
make
make -f Makefile.platform
./scripts/stage_sidecars.sh
cd client && pnpm install && pnpm build
cd src-tauri && cargo build --release
```

Windows：先用 `scripts\build_llamacpp_win.bat` 构建引擎，构建 coordinator、worker 与平台 agent，再运行 `scripts\build_client_release.bat`。

打包：`scripts\build_client_release.bat`（Windows NSIS 安装包）、`scripts/build_client_release.sh`（Linux `.deb` / `.rpm`）、`scripts/package_client_mac.sh`（macOS `.dmg`）。

## 疑难排查

**安装时的 SmartScreen / Gatekeeper 警告。** Windows 安装包尚未做 Authenticode 签名，macOS dmg 尚未公证。Windows：点击**更多信息 → 仍要运行**；macOS：右键应用选择**打开**，或执行 `xattr -d com.apple.quarantine /Applications/IdleToken.app`。

**本机开着 HTTP 代理时聊天流卡住。** Clash 等代理可能吞掉 loopback 的 SSE 流。在运行 `claude` 或 `curl` 的终端设置 `NO_PROXY=127.0.0.1,localhost`，或在代理中为 `127.0.0.1` 添加直连规则。

**Windows 防火墙拦截组网或集群流量。** 以管理员身份运行一次 IdleToken，或以管理员身份执行日志中打印的 `netsh` 命令。端口：UDP 14097、14099（发现与组网），TCP 14100、14101（集群控制），TCP 50052（工作节点 rpc-server，可配置），TCP 8000（API，仅当其它设备调用时需要）。

**Linux 客户端白屏。** 启动时加 `WEBKIT_DISABLE_DMABUF_RENDERER=1 idletoken-client`。

## 许可

[Apache-2.0](LICENSE)

## 致谢

IdleToken 的实现离不开许多优秀的开源项目，特别感谢：

- [ds4](https://github.com/antirez/ds4)
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)
- [Ollama](https://github.com/ollama/ollama)
- [Tauri](https://github.com/tauri-apps/tauri)
- [TweetNaCl](https://tweetnacl.cr.yp.to/)
- [BLAKE2](https://github.com/BLAKE2/BLAKE2)
- [DeepSeek](https://github.com/deepseek-ai)
- [Qwen](https://github.com/QwenLM)

以及所有 IdleToken 所依赖的开源项目和它们的贡献者。
