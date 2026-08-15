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

## 怎么用

IdleToken 是一个桌面客户端，正常使用不需要命令行。

**先选一个模型。** 可以直接从客户端内置的模型里选，也可以打开本地 GGUF 文件，或者粘贴 Hugging Face 的仓库或文件链接。IdleToken 会扣除系统占用，读取机器真正可用的显存和内存，告诉你模型能不能放下、可以开多长上下文；如果放不下，也会直接告诉你还差多少。

**然后启动它。** 缺少的权重会自动下载。如果一台机器就能装下，IdleToken 会直接在本机运行，不额外引入集群开销。模型就绪后，本地 API 默认运行在 `:8000`；客户端会显示访问地址和为你生成的 API key。

Claude Code 可以直接接入：

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_API_KEY='把 IdleToken 中显示的 key 粘贴到这里'
claude
```

也可以通过 OpenAI 兼容接口调用：

```sh
export IDLETOKEN_API_KEY='把 IdleToken 中显示的 key 粘贴到这里'
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "authorization: Bearer $IDLETOKEN_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"你好"}]}'
```

如果一台机器装不下一个支持集群运行的模型，可以再加一台同一局域网里的机器。Windows、Linux 和 Apple Silicon Mac 可以混合组网。在一台机器上创建集群，其它机器用六位验证码或同一个账号加入。之后的节点发现、可用容量探测和模型切分都会自动完成。所有节点必须使用相同版本的 IdleToken 推理引擎；版本不一致时会明确提示升级并拒绝加入，避免输出未经验证的结果。

运行 API 的机器会保留完整的 GGUF 文件，工作节点不需要各自下载一份；模型数据通过局域网内经过认证和加密的连接发送过去。Token embedding 也固定留在 API 所在的机器上，prompt 不会以明文跨节点传输。

如果有闲置算力，也可以主动打开共享。分享会赚取火花，需要更多算力时，再用火花使用别人分享出来的资源。共享默认关闭，不需要时随时可以关掉。

**自己没有机器怎么办？** 那就从另一头用：不需要显卡，也不用装任何东西。在 [idletoken.ai](https://idletoken.ai) 注册、创建一个 API key，把上面那些客户端指向平台而不是本机地址：

```sh
export ANTHROPIC_BASE_URL=https://api.idletoken.ai
export ANTHROPIC_API_KEY='把你的平台 API key 粘贴到这里'
claude
```

OpenAI 兼容接口同理，用同一个地址即可。每个请求会被转发到别人正在分享的集群，按火花计费；新账户注册赠 100 火花。

试之前先知道一件事：平台上的每个集群都是别人自己的机器，人家不用的时候才开着。供给时有时无；没有合适的集群在线时，请求会明确失败，而不是被悄悄换到别的地方跑。

## 模型

IdleToken 可以运行当前固定版本 llama.cpp 所支持的文本生成 GGUF 模型。模型不必先由 IdleToken 登记：打开本地文件或使用 Hugging Face 链接后，客户端会直接读取 GGUF 文件头里的模型信息。

客户端也内置了一组已经配置好下载地址和默认版本的模型：

| 模型 | 默认权重大小 | 默认量化 | 说明 |
| --- | --- | --- | --- |
| Qwen3.5-0.8B | 0.49 GiB | Q4_K_M |  |
| Qwen3.5-4B | 2.54 GiB | Q4_K_M |  |
| Qwen3-8B | 4.68 GiB | Q4_K_M |  |
| Qwen3.5-9B | 5.28 GiB | Q4_K_M |  |
| Qwen3.5-27B | 15.58 GiB | Q4_K_M |  |
| Qwen3.5-35B-A3B | 20.49 GiB | Q4_K_M | 3B 激活参数 |
| DeepSeek-V4-Flash-0731 | 80.76 GiB | IQ2_XXS + Q2_K | 304B 总参数、13B 激活参数 |

直接打开的 GGUF 会在本机运行。DeepSeek-V4-Flash 可以走集群路径；进入这条路径后，调度由权重的真实大小和各节点当前可用的内存共同决定。能在一台机器上放下，就走更短的本机路径；只有放不下时，才把模型分布到集群。集群增加的是容量，不是凭空增加速度——把小模型拆到局域网上通常比直接在本机运行更慢，所以 IdleToken 会尽量让它留在本机。

多款 Qwen 模型还提供 Q5、Q6、Q8 和 BF16 版本。暂不支持多模态输入。

## 硬件要求

以下都是针对**跑推理的机器**。通过平台调用别人分享的集群不需要这些——不用显卡，也不用装东西。

计算节点支持 **Windows、Linux 和 Apple Silicon Mac**，一台或多台都行：

| 平台 | 计算硬件 | 另需 |
| --- | --- | --- |
| Windows 10/11 | NVIDIA，计算能力 ≥ 7.5（RTX 20 系及以后），显存 ≥ 4 GB | 驱动 ≥ 527.41，CUDA Toolkit 12.x |
| Linux | NVIDIA，计算能力 ≥ 7.5（RTX 20 系及以后），显存 ≥ 4 GB | 驱动 ≥ 580.65，CUDA Toolkit 13.0 |
| macOS | 支持 Metal 的 Apple Silicon，统一内存足以容纳所选模型 | 不需要安装 CUDA |

Windows、Linux 和 macOS 节点可以混合组成一个集群。真正需要一致的是推理引擎版本，而不是操作系统。机器之间还要有可以直接互通的局域网地址；模型张量会直接走真实局域网，不经过 Tailscale 或其它覆盖网络。千兆有线网络可以工作，2.5G 或万兆网络会给分布式大模型留下更多余量。

纯 CPU 机器、AMD 显卡和 Intel Mac 暂不能作为计算节点，但仍可以用客户端登录、聊天和控制其它集群；iPhone 和 Android 手机也可以作为控制端。

即使模型分布到多台机器，运行 API 的机器仍需要有足够的磁盘空间保存完整 GGUF。工作节点只需要在显存或内存里容纳分配给自己的张量。

## 从源码构建

Windows、Linux 和 macOS 安装包会发布在 [Releases](https://github.com/idletoken/IdleToken/releases) 页面。在 Linux 或 macOS 上自行构建推理引擎和客户端：

```sh
./scripts/build_llamacpp.sh
make
make -f Makefile.platform
./scripts/stage_sidecars.sh
cd client && pnpm install && pnpm build
cd src-tauri && cargo build --release
```

Windows 使用 `scripts\build_llamacpp_win.bat` 构建固定版本的 llama.cpp 推理引擎，再构建 coordinator、worker 和平台 agent，最后运行 `scripts\build_client_release.bat`。

Windows、Linux 和 macOS 分别使用 `scripts\build_client_release.bat`、`scripts/build_client_release.sh` 和 `scripts/package_client_mac.sh` 打包，产物包括 Windows NSIS 安装包、Linux `.deb` / `.rpm` 和 macOS `.dmg`。每个候选版本都会通过 `scripts/acceptance.sh` 在真实硬件上完成验收。

## 疑难排查

**安装时的 Windows SmartScreen / macOS Gatekeeper 警告。** Windows 安装包尚未做 Authenticode 签名，SmartScreen 会显示"Windows 已保护你的电脑"——点击**更多信息 → 仍要运行**。macOS 的 dmg 尚未公证，Gatekeeper 会拒绝首次打开：右键点击应用选择**打开**，或执行 `xattr -d com.apple.quarantine /Applications/IdleToken.app`。这两者都只是"未签名身份"提示，不是恶意软件判定；应用内更新通道对每个安装的包都做密码学校验，与此无关。

**本机开着 HTTP 代理时聊天流永久卡住。** 系统级代理（Clash 等）可能劫持 loopback 的 SSE 流量并吞掉流结束信号。在运行 `claude` 或 `curl` 的终端里设置 `NO_PROXY=127.0.0.1,localhost`，或在代理里给 `127.0.0.1` 添加直连规则。客户端为它自己拉起的引擎进程已经设置了这个变量。

**Windows 防火墙拦住了组网或集群流量。** IdleToken 在有管理员权限时会自行添加入站规则；否则日志会打印需要以管理员身份执行一次的 `netsh` 命令。涉及的端口：UDP 14097（引擎发现）和 UDP 14099（客户端组网广播）；TCP 14100 和 14101（coordinator/worker 控制通道）；TCP 50052（工作节点的 rpc-server，可配置）；TCP 8000（API，仅当其它设备要调用时才需要）。

**Linux 客户端打开后白屏。** 启动时加 `WEBKIT_DISABLE_DMABUF_RENDERER=1 idletoken-client`——部分驱动栈上的 WebKitGTK dmabuf 问题。

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
