<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-dark.svg">
    <img src="docs/images/logo.svg" alt="IdleToken" width="380">
  </picture>
</p>

<h1 align="center">分享你的空闲 Token。</h1>

<p align="center">
  本地运行大模型，闲时分享给别人，忙时调用别人的。
</p>

<p align="center">
  <a href="https://github.com/idletoken/IdleToken/releases">下载客户端</a>
  · <a href="https://idletoken.ai">项目主页</a>
  · <a href="README.md">English</a>
</p>

---

## 为什么做 IdleToken

智能体的算力需求有自己的峰谷：日常可能只有一两个任务，复杂任务到来时，多个智能体会同时工作，推理请求在短时间内集中，本地任务因此开始排队。

家用电脑也有另一套独立的峰谷：机器通常按高峰需求配置，却不会全天满载，GPU 和内存在许多时段没有被使用。机器的空闲不是智能体负载波动造成的，但这两种独立的波动恰好可以互补。

IdleToken 把这些机器连接起来。平时在自己的电脑上运行模型；本机繁忙时，请求别人分享的模型完成排队中的任务；自己的机器空闲时，也可以把富余算力分享给别人。它既支持一台电脑独立运行，也支持多台电脑组成局域网集群，共同运行单机装不下的大模型。

<table>
<tr>
<td width="50%" align="center">
<img src="docs/images/why-idle.svg" alt="空闲 GPU 分享富余算力" width="320"><br>
<strong>机器空闲时</strong><br>
<sub>把富余算力分享给别人。</sub>
</td>
<td width="50%" align="center">
<img src="docs/images/why-busy.svg" alt="繁忙机器请求其他机器协助" width="320"><br>
<strong>本机繁忙时</strong><br>
<sub>请求别人完成排队中的任务。</sub>
</td>
</tr>
</table>

我们使用“火花”来结算实际完成的 Token。

## 两种开始方式

### 部署自己的模型

1. 从 [Releases](https://github.com/idletoken/IdleToken/releases) 下载并安装客户端。当前支持带 NVIDIA GPU 的 Windows 10/11、Linux x86_64 / arm64，以及 Apple Silicon Mac。
2. 启动客户端，选择当前支持的 Qwen、GPT-OSS、DeepSeek、GLM 或 Kimi 模型，并选择量化精度与上下文。模型参数会由客户端准备。
3. 打开“外援”按钮，可在本地任务排队时请求别人完成；打开“共享”按钮，可让空闲机器帮助别人完成任务。

### 直接调用别人分享的模型

如果你不想自己部署，例如希望在手机、平板或没有受支持 GPU 的电脑上使用 IdleToken，可以直接调用别人分享的模型：

1. 在 [idletoken.ai](https://idletoken.ai) 注册账号。
2. 在账户中创建 API 密钥。
3. 将 API 地址和密钥填入任意兼容 Anthropic 或 OpenAI API 的客户端。无需安装桌面客户端。

## 加入社区

欢迎加入 IdleToken Discord 社区，交流模型部署、局域网组网、客户端接入与项目开发，也可以直接反馈使用中遇到的问题。

## 接入现有工具

上面的两种使用方式最终都提供 OpenAI 与 Anthropic 兼容 API。部署自己的模型时，工具连接本机地址；不自行部署、直接调用别人分享的模型时，工具连接 IdleToken API：

| 使用方式 | Base URL | API key |
| --- | --- | --- |
| 自己部署的模型 | `http://127.0.0.1:8000` | 任意非空值；若自行设置了本地 token，则使用该 token |
| 别人分享的模型 | `https://api.idletoken.ai` | 在账户中创建的 API 密钥 |

以 Claude Code 直接调用别人分享的模型为例：

```sh
export ANTHROPIC_BASE_URL=https://api.idletoken.ai
export ANTHROPIC_API_KEY='sk-idletoken-********************************'
claude
```

`GET /v1/models` 返回对应地址当前可用的模型 ID。

## 当前支持的模型

当前支持 Qwen3 8B、Qwen3.5（0.8B、2B、4B、9B、27B、35B-A3B、122B-A10B、397B-A17B）、Qwen3.8 27B、GPT-OSS（20B、120B）、DeepSeek V4（Flash、Pro）、GLM-5.2 和 Kimi K2.5。

需要其它模型？请[提交 issue](https://github.com/idletoken/IdleToken/issues)。

## 硬件要求

实际资源需求取决于所选模型、量化精度和上下文长度。客户端会根据当前机器或集群的可用显存、统一内存和必要运行开销给出估算；运行时仍会按用户选择的精确配置检查资源。

| 平台 | 计算要求 |
| --- | --- |
| **Windows 10/11** | NVIDIA GPU，计算能力不低于 7.5，显存至少 4 GB，并安装当前驱动。安装包已包含 CUDA 运行时 DLL。 |
| **Linux x86_64** | NVIDIA GPU，计算能力不低于 7.5，显存至少 4 GB；驱动 570.26+，CUDA 12.8。 |
| **Linux arm64** | NVIDIA GPU，计算能力不低于 7.5，显存至少 4 GB；驱动 580.65+，CUDA 13.0。 |
| **macOS** | Apple Silicon；可用统一内存需满足所选模型、精度和上下文的资源需求。 |

4 GB 是计算节点的最低硬件门槛，并不代表所有模型都能在 4 GB 显存上运行。

## 单机与联机运行

IdleToken 使用同一套客户端支持两种部署方式：

- **单机运行：**选择模型、量化精度和上下文后，选择单机启动。模型直接在本机运行，不经过 RPC。
- **联机运行：**Windows、Linux 和 macOS 计算节点可以在同一局域网内混合组网，共同运行一个模型。

联机流程：

1. 在所有参与计算的机器上安装同一版本的 IdleToken，并确保它们可以通过真实局域网地址互相直连。
2. 在一台机器上创建集群，其余机器登录同一账号或使用验证码加入。
3. 每台机器选择完全相同的模型、量化精度和上下文，并等待模型参数下载完成、节点显示就绪。
4. 由创建者选择联机并启动。IdleToken 探测各节点资源后自动分配模型层；启动后，本机 API 仍使用 `http://127.0.0.1:8000`。

联机计算流量不走 Tailscale 等 VPN 或覆盖网络。所有节点必须使用同一 IdleToken 版本。

## 从源码构建

### 依赖

- Git、CMake、Node.js 18+、pnpm、Rust 1.77+
- [Tauri v2 系统依赖](https://tauri.app/start/prerequisites/)
- Linux：C 编译器与对应架构的 CUDA Toolkit
- macOS：Apple Silicon Mac 与 Xcode Command Line Tools
- Windows：Visual Studio 2022 Build Tools（Desktop development with C++）、Windows SDK、CUDA Toolkit 12.8（含 Visual Studio Integration）以及提供 `gcc`、`windres` 的 WinLibs/MinGW

### 获取源码

```sh
git clone https://github.com/idletoken/IdleToken.git
cd IdleToken
```

### Linux

```sh
./scripts/build_llamacpp.sh
make all IDLETOKEN_PLATFORM_VERIFY_KEY_B64="$(tr -d '\r\n' < scripts/platform-verify-key.b64)"
make -f Makefile.platform
./scripts/build_client_release.sh
```

安装包位于 `client/src-tauri/target/release/bundle/deb/` 和 `client/src-tauri/target/release/bundle/rpm/`。

### macOS

```sh
./scripts/build_llamacpp.sh
./scripts/package_client_mac.sh
```

安装包位于 `client/src-tauri/target/release/bundle/dmg/`。

### Windows

在 x64 Native Tools Command Prompt 中执行：

```bat
scripts\build_llamacpp_win.bat
cd client
pnpm install
pnpm build:release
cd ..
scripts\build_client_release.bat
```

安装包位于 `client\src-tauri\target\release\bundle\nsis\`。

若构建工具不在默认位置，可设置 `IDLETOKEN_MINGW_BIN`、`IDLETOKEN_CUDA_RUNTIME_DIR` 或 `IDLETOKEN_MBEDTLS_SRC`。

## 获取帮助

遇到安装、组网、API 或源码构建问题时，请先搜索[已有 issue](https://github.com/idletoken/IdleToken/issues)；若需新建 issue，请附上操作系统、IdleToken 版本、硬件信息、失败命令与相关报错或日志片段。

---

<p align="center">
  <a href="https://idletoken.ai">项目主页</a>
  · <a href="https://github.com/idletoken/IdleToken/releases">版本发布</a>
  · <a href="https://github.com/idletoken/IdleToken/issues">问题反馈</a>
  · <a href="LICENSE">Apache-2.0 许可</a>
</p>

<p align="center"><sub>项目基于 <a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a> 与 <a href="https://github.com/tauri-apps/tauri">Tauri</a> 构建；第三方项目致谢见 <a href="NOTICE">NOTICE</a>。</sub></p>
