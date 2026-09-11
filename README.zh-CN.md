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
  <a href="https://github.com/idletoken/IdleToken/releases">⬇️ 下载客户端</a>
  · <a href="https://idletoken.ai">🏠 项目主页</a>
  · <a href="README.md">🌐 English</a>
</p>

---

## ✨ 为什么做 IdleToken

智能体对算力的需求天然有峰谷。平时只有一两个推理任务在运行；遇到复杂任务时，多个智能体同时工作，短时间内发出大量推理请求。

设备的使用也有自己的峰谷：用户离开或夜间不用时闲置吃灰，用户需要时又经常不够用（如同时开启多个应用还要部署大模型）。

IdleToken 想做的，就是把这些错开的峰谷连接起来：让闲置的机器，为正处于算力高峰的智能体提供算力。

<table>
<tr>
<td width="50%" align="center">
<img src="docs/images/why-idle.svg" alt="空闲 GPU 分享富余算力" width="320"><br>
<strong>🌿 机器空闲时</strong><br>
<sub>把富余算力分享给别人。</sub>
</td>
<td width="50%" align="center">
<img src="docs/images/why-busy.svg" alt="繁忙机器请求其他机器协助" width="320"><br>
<strong>⚡ 机器繁忙时</strong><br>
<sub>请求别人完成排队中的任务。</sub>
</td>
</tr>
</table>

我们使用“火花”来结算 Token。

## 💬 加入社区

加入 IdleToken Discord，交流使用与开发。

[![Discord](https://img.shields.io/badge/Discord-%E5%8A%A0%E5%85%A5%E7%A4%BE%E5%8C%BA-5865F2?logo=discord&logoColor=white)](https://discord.gg/XbaWCH4t2J)

## 🚀 两种开始方式

### 🖥️ 部署并分享自己的模型

1. [下载并安装客户端](https://github.com/idletoken/IdleToken/releases)。
2. 启动客户端，下载并部署模型。
3. 打开“外援”按钮，请求别人完成本地排队的任务。
4. 打开“共享”按钮，帮助别人完成任务。

### ☁️ 调用别人分享的模型

如果你不想自己部署，例如希望在手机、平板或没有受支持 GPU 的电脑上使用 IdleToken，可以直接调用别人分享的模型。

1. 在 [idletoken.ai](https://idletoken.ai/register) 注册账号。
2. 在账户中创建 API 密钥。
3. 使用 API 调用别人分享的模型。

## 🔌 接入现有工具

上面的两种方式都可以接入现有工具：部署自己的模型时连接本机地址；直接调用别人分享的模型时连接 IdleToken API。两者都兼容 OpenAI 与 Anthropic API。

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

## 🧠 模型

IdleToken 提供精选的 Qwen、OpenAI 与 DeepSeek GGUF 文本生成模型。版本化清单位于 [`models/`](models/)；客户端下载权重并完成完整性校验、资源估算和量化精度选择。

每个入列模型都可在装得下时单机运行，也可由用户选择集群部署。当前不支持多模态输入或任意本地 GGUF 文件。需要其它模型？欢迎[提交 issue](https://github.com/idletoken/IdleToken/issues)。

## 💻 硬件

实际资源需求取决于所选模型、量化精度和上下文长度。客户端会根据当前机器或集群的可用资源给出估算，并在启动模型时检查所选配置。

| 平台 | 计算要求 |
| --- | --- |
| **Windows 10/11** | NVIDIA GPU，计算能力不低于 7.5，显存至少 4 GB，并安装当前驱动。安装包已包含 CUDA 运行时 DLL。 |
| **Linux x86_64** | NVIDIA GPU，计算能力不低于 7.5，显存至少 4 GB；驱动 570.26+，CUDA 12.8。 |
| **Linux arm64** | NVIDIA GPU，计算能力不低于 7.5，显存至少 4 GB；驱动 580.65+，CUDA 13.0。 |
| **macOS** | Apple Silicon；可用统一内存需满足所选模型、精度和上下文的资源需求。 |

## 🖧 单机与联机

单机运行时，选择模型、量化精度和上下文后即可在本机启动。

联机运行时：

1. 在所有参与计算的机器上安装同一版本的 IdleToken，并确保它们可以通过局域网互相直连。
2. 在一台机器上创建集群，其余机器登录同一账号或使用验证码加入。
3. 每台机器选择相同的模型、量化精度和上下文，并完成模型参数下载。
4. 所有节点显示就绪后，由创建者选择联机并启动。IdleToken 会探测各节点资源并自动分配模型层。

Windows、Linux 和 macOS 节点可以混合组网。联机计算流量不走 Tailscale 等 VPN 或覆盖网络。

---

## 🛠️ 从源码构建

### 前置依赖

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

## 💬 获取帮助

遇到安装、组网、API 或源码构建问题时，请先搜索[已有 issue](https://github.com/idletoken/IdleToken/issues)；若需新建 issue，请附上操作系统、IdleToken 版本、硬件信息、失败命令与相关报错或日志片段。

---

<p align="center">
  <a href="https://idletoken.ai">项目主页</a>
  · <a href="https://github.com/idletoken/IdleToken/releases">版本发布</a>
  · <a href="https://github.com/idletoken/IdleToken/issues">问题反馈</a>
  · <a href="LICENSE">Apache-2.0 许可</a>
</p>

<p align="center"><sub>项目基于 <a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a> 与 <a href="https://github.com/tauri-apps/tauri">Tauri</a> 构建；第三方项目致谢见 <a href="NOTICE">NOTICE</a>。</sub></p>
