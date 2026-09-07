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

<p align="center">
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-2563eb?style=flat-square" alt="支持 Windows、Linux 与 macOS">
  <img src="https://img.shields.io/badge/API-OpenAI%20%2B%20Anthropic-7c3aed?style=flat-square" alt="兼容 OpenAI 与 Anthropic API">
  <img src="https://img.shields.io/badge/engine-llama.cpp-111827?style=flat-square" alt="使用 llama.cpp 引擎">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-0f766e?style=flat-square" alt="Apache-2.0 许可"></a>
</p>

---

## ✨ 为什么做 IdleToken

智能体对算力的需求有明显峰谷，家里的 GPU 却在大部分时间里处于闲置状态。IdleToken 让不同机器的算力彼此补位：闲时分享算力赚取火花，需要时再用火花调用别人分享的模型服务。

<table>
<tr>
<td width="50%" align="center">
<img src="docs/images/why-idle.svg" alt="空闲 GPU 分享富余算力" width="320"><br>
<strong>🌿 机器空闲时</strong><br>
<sub>分享富余算力，赚取火花。</sub>
</td>
<td width="50%" align="center">
<img src="docs/images/why-busy.svg" alt="繁忙 GPU 获得共享算力" width="320"><br>
<strong>⚡ 机器繁忙时</strong><br>
<sub>调用社区分享的模型服务。</sub>
</td>
</tr>
</table>

## 🚀 两种开始方式

<table>
<tr>
<th width="50%" align="center">🖥️ 部署并分享自己的模型</th>
<th width="50%" align="center">☁️ 调用别人分享的模型</th>
</tr>
<tr>
<td width="50%" valign="top">
<ol>
<li>在带 NVIDIA 显卡的 Windows / Linux 电脑，或 Apple Silicon Mac 上安装桌面客户端。</li>
<li>选择精选模型、量化精度和上下文，在一台电脑或异构系统组成的局域网集群上运行。</li>
<li>打开共享赚取火花；需要更多算力时打开外援，调用他人分享的资源。</li>
</ol>
</td>
<td width="50%" valign="top">
<ol>
<li>在 <a href="https://idletoken.ai">idletoken.ai</a> 注册账号。</li>
<li>进入<strong>火花 → API 密钥</strong>并创建密钥。</li>
<li>接入任意兼容 Anthropic 或 OpenAI 的客户端。无需显卡，也无需安装桌面客户端。</li>
</ol>
<p>新账号的火花余额为零。你可以通过分享算力赚取。</p>
</td>
</tr>
</table>

## 🔌 接入现有工具

| 用途 | Base URL | API key |
| --- | --- | --- |
| 🖥️ 自己的本地模型 | `http://127.0.0.1:8000` | 任意非空值 |
| ☁️ 共享算力市场 | `https://api.idletoken.ai` | 在门户创建的密钥 |

Claude Code 示例：

```sh
export ANTHROPIC_BASE_URL=https://api.idletoken.ai
export ANTHROPIC_API_KEY='sk-idletoken-********************************'
claude
```

`GET /v1/models` 会返回当前 endpoint 可用的模型 ID。

## 🧠 模型

IdleToken 提供精选的 Qwen、OpenAI 与 DeepSeek GGUF 文本生成模型。版本化清单位于 [`models/`](models/)；客户端下载权重并完成完整性校验、资源估算和量化精度选择。

每个入列模型都可在装得下时单机运行，也可由用户选择集群部署。当前不支持多模态输入或任意本地 GGUF 文件。需要其它模型？欢迎[提交 issue](https://github.com/idletoken/IdleToken/issues)。

## 💻 硬件

| 平台 | 计算要求 |
| --- | --- |
| 🪟 **Windows 10/11** | NVIDIA GPU，计算能力不低于 7.5、显存至少 4 GB，并安装当前驱动。安装包已包含 CUDA 运行时 DLL。 |
| 🐧 **Linux** | 显卡要求相同；x86_64 需要 570.26+ 驱动与 CUDA 12.8，arm64 需要 580.65+ 驱动与 CUDA 13.0。 |
| 🍎 **macOS** | Apple Silicon，统一内存足以容纳所选模型。 |
| 🎛️ **仅作控制端** | 纯 CPU 电脑、AMD / Intel GPU、Intel Mac 与手机可以控制集群，但不参与推理。 |

> [!NOTE]
> 集群节点必须运行同一版本的 IdleToken，并能通过局域网直连。张量流量不走 Tailscale 等 VPN 或覆盖网络。

---

## 🛠️ 从源码构建

> [!IMPORTANT]
> 公开源码只有一个用途：构建完整的 IdleToken 桌面客户端。各平台的构建都会生成对应的原生安装包，并将客户端界面、必需的原生进程和钉住版本的 llama.cpp 引擎一并打包。

请在目标操作系统上直接构建；当前不支持交叉编译。

### 前置依赖

所有平台都需要 Git、CMake、Node.js 18 或更新版本、pnpm、Rust 1.77 或更新版本，以及目标操作系统对应的 [Tauri v2 系统依赖](https://tauri.app/start/prerequisites/)。请从干净的代码仓库开始，并在仓库根目录执行以下命令。

- **Linux：** C 编译器，以及“硬件”一节中对应架构的 CUDA Toolkit。
- **macOS：** Apple Silicon Mac 与 Xcode Command Line Tools。
- **Windows：** Visual Studio 2022 Build Tools（勾选 **Desktop development with C++**）、Windows SDK、带 Visual Studio Integration 的 CUDA Toolkit 12.8，以及提供 `gcc` 和 `windres` 的 WinLibs/MinGW。若这些工具不在 `PATH`，请设置 `IDLETOKEN_MINGW_BIN`。

llama.cpp 的传输层补丁会在 CMake 配置阶段获取 mbedTLS。若构建机器无法从 GitHub 下载，请将 `IDLETOKEN_MBEDTLS_SRC` 指向已有的 mbedTLS 3.6.7 源码目录。

### 获取源码

```sh
git clone https://github.com/idletoken/IdleToken.git
cd IdleToken
```

引擎构建脚本会读取 `scripts/llamacpp-patches/UPSTREAM` 中钉住的 llama.cpp commit，应用仓库内的补丁，并编译客户端使用的两个引擎 sidecar。

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

请在 x64 Native Tools Command Prompt 中执行：

```bat
scripts\build_llamacpp_win.bat
cd client
pnpm install
pnpm build:release
cd ..
scripts\build_client_release.bat
```

安装包位于 `client\src-tauri\target\release\bundle\nsis\`。

若 CUDA 安装在非标准目录，请将 `IDLETOKEN_CUDA_RUNTIME_DIR` 指向包含 `cudart64_12.dll`、`cublas64_12.dll` 与 `cublasLt64_12.dll` 的目录。

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
