<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-dark.svg">
    <img src="docs/images/logo.svg" alt="IdleToken" width="380">
  </picture>
</p>

<h1 align="center">Share your idle Token.</h1>

<p align="center">
  Run large models locally. Share yours when idle; call someone else's when busy.
</p>

<p align="center">
  <a href="https://github.com/idletoken/IdleToken/releases">⬇️ Download</a>
  · <a href="https://idletoken.ai">🏠 Project website</a>
  · <a href="README.zh-CN.md">🌐 中文</a>
</p>

---

## ✨ Why IdleToken

Agent workloads are bursty by nature. Most of the time, only one or two inference tasks may be running. Then a complex job starts several agents at once, and a batch of parallel requests drives compute demand sharply upward.

Home machines have their own rhythm: busy sometimes, idle most of the time. Few people run models around the clock, so a machine sized for peak demand often has GPU and memory to spare; in a home with several computers, those idle hours are spread across different machines.

IdleToken connects one person's idle hours to another's busy ones: share spare compute when you are not using it, and call a model shared by someone else when local tasks begin to queue.

<table>
<tr>
<td width="50%" align="center">
<img src="docs/images/why-idle.svg" alt="An idle GPU sharing spare compute" width="320"><br>
<strong>🌿 When your machines are idle</strong><br>
<sub>Share spare compute with others.</sub>
</td>
<td width="50%" align="center">
<img src="docs/images/why-busy.svg" alt="A busy machine asking another machine for help" width="320"><br>
<strong>⚡ When your machines are busy</strong><br>
<sub>Ask others to complete queued tasks.</sub>
</td>
</tr>
</table>

IdleToken uses Sparks to settle Tokens.

## 🚀 Two ways to start

<table>
<tr>
<th width="50%" align="center">🖥️ Deploy and share your own model</th>
<th width="50%" align="center">☁️ Call a model shared by someone else</th>
</tr>
<tr>
<td width="50%" valign="top">
<ol>
<li>Download and install the client. It currently supports Windows 10/11 and Linux x86_64/arm64 with an NVIDIA GPU, plus Apple Silicon Macs.</li>
<li>Start the client and choose a model, quantization, and context from the built-in list. The client downloads any missing model weights.</li>
<li>Turn on <strong>Request help</strong> to ask someone else to complete a task waiting locally. Turn on <strong>Share compute</strong> to help others complete their tasks.</li>
</ol>
</td>
<td width="50%" valign="top">
<p>If you do not want to deploy a model yourself—for example, if you want to use IdleToken from a phone, tablet, or computer without a supported GPU—you can call a model shared by someone else.</p>
<ol>
<li>Create an account at <a href="https://idletoken.ai">idletoken.ai</a>.</li>
<li>Create an API key in your account.</li>
<li>Enter the API address and key in any Anthropic- or OpenAI-compatible client. No desktop installation is required.</li>
</ol>
</td>
</tr>
</table>

## 💬 Join the community

Join the IdleToken Discord community to discuss model deployment, LAN clustering, client integrations, and project development, or to share feedback and ask for help.

## 🔌 Connect existing tools

Both paths above work with existing tools. Connect to the local address when you deploy your own model, or to the IdleToken API when you call a model shared by someone else. Both are compatible with the OpenAI and Anthropic APIs.

| Usage | Base URL | API key |
| --- | --- | --- |
| A model you deploy | `http://127.0.0.1:8000` | Any non-empty value, or your local token if you configured one |
| A model shared by someone else | `https://api.idletoken.ai` | An API key created in your account |

For example, to call a shared model from Claude Code:

```sh
export ANTHROPIC_BASE_URL=https://api.idletoken.ai
export ANTHROPIC_API_KEY='sk-idletoken-********************************'
claude
```

`GET /v1/models` returns the model IDs currently available at that address.

## 🧠 Models

IdleToken offers a curated list of GGUF text-generation models from Qwen, OpenAI, and DeepSeek. The versioned manifests live in [`models/`](models/); the client handles downloads, integrity checks, resource estimates, and supported quantizations.

Every listed model can run on one machine when it fits or across a cluster when you choose that deployment. Multimodal input and arbitrary local GGUF files are outside the current scope. To request another model, [open an issue](https://github.com/idletoken/IdleToken/issues).

## 💻 Hardware

Resource requirements depend on the selected model, quantization, and context length. The client estimates them from the resources available on the current machine or cluster and checks the selected configuration when the model starts.

| Platform | Compute requirements |
| --- | --- |
| **Windows 10/11** | NVIDIA GPU with compute capability 7.5 or newer and at least 4 GB VRAM, plus a current driver. CUDA runtime DLLs are included. |
| **Linux x86_64** | NVIDIA GPU with compute capability 7.5 or newer and at least 4 GB VRAM; driver 570.26+ and CUDA 12.8. |
| **Linux arm64** | NVIDIA GPU with compute capability 7.5 or newer and at least 4 GB VRAM; driver 580.65+ and CUDA 13.0. |
| **macOS** | Apple Silicon with enough available unified memory for the selected model, quantization, and context. |

## 🖧 Single-machine and cluster deployment

For a single-machine deployment, select a model, quantization, and context, then start it on that computer.

For a cluster deployment:

1. Install the same IdleToken version on every participating computer and make sure they can reach each other over the LAN.
2. Create a cluster on one computer. Join from the others with the same account or a verification code.
3. Select the same model, quantization, and context on every computer, and finish downloading the model weights.
4. When every node reports ready, choose cluster deployment on the creator and start the model. IdleToken detects each node's resources and assigns model layers automatically.

Windows, Linux, and macOS nodes can be mixed in one cluster. Cluster compute traffic does not use VPN or overlay networks such as Tailscale.

---

## 🛠️ Build from source

### Prerequisites

- Git, CMake, Node.js 18+, pnpm, and Rust 1.77+
- [Tauri v2 system prerequisites](https://tauri.app/start/prerequisites/)
- Linux: a C compiler and the CUDA Toolkit for the target architecture
- macOS: an Apple Silicon Mac and Xcode Command Line Tools
- Windows: Visual Studio 2022 Build Tools with Desktop development with C++, Windows SDK, CUDA Toolkit 12.8 with Visual Studio Integration, and WinLibs/MinGW providing `gcc` and `windres`

### Clone

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

Installers are written below `client/src-tauri/target/release/bundle/deb/` and `client/src-tauri/target/release/bundle/rpm/`.

### macOS

```sh
./scripts/build_llamacpp.sh
./scripts/package_client_mac.sh
```

The installer is written below `client/src-tauri/target/release/bundle/dmg/`.

### Windows

Run from an x64 Native Tools Command Prompt:

```bat
scripts\build_llamacpp_win.bat
cd client
pnpm install
pnpm build:release
cd ..
scripts\build_client_release.bat
```

The installer is written below `client\src-tauri\target\release\bundle\nsis\`.

Set `IDLETOKEN_MINGW_BIN`, `IDLETOKEN_CUDA_RUNTIME_DIR`, or `IDLETOKEN_MBEDTLS_SRC` when the corresponding build tools are not in their default locations.

## 💬 Help

For installation, pairing, API, or source-build problems, search the [existing issues](https://github.com/idletoken/IdleToken/issues). When opening a new issue, include your operating system, IdleToken version, hardware, the command that failed, and the relevant error or log excerpt.

---

<p align="center">
  <a href="https://idletoken.ai">Project website</a>
  · <a href="https://github.com/idletoken/IdleToken/releases">Releases</a>
  · <a href="https://github.com/idletoken/IdleToken/issues">Issues</a>
  · <a href="LICENSE">Apache-2.0 license</a>
</p>

<p align="center"><sub>Built on <a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a> and <a href="https://github.com/tauri-apps/tauri">Tauri</a>. See <a href="NOTICE">NOTICE</a> for third-party acknowledgements.</sub></p>
