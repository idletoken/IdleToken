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
  <a href="https://github.com/idletoken/IdleToken/releases">Download</a>
  · <a href="https://idletoken.ai">Project website</a>
  · <a href="README.zh-CN.md">中文</a>
</p>

---

## Why IdleToken

Agent workloads have their own peaks and valleys. There may be only one or two tasks running most of the time, but a complex job can start several agents at once, concentrate inference requests into a short burst, and leave local tasks waiting in a queue.

Home computers follow a separate cycle. They are usually sized for peak demand but rarely run at full capacity all day, leaving GPUs and memory unused for long periods. Machine idle time is not caused by fluctuations in agent workloads, but these two independent patterns can complement each other.

IdleToken connects those machines. Run models on your own computers day to day; ask models shared by others to complete queued tasks when your machines are busy; and share spare capacity with others when your machines are idle. One computer can run independently, while several computers can form a LAN cluster to run a model that does not fit on a single machine.

<table>
<tr>
<td width="50%" align="center">
<img src="docs/images/why-idle.svg" alt="An idle GPU sharing spare compute" width="320"><br>
<strong>When your machines are idle</strong><br>
<sub>Share spare compute with others.</sub>
</td>
<td width="50%" align="center">
<img src="docs/images/why-busy.svg" alt="A busy machine asking another machine for help" width="320"><br>
<strong>When your machines are busy</strong><br>
<sub>Ask others to complete queued tasks.</sub>
</td>
</tr>
</table>

IdleToken uses Sparks to settle the Tokens actually completed.

## Two ways to start

### Deploy your own model

1. Download and install the client from [Releases](https://github.com/idletoken/IdleToken/releases). It currently supports Windows 10/11 and Linux x86_64/arm64 with an NVIDIA GPU, plus Apple Silicon Macs.
2. Start the client, choose a supported Qwen, GPT-OSS, DeepSeek, GLM, or Kimi model, then select its quantization and context. The client prepares the model weights.
3. Turn on **Request help** to ask someone else to complete a task waiting locally. Turn on **Share compute** to let an idle machine help others.

### Call a model shared by someone else

If you do not want to deploy a model yourself—for example, if you want to use IdleToken from a phone, tablet, or computer without a supported GPU—you can call a model shared by someone else:

1. Create an account at [idletoken.ai](https://idletoken.ai).
2. Create an API key in your account.
3. Enter the API address and key in any Anthropic- or OpenAI-compatible client. No desktop installation is required.

## Join the community

Join the IdleToken Discord community to discuss model deployment, LAN clustering, client integrations, and project development, or to share feedback and ask for help.

## Connect existing tools

Both paths above expose the same OpenAI- and Anthropic-compatible APIs. Tools connect to a local address when you deploy your own model, or to the IdleToken API when you call a model shared by someone else:

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

## Supported models

IdleToken currently supports Qwen3 8B; Qwen3.5 0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B, and 397B-A17B; Qwen3.8 27B; GPT-OSS 20B and 120B; DeepSeek V4 Flash and Pro; GLM-5.2; and Kimi K2.5.

Need another model? [Open an issue](https://github.com/idletoken/IdleToken/issues).

## Hardware requirements

Resource requirements depend on the selected model, quantization, and context length. The client estimates them from the usable GPU or unified memory of the current machine or cluster, including required runtime overhead. The exact configuration selected by the user is checked again when the model starts.

| Platform | Compute requirements |
| --- | --- |
| **Windows 10/11** | NVIDIA GPU with compute capability 7.5 or newer and at least 4 GB VRAM, plus a current driver. CUDA runtime DLLs are included. |
| **Linux x86_64** | NVIDIA GPU with compute capability 7.5 or newer and at least 4 GB VRAM; driver 570.26+ and CUDA 12.8. |
| **Linux arm64** | NVIDIA GPU with compute capability 7.5 or newer and at least 4 GB VRAM; driver 580.65+ and CUDA 13.0. |
| **macOS** | Apple Silicon with enough available unified memory for the selected model, quantization, and context. |

The 4 GB figure is the minimum hardware threshold for a compute node; it does not mean every model can run in 4 GB of VRAM.

## Single-machine and cluster deployment

The same IdleToken client supports both deployment modes:

- **Single machine:** select a model, quantization, and context, then start it in single-machine mode. The model runs directly on that computer without RPC.
- **Cluster:** Windows, Linux, and macOS compute nodes can form a mixed-OS LAN cluster and run one model together.

To start a cluster:

1. Install the same IdleToken version on every participating computer and make sure they can reach each other over real LAN addresses.
2. Create a cluster on one computer. Join from the others with the same account or a verification code.
3. Select the exact same model, quantization, and context on every computer. Wait until the model weights are downloaded and every node reports ready.
4. On the creator, choose cluster deployment and start the model. IdleToken detects each node's resources and assigns model layers automatically. The local API remains at `http://127.0.0.1:8000`.

Cluster compute traffic does not use VPN or overlay networks such as Tailscale. Every node must run the same IdleToken version.

## Build from source

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

## Help

For installation, pairing, API, or source-build problems, search the [existing issues](https://github.com/idletoken/IdleToken/issues). When opening a new issue, include your operating system, IdleToken version, hardware, the command that failed, and the relevant error or log excerpt.

---

<p align="center">
  <a href="https://idletoken.ai">Project website</a>
  · <a href="https://github.com/idletoken/IdleToken/releases">Releases</a>
  · <a href="https://github.com/idletoken/IdleToken/issues">Issues</a>
  · <a href="LICENSE">Apache-2.0 license</a>
</p>

<p align="center"><sub>Built on <a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a> and <a href="https://github.com/tauri-apps/tauri">Tauri</a>. See <a href="NOTICE">NOTICE</a> for third-party acknowledgements.</sub></p>
