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
  <a href="https://github.com/idletoken/IdleToken/releases">Download the desktop app</a>
  · <a href="https://idletoken.ai">Project website</a>
  · <a href="README.zh-CN.md">中文</a>
</p>

## Why IdleToken

Agent workloads arrive in bursts, while home GPUs spend much of their time unused. IdleToken lets those machines back each other up: share idle compute to earn Sparks, then spend Sparks on models shared by others when your own machines are busy.

<p align="center">
  <img src="docs/images/why-idle.svg" alt="An idle GPU sharing spare compute" width="360">
  &nbsp;&nbsp;
  <img src="docs/images/why-busy.svg" alt="A busy GPU receiving shared compute" width="360">
</p>

<p align="center"><sub>Share spare compute when idle · Draw on shared compute when busy</sub></p>

## Two ways to start

<table>
<tr>
<td width="50%" valign="top">
<h3>Deploy and share your own model</h3>
<ol>
<li>Install the desktop app on Windows or Linux with an NVIDIA GPU, or on an Apple Silicon Mac.</li>
<li>Choose a curated model, quantization, and context, then run it on one computer or a mixed-OS LAN cluster.</li>
<li>Turn on sharing to earn Sparks. Turn on backup when you want IdleToken to draw on shared capacity.</li>
</ol>
</td>
<td width="50%" valign="top">
<h3>Call a model shared by someone else</h3>
<ol>
<li>Create an account at <a href="https://idletoken.ai">idletoken.ai</a>.</li>
<li>Open <strong>Sparks → API keys</strong> and create a key.</li>
<li>Connect any Anthropic- or OpenAI-compatible client. No GPU or desktop installation is required.</li>
</ol>
<p>New accounts start with zero Sparks. Earn them by sharing compute.</p>
</td>
</tr>
</table>

## Connect your tools

| Use | Base URL | API key |
| --- | --- | --- |
| Your local model | `http://127.0.0.1:8000` | Any non-empty value |
| Shared marketplace | `https://api.idletoken.ai` | A key created in the portal |

For Claude Code:

```sh
export ANTHROPIC_BASE_URL=https://api.idletoken.ai
export ANTHROPIC_API_KEY='sk-idletoken-********************************'
claude
```

`GET /v1/models` returns the model IDs available at that endpoint.

## Models

IdleToken offers a curated list of GGUF text-generation models from Qwen, OpenAI, and DeepSeek. The versioned manifests live in [`models/`](models/); the client handles downloads, integrity checks, resource estimates, and supported quantizations.

Every listed model can run on one machine when it fits or across a cluster when you choose that deployment. Multimodal input and arbitrary local GGUF files are outside the current scope. To request another model, [open an issue](https://github.com/idletoken/IdleToken/issues).

## Hardware

- Windows 10/11: NVIDIA GPU with compute capability 7.5 or newer, at least 4 GB VRAM, and a current driver; CUDA runtime DLLs are included.
- Linux: the same GPU requirement, plus driver 570.26 or newer / CUDA 12.8 on x86_64, or driver 580.65 or newer / CUDA 13.0 on arm64.
- macOS: Apple Silicon with enough unified memory for the selected model.
- CPU-only computers, AMD or Intel GPUs, Intel Macs, and phones can control a cluster but do not run inference.
- Cluster nodes need the same IdleToken version and a direct LAN route; tensor traffic does not use VPN or overlay networks such as Tailscale.

## Build from source

The public source tree has one purpose: build the complete IdleToken desktop client. Each platform build produces its native installer with the interface, required native processes, and the pinned llama.cpp engine bundled together.

Build on the operating system you are targeting. Cross-compilation is not supported.

### Prerequisites

All platforms need Git, CMake, Node.js 18 or newer, pnpm, Rust 1.77 or newer, and the [Tauri v2 system prerequisites](https://tauri.app/start/prerequisites/). Start from a clean checkout and run the commands from the repository root.

- **Linux:** a C compiler and the CUDA Toolkit listed in the hardware section.
- **macOS:** an Apple Silicon Mac and Xcode Command Line Tools.
- **Windows:** Visual Studio 2022 Build Tools with **Desktop development with C++**, Windows SDK, CUDA Toolkit 12.8 with Visual Studio Integration, and WinLibs/MinGW tools providing `gcc` and `windres`. Set `IDLETOKEN_MINGW_BIN` if those tools are not on `PATH`.

The llama.cpp transport patch fetches mbedTLS during configuration. Set `IDLETOKEN_MBEDTLS_SRC` to an existing mbedTLS 3.6.7 source tree when the build machine cannot fetch it from GitHub.

### Clone

```sh
git clone https://github.com/idletoken/IdleToken.git
cd IdleToken
```

The engine scripts fetch the llama.cpp commit recorded in `scripts/llamacpp-patches/UPSTREAM`, apply the included patches, and compile the two engine sidecars used by the client.

### Linux

```sh
./scripts/build_llamacpp.sh
make all IDLETOKEN_PLATFORM_VERIFY_KEY_B64="$(tr -d '\r\n' < scripts/platform-verify-key.b64)"
make -f Makefile.platform
./scripts/build_client_release.sh
```

The installers are written below `client/src-tauri/target/release/bundle/deb/` and `client/src-tauri/target/release/bundle/rpm/`.

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

If CUDA is installed in a nonstandard location, set `IDLETOKEN_CUDA_RUNTIME_DIR` to the directory containing `cudart64_12.dll`, `cublas64_12.dll`, and `cublasLt64_12.dll`.

## Help

For installation, pairing, API, or source-build problems, search the [existing issues](https://github.com/idletoken/IdleToken/issues) or open one with your OS, IdleToken version, hardware, the command that failed, and the relevant error or log excerpt.

## Project links

[Project website](https://idletoken.ai) · [Releases](https://github.com/idletoken/IdleToken/releases) · [Issues](https://github.com/idletoken/IdleToken/issues) · [Apache-2.0 license](LICENSE)

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) and [Tauri](https://github.com/tauri-apps/tauri). See [NOTICE](NOTICE) for third-party acknowledgements.
