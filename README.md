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
  · <a href="https://idletoken.ai">Open the marketplace</a>
  · <a href="README.zh-CN.md">中文</a>
</p>

## Why IdleToken

Agent workloads arrive in bursts, while home GPUs spend much of their time unused. IdleToken lets those machines back each other up: share idle compute to earn Sparks, then spend Sparks on models shared by others when your own machines are busy.

## Two ways to start

### Deploy and share your own model

1. Install the desktop app on Windows or Linux with an NVIDIA GPU, or on an Apple Silicon Mac.
2. Choose a curated model, quantization, and context, then run it on one computer or a mixed-OS LAN cluster.
3. Turn on sharing to earn Sparks. Turn on backup when you want IdleToken to draw on shared capacity.

### Call a model shared by someone else

1. Create an account at [idletoken.ai](https://idletoken.ai).
2. Open **Sparks → API keys** and create a key.
3. Connect any Anthropic- or OpenAI-compatible client. No GPU or desktop installation is required.

New accounts start with zero Sparks. Earn them by sharing compute or use a redemption code; purchasing Sparks is not currently supported.

## Connect your tools

| Use | Base URL | API key |
| --- | --- | --- |
| Your local model | `http://127.0.0.1:8000` | Any non-empty value |
| Shared marketplace | `https://api.idletoken.ai` | A key created in the portal |

For Claude Code:

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_API_KEY=idletoken
claude
```

Use the same base URL with the OpenAI-compatible API. `GET /v1/models` returns the model IDs available at that endpoint.

## What you get

- **One app, one machine or many.** Single-machine inference talks directly to llama.cpp; clusters split the model across Windows, Linux, and macOS computers on the same LAN.
- **Two familiar APIs.** OpenAI and Anthropic compatibility includes Claude Code as a first-class use case.
- **Local-first boundaries.** The local API listens only on `127.0.0.1`; cluster tensor traffic stays on the direct LAN and is protected with PSK-TLS.
- **Explicit resource choices.** Pick 256K or, where supported, 1M context. IdleToken refuses insufficient configurations instead of silently shrinking the window or changing deployment mode.

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

The repository contains the Tauri client and every native sidecar it needs: the coordinator, worker supervisor, platform agent, and the pinned llama.cpp server and RPC server. Build commands below run from the repository root.

### Prerequisites

Frontend-only development needs Node.js and pnpm. A full native build also needs Git, CMake, Rust 1.77 or newer, and the [Tauri v2 system prerequisites](https://tauri.app/start/prerequisites/) for the target operating system.

- **Linux:** a C compiler and the CUDA toolkit listed in the hardware section.
- **macOS:** Apple Silicon and Xcode Command Line Tools.
- **Windows:** Visual Studio 2022 Build Tools with **Desktop development with C++**, Windows SDK, CUDA Toolkit 12.8 with Visual Studio Integration, and WinLibs/MinGW tools providing `gcc` and `windres`. Set `IDLETOKEN_MINGW_BIN` if those tools are not on `PATH`.

### Frontend-only development

This path does not start a native engine. The browser UI uses a clearly marked development fixture.

```sh
cd client
pnpm install
pnpm dev
```

Open `http://localhost:1420`.

### Full desktop development on Linux or macOS

```sh
./scripts/build_llamacpp.sh
make
make -f Makefile.platform
./scripts/stage_sidecars.sh
cd client
pnpm install
pnpm tauri dev
```

Linux builds the CUDA backend; macOS builds the Metal backend. A missing GPU toolchain is a hard error—there is no CPU fallback.

### Installable packages

The packaging scripts verify that all sidecars are present and that the bundled engine matches the pinned llama.cpp revision. Linux and macOS release packaging requires a clean Git worktree so the artifact maps to an exact source commit.

Linux (`.deb` and `.rpm`):

```sh
./scripts/build_llamacpp.sh
make IDLETOKEN_PLATFORM_VERIFY_KEY_B64="$(tr -d '\r\n' < scripts/platform-verify-key.b64)"
make -f Makefile.platform
./scripts/build_client_release.sh
```

Artifacts are written below `client/src-tauri/target/release/bundle/`.

macOS (`.dmg`):

```sh
./scripts/build_llamacpp.sh
./scripts/package_client_mac.sh
```

Windows (`.exe`, from an x64 Native Tools Command Prompt):

```bat
scripts\build_llamacpp_win.bat
cd client
pnpm install
pnpm build:release
cd ..
scripts\build_client_release.bat
```

The Windows installer is written below `client\src-tauri\target\release\bundle\nsis\`. The release script rebuilds the coordinator, worker, and platform agent; stages the pinned llama.cpp sidecars and runtime DLLs; and runs the NSIS bundler.

### Checks without inference hardware

```sh
scripts/acceptance.sh --gate G_MODEL
python3 scripts/model_manifest_check.py
cd client && npx tsc --noEmit
```

The complete acceptance ladder is implemented by [`scripts/acceptance.sh`](scripts/acceptance.sh). Hardware gates use the machine definitions shown in [`scripts/testbed.env.example`](scripts/testbed.env.example).

## Help

For installation, pairing, or API problems, search the [existing issues](https://github.com/idletoken/IdleToken/issues) or open one with your OS, IdleToken version, hardware, and the relevant error or log excerpt.

## Project links

[Releases](https://github.com/idletoken/IdleToken/releases) · [Issues](https://github.com/idletoken/IdleToken/issues) · [Security](SECURITY.md) · [Contributing](CONTRIBUTING.md) · [Apache-2.0 license](LICENSE)

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) and [Tauri](https://github.com/tauri-apps/tauri). See [NOTICE](NOTICE) for third-party acknowledgements.
