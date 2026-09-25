<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-dark.svg">
    <img src="docs/images/logo.svg" alt="IdleToken" width="380">
  </picture>
</p>

<h1 align="center">Share your idle tokens.</h1>

<p align="center">
  Run large language models locally. Share yours when idle; use others' when busy.
</p>

<p align="center">
  <a href="https://github.com/idletoken/IdleToken/releases">⬇️ Download the client</a>
  · <a href="https://idletoken.ai">🏠 Project website</a>
  · <a href="README.zh-CN.md">🌐 中文</a>
</p>

---

## ✨ Why IdleToken

Agents' compute needs naturally rise and fall. Most of the time, only one or two inference tasks are running. For complex tasks, several agents work at once and send many inference requests in a short period.

Device usage has its own peaks and valleys: machines sit idle when users step away or stop using them for the night, yet often lack enough resources when needed, such as when running several applications while also hosting a large language model.

IdleToken aims to connect these offset peaks and valleys, letting idle machines supply compute to agents experiencing peak demand.

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

We use Sparks to settle token usage.

## 💬 Join the community

Join the IdleToken Discord to discuss using and developing the project.

[![Discord](https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white)](https://discord.gg/XbaWCH4t2J)

## 🚀 Two ways to start

> 📖 **New here? The [step-by-step guide](docs/user-guide.md) walks through the
> whole path with screenshots** — install, account, your first model, clusters,
> sharing, and connecting Claude Code.

### 🖥️ Deploy and share your own model

1. [Download and install the client](https://github.com/idletoken/IdleToken/releases).
2. Start the client, then download and deploy a model.
3. Turn on **Request help** to ask others to complete tasks queued locally.
4. Turn on **Share compute** to help others complete their tasks.

### ☁️ Use models shared by others

If you do not want to deploy a model yourself, for example when using IdleToken on a phone, tablet, or computer without a supported GPU, you can use models shared by others directly.

1. Create an account at [idletoken.ai](https://idletoken.ai/register).
2. Create an API key in your account.
3. Use the API to access models shared by others.

## 🔌 Connect existing tools

- **Connect to a model you deploy:** `http://127.0.0.1:8000`. Once the model starts, you can connect through this local address.
- **Use models shared by others:** `https://api.idletoken.ai`. After signing in, click your profile avatar in the top right, open **Sparks** → **API keys**, and click **New key**.

### Connect DeepSeek Harness

[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`) reaches IdleToken through a custom provider. Add one to `$DSH_HOME/settings.yaml` (`~/.dsh/settings.yaml` by default), replacing `MODEL_ID` with a model ID returned by `GET /v1/models`:

```yaml
llm-pi-ai:
  providers:
    idletoken:
      api: openai-completions
      baseURL: https://api.idletoken.ai/v1
      apiKeyEnv: IDLETOKEN_API_KEY
      models:
        - id: MODEL_ID
```

```sh
export IDLETOKEN_API_KEY='sk-idletoken-********************************'
npx @deepseek-ai/dsh web
```

The Web UI opens at `http://127.0.0.1:3080`; pick the model under the
`idletoken` provider. A model added by hand is treated as text-only, so give a
vision model `input: [text, image]` next to its `id`.

### Connect OpenCode

Create `opencode.json` in your project root, replacing `MODEL_ID` with a model ID returned by `GET /v1/models`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "idletoken/MODEL_ID",
  "provider": {
    "idletoken": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "IdleToken",
      "options": {
        "baseURL": "https://api.idletoken.ai/v1",
        "apiKey": "{env:IDLETOKEN_API_KEY}"
      },
      "models": {
        "MODEL_ID": {
          "name": "IdleToken"
        }
      }
    }
  }
}
```

```sh
export IDLETOKEN_API_KEY='sk-idletoken-********************************'
opencode
```

## 🧠 Models

[Currently supported models](models/)

Need another model? [Open an issue](https://github.com/idletoken/IdleToken/issues).

## 💻 Hardware

Minimum requirements for deploying a model on your own machine:

| Platform | Minimum requirements |
| --- | --- |
| **Windows 10/11 · Linux x86_64/arm64** | An NVIDIA GPU from the RTX 20 series or newer (Turing, compute capability 7.5) with 8 GB of VRAM, and a current NVIDIA driver. The published v0.1.90 Linux binaries require glibc 2.39; builds from the current source target and declare glibc 2.35. |
| **macOS** | An Apple Silicon Mac with 16 GB of unified memory. |

## 🖧 Cluster deployment

Our client lets you connect multiple devices into a cluster to deploy large language models. Follow these steps:

1. Install the same IdleToken version on every participating computer and make sure they can reach each other directly over the LAN.
2. Select the same model, quantization, and context on every computer, and finish downloading the model weights.
3. Create a cluster on one computer. Join from the others with the same account or a verification code.
4. When every node reports ready, choose cluster deployment on the creator and start the model. IdleToken detects each node's resources and assigns model layers automatically.

---

## 🛠️ Build from source

### Prerequisites

- Git, CMake, Node.js 18+, pnpm, and Rust 1.77+
- [Tauri v2 system prerequisites](https://tauri.app/start/prerequisites/)
- Linux: a C compiler, the CUDA Toolkit for the target architecture, pkg-config, libcurl development headers (7.68+ with TLS and asynchronous DNS), GLib 2.72+, Duktape 2.7, and CA certificates
- macOS: an Apple Silicon Mac and Xcode Command Line Tools
- Windows: Visual Studio 2022 Build Tools with Desktop development with C++, Windows SDK, CUDA Toolkit 12.8 with Visual Studio Integration, and WinLibs/MinGW providing `gcc` and `windres`

Desktop account/control requests, the provider agent and coordinator overflow share one transport. Public API connections preserve HTTPS and use the machine's proxy policy, including HTTP/SOCKS proxies and PAC. Windows builds fetch a pinned static curl with the Windows certificate store; macOS uses system libcurl. Proxy failures are reported without silently switching to direct access. Loopback inference bypasses proxies. Linux installers bundle a pinned private libproxy instead of requiring the distribution to upgrade its copy. Source builds can stage the same resolver with `scripts/build_platform_proxy_linux.sh`; its Debian/Ubuntu build dependencies are `libcurl4-openssl-dev`, `libglib2.0-dev`, `duktape-dev`, `gsettings-desktop-schemas-dev`, `pkg-config`, `meson`, `ninja-build`, `patchelf`, `curl`, and `ca-certificates`. Native proxy discovery runs in bounded subprocesses so a stuck PAC resolver cannot permanently exhaust the caller. Setting `CURL_CA_BUNDLE` in the environment overrides the certificate store used for every platform connection; leave it unset to use the system store.

### Get the source

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

Installers are written below `client/src-tauri/target/release/bundle/deb/` and
`client/src-tauri/target/release/bundle/rpm/`. Other Linux distributions can
adapt the open-source build, but their packages are not official release assets.

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

<p align="center"><sub>See <a href="NOTICE">NOTICE</a> for third-party acknowledgements.</sub></p>
