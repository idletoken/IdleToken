<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-dark.svg">
    <img src="docs/images/logo.svg" alt="IdleToken" width="380">
  </picture>
</p>

**Share when idle. Scale when busy.**

Put your idle compute to work, then tap into more Tokens when demand spikes.

[中文](README.zh-CN.md)

<p align="center">
  <img src="docs/images/screenshot.png" alt="IdleToken" width="820">
</p>

---

## Why IdleToken

Agent workloads are peaky by nature. Most of the time only one or two inference tasks are running; then a complex job arrives, several agents start working at once, and a burst of parallel requests drives the demand for compute straight up.

Machines deployed at home are peaky in the same way — busy sometimes, idle most of the time. Nobody runs a model around the clock, so a machine spends most of its life with capacity to spare, and then turns out to be short of it exactly when a lot of inference is needed.

What IdleToken sets out to do is connect one person's idle hours to another's busy ones: **share your compute when you are not using it and earn Sparks; spend Sparks to use someone else's idle machine when you need more.**

## Getting started

There are two ways to use IdleToken, depending on whether you have a machine with a supported GPU.

### No machine: use the platform

Sign up at [idletoken.ai](https://idletoken.ai) and create an API key, then use it like any third-party model provider:

```sh
export ANTHROPIC_BASE_URL=https://api.idletoken.ai
export ANTHROPIC_API_KEY='<your platform API key>'
claude
```

The OpenAI-compatible endpoint works at the same base URL. Requests run on clusters other people share and are billed in Sparks; new accounts start with 100.

### Your own machine: deploy IdleToken

IdleToken is a desktop app; normal use needs no command line. Installers for Windows, Linux and macOS are on the [Releases](https://github.com/idletoken/IdleToken/releases) page.

1. **Pick a model** — from the built-in list, a local GGUF file, or a Hugging Face link. The app checks it against this machine's available VRAM and RAM.
2. **Start serving** — missing weights download automatically. The API listens on `:8000`; the app shows the address and the generated API key.

Connect Claude Code:

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_API_KEY='<key shown in IdleToken>'
claude
```

Or use the OpenAI-compatible endpoint:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'authorization: Bearer <key shown in IdleToken>' \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"Hello"}]}'
```

When a model is too large for one machine, machines on the same LAN can serve it together. Create a cluster on one machine and join the others with a six-character code or the same account; IdleToken measures each machine's usable memory and splits the model accordingly. Windows, Linux and macOS nodes can mix in one cluster; every node must run the same IdleToken version. Workers receive their model shards from the API machine over an encrypted connection, and prompts do not cross the cluster in plain text.

To earn Sparks, turn on sharing while your cluster is idle — it is off by default.

## Models

IdleToken runs text-generation models in GGUF format — from the built-in list, a local file, or a Hugging Face link; model metadata is read from the GGUF header. Multimodal input is not supported.

The built-in models:

| Model                  |   Default weights | Default quant | Notes                  |
| ---------------------- | ----------------: | ------------- | ---------------------- |
| Qwen3.5-0.8B           |          0.49 GiB | Q4_K_M        |                        |
| Qwen3.5-4B             |          2.54 GiB | Q4_K_M        |                        |
| Qwen3-8B               |          4.68 GiB | Q4_K_M        |                        |
| Qwen3.5-9B             |          5.28 GiB | Q4_K_M        |                        |
| Qwen3.5-27B            |         15.58 GiB | Q4_K_M        |                        |
| Qwen3.5-35B-A3B        |         20.49 GiB | Q4_K_M        | 3B active parameters   |
| DeepSeek-V4-Flash-0731 |         80.76 GiB | IQ2_XXS + Q2_K | 304B total, 13B active |

Several Qwen models also offer Q5, Q6, Q8 and BF16 variants. DeepSeek-V4-Flash can be distributed across a cluster; the other built-in models run on a single machine.

## Requirements

These apply to machines that run inference; using the platform requires no GPU and no install.

| Platform | Compute hardware | Also needs |
| --- | --- | --- |
| Windows 10/11 | NVIDIA, compute capability ≥ 7.5 (RTX 20-series or newer), ≥ 4 GB VRAM | driver ≥ 527.41 and CUDA Toolkit 12.x |
| Linux | NVIDIA, compute capability ≥ 7.5 (RTX 20-series or newer), ≥ 4 GB VRAM | driver ≥ 580.65 and CUDA Toolkit 13.0 |
| macOS | Apple Silicon with Metal and enough unified memory for the selected model | no CUDA installation |

- Operating systems can mix in one cluster; the IdleToken version must match on every node.
- Cluster machines need a direct LAN route to each other; tensor traffic does not go through VPN or overlay networks such as Tailscale. Wired gigabit or faster is recommended.
- CPU-only machines, AMD GPUs, Intel Macs and phones cannot compute, but can run the client to sign in, chat and control a cluster.
- The API machine stores the complete GGUF; a worker only needs memory for its assigned shard.

## Building from source

On Linux or macOS:

```sh
./scripts/build_llamacpp.sh
make
make -f Makefile.platform
./scripts/stage_sidecars.sh
cd client && pnpm install && pnpm build
cd src-tauri && cargo build --release
```

On Windows, build the engine with `scripts\build_llamacpp_win.bat`, build the coordinator, worker and platform agent, then run `scripts\build_client_release.bat`.

Packaging: `scripts\build_client_release.bat` (Windows NSIS installer), `scripts/build_client_release.sh` (Linux `.deb` / `.rpm`), `scripts/package_client_mac.sh` (macOS `.dmg`).

## Troubleshooting

**Installer warnings (SmartScreen / Gatekeeper).** The Windows installer is not Authenticode-signed and the macOS dmg is not notarized yet. Windows: choose **More info → Run anyway**. macOS: right-click the app and choose **Open**, or run `xattr -d com.apple.quarantine /Applications/IdleToken.app`.

**Chat streams hang while a local HTTP proxy is running.** Proxies such as Clash can swallow loopback SSE streams. Set `NO_PROXY=127.0.0.1,localhost` in the terminal running `claude` or `curl`, or add a direct-connection rule for `127.0.0.1` to the proxy.

**Windows Firewall blocks pairing or cluster traffic.** Run IdleToken elevated once, or run the `netsh` command printed in the log as administrator. Ports: UDP 14097 and 14099 (discovery and pairing), TCP 14100 and 14101 (cluster control), TCP 50052 (worker rpc-server, configurable), TCP 8000 (API, only when other devices call it).

**Linux client opens a blank window.** Launch with `WEBKIT_DISABLE_DMABUF_RENDERER=1 idletoken-client`.

## License

[Apache-2.0](LICENSE)

## Acknowledgements

IdleToken would not exist without a number of excellent open-source projects. Particular thanks to:

* [ds4](https://github.com/antirez/ds4)
* [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)
* [Ollama](https://github.com/ollama/ollama)
* [Tauri](https://github.com/tauri-apps/tauri)
* [TweetNaCl](https://tweetnacl.cr.yp.to/)
* [BLAKE2](https://github.com/BLAKE2/BLAKE2)
* [DeepSeek](https://github.com/deepseek-ai)
* [Qwen](https://github.com/QwenLM)

and to every open-source project IdleToken depends on, and everyone who contributes to them.
