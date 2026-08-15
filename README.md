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

## How to use it

IdleToken is a desktop app; normal use involves no command line.

**Pick a model first.** Start with one of the models in the app, open a local GGUF file, or paste a Hugging Face repository or file link. IdleToken reads the memory that is actually free — VRAM and RAM, after the operating system's share — and tells you what will fit, how much context it can open, and what the shortfall is when it cannot.

**Then start it.** Missing weights download automatically. If the model fits on one machine, IdleToken runs it there directly, without paying any cluster overhead. Once it is ready, the local API runs on `:8000` by default; the app shows both its address and the API key it generated for you.

Claude Code connects to it directly:

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_API_KEY='paste the key shown in IdleToken here'
claude
```

Or call it through the OpenAI-compatible endpoint:

```sh
export IDLETOKEN_API_KEY='paste the key shown in IdleToken here'
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "authorization: Bearer $IDLETOKEN_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"Hello"}]}'
```

If one machine cannot hold a model offered with cluster support, add another machine on the same LAN. Windows, Linux and Apple Silicon Macs can work in the same cluster. Create a cluster on one machine and join the others with a six-character code or the same account. IdleToken discovers the machines, measures their usable memory and divides the model between them. Every node must run the same IdleToken engine version; a mismatched node is refused with an upgrade message instead of being allowed to produce unverified output.

The machine running the API keeps the complete GGUF file. Workers do not need their own copy: model data is sent to them over an authenticated, encrypted connection on the local network. The API machine also keeps the token embedding stage local, so prompts are not sent across the cluster as plain text.

If you have spare capacity, you can turn sharing on. Sharing earns Sparks, and when you need more compute you spend Sparks on what other people have shared. Sharing is off by default and can be turned off again at any time.

**No machine of your own?** Then use IdleToken from the other side: no GPU, nothing to install. Sign up at [idletoken.ai](https://idletoken.ai), create an API key, and point the same clients at the platform instead of at a local address:

```sh
export ANTHROPIC_BASE_URL=https://api.idletoken.ai
export ANTHROPIC_API_KEY='paste your platform API key here'
claude
```

The OpenAI-compatible endpoint works the same way against the same base URL. Each request is routed to a cluster somebody else is sharing and billed in Sparks; new accounts start with 100 of them.

One thing to know before you try it: every cluster on the platform is somebody's own machine, switched on when they are not using it. Supply comes and goes; when no suitable cluster is online, a request fails clearly rather than being quietly served by something else.

## Models

IdleToken can run text-generation GGUF models supported by its pinned version of llama.cpp. A model does not have to be added to IdleToken first: you can open a local file or use a Hugging Face link, and its metadata is read from the GGUF header.

The app also includes a curated starting set with known downloads and sensible defaults:

| Model                  |   Default weights | Default quant | Notes                  |
| ---------------------- | ----------------: | ------------- | ---------------------- |
| Qwen3.5-0.8B           |          0.49 GiB | Q4_K_M        |                        |
| Qwen3.5-4B             |          2.54 GiB | Q4_K_M        |                        |
| Qwen3-8B               |          4.68 GiB | Q4_K_M        |                        |
| Qwen3.5-9B             |          5.28 GiB | Q4_K_M        |                        |
| Qwen3.5-27B            |         15.58 GiB | Q4_K_M        |                        |
| Qwen3.5-35B-A3B        |         20.49 GiB | Q4_K_M        | 3B active parameters   |
| DeepSeek-V4-Flash-0731 |         80.76 GiB | IQ2_XXS + Q2_K | 304B total, 13B active |

Several Qwen models also have Q5, Q6, Q8 and BF16 variants. Multimodal input is not supported yet.

A GGUF opened directly runs on the local machine. DeepSeek-V4-Flash can use the cluster path; there, placement is decided from the model's real size and the machines' usable memory. If it fits on one machine, it stays there, and distribution is used only when it is needed. A cluster adds capacity, not free speed — splitting a small model across a LAN is usually slower than running it locally, so IdleToken takes the shorter path whenever it can.

## Requirements

All of this applies to machines that run inference. Calling shared clusters through the platform needs none of it — no GPU, no install.

Compute nodes can run on **Windows, Linux and Apple Silicon Macs**, one machine or several:

| Platform | Compute hardware | Also needs |
| --- | --- | --- |
| Windows 10/11 | NVIDIA, compute capability ≥ 7.5 (RTX 20-series or newer), ≥ 4 GB VRAM | driver ≥ 527.41 and CUDA Toolkit 12.x |
| Linux | NVIDIA, compute capability ≥ 7.5 (RTX 20-series or newer), ≥ 4 GB VRAM | driver ≥ 580.65 and CUDA Toolkit 13.0 |
| macOS | Apple Silicon with Metal and enough unified memory for the selected model | no CUDA installation |

Windows, Linux and macOS nodes may be mixed in one cluster. What has to match is the inference engine build, not the operating system. Every machine also needs a direct local-network route to the others; tensor traffic deliberately stays on the real LAN instead of going through Tailscale or another overlay network. Wired gigabit works, while 2.5 GbE or 10 GbE gives large distributed models more room.

CPU-only machines, AMD GPUs and Intel Macs are not supported as compute nodes. They can still use the client to sign in, chat and control another cluster, as can iPhones and Android phones.

The API machine needs enough disk space for the complete GGUF even when the model is distributed. A worker only needs room for its assigned tensors in VRAM or RAM.

## Building from source

Installers for Windows, Linux and macOS are published on the [Releases](https://github.com/idletoken/IdleToken/releases) page. To build the engine and client yourself on Linux or macOS:

```sh
./scripts/build_llamacpp.sh
make
make -f Makefile.platform
./scripts/stage_sidecars.sh
cd client && pnpm install && pnpm build
cd src-tauri && cargo build --release
```

On Windows, build the pinned llama.cpp engine with `scripts\build_llamacpp_win.bat`, then build the coordinator, worker and platform agent before running `scripts\build_client_release.bat`.

Release packages are built with `scripts\build_client_release.bat` on Windows, `scripts/build_client_release.sh` on Linux and `scripts/package_client_mac.sh` on macOS. They produce Windows NSIS installers, Linux `.deb` and `.rpm` packages, and macOS `.dmg` images. Every release candidate is accepted by running `scripts/acceptance.sh` on real hardware.

## Troubleshooting

**Windows SmartScreen / macOS Gatekeeper warnings on install.** The Windows installer is not Authenticode-signed yet, so SmartScreen shows "Windows protected your PC" — choose **More info → Run anyway**. The macOS dmg is not notarized yet, so Gatekeeper refuses the first launch: right-click the app and choose **Open**, or run `xattr -d com.apple.quarantine /Applications/IdleToken.app`. Both are unsigned-identity warnings, not malware verdicts, and both are separate from the in-app update channel, which cryptographically verifies every package it installs.

**Chat streams hang forever while a local HTTP proxy is running.** A system-wide proxy (Clash and similar) can intercept loopback SSE traffic and swallow the end of the stream. Set `NO_PROXY=127.0.0.1,localhost` in the terminal that runs `claude` or `curl`, or add a direct-connection rule for `127.0.0.1` in the proxy. The client already sets this for the engine processes it starts itself.

**Windows Firewall blocks pairing or cluster traffic.** IdleToken adds its own inbound rules when it runs elevated; otherwise the log prints the exact `netsh` command to run once as administrator. The ports involved: UDP 14097 (engine discovery) and UDP 14099 (client pairing beacon); TCP 14100 and 14101 (coordinator/worker control channel); TCP 50052 (a worker's rpc-server, configurable); and TCP 8000 (the API, needed only when other devices call it).

**Linux client opens a blank window.** Launch with `WEBKIT_DISABLE_DMABUF_RENDERER=1 idletoken-client` — a WebKitGTK dmabuf issue on some driver stacks.

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
