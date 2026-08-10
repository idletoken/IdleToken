<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-dark.svg">
    <img src="docs/images/logo.svg" alt="IdleToken" width="380">
  </picture>
</p>

<p align="center">
  <b>Turn the idle NVIDIA GPUs in your house into one inference cluster.</b><br>
  Run models no single machine can hold, behind an OpenAI- and Anthropic-compatible API.
</p>

<p align="center">
  <a href="https://github.com/idletoken/IdleToken/releases/latest">Download</a> ·
  <a href="https://github.com/idletoken/IdleToken/issues">Issues</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

<p align="center">
  <img src="docs/images/screenshot.png" alt="IdleToken" width="820">
</p>

---

A 16 GB card cannot hold a 300B model. Four of them can. IdleToken measures what
each machine actually has free, splits the model across them by capacity, runs
pipeline parallelism over your LAN, and serves the result as an API — from a
desktop app, not a config file.

**One machine is enough to start.** Add more later; the split re-plans itself.

## Quick start

1. Install from [Releases](https://github.com/idletoken/IdleToken/releases/latest)
2. Sign in, pick a model that fits your card
3. Start — the API comes up on `:8000`

```sh
export ANTHROPIC_BASE_URL=http://<your-machine>:8000
export ANTHROPIC_API_KEY=whatever
claude
```

To add machines, generate a pairing code on one and enter it on the others.
Discovery, capacity probing, splitting and weight download are automatic.

## What you can run

| Model | Weights | Runs on |
| --- | ---: | --- |
| Qwen3.5-0.8B | 0.49 GiB | any supported card |
| Qwen3.5-4B | 2.54 GiB | 8 GB |
| Qwen3-8B | 4.68 GiB | 8 GB |
| Qwen3.5-9B | 5.28 GiB | 12 GB |
| Qwen3.5-27B | 15.58 GiB | 24 GB |
| Qwen3.5-35B-A3B | 20.49 GiB | 24 GB, or two machines |
| **DeepSeek-V4-Flash-0731** | **80.76 GiB** | **a cluster** — 304B, 13B active |

Q4_K_M except DeepSeek-V4-Flash, which is Q2. The app reports which of these fit
your hardware, at what context length, and for the rest — how many GB you are
short.

## Requirements

- **NVIDIA GPU on every compute node**: compute capability ≥ 7.5 (RTX 20-series
  or newer), ≥ 4 GB VRAM. CPU-only nodes are refused, not silently downgraded.
- Driver: Windows ≥ 527.41 · Linux ≥ 580.65
- Windows or Linux for compute; macOS and phones can pair and monitor
- Gigabit LAN is enough; nothing leaves your network
- CUDA Toolkit optional — noticeably faster prompt processing when present

Linux has no prebuilt package yet — build from source:

```sh
make                                        # engine (CUDA toolkit + nvcc)
cd client && pnpm install && pnpm build
cd src-tauri && cargo build --release       # desktop app
```

## How it works

Only hidden states cross the wire — 64 KB per token per stage boundary for
DeepSeek-V4-Flash, so gigabit is not the bottleneck. A machine joining the
cluster downloads **only its assigned layers** over HTTP Range: a node holding 4
of 43 layers fetched 9.54 GB instead of 86.72 GB. Planning subtracts what the
operating system already occupies rather than dividing total VRAM.

Gigabit LAN at 32K context misses our latency target under pure pipeline
parallelism — that needs 2.5G+ or sequence parallelism, which is not in v0.1.
Speed depends entirely on your cards and your network, so measure your own with
`scripts/bench.py` rather than trusting a number from someone else's machines.

exo and llama.cpp's RPC backend support far more hardware. IdleToken trades that
breadth for a desktop app, capacity-aware splitting, and native Anthropic
support.

## Status

Beta. Releases are gated by an acceptance ladder that runs on real machines,
including a walkthrough of the shipped installer on a clean one:
`scripts/acceptance.sh`.

Known gaps: the installer is unsigned — verify the SHA256 on the release page;
automatic weight download is new in v0.1.0 and not yet walked through end to end
(fall back to setting the GGUF path in Settings); there is no auto-update.

Questions and bug reports belong in
[Issues](https://github.com/idletoken/IdleToken/issues). A change that breaks one
of the requirements above — CPU fallback, guessing at capacity instead of
measuring it — will not be merged even if it works.

## License

Apache-2.0. The "IdleToken" name and marks are not licensed with the code — ship
derivatives under your own name. Vendored: [ds4](https://github.com/antirez/ds4)
(MIT), TweetNaCl, BLAKE2b. Model weights are not distributed with this software.
