<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-dark.svg">
    <img src="docs/images/logo.svg" alt="IdleToken" width="380">
  </picture>
</p>

<p align="center">
  <b>Run models on the machines you already own.</b><br>
  Share the tokens you don't use. Use someone else's when you run short.
</p>

<p align="center">
  <a href="https://github.com/idletoken/IdleToken/issues">Issues</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

<p align="center">
  <img src="docs/images/screenshot.png" alt="IdleToken" width="820">
</p>

---

Your GPU is idle most of the day. IdleToken puts a model on it — one machine, or
several pooled into one cluster — and serves what it produces behind an OpenAI-
and Anthropic-compatible API. Point Claude Code at it and the tokens come from
your own hardware.

Capacity you don't use can be shared. Capacity you're short of can come from
someone else's idle machine. Sharing earns credits; borrowing spends them.

**The client is open source so you can check it.** It runs on your machine, sees
your prompts and drives your GPU — that is exactly the kind of software whose
source you should be able to read. The marketplace that matches sharers with
borrowers is a hosted service and is not part of this repository.

## Quick start

> **No installer is published yet.** Build from source (see Requirements below)
> until the first release ships.

1. Install and launch the app
2. Sign in, pick a model that fits your hardware
3. Start — the API comes up on `:8000`

```sh
export ANTHROPIC_BASE_URL=http://<your-machine>:8000
export ANTHROPIC_API_KEY=whatever
claude
```

One machine is enough. To add more, generate a pairing code on one and enter it
on the others; discovery, capacity probing, splitting and weight download are
automatic, and the split re-plans itself as machines come and go.

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
short. Pooling machines is what puts the bottom row in reach: no consumer card
holds 80 GiB on its own.

## Sharing and borrowing

Off by default. Nothing leaves your network until you turn it on, and turning it
off ends it.

- **What a borrower's request looks like to you.** It arrives envelope-encrypted
  (X25519 sealed box), is opened only in your coordinator's memory, and is never
  written to a log. The other machines in your cluster see hidden states only —
  no text, no keys, no tokenizer.
- **What the platform sees.** Plaintext, because it moderates content and meters
  usage. That is the price of a shared marketplace, and it belongs in plain sight
  rather than in a footnote. A cluster you run for yourself involves no
  encryption and no third party at all.
- **Credits, not money.** Sharing earns them, borrowing spends them, and a new
  account starts with some. There is no payout.

## Requirements

- **NVIDIA GPU on every compute node**: compute capability ≥ 7.5 (RTX 20-series
  or newer), ≥ 4 GB VRAM. CPU-only nodes are refused, not silently downgraded.
- Driver: Windows ≥ 527.41 · Linux ≥ 580.65
- Windows or Linux for compute; macOS and phones can pair and monitor
- Gigabit LAN between machines; a shared cluster reaches the platform over an
  outbound connection, so no port forwarding
- CUDA Toolkit optional — noticeably faster prompt processing when present

Until the first release ships, building from source is the only route (a Windows
installer and a Linux package follow):

```sh
make                                        # engine (CUDA toolkit + nvcc)
cd client && pnpm install && pnpm build
cd src-tauri && cargo build --release       # desktop app
```

## How it works

Only hidden states cross the wire — 64 KB per token per stage boundary for
DeepSeek-V4-Flash, so gigabit is not the bottleneck. A machine joining a cluster
downloads **only its assigned layers** over HTTP Range: a node holding 4 of 43
layers fetched 9.54 GB instead of 86.72 GB. Planning subtracts what the operating
system already occupies rather than dividing total VRAM, which is why the app can
tell you what fits before you spend an hour downloading.

Gigabit at 32K context misses our latency target under pure pipeline parallelism —
that needs 2.5G+ or sequence parallelism, which is not in v0.1. Speed depends
entirely on your hardware and your network, so measure your own with
`scripts/bench.py` rather than trusting a number from someone else's machines.

## Status

Beta. Releases are gated by an acceptance ladder that runs on real machines,
including a walkthrough of the shipped installer on a clean one:
`scripts/acceptance.sh`.

Known gaps: no installer is published yet; the build that ships will be unsigned,
so Windows will warn once; automatic weight download is new and not yet walked
through end to end (fall back to setting the GGUF path in Settings); there is no
auto-update. Planned: macOS as a compute node — today it can pair and monitor,
but not compute.

Questions and bug reports belong in
[Issues](https://github.com/idletoken/IdleToken/issues). A change that breaks one
of the requirements above — CPU fallback, guessing at capacity instead of
measuring it — will not be merged even if it works.

## License

Apache-2.0. The "IdleToken" name and marks are not licensed with the code — ship
derivatives under your own name. Vendored: [ds4](https://github.com/antirez/ds4)
(MIT), TweetNaCl, BLAKE2b. Model weights are not distributed with this software.
