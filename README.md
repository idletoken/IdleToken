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

## What it does

- **Runs a model locally** and serves it on `:8000` as an OpenAI- and
  Anthropic-compatible API.
- **Pools several machines** into one cluster when a model does not fit on one.
  Pair them with a code; layer splitting and weight download are automatic.
- **Tells you what fits** before you download anything — which models your
  hardware can run, at what context length, and how many GB the rest are short.
- **Downloads only what each machine needs**: a node assigned 4 of 43 layers
  fetches those 4, not the whole file.
- **Shares spare capacity** to the marketplace when you turn it on, and lets you
  use other people's when you run short. Off by default; settled in credits, not
  money.

## How to use it

**1. Install and launch.** No installer is published yet — build from source
(below) until the first release ships.

**2. Sign in and pick a model.** The capability panel marks which ones fit this
machine. Press Start; the app downloads the weights if they are missing and
brings the API up on `:8000`.

**3. Point a client at it.**

```sh
export ANTHROPIC_BASE_URL=http://<your-machine>:8000
export ANTHROPIC_API_KEY=whatever
claude
```

Or OpenAI-style:

```sh
curl http://<your-machine>:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"Hello"}]}'
```

**4. Add machines (optional).** On one machine choose *Create cluster* and note
the join code; on the others choose *Join cluster* and enter it. They find each
other on the LAN, split the model by what each has free, and the API comes back
up when every machine reports ready.

**5. Share or borrow (optional).** Turn sharing on to offer your spare capacity;
turn it off and it stops. Requests from others arrive encrypted and are never
written to a log, though the marketplace itself sees plaintext because it
moderates and meters usage.

## Models

| Model | Weights | Runs on |
| --- | ---: | --- |
| Qwen3.5-0.8B | 0.49 GiB | any supported card |
| Qwen3.5-4B | 2.54 GiB | 8 GB |
| Qwen3-8B | 4.68 GiB | 8 GB |
| Qwen3.5-9B | 5.28 GiB | 12 GB |
| Qwen3.5-27B | 15.58 GiB | 24 GB |
| Qwen3.5-35B-A3B | 20.49 GiB | 24 GB, or two machines |
| **DeepSeek-V4-Flash-0731** | **80.76 GiB** | **a cluster** — 304B, 13B active |

Q4_K_M except DeepSeek-V4-Flash, which is Q2.

## Requirements

- **NVIDIA GPU on every compute node**: compute capability ≥ 7.5 (RTX 20-series
  or newer), ≥ 4 GB VRAM. CPU-only nodes are refused, not silently downgraded.
- Driver: Windows ≥ 527.41 · Linux ≥ 580.65
- Windows or Linux to compute; macOS and phones can pair and monitor
- Gigabit LAN between machines. Sharing uses an outbound connection, so no port
  forwarding.
- CUDA Toolkit optional — faster prompt processing when present

## Build from source

```sh
make                                        # engine (CUDA toolkit + nvcc)
cd client && pnpm install && pnpm build
cd src-tauri && cargo build --release       # desktop app
```

## Status

Beta. No installer published yet; when one ships it will be unsigned, and there
is no auto-update. macOS as a compute node is planned — today it can pair and
monitor only.

Bug reports and questions: [Issues](https://github.com/idletoken/IdleToken/issues).

## License

Apache-2.0. The "IdleToken" name and marks are not licensed with the code.
Vendored: [ds4](https://github.com/antirez/ds4) (MIT), TweetNaCl, BLAKE2b. Model
weights are not distributed with this software.
