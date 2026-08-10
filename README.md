# IdleToken

**Run models on your own machines and produce tokens. Share the ones you don't use; use someone else's when you run short.**

IdleToken is an open-source local LLM client and inference engine.

[中文](README.zh-CN.md)

---

## Why IdleToken

Agent workloads are peaky by nature. Most of the time only one or two inference tasks are running; then a complex job arrives, several agents start working at once, and a burst of parallel requests drives the demand for compute straight up.

Machines deployed at home are peaky in the same way — busy sometimes, idle most of the time. Nobody runs a model around the clock, so a machine spends most of its life with capacity to spare, and then turns out to be short of it exactly when a lot of inference is needed.

What IdleToken sets out to do is connect one person's idle hours to another's busy ones: **share your compute when you are not using it and earn Sparks; spend Sparks to use someone else's idle machine when you need more.** Capacity that would otherwise sit idle flows to where it is actually needed.

## How to use it

IdleToken is a desktop app; normal use involves no command line.

**Pick a model first.** IdleToken reads the VRAM the machine currently has free and tells you which models it can run and how long a context they can open; for the ones that do not fit, it tells you how much VRAM you are short.

**Then start it.** Missing weights download automatically, and once the model is ready the local service runs on `:8000` by default.

Claude Code connects to it directly:

```sh
export ANTHROPIC_BASE_URL=http://<your-machine>:8000
export ANTHROPIC_API_KEY=whatever
claude
```

Or call it through the OpenAI-compatible endpoint:

```sh
curl http://<your-machine>:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"Hello"}]}'
```

If one machine cannot hold the model, add another. Create a cluster on one machine to get a join code, and enter that code on the others. Node discovery, capacity probing, model splitting and weight download all happen automatically from there.

If you have spare capacity, you can turn sharing on. Sharing earns Sparks, and when you need more compute you spend Sparks on what other people have shared. Sharing is off by default and can be turned off again at any time.

## Recent updates

<!--
Keep the last few updates worth knowing about here, for example:

- YYYY-MM-DD · what changed
- YYYY-MM-DD · what changed
- YYYY-MM-DD · what changed
-->

## Models

IdleToken runs everything from small single-card models to large ones that need several machines working together. For example:

| Model                  |   Weights | Notes                                          |
| ---------------------- | --------: | ---------------------------------------------- |
| Qwen3.5-0.8B           |  0.49 GiB |                                                |
| Qwen3-8B               |  4.68 GiB |                                                |
| Qwen3.5-35B-A3B        | 20.49 GiB | 24 GB, or two machines                         |
| DeepSeek-V4-Flash-0731 | 80.76 GiB | 304B total, 13B active; needs a cluster        |

Every model is Q4_K_M except DeepSeek-V4-Flash-0731, which is Q2.

The full model list and its requirements will be maintained on a page of its own.

## Requirements

Compute nodes currently run on **Windows and Linux**. Each machine needs:

* an NVIDIA GPU with compute capability ≥ 7.5 (RTX 20-series or newer)
* ≥ 4 GB of VRAM
* Windows driver ≥ 527.41
* Linux driver ≥ 580.65

CPU-only nodes are refused outright; there is no silent fallback to CPU inference.

## Building from source

IdleToken is in **Beta** and no installer has been published yet. Until the first release ships, build from source:

```sh
make
cd client && pnpm install && pnpm build
cd src-tauri && cargo build --release
```

Worth knowing for now:

* the first installer will not be code signed, so Windows will warn the first time it runs;
* automatic weight download is a new feature and has not been walked through end to end;
* there is no auto-update yet;
* releases are accepted by running `scripts/acceptance.sh` against real machines.

## License

**Apache-2.0**

## Acknowledgements

IdleToken would not exist without a number of excellent open-source projects. Particular thanks to:

* Ollama
* [ds4](https://github.com/antirez/ds4)

and to every open-source project IdleToken depends on, and everyone who contributes to them.
