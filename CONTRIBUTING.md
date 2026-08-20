# Contributing

## Before you change anything

These constraints are not negotiable. A change that violates one of them will
not be merged even if it works:

- **The engine is llama.cpp**, pinned to the commit recorded in
  `scripts/llamacpp-patches/UPSTREAM` and built with the patches in that
  directory. Upstream gaps are filled by adding a patch there, not by forking.
  Moving the pin is its own decision with its own acceptance run — it does not
  ride along with another change.
- **A supported GPU on every compute node**: NVIDIA with compute capability
  ≥ 7.5 and ≥ 4 GB of VRAM on Windows and Linux, or Apple Silicon with Metal on
  macOS. A machine that cannot serve layers is refused, never silently
  downgraded to CPU or to a mock backend. Operating systems may be mixed in one
  cluster; the engine version may not.
- **Pipeline parallelism over the LAN.** No tensor or expert parallelism — both
  need interconnect bandwidth a home network does not have. Tensor traffic goes
  over a real LAN route, never through a VPN or overlay network.
- **Both API protocols.** OpenAI *and* Anthropic; Claude Code is a primary use
  case, not an afterthought.
- **Capacity is measured, never assumed.** Planning subtracts what the OS
  already occupies instead of dividing total VRAM.
- **No silent fallback.** If the engine, the GPU or the TLS link cannot come
  up, the answer is an error — not a mock, not a downgrade.

## Build

```sh
./scripts/build_llamacpp.sh   # the pinned engine, with our patches
make                          # coordinator + worker
cd client && pnpm install && pnpm build
```

## Checks you can run without any hardware

```sh
scripts/acceptance.sh --gate G_MODEL   # model registry + planner
python3 scripts/model_manifest_check.py
cd client && npx tsc --noEmit
```

All three run in CI. Everything else in the acceptance ladder needs a set of
machines with supported GPUs; the gate names are listed near the top of
`scripts/acceptance.sh`, and `scripts/testbed.env.example` shows how to point
the ladder at your own machines.

## Two rules that matter more than style here

**A check that cannot fail is not a check.** When you add or change one, break
it on purpose once and confirm it goes red. This repo has shipped several
green-but-blind checks — a sweep that matched no processes for four days, a
forbidden-pattern scan whose escaping matched nothing, a freshness assertion
reading a stale artifact. Every one of them looked fine.

**Numbers come from measurement, not reasoning.** Performance claims must come
from [`scripts/bench.py`](scripts/bench.py) on real hardware, and a number
without a recorded run behind it does not go into a document.

## Adding a model

The model list is curated on purpose: a shared endpoint is only trustworthy if
someone has actually run what it serves. Requests go through a
[GitHub issue](https://github.com/idletoken/IdleToken/issues) — the bar for
being added is a real-machine perplexity gate plus a testbed smoke run, and
`available` stays `false` until both exist.

Mechanically, a model is a `models/<id>.json` manifest plus a backend that can
run the architecture; reusing an architecture that is already supported is much
cheaper. Keep the manifest, the engine registry (`src/common/model.c`) and the
client list (`client/src/models.ts`) in sync — `scripts/model_manifest_check.py`
enforces it, and it also requires a sha256 for every available model.
