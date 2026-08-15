/* idletoken_plan.h — cluster mode decision + PP layer split.
 *
 * The capacity model of docs/architecture.md §5 as pure functions: no sockets, no GPU, no
 * ds4 — so the core planning algorithms unit-test on any host
 * (src/tools/plan_test.c, runs on macOS control machine and Linux alike).
 *
 * Multi-model: every model-shaped number (layer count, weight bytes, KV
 * overhead) comes from the idletoken_model_spec passed in — the planner itself
 * knows no model (docs/multi-model-design.md §3.3).
 */
#ifndef IDLETOKEN_PLAN_H
#define IDLETOKEN_PLAN_H

#include <stdint.h>
#include <stddef.h>

#include "idletoken_model.h"

typedef enum {
    IDLETOKEN_MODE_REFUSE   = 0,
    IDLETOKEN_MODE_GPU_ONLY = 1,   /* value matches the ASSIGN_PLAN mode byte */
    IDLETOKEN_MODE_HYBRID   = 2,
} idletoken_mode;

typedef struct {
    uint64_t vram_usable;
    uint64_t ram_usable;
    /* Measured ceiling on pinned host memory; 0 = unknown/unconstrained.
     * HYBRID spills into pinned memory, so this — not ram_usable — bounds the
     * host side of a node's shard. See idletoken_resource.h for why it must be
     * measured rather than derived. */
    uint64_t ram_pinnable;
    uint8_t  unified;   /* 1 = unified memory host (vram aliases ram) */
} idletoken_node_mem;

/* needed = model layer weights + per-node duplicated shared weights (every
 * stage loads embd/head) + per-node inference overhead (KV + compressed
 * state + activations + CUDA workspace + comms — idletoken_model_overhead()
 * with an average layers-per-node estimate).
 * ⚠ Overheads are ESTIMATES pending real-machine calibration — Spark's
 * unified memory can't produce valid numbers. */
uint64_t idletoken_needed_bytes(const idletoken_model_spec *model,
                             uint32_t ctx_size, int n_nodes);

/* Same, for an explicitly chosen precision. Small models ship several quants
 * whose weight bytes differ by 3-4x, so "can I run this?" is only answerable
 * per (model, quant). `quant` NULL/unknown falls back to the model default,
 * making this a strict superset of idletoken_needed_bytes. */
uint64_t idletoken_needed_bytes_quant(const idletoken_model_spec *model,
                                   const char *quant,
                                   uint32_t ctx_size, int n_nodes);

/* Mode decision (docs/architecture.md §5):
 *   Σ usable_vram              >= needed  → GPU_ONLY
 *   Σ (usable_vram+usable_ram) >= needed  → HYBRID, but every node must keep
 *                                           ≥4 GiB usable VRAM (CUDA context
 *                                           + workspace + ≥1 GPU layer)
 *   else                                  → REFUSE
 * Unified-memory nodes contribute max(vram, ram) exactly once — the pool is
 * the same physical memory, summing both would double-count.
 * `why` (optional) receives a short human-readable reason. */
/* Quant-aware mode decision + the shortfall when the answer is REFUSE.
 * `idletoken_mode_decide` below is this with quant=NULL and no shortfall — one
 * implementation, so the advisor's verdict can never drift from the planner's
 * (that drift is the easiest way for a capability table to lie). */
idletoken_mode idletoken_mode_decide_quant(const idletoken_model_spec *model,
                                     const char *quant,
                                     const idletoken_node_mem *nodes, int n,
                                     uint32_t ctx_size,
                                     uint64_t *shortfall_out,
                                     char *why, size_t whylen);

idletoken_mode idletoken_mode_decide(const idletoken_model_spec *model,
                               const idletoken_node_mem *nodes, int n,
                               uint32_t ctx_size, char *why, size_t whylen);

/* Resource-proportional contiguous split of model->n_layers across n nodes
 * (callers pass nodes sorted strongest-first). Every node gets ≥1 layer;
 * remainders land on the strongest nodes. out_counts[i] = node i's layer
 * count, Σ == n_layers. Returns 0, or -1 when n <= 0 or n > n_layers.
 * `mode` sets the sizing weight: GPU_ONLY splits by usable VRAM (all layers in
 * fast device memory); HYBRID splits by total usable capacity (VRAM+RAM) so a
 * node holds layers up to its combined memory and offloads the overflow.
 * `ctx_size` feeds the per-node capacity cap: after the proportional pass the
 * split is repaired so no node is assigned more layers than
 * (its usable memory − shared weights − per-tier overhead) can hold — the
 * Per-node capacity-cap bin-packing repair. Excess moves to nodes with headroom,
 * strongest first. When even the caps can't fit n_layers (coarse estimates,
 * borderline clusters) the repair degrades gracefully back toward the
 * proportional split instead of failing — mode_decide already admitted the
 * cluster by total capacity, so refusing here on estimates would brick it.
 * When model->split_boundary_multiple > 1 a final pass nudges stage
 * boundaries onto multiples of it where capacity allows (best-effort — e.g.
 * GLM-5.2's 4-layer shared-indexer groups); it never violates caps or the
 * ≥1-layer floor. */
int idletoken_plan_layers(const idletoken_model_spec *model,
                       const idletoken_node_mem *nodes, int n,
                       uint32_t ctx_size, int *out_counts, idletoken_mode mode);

/* ===== llama.cpp-engine scheduling (v2 rebuild WS-B2) ======================
 *
 * The v2 mandate (docs/v2-rebuild-plan-2026-08.md §4 WS-B2, §5 invariants
 * 1/5): given a model's REAL byte size (from the GGUF header, WS-B4) and the
 * probed node memories, decide single-machine vs cluster, produce per-node
 * --tensor-split ratios, pin layer 0 to the coordinator, and refuse with a
 * sentence a human can act on. Pure functions — unit-tested in plan_test.c.
 *
 * Usable memory per node (the formula; calibration constants are estimates):
 *
 *   unified == 0 (discrete GPU): usable = vram_usable + ram_usable.
 *       llama.cpp keeps -ngl layers in VRAM and the remainder in host RAM;
 *       both pools genuinely hold weights, so both count. The probe already
 *       subtracted other processes + safety margins (idletoken_resource.h).
 *
 *   unified == 1 (Apple Silicon / Grace): usable = max(vram_usable,
 *       ram_usable), counted ONCE — one physical pool, summing would
 *       double-count. On macOS the probe reports vram_usable =
 *       min(Metal recommendedMaxWorkingSetSize, ram_usable) − workspace
 *       reserve (src/common/resource.c, Darwin probe_gpu): the working-set
 *       recommendation IS Apple's "total minus OS reserve" number (measured
 *       ~74% of physical on the 16 GiB M4). Exact per-machine calibration of
 *       the workspace constant is TODO (v2 plan §1.3) — the structure is here,
 *       the constant is a reasoned reuse of the CUDA workspace reserve.
 */

#define IDLETOKEN_LLPLAN_MAX_NODES 16

typedef enum {
    IDLETOKEN_LLPLAN_SINGLE  = 0,  /* run on one machine, no RPC */
    IDLETOKEN_LLPLAN_CLUSTER = 1,  /* span nodes[] via ggml-RPC + tensor-split */
    IDLETOKEN_LLPLAN_REFUSE  = 2,  /* cannot run; `why` says what is missing */
} idletoken_llplan_kind;

/* What the scheduler needs to know about a model — all three numbers come
 * from idletoken_model_from_gguf() (WS-B4) for open models. */
typedef struct {
    uint64_t total_bytes;        /* whole GGUF on disk (llama.cpp keeps it all) */
    uint32_t n_layers;           /* transformer blocks */
    uint64_t kv_bytes_per_token; /* whole-model KV bytes per context token;
                                  * 0 = unknown (charged as unknown, not free) */
} idletoken_llm_model_size;

typedef struct {
    idletoken_llplan_kind kind;

    /* SINGLE: the chosen node (index into the caller's nodes[]). */
    int single_node;

    /* CLUSTER: participating nodes in tensor-split order. order[0] is ALWAYS
     * the coordinator — llama.cpp assigns the first split slice (layer 0 and
     * the embedding lookup with it) to the first device, which is how hard
     * invariant #1 (layer 0 + embedding stay on the coordinator) is enforced
     * by construction rather than by hope. */
    int    n_nodes;
    int    order[IDLETOKEN_LLPLAN_MAX_NODES];        /* indices into nodes[] */
    double tensor_split[IDLETOKEN_LLPLAN_MAX_NODES]; /* proportional, Σ = 1.0 */
    int    layer0_node;   /* == order[0] == coordinator (CLUSTER); == single_node (SINGLE) */

    char why[512];        /* human-readable decision / refusal reason */
} idletoken_llama_plan;

/* Decide how to run `model` on `nodes` (n of them, coordinator = index of the
 * node this coordinator process runs on).
 *
 *   - Fits the coordinator → SINGLE (hard invariant #5: fits → don't
 *     cluster). The coordinator is the only machine that can legally run
 *     single: layer 0 + embedding may not move to a worker, so a strong
 *     worker never substitutes. `allow_small_cluster` (the acceptance
 *     harness's IDLETOKEN_ALLOW_SMALL_CLUSTER=1) forces CLUSTER even then —
 *     test vehicle only; callers plumb the env var, this function reads none.
 *   - Needs several nodes → CLUSTER with tensor_split; coordinator first and
 *     holding at least one layer's worth. A coordinator with no usable local
 *     compute memory → REFUSE (layer 0 + embedding may not leave it).
 *   - Total memory insufficient → REFUSE; `why` names the need, the have and
 *     the shortfall in GiB plus what to do about it.
 *
 * Returns 0 (out fully filled, including REFUSE) or -1 on invalid arguments. */
int idletoken_plan_llamacpp(const idletoken_llm_model_size *model,
                            const idletoken_node_mem *nodes, int n,
                            int coordinator, uint32_t ctx_size,
                            int allow_small_cluster,
                            idletoken_llama_plan *out);

/* Largest context that fits `usable` bytes next to the weights + fixed
 * overhead, capped at ctx_want. Returns the granted context (a multiple of
 * 1024, ≥ ctx_floor), or 0 when even ctx_floor does not fit — the caller must
 * then refuse loudly, never silently serve a smaller window (a Claude Code
 * session needs ~13K of input before the first reply; a quietly shrunken
 * context fails mid-conversation instead of at startup, which is worse).
 * kv_bytes_per_token == 0 (unknown KV shape) grants ctx_want unchanged:
 * inventing a KV cost would refuse machines that are actually fine, and the
 * engine itself still fails loudly if it cannot allocate. */
uint32_t idletoken_llama_fit_ctx(uint64_t usable_bytes,
                                 const idletoken_llm_model_size *model,
                                 uint32_t ctx_want, uint32_t ctx_floor);

/* The usable-memory formula documented above, exposed so the coordinator's
 * preflight and the planner can never drift apart. */
uint64_t idletoken_llama_node_usable(const idletoken_node_mem *node);

#endif /* IDLETOKEN_PLAN_H */
