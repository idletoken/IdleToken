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

#endif /* IDLETOKEN_PLAN_H */
