/* plan.c — cluster mode decision + PP layer split (pure, unit-testable).
 * See include/idletoken_plan.h for contracts and docs/architecture.md §5.
 * Model-shaped numbers all come from the idletoken_model_spec (multi-model
 * design §3.3) — this file must stay free of per-model constants. */
#include "idletoken_plan.h"

#include <stdio.h>

#define GiB (1024ull * 1024 * 1024)

/* HYBRID floor: CUDA context + workspace + at least one GPU layer. */
#define IDLETOKEN_HYBRID_MIN_VRAM      (4ull * GiB)

/* Average layers per node — the overhead estimate used before the actual
 * split exists (needed_bytes, capacity caps). Ceil so we err conservative. */
static int avg_layers_per_node(const idletoken_model_spec *m, int n_nodes) {
    if (n_nodes < 1) n_nodes = 1;
    return (int)(((int)m->n_layers + n_nodes - 1) / n_nodes);
}

uint64_t idletoken_needed_bytes_quant(const idletoken_model_spec *model,
                                   const char *quant,
                                   uint32_t ctx_size, int n_nodes) {
    if (n_nodes < 1) n_nodes = 1;
    uint64_t layer_bytes = model->layer_weight_bytes;
    uint64_t shared_bytes = model->shared_weight_bytes;
    idletoken_model_weight_bytes(model, quant, &layer_bytes, &shared_bytes);
    const uint64_t per_node = shared_bytes +
        idletoken_model_overhead(model, ctx_size, avg_layers_per_node(model, n_nodes));
    return layer_bytes + (uint64_t)n_nodes * per_node;
}

uint64_t idletoken_needed_bytes(const idletoken_model_spec *model,
                             uint32_t ctx_size, int n_nodes) {
    /* The model's own default precision — one code path, no drift. */
    return idletoken_needed_bytes_quant(model, NULL, ctx_size, n_nodes);
}

idletoken_mode idletoken_mode_decide_quant(const idletoken_model_spec *model,
                                     const char *quant,
                                     const idletoken_node_mem *nodes, int n,
                                     uint32_t ctx_size,
                                     uint64_t *shortfall_out,
                                     char *why, size_t whylen) {
    if (!nodes || n <= 0) {
        if (why) snprintf(why, whylen, "no nodes");
        return IDLETOKEN_MODE_REFUSE;
    }
    if (!model->available || model->n_layers == 0) {
        if (why) snprintf(why, whylen,
                          "refuse: model '%s' is not runnable in this build",
                          model->id);
        return IDLETOKEN_MODE_REFUSE;
    }

    const uint64_t needed = idletoken_needed_bytes_quant(model, quant, ctx_size, n);
    if (shortfall_out) *shortfall_out = 0;
    uint64_t sum_vram = 0, sum_all = 0;
    uint64_t min_vram = UINT64_MAX;
    for (int i = 0; i < n; i++) {
        const uint64_t v = nodes[i].vram_usable;
        const uint64_t r = nodes[i].ram_usable;
        sum_vram += v;
        /* Unified hosts: one physical pool — count it once, not twice. */
        sum_all  += nodes[i].unified ? (v > r ? v : r) : (v + r);
        if (v < min_vram) min_vram = v;
    }

    if (sum_vram >= needed) {
        if (why) snprintf(why, whylen,
                          "GPU_ONLY: %.1f GiB VRAM >= %.1f GiB needed (%s)",
                          (double)sum_vram / (double)GiB,
                          (double)needed   / (double)GiB, model->id);
        return IDLETOKEN_MODE_GPU_ONLY;
    }
    if (sum_all >= needed) {
        if (min_vram < IDLETOKEN_HYBRID_MIN_VRAM) {
            if (why) snprintf(why, whylen,
                              "refuse: HYBRID possible by memory (%.1f GiB) but a "
                              "node has only %.1f GiB usable VRAM (< 4 GiB floor)",
                              (double)sum_all  / (double)GiB,
                              (double)min_vram / (double)GiB);
            return IDLETOKEN_MODE_REFUSE;
        }
        if (why) snprintf(why, whylen,
                          "HYBRID: VRAM %.1f GiB short of %.1f GiB, VRAM+RAM "
                          "%.1f GiB covers it (%s)",
                          (double)sum_vram / (double)GiB,
                          (double)needed   / (double)GiB,
                          (double)sum_all  / (double)GiB, model->id);
        return IDLETOKEN_MODE_HYBRID;
    }
    if (why) snprintf(why, whylen,
                      "refuse: %s needs %.1f GiB for ctx=%u on %d node(s); have "
                      "%.1f GiB VRAM / %.1f GiB VRAM+RAM. Add nodes, free "
                      "memory, or lower the context tier.",
                      model->id,
                      (double)needed   / (double)GiB, ctx_size, n,
                      (double)sum_vram / (double)GiB,
                      (double)sum_all  / (double)GiB);
    /* How much more memory would make it fit — the number the capability
     * advisor turns into "37 GB short". Reported against VRAM+RAM because that
     * is the boundary between REFUSE and HYBRID. */
    if (shortfall_out) *shortfall_out = needed - sum_all;
    return IDLETOKEN_MODE_REFUSE;
}

idletoken_mode idletoken_mode_decide(const idletoken_model_spec *model,
                               const idletoken_node_mem *nodes, int n,
                               uint32_t ctx_size, char *why, size_t whylen) {
    return idletoken_mode_decide_quant(model, NULL, nodes, n, ctx_size, NULL,
                                    why, whylen);
}

/* Per-node weight used to size its layer share.
 * - GPU_ONLY: usable VRAM only — keep every layer in fast device memory.
 * - HYBRID:   total usable capacity (VRAM + RAM for a discrete card; the single
 *             unified pool for a unified host) — a node with modest VRAM but
 *             ample RAM can then hold MORE layers, spilling the overflow to RAM
 *             (the worker auto-offloads via cudaHostAlloc). This is what makes
 *             the cluster use every machine's RAM, not just its VRAM. */
static uint64_t node_split_weight(const idletoken_node_mem *m, idletoken_mode mode) {
    uint64_t w;
    if (mode == IDLETOKEN_MODE_HYBRID) {
        /* The host-side share is bounded by the **pinned-memory ceiling**, not
         * by ram_usable: every spilled layer goes through cudaHostAlloc, and
         * that ceiling sits well below physical RAM and cannot be derived
         * (measured 73.0% vs 48.2% of physical RAM on the same GPU model, each
         * reproducible across runs). Exceeding it shows up mid-load as
         * `cudaHostAlloc failed: out of memory`, a failure the capacity model
         * did not account for at all. 0 = not measured, so fall back to the old
         * behaviour (no constraint) rather than shrinking a node because a
         * probe failed. Unified-memory hosts skip this path entirely: they have
         * no separate VRAM pool to spill from. */
        uint64_t host = m->ram_usable;
        if (!m->unified && m->ram_pinnable > 0 && host > m->ram_pinnable)
            host = m->ram_pinnable;
        w = m->unified ? (m->vram_usable > m->ram_usable ? m->vram_usable : m->ram_usable)
                       : (m->vram_usable + host);
    } else {
        w = m->vram_usable > 0 ? m->vram_usable : m->ram_usable;
    }
    return w == 0 ? 1 : w;   /* probe-failure guard: avoid div by zero */
}

/* Max layers node i can actually hold in `mode`:
 * (mode-appropriate usable memory − shared weights − per-node overhead) /
 * per-layer bytes. Floor of 1: mode_decide admitted the node, and a PP stage
 * needs ≥1 layer — the estimates are too coarse to evict a member here. */
static int node_layer_cap(const idletoken_model_spec *model,
                          const idletoken_node_mem *m, idletoken_mode mode,
                          uint32_t ctx_size, int avg_layers,
                          uint64_t per_layer_bytes) {
    const uint64_t budget   = node_split_weight(m, mode);
    const uint64_t reserved = model->shared_weight_bytes +
                              idletoken_model_overhead(model, ctx_size, avg_layers);
    if (budget <= reserved + per_layer_bytes) return 1;
    int cap = (int)((budget - reserved) / per_layer_bytes);
    return cap < 1 ? 1 : cap;
}

int idletoken_plan_layers(const idletoken_model_spec *model,
                       const idletoken_node_mem *nodes, int n,
                       uint32_t ctx_size, int *out_counts, idletoken_mode mode) {
    const int n_layers = (int)model->n_layers;
    if (!nodes || !out_counts || n <= 0 || n > n_layers) return -1;

    uint64_t total_weight = 0;
    for (int i = 0; i < n; i++) total_weight += node_split_weight(&nodes[i], mode);

    /* Floor allocation with a 1-layer minimum per node. */
    int total_assigned = 0;
    for (int i = 0; i < n; i++) {
        uint64_t w = node_split_weight(&nodes[i], mode);
        int share = (int)((uint64_t)n_layers * w / total_weight);
        if (share < 1) share = 1;
        out_counts[i] = share;
        total_assigned += share;
    }
    /* Hand remainders to the strongest (front) nodes. */
    while (total_assigned < n_layers) {
        for (int i = 0; i < n && total_assigned < n_layers; i++) {
            out_counts[i]++;
            total_assigned++;
        }
    }
    /* Claw back overshoot (from the ≥1 floor) from the weakest nodes. */
    while (total_assigned > n_layers) {
        int took = 0;
        for (int i = n - 1; i >= 0; i--) {
            if (out_counts[i] > 1 && total_assigned > n_layers) {
                out_counts[i]--;
                total_assigned--;
                took = 1;
            }
        }
        if (!took) break;   /* everyone at 1 — n == n_layers */
    }

    /* ---- capacity-cap repair (bin-packing against per-node ceilings) ----
     * Proportional shares size by RELATIVE weight; a node can still land above
     * what its memory absolutely holds (e.g. remainder handing, or a cluster
     * where every node is near its limit). Move the excess of over-cap nodes
     * to nodes with headroom, strongest first. If total cap < n_layers the
     * loop simply stops when nobody has headroom — graceful degradation to
     * the proportional split (see header contract). */
    int caps[64];
    if (n > (int)(sizeof(caps) / sizeof(caps[0]))) return 0; /* absurd n: skip repair */
    const uint64_t per_layer = model->layer_weight_bytes / (uint64_t)n_layers;
    const int avg = avg_layers_per_node(model, n);
    for (int i = 0; i < n; i++)
        caps[i] = node_layer_cap(model, &nodes[i], mode, ctx_size, avg, per_layer);

    for (int i = 0; i < n; i++) {
        while (out_counts[i] > caps[i]) {
            int moved = 0;
            for (int j = 0; j < n; j++) {
                if (j != i && out_counts[j] < caps[j]) {
                    out_counts[j]++;
                    out_counts[i]--;
                    moved = 1;
                    break;
                }
            }
            if (!moved) break;   /* no headroom anywhere — degrade gracefully */
        }
    }

    /* ---- boundary-multiple snap (align split boundaries, best-effort) ---
     * Some models prefer PP cuts on multiples of k (GLM-5.2: k=4, layers in a
     * group share a DSA indexer). Nudge each cumulative boundary to the
     * nearest multiple by shifting layers between the two adjacent stages,
     * only when both sides stay within [1, cap]. Estimates never get worse
     * than the proportional split by more than k-1 layers per stage. */
    const int k = (int)model->split_boundary_multiple;
    if (k > 1) {
        int cum = 0;
        for (int i = 0; i < n - 1; i++) {
            cum += out_counts[i];
            const int rem = cum % k;
            if (rem == 0) continue;
            const int up = k - rem;   /* grow stage i by `up` … */
            const int dn = rem;       /* … or shrink it by `dn` */
            if (up <= dn && out_counts[i] + up <= caps[i] &&
                out_counts[i + 1] - up >= 1) {
                out_counts[i] += up; out_counts[i + 1] -= up; cum += up;
            } else if (out_counts[i] - dn >= 1 &&
                       out_counts[i + 1] + dn <= caps[i + 1]) {
                out_counts[i] -= dn; out_counts[i + 1] += dn; cum -= dn;
            }
            /* neither direction legal — leave the ragged boundary */
        }
    }
    return 0;
}
