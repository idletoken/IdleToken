/* plan.c — cluster mode decision + PP layer split (pure, unit-testable).
 * See include/idletoken_plan.h for contracts and docs/architecture.md §5.
 * Model-shaped numbers all come from the idletoken_model_spec (multi-model
 * design §3.3) — this file must stay free of per-model constants. */
#include "idletoken_plan.h"

#include <stdio.h>
#include <string.h>

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

/* ===== llama.cpp-engine scheduling (v2 rebuild WS-B2) ======================
 * Contracts + the usable-memory formula: include/idletoken_plan.h. */

/* Per-node engine overhead beyond weights + KV. CALIBRATED 2026-08-15 on all
 * three backends (Qwen3.5-0.8B Q4_K_M, ctx 4096, llama-server -lv 5 buffer
 * report + nvidia-smi / RSS; results/resource-calibration-20260815.md):
 *   - CUDA context on a discrete card:      ~550 MiB (win_PC RTX 5060 Ti:
 *     1171 GPU-MiB used − 497 weights − 122 engine buffers)
 *   - engine buffers (RS+compute+output):   ~133 MiB on every backend at 0.8B,
 *     and they scale with model width — so a fixed constant alone is wrong in
 *     both directions (overcharges a phone-sized model, undercharges DSv4).
 *   - Metal has no CUDA-context analogue (~0): the fixed term is conservative
 *     there, which is the safe direction on a 16 GiB unified Mac.
 * Formula: 768 MiB fixed (CUDA context + small-model buffers, ~15% margin
 * over the worst measurement) + weights/64 (~1.6%, covers the width-scaled
 * compute buffers; 80 GiB DSv4 → ~2 GiB/node total).
 * Deliberately shared between the fit check and the ctx sizing so the two
 * cannot disagree about what "fits" means. */
#define IDLETOKEN_LLAMA_NODE_OVERHEAD_FIXED (768ull * 1024 * 1024)
#define idletoken_llama_node_overhead(model_bytes) \
    (IDLETOKEN_LLAMA_NODE_OVERHEAD_FIXED + (model_bytes) / 64)

uint64_t idletoken_llama_node_usable(const idletoken_node_mem *node) {
    if (!node) return 0;
    if (node->unified)   /* one physical pool — count it once */
        return node->vram_usable > node->ram_usable ? node->vram_usable
                                                    : node->ram_usable;
    return node->vram_usable + node->ram_usable;
}

/* Bytes this model needs on `n_nodes` machines at `ctx_size`:
 * weights (whole file — llama.cpp keeps everything resident) + KV cache for
 * the requested context + fixed per-node engine overhead. */
static uint64_t llplan_needed(const idletoken_llm_model_size *model,
                              uint32_t ctx_size, int n_nodes) {
    if (n_nodes < 1) n_nodes = 1;
    return model->total_bytes +
           model->kv_bytes_per_token * (uint64_t)ctx_size +
           (uint64_t)n_nodes * idletoken_llama_node_overhead(model->total_bytes);
}

uint32_t idletoken_llama_fit_ctx(uint64_t usable_bytes,
                                 const idletoken_llm_model_size *model,
                                 uint32_t ctx_want, uint32_t ctx_floor) {
    if (!model || ctx_want == 0) return 0;
    if (model->kv_bytes_per_token == 0) return ctx_want;  /* unknown ≠ free; the
                                                           * engine still fails
                                                           * loudly if wrong */
    const uint64_t fixed = model->total_bytes +
                           idletoken_llama_node_overhead(model->total_bytes);
    if (usable_bytes <= fixed) return 0;
    uint64_t tokens = (usable_bytes - fixed) / model->kv_bytes_per_token;
    if (tokens > ctx_want) tokens = ctx_want;
    tokens -= tokens % 1024;             /* engine-friendly granularity */
    if (tokens < ctx_floor) return 0;
    return (uint32_t)tokens;
}

int idletoken_plan_llamacpp(const idletoken_llm_model_size *model,
                            const idletoken_node_mem *nodes, int n,
                            int coordinator, uint32_t ctx_size,
                            int allow_small_cluster,
                            idletoken_llama_plan *out) {
    if (!model || !nodes || !out || n <= 0 || n > IDLETOKEN_LLPLAN_MAX_NODES ||
        coordinator < 0 || coordinator >= n || model->total_bytes == 0)
        return -1;
    memset(out, 0, sizeof(*out));
    out->single_node = -1;
    out->layer0_node = -1;

    const uint64_t need1 = llplan_needed(model, ctx_size, 1);
    const uint64_t coord_usable = idletoken_llama_node_usable(&nodes[coordinator]);

    /* ---- hard invariant #1 first: layer 0 + the embedding lookup stay with
     * the coordinator. That rules out BOTH "cluster around a memoryless
     * coordinator" and "run single on a strong worker instead" — either way
     * the raw prompt would leave the machine that decrypted it, and a remote
     * layer 0 lets any worker recover it from the public GGUF. */
    if (coord_usable == 0) {
        out->kind = IDLETOKEN_LLPLAN_REFUSE;
        snprintf(out->why, sizeof(out->why),
                 "refuse: the coordinator machine has no usable compute memory, "
                 "but layer 0 and the embedding table must stay on the "
                 "coordinator (privacy invariant — a remote layer 0 lets any "
                 "worker recover the prompt from the public GGUF). Run the "
                 "coordinator on a machine with a supported GPU, or free "
                 "memory on this one.");
        return 0;
    }

    /* ---- hard invariant #5: fits the coordinator alone → don't cluster.
     * "The best single node" is the coordinator by construction: a lone
     * worker could hold the bytes, but layer 0 may not move there (above),
     * so the only machine that can legally run single IS the coordinator. */
    if (coord_usable >= need1 && !(allow_small_cluster && n > 1)) {
        out->kind = IDLETOKEN_LLPLAN_SINGLE;
        out->single_node = coordinator;
        out->layer0_node = coordinator;
        snprintf(out->why, sizeof(out->why),
                 "SINGLE: %.2f GiB needed (weights %.2f + KV@%u + overhead) fits "
                 "the coordinator's %.2f GiB usable — clustering would only add "
                 "round-trip cost",
                 (double)need1 / (double)GiB,
                 (double)model->total_bytes / (double)GiB, ctx_size,
                 (double)coord_usable / (double)GiB);
        return 0;
    }

    /* ---- cluster path ---------------------------------------------------- */
    if (n == 1) {
        out->kind = IDLETOKEN_LLPLAN_REFUSE;
        snprintf(out->why, sizeof(out->why),
                 "refuse: this model needs %.2f GiB at ctx %u (weights %.2f GiB "
                 "+ KV cache + overhead) but this machine has %.2f GiB usable — "
                 "%.2f GiB short. Free memory, pick a smaller quantization, "
                 "lower the context size, or add machines to form a cluster.",
                 (double)need1 / (double)GiB, ctx_size,
                 (double)model->total_bytes / (double)GiB,
                 (double)coord_usable / (double)GiB,
                 (double)(need1 - coord_usable) / (double)GiB);
        return 0;
    }

    uint64_t total_usable = 0;
    for (int i = 0; i < n; i++) total_usable += idletoken_llama_node_usable(&nodes[i]);
    const uint64_t need_n = llplan_needed(model, ctx_size, n);
    if (total_usable < need_n) {
        out->kind = IDLETOKEN_LLPLAN_REFUSE;
        snprintf(out->why, sizeof(out->why),
                 "refuse: this model needs %.2f GiB at ctx %u (weights %.2f GiB "
                 "+ KV cache + per-node overhead) but the %d machine(s) have "
                 "%.2f GiB usable in total — %.2f GiB short. Add machines, "
                 "free memory, pick a smaller quantization, or lower the "
                 "context size.",
                 (double)need_n / (double)GiB, ctx_size,
                 (double)model->total_bytes / (double)GiB,
                 n, (double)total_usable / (double)GiB,
                 (double)(need_n - total_usable) / (double)GiB);
        return 0;
    }

    /* Participation order: coordinator first (see the header for why that IS
     * the layer-0 pin), then the rest strongest-first. */
    out->kind = IDLETOKEN_LLPLAN_CLUSTER;
    out->n_nodes = n;
    out->order[0] = coordinator;
    int k = 1;
    for (int i = 0; i < n; i++) if (i != coordinator) out->order[k++] = i;
    for (int i = 1; i < n - 1; i++)          /* insertion sort, strongest first */
        for (int j = i + 1; j < n; j++)
            if (idletoken_llama_node_usable(&nodes[out->order[j]]) >
                idletoken_llama_node_usable(&nodes[out->order[i]])) {
                int t = out->order[i]; out->order[i] = out->order[j]; out->order[j] = t;
            }
    out->layer0_node = coordinator;

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        out->tensor_split[i] =
            (double)idletoken_llama_node_usable(&nodes[out->order[i]]);
        sum += out->tensor_split[i];
    }
    for (int i = 0; i < n; i++) out->tensor_split[i] /= sum;

    /* The coordinator's slice must cover layer 0 in practice, not just in
     * intent: bump it to at least one layer's share and renormalize. */
    if (model->n_layers > 0) {
        const double min_frac = 1.0 / (double)model->n_layers;
        if (out->tensor_split[0] < min_frac) {
            const double deficit = min_frac - out->tensor_split[0];
            const double others = 1.0 - out->tensor_split[0];
            for (int i = 1; i < n; i++)
                out->tensor_split[i] -= deficit * (out->tensor_split[i] / others);
            out->tensor_split[0] = min_frac;
        }
    }

    if (coord_usable >= need1) {
        /* Only reachable through the small-cluster override: the model fits
         * the coordinator, so say that — the generic wording below would
         * claim need > usable with the numbers contradicting it. */
        snprintf(out->why, sizeof(out->why),
                 "CLUSTER (forced): %.2f GiB needed FITS the coordinator's "
                 "%.2f GiB usable — clustering anyway because "
                 "IDLETOKEN_ALLOW_SMALL_CLUSTER=1 (acceptance override); "
                 "splitting across %d nodes (%.2f GiB total), layer 0 pinned "
                 "to the coordinator (node %d).",
                 (double)need1 / (double)GiB, (double)coord_usable / (double)GiB,
                 n, (double)total_usable / (double)GiB, coordinator);
    } else {
        snprintf(out->why, sizeof(out->why),
                 "CLUSTER: %.2f GiB needed exceeds the coordinator's %.2f GiB "
                 "usable; splitting across %d nodes (%.2f GiB total), layer 0 "
                 "pinned to the coordinator (node %d).",
                 (double)need1 / (double)GiB, (double)coord_usable / (double)GiB,
                 n, (double)total_usable / (double)GiB, coordinator);
    }
    return 0;
}
