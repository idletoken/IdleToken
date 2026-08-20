/* plan.c — cluster mode decision + PP layer split (pure, unit-testable).
 * See include/idletoken_plan.h for contracts and docs/architecture.md §5.
 * Model-shaped numbers all come from the idletoken_model_spec (multi-model
 * design §3.3) — this file must stay free of per-model constants. */
#include "idletoken_plan.h"

#include <stdio.h>
#include <stdlib.h>   /* getenv — the over-subscription override */
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
    int min_vram_at = 0;               /* which node holds that minimum */
    for (int i = 0; i < n; i++) {
        const uint64_t v = nodes[i].vram_usable;
        const uint64_t r = nodes[i].ram_usable;
        sum_vram += v;
        /* Unified hosts: one physical pool — count it once, not twice. */
        sum_all  += nodes[i].unified ? (v > r ? v : r) : (v + r);
        if (v < min_vram) { min_vram = v; min_vram_at = i; }
    }
    /* Name the machine, or say which index it was. A refusal the user cannot
     * act on is only marginally better than no refusal at all. */
    char who[80];
    if (nodes[min_vram_at].label && nodes[min_vram_at].label[0])
        snprintf(who, sizeof who, "%s", nodes[min_vram_at].label);
    else
        snprintf(who, sizeof who, "node #%d", min_vram_at + 1);

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
                              "refuse: HYBRID possible by memory (%.1f GiB) but "
                              "%s reports only %.1f GiB usable VRAM (< 4 GiB floor). "
                              "Check that machine: close what is using its GPU, "
                              "raise its VRAM limit in Settings, or leave it out "
                              "of the cluster.",
                              (double)sum_all  / (double)GiB, who,
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
 *   - CUDA context on a discrete card:      ~550 MiB (RTX 5060 Ti:
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
 * cannot disagree about what "fits" means.
 *
 * RECHECKED 2026-08-19 at the top end, which 08-15 could only extrapolate
 * (results/resource-calibration-20260819.md; 5.29 / 15.59 / 80.76 GiB of
 * weights). Two results:
 *
 *  - The slope is SAFE where it was unmeasured: DeepSeek-V4-Flash at 80.76 GiB
 *    wants ~321 MiB of non-weight buffers against the ~2060 MiB budgeted here,
 *    a 6.4x margin. Nothing to raise. The constants are therefore unchanged.
 *
 *  - But the real driver is NOT model size, it is SLOT COUNT, and this formula
 *    has no slot term. The engine's per-sequence state buffer scales exactly
 *    with -np (measured 4.000x from -np 1 to -np 4 on both Qwen3.5 sizes), and
 *    on the hybrid-linear-attention Qwen3.5 family it dominates: ~9.65 MiB per
 *    GiB of weights PER SLOT, i.e. ~38.6 MiB/GiB at the cap of 4, against the
 *    16 MiB/GiB this formula charges. Crossover is around 29 GiB of weights.
 *    The largest curated Qwen3.5 (35b-a3b, 20.5 GiB) still fits with a 1.22x
 *    margin, so nothing shipped is mis-budgeted and nothing is changed here.
 *
 *    ⚠ If that margin is ever spent — a bigger hybrid-attention model, or a
 *    slot cap above 4 — this needs a slot term, not a bigger constant. Note
 *    that making it slot-aware closes a loop: idletoken_llama_seq_slots() calls
 *    this to decide the slot count. That is a design decision, not a
 *    recalibration; take it deliberately. */
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

uint64_t idletoken_llama_kv_pool(const idletoken_node_mem *node) {
    if (!node) return 0;
    /* Unified: one physical pool, so where the KV lives and what the model can
     * be laid out across are the same number. Discrete: the KV is allocated on
     * the device (-ngl 99), and host RAM does not back it. */
    if (node->unified) return idletoken_llama_node_usable(node);
    return node->vram_usable;
}

/* Bytes this model needs on `n_nodes` machines at `ctx_size`:
 * weights (whole file) + KV cache for the requested context + fixed per-node
 * engine overhead.
 *
 * ⚠ This is the FULL-SPEED requirement, not the feasibility bound — see
 * idletoken_llama_hard_need() for what actually has to be resident. It is
 * still the right number for "will this run WELL", which is what ctx sizing
 * and the single-node fast path ask. */
static uint64_t llplan_needed(const idletoken_llm_model_size *model,
                              uint32_t ctx_size, int n_nodes) {
    if (n_nodes < 1) n_nodes = 1;
    return model->total_bytes +
           model->kv_bytes_per_token * (uint64_t)ctx_size +
           (uint64_t)n_nodes * idletoken_llama_node_overhead(model->total_bytes);
}

uint64_t idletoken_llama_hard_need(const idletoken_llm_model_size *model,
                                   uint32_t ctx_size, int n_nodes) {
    if (!model) return 0;
    if (n_nodes < 1) n_nodes = 1;
    return model->kv_bytes_per_token * (uint64_t)ctx_size +
           (uint64_t)n_nodes * idletoken_llama_node_overhead(model->total_bytes);
}

uint64_t idletoken_llama_working_set(const idletoken_llm_model_size *model) {
    if (!model) return 0;
    /* Dense (or an MoE we cannot characterise): every byte is touched. Falling
     * back to the whole file is the SAFE direction — it can only make the
     * verdict more conservative, never promise a speed the machine cannot
     * deliver. */
    if (model->n_expert < 2 || model->n_expert_used == 0 ||
        model->n_expert_used >= model->n_expert)
        return model->total_bytes;
    /* MoE. The expert FFN dominates the file, but attention/norms/shared
     * experts are touched by EVERY token, and no field here separates them —
     * so charge a floor of 1/8 of the file for the always-resident part on top
     * of the consulted experts' share. Deliberately crude and deliberately
     * high: over-charging costs a "slow" label on a model that would have been
     * fast, while under-charging promises a speed the disk cannot deliver.
     * (GLM-5.2 256/8: 12.5% floor + 3.1% experts ≈ 35 GiB of a 222 GiB file;
     * measured working set from the 08-16 run is ~12 GiB, so this is ~3x
     * conservative. Tighten it when a GGUF-side split of expert vs shared
     * tensor bytes exists — see the doc's closed-loop item.) */
    const uint64_t experts = model->total_bytes / (uint64_t)model->n_expert *
                             (uint64_t)model->n_expert_used;
    return model->total_bytes / 8 + experts;
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

int idletoken_llama_seq_slots(const idletoken_node_mem *node,
                              const idletoken_llm_model_size *model,
                              uint32_t ctx_size, double layer_share, int cap) {
    if (cap <= 0) cap = IDLETOKEN_LLAMA_SLOT_CAP;
    if (!node || !model || ctx_size == 0 || model->kv_bytes_per_token == 0)
        return 1;
    if (!(layer_share > 0.0)) layer_share = 1.0;
    if (layer_share > 1.0)    layer_share = 1.0;

    /* The pool the KV will be allocated in — NOT the node's total capacity.
     * On a discrete card that is VRAM alone, because -ngl 99 puts the KV
     * there; see idletoken_llama_kv_pool for the machine this taught us. */
    const uint64_t pool = idletoken_llama_kv_pool(node);

    const uint64_t weights = (uint64_t)((double)model->total_bytes * layer_share);
    /* The overhead term is per-NODE and does not scale with the share: a CUDA
     * context costs the same on a machine holding three layers as on one
     * holding all of them (results/resource-calibration-20260815.md). */
    const uint64_t fixed = weights + idletoken_llama_node_overhead(model->total_bytes);
    /* Also the slow-tier rule: weights that do not fit the KV pool are weights
     * spilling to host RAM, and a machine already paying PCIe per token must
     * not also be asked to hold a second context in the memory it ran out of. */
    if (pool <= fixed) return 1;

    const uint64_t kv_per_seq =
        (uint64_t)((double)model->kv_bytes_per_token * layer_share) *
        (uint64_t)ctx_size;
    if (kv_per_seq == 0) return 1;

    uint64_t slots = (pool - fixed) / kv_per_seq;
    if (slots < 1) return 1;
    if (slots > (uint64_t)cap) slots = (uint64_t)cap;
    return (int)slots;
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

    /* ---- one machine, weights beyond its memory --------------------------
     * Not automatically a refusal any more (2026-08-16). The weights are
     * mmap'd, so what must be RESIDENT is only KV + engine overhead; the rest
     * streams from disk. This exact case was measured: GLM-5.2 (222 GiB) on a
     * 119 GiB DGX loaded in 3m11s and generated at 0.91 tok/s, while the old
     * rule here called it "119.35 GiB short". Same ceiling as the cluster
     * path — past 2x the cache it is honest to refuse. */
    if (n == 1) {
        const uint64_t hard1 = idletoken_llama_hard_need(model, ctx_size, 1);
        const uint64_t wset1 = idletoken_llama_working_set(model);
        out->hard_need_bytes   = hard1;
        out->working_set_bytes = wset1;
        out->working_set_fits  = (coord_usable >= hard1 + wset1);

        if (coord_usable < hard1) {
            out->kind = IDLETOKEN_LLPLAN_REFUSE;
            snprintf(out->why, sizeof(out->why),
                     "refuse: this model needs %.2f GiB resident at ctx %u (KV "
                     "cache + engine overhead; the %.2f GiB of weights stream "
                     "from disk and do not have to fit) but this machine has "
                     "%.2f GiB usable — %.2f GiB short. Free memory, lower the "
                     "context size, or add machines.",
                     (double)hard1 / (double)GiB, ctx_size,
                     (double)model->total_bytes / (double)GiB,
                     (double)coord_usable / (double)GiB,
                     (double)(hard1 - coord_usable) / (double)GiB);
            return 0;
        }
        const uint64_t cache1 = coord_usable - hard1;
        if (!out->working_set_fits && (cache1 == 0 || wset1 > 2 * cache1) &&
            !getenv("IDLETOKEN_ALLOW_SLOW_OVERSUBSCRIBE")) {
            out->kind = IDLETOKEN_LLPLAN_REFUSE;
            snprintf(out->why, sizeof(out->why),
                     "refuse: it would start, but unusably slowly. This machine "
                     "has %.2f GiB usable and %.2f GiB of that can cache weights, "
                     "against a %.2f GiB working set at ctx %u — %.1fx over, so "
                     "every token would re-read most of the model from disk. Pick "
                     "a smaller quantization or add machines. Set "
                     "IDLETOKEN_ALLOW_SLOW_OVERSUBSCRIBE=1 to run it anyway.",
                     (double)coord_usable / (double)GiB,
                     (double)cache1 / (double)GiB,
                     (double)wset1 / (double)GiB, ctx_size,
                     cache1 ? (double)wset1 / (double)cache1 : 0.0);
            return 0;
        }
        /* Runs on this one machine, streaming part of the weights. */
        out->kind = IDLETOKEN_LLPLAN_SINGLE;
        out->single_node = coordinator;
        out->layer0_node = coordinator;
        snprintf(out->why, sizeof(out->why),
                 "SINGLE (slow): %.2f GiB resident fits this machine's %.2f GiB "
                 "usable, but the %.2f GiB working set does not — weights stream "
                 "from disk each token. Measured ~1 token/s at this ratio; a "
                 "smaller quantization or another machine removes it.",
                 (double)hard1 / (double)GiB, (double)coord_usable / (double)GiB,
                 (double)wset1 / (double)GiB);
        return 0;
    }

    uint64_t total_usable = 0;
    for (int i = 0; i < n; i++) total_usable += idletoken_llama_node_usable(&nodes[i]);
    const uint64_t need_n = llplan_needed(model, ctx_size, n);
    (void)need_n;   /* full-speed figure; the CLUSTER wording below quotes need1 */

    /* Feasibility is decided by what must be RESIDENT — KV + per-node engine
     * overhead — not by the weight bytes (2026-08-16). llama.cpp mmaps the
     * weights, so they are reclaimable page cache: a 222 GiB model served from
     * a 119 GiB machine at 0.91 tok/s is the measurement that retired the old
     * rule, which had refused the same model on THREE machines. */
    const uint64_t hard = idletoken_llama_hard_need(model, ctx_size, n);
    const uint64_t wset = idletoken_llama_working_set(model);
    out->hard_need_bytes   = hard;
    out->working_set_bytes = wset;
    /* Full speed needs the working set cached NEXT TO the resident part. */
    out->working_set_fits  = (total_usable >= hard + wset);

    if (total_usable < hard) {
        out->kind = IDLETOKEN_LLPLAN_REFUSE;
        snprintf(out->why, sizeof(out->why),
                 "refuse: this model needs %.2f GiB resident at ctx %u (KV cache "
                 "+ per-node engine overhead; the %.2f GiB of weights stream from "
                 "disk and do not have to fit) but the %d machine(s) have %.2f GiB "
                 "usable in total — %.2f GiB short. Add machines, free memory, or "
                 "lower the context size.",
                 (double)hard / (double)GiB, ctx_size,
                 (double)model->total_bytes / (double)GiB,
                 n, (double)total_usable / (double)GiB,
                 (double)(hard - total_usable) / (double)GiB);
        return 0;
    }

    /* Over-subscription ceiling. "It runs" stops being a useful promise well
     * before it stops being true: every token re-reads the part of the working
     * set that did not stay cached, so the rate collapses toward
     * disk_bandwidth / uncached_bytes.
     *
     * The bound is anchored to the ONE ratio actually measured (2026-08-16):
     * GLM-5.2, working set ~1.9x the machine, 0.91 tok/s — slow but usable.
     * Past 2x we would be extrapolating from a single point, and the wrong
     * direction to extrapolate is the one that tells a user "sure, go ahead"
     * before a 40 s/token experience. Beyond it we refuse and say why, with an
     * override for whoever wants to measure the next point. */
    const uint64_t cache_avail = total_usable - hard;
    if (!out->working_set_fits && cache_avail > 0 &&
        wset > 2 * cache_avail && !getenv("IDLETOKEN_ALLOW_SLOW_OVERSUBSCRIBE")) {
        out->kind = IDLETOKEN_LLPLAN_REFUSE;
        snprintf(out->why, sizeof(out->why),
                 "refuse: it would start, but unusably slowly. The %d machine(s) "
                 "have %.2f GiB usable and %.2f GiB of that can cache weights, "
                 "against a %.2f GiB working set at ctx %u — %.1fx over, so every "
                 "token would re-read most of the model from disk (measured: %.2f GiB "
                 "short of the cache it wants). Add machines, free memory, or pick a "
                 "smaller quantization. Set IDLETOKEN_ALLOW_SLOW_OVERSUBSCRIBE=1 to "
                 "run it anyway.",
                 n, (double)total_usable / (double)GiB,
                 (double)cache_avail / (double)GiB,
                 (double)wset / (double)GiB, ctx_size,
                 (double)wset / (double)cache_avail,
                 (double)(wset - cache_avail) / (double)GiB);
        return 0;
    }

    /* Participation order: coordinator first (see the header for why that IS
     * the layer-0 pin), then the rest strongest-first — "strongest" measured in
     * the pool the node can be handed layers in (below), not in what the
     * machine owns. */
    out->kind = IDLETOKEN_LLPLAN_CLUSTER;
    out->n_nodes = n;
    out->order[0] = coordinator;
    int k = 1;
    for (int i = 0; i < n; i++) if (i != coordinator) out->order[k++] = i;
    for (int i = 1; i < n - 1; i++)          /* insertion sort, strongest first */
        for (int j = i + 1; j < n; j++)
            if (idletoken_llama_kv_pool(&nodes[out->order[j]]) >
                idletoken_llama_kv_pool(&nodes[out->order[i]])) {
                int t = out->order[i]; out->order[i] = out->order[j]; out->order[j] = t;
            }
    out->layer0_node = coordinator;

    /* ---- the split is proportional to the pool a node can be HANDED LAYERS
     * IN, which is not what the machine owns (2026-08-20, T16) ---------------
     *
     * In cluster mode every layer is a real allocation on a compute device:
     * the coordinator's llama-server runs `-ngl 99`, and a worker's
     * rpc-server is started `-d CUDA0`. A discrete worker's system RAM
     * therefore cannot hold a single layer — and unlike the single-node path
     * there is no mmap to fall back on, because the far side receives its
     * slice over the wire into a device buffer rather than mapping the file.
     *
     * That is exactly the question idletoken_llama_kv_pool() answers — "what
     * can the engine backend on this node allocate" — so this reuses it
     * instead of adding a second function that could drift from it. The KV was
     * merely the first caller to discover the distinction (2026-08-18, the
     * WDDM freeze on a Windows desktop); the weight slice is the second.
     *
     * Measured (results/t14-engine-bump-phaseb-20260820.md): DSv4-Flash,
     * 80.76 GiB, across a unified-memory node (107.61 GiB) and a discrete-GPU
     * node (13.2 GiB VRAM + 37.3 GiB RAM). Budgeted against the machine, the
     * joiner's share came
     * out 50.48/158.09 = 0.3193 = 25.8 GiB onto a 13.2 GiB card; the driver
     * paged VRAM out to system memory and the rpc-server died mid-decode. The
     * same split was produced for a 0.5 GiB model and an 80.76 GiB one, which
     * is the tell: it never looked at the bytes it was handing out. */
    const uint64_t kv_total = model->kv_bytes_per_token * (uint64_t)ctx_size;
    const uint64_t slice_all = model->total_bytes + kv_total;  /* what gets split */
    const uint64_t per_node_oh = idletoken_llama_node_overhead(model->total_bytes);

    /* cap[i] = the largest layer fraction node i can actually hold.
     *
     * Discrete: (VRAM − per-node engine overhead) / (weights + KV). Unified:
     * 1.0, i.e. "the whole model, as far as THIS check is concerned" — whether
     * an over-subscribed unified pool is feasible is the 2026-08-16 page-cache
     * question, already decided by the hard-need and over-subscription checks
     * above, and this change deliberately does not re-open it. */
    double   cap[IDLETOKEN_LLPLAN_MAX_NODES];
    uint64_t pool_of[IDLETOKEN_LLPLAN_MAX_NODES];
    double   cap_total = 0.0;
    int      tightest = 0;
    for (int i = 0; i < n; i++) {
        const idletoken_node_mem *nd = &nodes[out->order[i]];
        pool_of[i] = idletoken_llama_kv_pool(nd);
        if (nd->unified) {
            cap[i] = 1.0;
        } else {
            const uint64_t room = pool_of[i] > per_node_oh
                                      ? pool_of[i] - per_node_oh : 0;
            cap[i] = slice_all ? (double)room / (double)slice_all : 1.0;
            if (cap[i] > 1.0) cap[i] = 1.0;
        }
        cap_total += cap[i];
        if (cap[i] < cap[tightest]) tightest = i;
    }

    const double min_frac = model->n_layers > 0
                                ? 1.0 / (double)model->n_layers : 0.0;

    /* Layer 0 may not leave the coordinator (hard invariant #1), so a
     * coordinator that cannot hold one layer's worth in its own pool is a
     * refusal, not a re-plan. */
    if (cap[0] < min_frac) {
        out->kind = IDLETOKEN_LLPLAN_REFUSE;
        snprintf(out->why, sizeof(out->why),
                 "refuse: the coordinator can only hold %.2f GiB in the memory "
                 "its GPU can address (%.2f GiB of video memory, minus %.2f GiB "
                 "of engine overhead), which is less than the %.2f GiB one layer "
                 "of this model needs at ctx %u — and layer 0 must stay on the "
                 "coordinator (privacy invariant). Free video memory on this "
                 "machine, pick a smaller quantization, or lower the context "
                 "size.",
                 cap[0] * (double)slice_all / (double)GiB,
                 (double)pool_of[0] / (double)GiB,
                 (double)per_node_oh / (double)GiB,
                 min_frac * (double)slice_all / (double)GiB, ctx_size);
        return 0;
    }

    /* Even handing every machine everything its engine can address, does the
     * model fit? Names the machine that runs out and WHICH memory it ran out
     * of — the useful sentence is "this node's card is too small", not "the
     * cluster is short". */
    if (cap_total < 1.0) {
        const idletoken_node_mem *t = &nodes[out->order[tightest]];
        out->kind = IDLETOKEN_LLPLAN_REFUSE;
        snprintf(out->why, sizeof(out->why),
                 "refuse: %.2f GiB of weights + KV at ctx %u must be split "
                 "across %d machines, but a machine can only be handed layers "
                 "in memory its GPU can address. Node %d has %.2f GiB of video "
                 "memory (its %.2f GiB of system RAM cannot hold layers: its "
                 "engine is started on the GPU device), and cluster-wide only "
                 "%.2f GiB is reachable — %.2f GiB short. Add a machine with "
                 "more video memory, pick a smaller quantization, or lower the "
                 "context size.",
                 (double)slice_all / (double)GiB, ctx_size, n,
                 out->order[tightest], (double)pool_of[tightest] / (double)GiB,
                 (double)t->ram_usable / (double)GiB,
                 cap_total * (double)slice_all / (double)GiB,
                 (1.0 - cap_total) * (double)slice_all / (double)GiB);
        return 0;
    }

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        out->tensor_split[i] = (double)pool_of[i];
        sum += out->tensor_split[i];
    }
    if (sum <= 0.0) {   /* cap_total >= 1.0 rules this out; belt and braces */
        out->kind = IDLETOKEN_LLPLAN_REFUSE;
        snprintf(out->why, sizeof(out->why),
                 "refuse: no machine in this cluster reports memory its engine "
                 "can allocate in. Check that each machine has a supported GPU "
                 "with free video memory.");
        return 0;
    }
    for (int i = 0; i < n; i++) out->tensor_split[i] /= sum;

    /* The coordinator's slice must cover layer 0 in practice, not just in
     * intent: bump it to at least one layer's share and renormalize. */
    if (min_frac > 0.0 && out->tensor_split[0] < min_frac) {
        const double deficit = min_frac - out->tensor_split[0];
        const double others = 1.0 - out->tensor_split[0];
        for (int i = 1; i < n; i++)
            out->tensor_split[i] -= deficit * (out->tensor_split[i] / others);
        out->tensor_split[0] = min_frac;
    }

    /* Water-filling: whatever a node cannot hold goes back to the nodes that
     * can, proportionally to the room they have left. Each round pins at least
     * one node at its cap and a pinned node never moves again, so n rounds is
     * enough; the loop bound is there so a rounding pathology cannot spin.
     * cap[0] >= min_frac (checked above) is what keeps the layer-0 floor
     * intact through this. */
    for (int round = 0; round < n; round++) {
        double excess = 0.0, room = 0.0;
        for (int i = 0; i < n; i++)
            if (out->tensor_split[i] > cap[i]) {
                excess += out->tensor_split[i] - cap[i];
                out->tensor_split[i] = cap[i];
            }
        if (excess <= 1e-12) break;
        for (int i = 0; i < n; i++) {
            const double r = cap[i] - out->tensor_split[i];
            if (r > 0.0) room += r;
        }
        if (room <= 1e-12) break;
        for (int i = 0; i < n; i++) {
            const double r = cap[i] - out->tensor_split[i];
            if (r > 0.0) out->tensor_split[i] += excess * (r / room);
        }
    }

    /* Fail closed if the layers did not all find a home. This is not a
     * formality: llama.cpp NORMALIZES --tensor-split itself, so a vector
     * summing to 0.9 does not under-load the cluster — it is rescaled, and the
     * node we just clamped is handed its old over-sized share again. A plan
     * that cannot be expressed as a ratio must not be shipped as one. */
    double placed = 0.0;
    for (int i = 0; i < n; i++) placed += out->tensor_split[i];
    if (placed < 0.999) {
        out->kind = IDLETOKEN_LLPLAN_REFUSE;
        snprintf(out->why, sizeof(out->why),
                 "refuse: only %.1f%% of this model's layers could be placed in "
                 "memory the machines' GPUs can address (node %d holds at most "
                 "%.2f GiB of the %.2f GiB total at ctx %u). Add a machine with "
                 "more video memory, pick a smaller quantization, or lower the "
                 "context size.",
                 placed * 100.0, out->order[tightest],
                 cap[tightest] * (double)slice_all / (double)GiB,
                 (double)slice_all / (double)GiB, ctx_size);
        return 0;
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
    } else if (out->working_set_fits) {
        snprintf(out->why, sizeof(out->why),
                 "CLUSTER: %.2f GiB needed exceeds the coordinator's %.2f GiB "
                 "usable; splitting across %d nodes (%.2f GiB total), layer 0 "
                 "pinned to the coordinator (node %d).",
                 (double)need1 / (double)GiB, (double)coord_usable / (double)GiB,
                 n, (double)total_usable / (double)GiB, coordinator);
    } else {
        /* Runs, but the weights do not all stay cached — pages stream from
         * disk. Said out loud with the numbers, because "it works" and "it
         * works at a usable speed" are different promises and the user is
         * about to wait on the difference. */
        snprintf(out->why, sizeof(out->why),
                 "CLUSTER (slow): splitting across %d nodes (%.2f GiB usable), "
                 "layer 0 pinned to the coordinator (node %d). %.2f GiB must stay "
                 "resident and does; but the %.2f GiB working set does not fit, so "
                 "weights stream from disk each token — expect roughly 1 token/s, "
                 "not interactive speed. A smaller quantization or another machine "
                 "removes this.",
                 n, (double)total_usable / (double)GiB, coordinator,
                 (double)out->hard_need_bytes / (double)GiB,
                 (double)out->working_set_bytes / (double)GiB);
    }
    return 0;
}
