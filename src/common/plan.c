/* plan.c — cluster mode decision + PP layer split (pure, unit-testable).
 * See include/idletoken_plan.h for contracts and docs/architecture.md §5.
 * Model-shaped numbers all come from the idletoken_model_spec (multi-model
 * design §3.3) — this file must stay free of per-model constants. */
#include "idletoken_plan.h"
#include "idletoken_modelsize.h"

#include <stdio.h>
#include <string.h>

#define GiB (1024ull * 1024 * 1024)

uint32_t idletoken_llama_product_ctx_ceiling(
        const idletoken_model_spec *model) {
    if (!model) return 0;
    uint32_t ceiling = model->ctx_yarn_max > model->ctx_max
        ? model->ctx_yarn_max : model->ctx_max;
    if (ceiling > IDLETOKEN_PRODUCT_CONTEXT_CAP)
        ceiling = IDLETOKEN_PRODUCT_CONTEXT_CAP;
    return ceiling;
}

/* Average layers per node — the overhead estimate used before the actual
 * split exists (needed_bytes, capacity caps). Ceil so we err conservative. */
static int avg_layers_per_node(const idletoken_model_spec *m, int n_nodes) {
    if (n_nodes < 1) n_nodes = 1;
    return (int)(((int)m->n_layers + n_nodes - 1) / n_nodes);
}

uint64_t idletoken_needed_bytes_quant(const idletoken_model_spec *model,
                                   const char *quant,
                                   uint32_t ctx_size, int n_nodes,
                                   uint8_t backend) {
    if (n_nodes < 1) n_nodes = 1;
    idletoken_llm_model_size size;
    if (!model || idletoken_model_size_resolve(model, quant, NULL, &size,
                                                NULL, 0) != 0)
        return UINT64_MAX;
    /* The shipped automatic KV tier is a function of selected weight
     * precision. This is the same closed mapping coord_main applies before it
     * asks the llama.cpp planner; using f16 here for an IQ2 row was a second,
     * hidden source of capability-table inflation. >=5/unknown remains f16
     * first because downgrade then depends on the particular machine. */
    const int bits = idletoken_quant_weight_bits(quant);
    if (bits >= 1 && bits <= 2)
        idletoken_llama_model_kv_scale(&size, 18.0 / 64.0);
    else if (bits >= 3 && bits <= 4)
        idletoken_llama_model_kv_scale(&size, 34.0 / 64.0);
    return size.total_bytes +
           idletoken_llama_hard_need(&size, ctx_size, n_nodes, backend);
}

uint64_t idletoken_needed_bytes(const idletoken_model_spec *model,
                             uint32_t ctx_size, int n_nodes, uint8_t backend) {
    /* The model's own default precision — one code path, no drift. */
    return idletoken_needed_bytes_quant(model, NULL, ctx_size, n_nodes, backend);
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

    const uint64_t needed = idletoken_needed_bytes_quant(
        model, quant, ctx_size, n, idletoken_llama_roster_backend(nodes, n));
    if (shortfall_out) *shortfall_out = 0;
    uint64_t sum_vram = 0;
    for (int i = 0; i < n; i++) {
        sum_vram += nodes[i].vram_usable;
    }
    if (sum_vram >= needed) {
        if (why) snprintf(why, whylen,
                          "GPU_ONLY: %.1f GiB VRAM >= %.1f GiB needed (%s)",
                          (double)sum_vram / (double)GiB,
                          (double)needed   / (double)GiB, model->id);
        return IDLETOKEN_MODE_GPU_ONLY;
    }
    if (why) snprintf(why, whylen,
                      "[RESOURCE_INSUFFICIENT] %s needs %.1f GiB of GPU memory "
                      "for ctx=%u on %d node(s); only %.1f GiB is available. "
                      "Free GPU memory, choose a smaller quantization, or add "
                      "nodes.",
                      model->id,
                      (double)needed   / (double)GiB, ctx_size, n,
                      (double)sum_vram / (double)GiB);
    if (shortfall_out) *shortfall_out = needed - sum_vram;
    return IDLETOKEN_MODE_REFUSE;
}

idletoken_mode idletoken_mode_decide(const idletoken_model_spec *model,
                               const idletoken_node_mem *nodes, int n,
                               uint32_t ctx_size, char *why, size_t whylen) {
    return idletoken_mode_decide_quant(model, NULL, nodes, n, ctx_size, NULL,
                                    why, whylen);
}

/* Per-node layer weight. Serving is GPU-only. */
static uint64_t node_split_weight(const idletoken_node_mem *m, idletoken_mode mode) {
    (void)mode;
    uint64_t w = m->vram_usable;
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

/* Per-node engine overhead beyond weights + KV + workspace.
 *
 * RETIRED 2026-09-01: `768 MiB + weights/64`. It was calibrated on ONE small
 * model at ctx 4096 and then extrapolated, and it had no context term at all.
 * Measured against the engine's own dry-run at 256K
 * (results/memory-need-measured-20260901.md) it was 59% high on Qwen3.5-0.8B
 * and 9.6x LOW on DeepSeek-V2-Lite — 1013 MiB charged against 9764 MiB real.
 * The workspace's growth rate is set by the model's ARCHITECTURE (DS-V2-Lite
 * grows linearly with context, Qwen3.5 is flat), so the failure was not a bad
 * constant; it was the belief that a constant existed. The workspace is now a
 * measured per-model input (`compute_bytes_*`), and what remains here is only
 * what genuinely is per-NODE.
 *
 * CUDA context: the driver's own allocation when a process initializes CUDA.
 * MEASURED 2026-09-01 on an RTX 5060 Ti (driver-current): NVML reported
 * 16050 MiB free from outside the process, the engine reported 15172 MiB free
 * from inside it, i.e. **878 MiB**. Metal has no analogue (~0), so charging
 * the CUDA figure everywhere is conservative in the safe direction on a Mac.
 *
 * ⚠ This is per-node because each machine runs its own engine process and so
 * pays its own context. Weights, KV and workspace are NOT per-node: the
 * tensor split divides them.
 *
 * The 100 MiB is the ONLY margin in the whole budget, and it is deliberately
 * small: the standing rule (2026-09-01) is no reserves anywhere, and 100 MB is
 * the stated ceiling for the one exception. It covers allocator fragmentation,
 * not a category of cost we failed to enumerate — if a real shortfall appears,
 * measure it and give it its own term rather than growing this one. */
#define IDLETOKEN_LLAMA_CUDA_CONTEXT_BYTES (878ull * 1024 * 1024)
#define IDLETOKEN_LLAMA_NODE_MARGIN_BYTES  (100ull * 1024 * 1024)
#define idletoken_llama_node_overhead(model_bytes) \
    ((void)(model_bytes), \
     IDLETOKEN_LLAMA_CUDA_CONTEXT_BYTES + IDLETOKEN_LLAMA_NODE_MARGIN_BYTES)

uint32_t idletoken_llama_ctx_tier_of(uint32_t ctx_size) {
    if (ctx_size == 0) return 0;
    if (ctx_size <= IDLETOKEN_CTX_TIER_128K) return IDLETOKEN_CTX_TIER_128K;
    if (ctx_size <= IDLETOKEN_CTX_TIER_256K) return IDLETOKEN_CTX_TIER_256K;
    if (ctx_size == IDLETOKEN_CTX_TIER_1M)   return IDLETOKEN_CTX_TIER_1M;
    /* Between 256K and 1M there is nothing measured and nothing offered.
     * Returning the 1M slot would be a guess; returning 0 makes the planner
     * refuse, which is the documented treatment of an unmeasured window. */
    return 0;
}

int idletoken_llama_is_ctx_tier(uint32_t ctx_size) {
    return ctx_size == IDLETOKEN_CTX_TIER_128K ||
           ctx_size == IDLETOKEN_CTX_TIER_256K ||
           ctx_size == IDLETOKEN_CTX_TIER_1M;
}

uint64_t idletoken_llama_compute_bytes(const idletoken_llm_model_size *model,
                                       uint32_t ctx_size, uint8_t backend) {
    if (!model) return 0;
    /* Three product tiers, three measurements (docs/ctx-tiers-2026-09.md).
     * Nothing is interpolated between them: the low tiers are dominated by an
     * n_ubatch floor and only the high end is linear in ctx, so a midpoint
     * would be wrong in both directions depending on where you stood. */
    uint64_t cuda = 0, metal = 0;
    switch (idletoken_llama_ctx_tier_of(ctx_size)) {
        case IDLETOKEN_CTX_TIER_128K:
            cuda  = model->compute_bytes_128k_cuda;
            metal = model->compute_bytes_128k_metal;
            break;
        case IDLETOKEN_CTX_TIER_256K:
            cuda  = model->compute_bytes_256k_cuda;
            metal = model->compute_bytes_256k_metal;
            break;
        case IDLETOKEN_CTX_TIER_1M:
            cuda  = model->compute_bytes_1m_cuda;
            metal = model->compute_bytes_1m_metal;
            break;
        default:
            return 0;   /* not a product tier — refuse, never round */
    }
    switch (backend) {
        case IDLETOKEN_NODE_BACKEND_CUDA:  return cuda;
        case IDLETOKEN_NODE_BACKEND_METAL: return metal;
        default: break;
    }
    /* Unknown backend: charge the larger. Picking one would be a guess about
     * the machine, and on GLM-5.2 that guess is wrong by 22x in the direction
     * that OOMs. If EITHER measurement is missing the answer is 0, which the
     * planner turns into a refusal — an unmeasured model must not become a
     * cheap one. */
    if (cuda == 0 || metal == 0) return 0;
    return cuda > metal ? cuda : metal;
}

uint8_t idletoken_llama_roster_backend(const idletoken_node_mem *nodes, int n) {
    if (!nodes || n <= 0) return IDLETOKEN_NODE_BACKEND_UNKNOWN;
    const uint8_t first = nodes[0].backend;
    if (first == IDLETOKEN_NODE_BACKEND_UNKNOWN)
        return IDLETOKEN_NODE_BACKEND_UNKNOWN;
    for (int i = 1; i < n; i++)
        if (nodes[i].backend != first) return IDLETOKEN_NODE_BACKEND_UNKNOWN;
    return first;
}

uint64_t idletoken_llama_node_usable(const idletoken_node_mem *node) {
    return node ? node->vram_usable : 0;
}

uint64_t idletoken_llama_kv_pool(const idletoken_node_mem *node) {
    if (!node) return 0;
    return node->vram_usable;
}

static uint64_t sat_add_u64(uint64_t a, uint64_t b) {
    return UINT64_MAX - a < b ? UINT64_MAX : a + b;
}

static uint64_t sat_mul_u64(uint64_t a, uint64_t b) {
    return a && b > UINT64_MAX / a ? UINT64_MAX : a * b;
}

static uint64_t round_cells_256(uint64_t cells) {
    if (cells > UINT64_MAX - 255) return UINT64_MAX;
    return (cells + 255) / 256 * 256;
}

uint64_t idletoken_llama_kv_bytes(const idletoken_llm_model_size *model,
                                  uint32_t ctx_size) {
    if (!model) return 0;
    uint64_t bytes = model->kv_fixed_bytes_per_seq;
    if (model->dsv4_raw_bytes_per_cell != 0) {
        /* Mirrors llama-kv-cache-dsv4.cpp exactly: raw has one cell per
         * context token; CSA/HCA use compressed cell counts, each padded to a
         * 256-cell allocation boundary. csa_bytes_per_cell already includes
         * the lightning-indexer cache attached to CSA layers. */
        const uint64_t raw_cells = ctx_size;
        const uint64_t csa_cells = round_cells_256(((uint64_t)ctx_size + 3) / 4);
        const uint64_t hca_cells = round_cells_256(((uint64_t)ctx_size + 127) / 128);
        bytes = sat_add_u64(bytes,
            sat_mul_u64(model->dsv4_raw_bytes_per_cell, raw_cells));
        bytes = sat_add_u64(bytes,
            sat_mul_u64(model->dsv4_csa_bytes_per_cell, csa_cells));
        bytes = sat_add_u64(bytes,
            sat_mul_u64(model->dsv4_hca_bytes_per_cell, hca_cells));
        return bytes;
    }
    return sat_add_u64(bytes,
        sat_mul_u64(model->kv_bytes_per_token, (uint64_t)ctx_size));
}

void idletoken_llama_model_kv_scale(idletoken_llm_model_size *model,
                                    double scale) {
    if (!model || !(scale > 0.0)) return;
#define SCALE_FIELD(field_) \
    model->field_ = (uint64_t)((double)model->field_ * scale + 0.5)
    SCALE_FIELD(kv_bytes_per_token);
    SCALE_FIELD(dsv4_raw_bytes_per_cell);
    SCALE_FIELD(dsv4_csa_bytes_per_cell);
    SCALE_FIELD(dsv4_hca_bytes_per_cell);
#undef SCALE_FIELD
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
                              uint32_t ctx_size, int n_nodes, uint8_t backend) {
    if (n_nodes < 1) n_nodes = 1;
    uint64_t need = sat_add_u64(model->total_bytes,
                                idletoken_llama_kv_bytes(model, ctx_size));
    /* The graph workspace is charged ONCE for the whole cluster, like weights
     * and KV: the tensor split divides the graph, so each node reserves its
     * slice rather than a private copy. Only the CUDA context and the margin
     * are per-node — every machine runs its own engine process. */
    need = sat_add_u64(need,
        idletoken_llama_compute_bytes(model, ctx_size, backend));
    return sat_add_u64(need,
        sat_mul_u64((uint64_t)n_nodes,
                    idletoken_llama_node_overhead(model->total_bytes)));
}

uint64_t idletoken_llama_hard_need(const idletoken_llm_model_size *model,
                                   uint32_t ctx_size, int n_nodes,
                                   uint8_t backend) {
    if (!model) return 0;
    if (n_nodes < 1) n_nodes = 1;
    /* The workspace is resident like the KV cache is: it is allocated, not
     * mmap'd, so it belongs on this side of the line even though the weights
     * do not. */
    uint64_t need = sat_add_u64(idletoken_llama_kv_bytes(model, ctx_size),
        idletoken_llama_compute_bytes(model, ctx_size, backend));
    return sat_add_u64(need,
        sat_mul_u64((uint64_t)n_nodes,
                    idletoken_llama_node_overhead(model->total_bytes)));
}

uint64_t idletoken_llama_working_set(const idletoken_llm_model_size *model) {
    if (!model) return 0;
    /* The GGUF manifests currently expose the total file size and MoE routing
     * counts, but not the byte-exact split between always-hot tensors and each
     * expert's tensors. A ratio derived from expert counts would therefore be
     * a guess. Use the whole mapped file until the tensor-directory parser can
     * provide those exact byte buckets. This is deliberately conservative: it
     * may label a runnable MoE configuration "slow", but it never promises a
     * full-speed working set that was not measured. */
    return model->total_bytes;
}

uint32_t idletoken_llama_fit_ctx(uint64_t usable_bytes,
                                 const idletoken_llm_model_size *model,
                                 uint32_t ctx_want, uint32_t ctx_floor,
                                 uint8_t backend) {
    if (!model || ctx_want == 0) return 0;
    if (model->kv_bytes_per_token == 0 &&
        model->dsv4_raw_bytes_per_cell == 0) return ctx_want; /* unknown ≠ free */
    const uint64_t fixed = model->total_bytes +
                           idletoken_llama_node_overhead(model->total_bytes);
    if (usable_bytes <= fixed) return 0;
    uint32_t lo = 0, hi = ctx_want / 1024;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo + 1) / 2;
        /* The workspace is a function of the context being tried, so it has to
         * move inside the search — hoisting it into `fixed` would size the
         * window against a different window's workspace. */
        const uint64_t need = sat_add_u64(fixed,
            sat_add_u64(idletoken_llama_kv_bytes(model, mid * 1024u),
                        idletoken_llama_compute_bytes(model, mid * 1024u,
                                                      backend)));
        if (need <= usable_bytes) lo = mid; else hi = mid - 1;
    }
    const uint32_t tokens = lo * 1024u;
    return tokens < ctx_floor ? 0 : tokens;
}

int idletoken_quant_bits_from_path(const char *path) {
    /* The client launches the coordinator with only a GGUF path (no
     * --model-id/--quant), so the weight-quant tier must read the file NAME:
     * curated downloads all carry the variant token ("...-UD-IQ2_XXS.gguf",
     * "...-Q4_K_M.gguf"). Scan for the LAST token shaped like a quant name —
     * (I)Q<digits>, BF16/F16, or MXFP4/FP4 at a word boundary — and return
     * its bits.
     * No match -> 0 (conservative tier), same never-guess-low rule. */
    if (!path || !path[0]) return 0;
    const char *base = path;
    for (const char *p = path; *p; p++)
        if (*p == '/' || *p == '\\') base = p + 1;
    int bits = 0;
    for (const char *p = base; *p; p++) {
        const int at_boundary =
            (p == base) || !((p[-1] >= 'A' && p[-1] <= 'Z') ||
                             (p[-1] >= 'a' && p[-1] <= 'z') ||
                             (p[-1] >= '0' && p[-1] <= '9'));
        if (!at_boundary) continue;
        const char *q = p;
        if ((q[0] == 'M' || q[0] == 'm') &&
            (q[1] == 'X' || q[1] == 'x') &&
            (q[2] == 'F' || q[2] == 'f') &&
            (q[3] == 'P' || q[3] == 'p')) q += 4;
        else if ((q[0] == 'F' || q[0] == 'f') &&
                 (q[1] == 'P' || q[1] == 'p')) q += 2;
        else if ((q[0] == 'B' || q[0] == 'b') &&
                 (q[1] == 'F' || q[1] == 'f')) q += 2;
        else {
            if (q[0] == 'I' || q[0] == 'i') q++;
            if (q[0] == 'Q' || q[0] == 'q') q++;
            else if (q[0] == 'F' || q[0] == 'f') q++;
            else continue;
        }
        if (!(*q >= '0' && *q <= '9')) continue;
        int b = 0;
        while (*q >= '0' && *q <= '9') b = b * 10 + (*q++ - '0');
        /* Must END the token here: "Qwen3" (Q + 3 + 'w'-less but followed by
         * '.') is a real trap only when letters follow; digits already
         * consumed, so reject a trailing letter that would make this a word. */
        if ((*q >= 'A' && *q <= 'Z' && *q != '_') || (*q >= 'a' && *q <= 'z'))
            continue;
        if (b >= 1 && b <= 32) bits = b;   /* keep scanning: LAST match wins */
    }
    return bits;
}

int idletoken_quant_weight_bits(const char *quant) {
    /* Use the same token parser as a GGUF filename. "first digit wins" read
     * the 3 in Qwen3.5 from the MXFP4 variant label and happened to choose the
     * same tier for the wrong reason. */
    return idletoken_quant_bits_from_path(quant);
}

/* REVISED 2026-09-02 (user decision), replacing the 08-26 three-tier rule:
 * q8_0 now covers EVERY quantized weight tier, 3 bit through 15, so Q5/Q6/Q8
 * no longer keep an f16 cache. f16 is left to BF16/F16 weights alone — an
 * unquantized model is the one case where a quantized cache would be the only
 * lossy thing in the pipeline. The 1-2 bit tier keeps q4_0.
 *
 * What this buys: on qwen3.8-27b Q8_K_XL @256K the growing KV drops 16.00 ->
 * 8.50 GiB, and the whole-cluster requirement 46.9 -> 39.4 GiB.
 *
 * ⚠ Not yet ppl-gated at the newly-covered tiers (5-15 bit). The 1-2 bit tier's
 * q4_0 has weak field evidence only (cluster-5 served weeks on it), and the
 * 3-4 bit tier's q8_0 likewise. Treat "q8_0 KV costs nothing measurable next to
 * Q8 weights" as an assumption on record, not a measurement.
 *
 * ⚠ Setting a type also makes the sidecar pass `-fa on` (a quantized V cache
 * requires flash attention), so that now applies to Q5-Q8 as well, where the
 * engine previously chose FA itself. */
int idletoken_llama_kv_tier_for_weight(int weight_bits) {
    if (weight_bits >= 1 && weight_bits <= 2) return IDLETOKEN_KV_TIER_Q4_0;
    if (weight_bits >= 3 && weight_bits <= 15) return IDLETOKEN_KV_TIER_Q8_0;
    /* 0 = unknown and >=16 = unquantized both keep f16. Unknown must not be
     * guessed DOWN: the planner and the engine agree only because both fall
     * back to f16 here, and a cheaper guess would under-charge the budget. */
    return IDLETOKEN_KV_TIER_F16;
}

const char *idletoken_llama_kv_tier_name(int tier) {
    switch (tier) {
        case IDLETOKEN_KV_TIER_F16:  return "f16";
        case IDLETOKEN_KV_TIER_Q8_0: return "q8_0";
        case IDLETOKEN_KV_TIER_Q4_0: return "q4_0";
        default: return "f16";
    }
}

int idletoken_llama_kv_tier_of_name(const char *name) {
    /* "" and bf16 both mean "the engine's default cache", which is what the
     * f16 slot was measured under. Everything else the escape hatch allows
     * (q5_1, q5_0, iq4_nl, q4_1) has NO measured workspace, and -1 says so
     * rather than letting a caller round it to a neighbour. */
    if (!name || !name[0]) return IDLETOKEN_KV_TIER_F16;
    for (int t = 0; t < IDLETOKEN_KV_TIER_COUNT; t++)
        if (strcmp(name, idletoken_llama_kv_tier_name(t)) == 0) return t;
    if (strcmp(name, "bf16") == 0) return IDLETOKEN_KV_TIER_F16;
    return -1;
}

const char *idletoken_llama_kv_type_for_weight(int weight_bits) {
    /* Derived, never a second boundary table: the f16 tier returns NULL so the
     * coordinator passes no -ctk/-ctv and the engine uses its own default,
     * which is the configuration the f16 workspace was measured under. */
    const int tier = idletoken_llama_kv_tier_for_weight(weight_bits);
    return tier == IDLETOKEN_KV_TIER_F16 ? NULL
                                         : idletoken_llama_kv_tier_name(tier);
}

/* Display rounding is retained for diagnostics. Product starts use the exact
 * selected 256K or 1M context and never call this as an automatic ladder. */
uint32_t idletoken_llama_ctx_display_tier(uint32_t max_ctx) {
    static const uint32_t tiers[] = {
        16384, 32768, 65536, 131072, 262144, 524288, 1048576,
    };
    uint32_t best = 0;
    for (size_t i = 0; i < sizeof(tiers) / sizeof(tiers[0]); i++) {
        if (tiers[i] > max_ctx) break;
        best = tiers[i];
    }
    return best;
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
    /* The workspace is split with the graph, so it takes the share; the CUDA
     * context does not — it costs the same on a machine holding three layers
     * as on one holding all of them. */
    const uint64_t workspace = (uint64_t)(
        (double)idletoken_llama_compute_bytes(model, ctx_size, node->backend) *
        layer_share);
    const uint64_t fixed = weights + workspace +
                           idletoken_llama_node_overhead(model->total_bytes);
    /* Also the slow-tier rule: weights that do not fit the KV pool are weights
     * spilling to host RAM, and a machine already paying PCIe per token must
     * not also be asked to hold a second context in the memory it ran out of. */
    if (pool <= fixed) return 1;

    const uint64_t kv_per_seq =
        (uint64_t)((double)idletoken_llama_kv_bytes(model, ctx_size) * layer_share);
    if (kv_per_seq == 0) return 1;

    uint64_t slots = (pool - fixed) / kv_per_seq;
    if (slots < 1) return 1;
    if (slots > (uint64_t)cap) slots = (uint64_t)cap;
    return (int)slots;
}

int idletoken_plan_llamacpp(const idletoken_llm_model_size *model,
                            const idletoken_node_mem *nodes, int n,
                            int coordinator, uint32_t ctx_size,
                            int force_cluster,
                            idletoken_llama_plan *out) {
    if (!model || !nodes || !out || n <= 0 || n > IDLETOKEN_LLPLAN_MAX_NODES ||
        coordinator < 0 || coordinator >= n || model->total_bytes == 0)
        return -1;
    memset(out, 0, sizeof(*out));
    out->single_node = -1;
    out->layer0_node = -1;

    /* No measurement, no admission (2026-09-01). A zero workspace is the ONE
     * value that must never be taken at face value: charging 0 means the
     * budget silently omits a term that measured 9764 MiB on a 15.7 GiB model
     * at 256K, so the plan would say "fits", the user would wait through a
     * full load, and the engine would OOM. Refusing is the honest answer and
     * the fix is mechanical — run the dry-run and record it.
     *
     * Deliberately BEFORE the invariant checks below: "we never measured this"
     * is a fact about the build, not about the user's machines, and a machine
     * report cannot make it true or false. */
    const uint8_t backend = idletoken_llama_roster_backend(nodes, n);
    if (idletoken_llama_compute_bytes(model, ctx_size, backend) == 0) {
        out->kind = IDLETOKEN_LLPLAN_REFUSE;
        snprintf(out->why, sizeof(out->why),
                 "[RESOURCE_INSUFFICIENT] this model has no measured GPU "
                 "workspace for a %u-token context, so its memory requirement "
                 "is unknown. Measure it against the pinned engine "
                 "(scripts/measure_model_memory.sh <gguf> %u) and record the "
                 "result in the model manifest; this build will not guess.",
                 ctx_size, ctx_size);
        return 0;
    }

    const uint64_t need1 = llplan_needed(model, ctx_size, 1, backend);
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

    /* ---- default: fits the coordinator alone → select SINGLE.
     * "The best single node" is the coordinator by construction: a lone
     * worker could hold the bytes, but layer 0 may not move there (above),
     * so the only machine that can legally run single IS the coordinator.
     * This is a default, not a veto: an explicit multi-machine choice wins. */
    if (coord_usable >= need1 && !(force_cluster && n > 1)) {
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

    /* A forced single-machine launch is allowed to reach the planner, but it
     * is not allowed to spill weights or cache to host RAM. Refuse the exact
     * selected window here; never silently shrink it. */
    if (n == 1) {
        out->kind = IDLETOKEN_LLPLAN_REFUSE;
        snprintf(out->why, sizeof(out->why),
                 "[RESOURCE_INSUFFICIENT] this model needs %.2f GiB of GPU "
                 "memory at ctx %u (weights + KV + engine overhead), but this "
                 "machine has %.2f GiB available — %.2f GiB short. Free GPU "
                 "memory, choose a smaller quantization, or add cluster nodes.",
                 (double)need1 / (double)GiB, ctx_size,
                 (double)coord_usable / (double)GiB,
                 (double)(need1 - coord_usable) / (double)GiB);
        return 0;
    }

    uint64_t total_usable = 0;
    for (int i = 0; i < n; i++) total_usable += idletoken_llama_node_usable(&nodes[i]);
    const uint64_t need_n = llplan_needed(model, ctx_size, n, backend);
    const uint64_t hard = idletoken_llama_hard_need(model, ctx_size, n, backend);
    const uint64_t wset = idletoken_llama_working_set(model);
    out->hard_need_bytes   = hard;
    out->working_set_bytes = wset;
    out->working_set_fits  = (total_usable >= need_n);

    if (total_usable < need_n) {
        out->kind = IDLETOKEN_LLPLAN_REFUSE;
        snprintf(out->why, sizeof(out->why),
                 "[RESOURCE_INSUFFICIENT] this model needs %.2f GiB of GPU "
                 "memory at ctx %u (weights + KV + per-node engine overhead), "
                 "but the %d machine(s) have %.2f GiB available — %.2f GiB "
                 "short. Free GPU memory, choose a smaller quantization, or "
                 "add cluster nodes.",
                 (double)need_n / (double)GiB, ctx_size,
                 n, (double)total_usable / (double)GiB,
                 (double)(need_n - total_usable) / (double)GiB);
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
    const uint64_t kv_total = idletoken_llama_kv_bytes(model, ctx_size);
    /* What gets split: weights, KV and the graph workspace all divide with the
     * layers. The CUDA context does not — it is charged per node below. */
    const uint64_t slice_all = model->total_bytes + kv_total +
                               idletoken_llama_compute_bytes(model, ctx_size,
                                                            backend);
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
        const uint64_t room = pool_of[i] > per_node_oh
                                  ? pool_of[i] - per_node_oh : 0;
        cap[i] = slice_all ? (double)room / (double)slice_all : 1.0;
        if (cap[i] > 1.0) cap[i] = 1.0;
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
                 "[RESOURCE_INSUFFICIENT] the coordinator can only hold %.2f GiB in the memory "
                 "its GPU can address (%.2f GiB of video memory, minus %.2f GiB "
                 "of engine overhead), which is less than the %.2f GiB one layer "
                 "of this model needs at ctx %u — and layer 0 must stay on the "
                 "coordinator (privacy invariant). Free video memory on this "
                 "machine, choose a smaller quantization, or add cluster nodes.",
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
        out->kind = IDLETOKEN_LLPLAN_REFUSE;
        snprintf(out->why, sizeof(out->why),
                 "[RESOURCE_INSUFFICIENT] %.2f GiB of weights + KV at ctx %u must be split "
                 "across %d machines, but a machine can only be handed layers "
                 "in memory its GPU can address. Node %d has %.2f GiB of video "
                 "memory, and cluster-wide only "
                 "%.2f GiB is reachable — %.2f GiB short. Add a machine with "
                 "more video memory or choose a smaller quantization.",
                 (double)slice_all / (double)GiB, ctx_size, n,
                 out->order[tightest], (double)pool_of[tightest] / (double)GiB,
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
                 "[RESOURCE_INSUFFICIENT] no machine in this cluster reports GPU memory its engine "
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
                 "[RESOURCE_INSUFFICIENT] only %.1f%% of this model's layers could be placed in "
                 "memory the machines' GPUs can address (node %d holds at most "
                 "%.2f GiB of the %.2f GiB total at ctx %u). Add a machine with "
                 "more video memory or choose a smaller quantization.",
                 placed * 100.0, out->order[tightest],
                 cap[tightest] * (double)slice_all / (double)GiB,
                 (double)slice_all / (double)GiB, ctx_size);
        return 0;
    }

    if (coord_usable >= need1) {
        /* The user explicitly selected a cluster even though the model fits
         * the coordinator. Say that without pretending capacity required it. */
        snprintf(out->why, sizeof(out->why),
                 "CLUSTER (selected): %.2f GiB needed FITS the coordinator's "
                 "%.2f GiB usable — using the user's multi-machine choice; "
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

int idletoken_llama_device_layer_range(
        unsigned n_layers, int n_gpu_layers,
        const double *shares, int n_devices,
        int dev_lo, int dev_hi,
        unsigned *layer_lo, unsigned *layer_hi) {
    if (!shares || n_devices < 1 ||
        n_devices > IDLETOKEN_LLPLAN_MAX_DEVICES ||
        dev_lo < 0 || dev_hi > n_devices || dev_lo >= dev_hi ||
        !layer_lo || !layer_hi) {
        return -1;
    }

    double sum = 0.0;
    double cumulative[IDLETOKEN_LLPLAN_MAX_DEVICES];
    for (int d = 0; d < n_devices; d++) {
        if (shares[d] < 0.0) return -1;
        sum += shares[d];
        cumulative[d] = sum;
    }
    if (sum <= 0.0) return -1;
    for (int d = 0; d < n_devices; d++) cumulative[d] /= sum;

    const int n_all = (int)n_layers;
    const int start = n_all + 1 - n_gpu_layers > 0
                          ? n_all + 1 - n_gpu_layers : 0;
    const int active = n_gpu_layers < n_all + 1
                           ? n_gpu_layers : n_all + 1;
    int first = -1, last = -1;
    for (int il = 0; il < n_all; il++) {
        if (il < start || il - start >= active) continue;
        const double pos = (double)(il - start) / (double)active;
        int owner = 0;
        while (owner < n_devices && pos >= cumulative[owner]) owner++;
        if (owner >= n_devices) owner = n_devices - 1;
        if (owner >= dev_lo && owner < dev_hi) {
            if (first < 0) first = il;
            last = il;
        }
    }
    if (first < 0) {
        *layer_lo = n_layers;
        *layer_hi = n_layers;
    } else {
        *layer_lo = (unsigned)first;
        *layer_hi = (unsigned)(last + 1);
    }
    return 0;
}
