/* idletoken_model.h — pluggable model registry (multi-model design §3.1).
 *
 * The engine-side mirror of the client's models/<id>.json manifests: one
 * idletoken_model_spec per model, looked up by stable id. The orchestration
 * layer (plan/coord/worker) reads ALL model-shaped numbers from here —
 * never from DS4_* compile-time constants — so adding a model touches this
 * table (+ a backend that can run it), not the planner or the protocol.
 *
 * Numbers for models with available=0 are pre-integration ESTIMATES used
 * only for UI/feasibility hints; they get corrected when the backend lands.
 * At load time the backend must cross-check n_layers/arch against the GGUF
 * metadata (manifest is for planning; the GGUF is runtime truth).
 *
 * C only. No C++. Pure data + lookups: unit-testable anywhere.
 */
#ifndef IDLETOKEN_MODEL_H
#define IDLETOKEN_MODEL_H

#include <stddef.h>
#include <stdint.h>

/* Which inference backend runs this model (ASSIGN_PLAN `backend` byte). */
typedef enum {
    IDLETOKEN_BACKEND_DS4  = 1,   /* vendor/ds4 DSv4-Flash-only kernel path */
    IDLETOKEN_BACKEND_DS4X = 2,   /* runtime-config generic MLA-MoE path (Phase B) */
    IDLETOKEN_BACKEND_LLAMACPP = 3, /* llama.cpp engine (v2 rebuild, 2026-08-14):
                                   * coord drives a local llama-server sidecar.
                                   * Registry migration to this backend is a
                                   * separate step (WS-B4); the value exists so
                                   * ASSIGN_PLAN and manifests can name it. */
} idletoken_backend;

/* Technical placement capability. This is deliberately NOT the default
 * deployment: the same medium-sized model may fit locally at a low-bit quant
 * and need several machines at a high-precision quant. The client derives a
 * quiet default from model + quant + current memory, then honors the user's
 * explicit choice. Current llama.cpp-backed models are all cluster-capable. */
typedef enum {
    IDLETOKEN_DEPLOY_UNSPECIFIED = 0, /* never valid in the registry — see
                                       * idletoken_model_may_cluster() */
    IDLETOKEN_DEPLOY_SINGLE_NODE = 1, /* reserved for an unsplittable backend */
    IDLETOKEN_DEPLOY_CLUSTER     = 2, /* may span a LAN cluster (N=1 is
                                       * the degenerate case, still allowed) */
} idletoken_deployment;

/* KV-cache shape family — drives the per-node overhead estimate. */
typedef enum {
    IDLETOKEN_KV_DSV4 = 1,  /* SWA/CSA/HCA tri-state: fixed per-tier table */
    IDLETOKEN_KV_MLA  = 2,  /* compressed latent KV: bytes/token/layer formula */
    IDLETOKEN_KV_GQA  = 3,  /* standard K/V cache: 2·n_head_kv·head_dim·dtype
                          * bytes/token/layer — same overhead formula as MLA,
                          * only the per-token-layer constant differs
                          * (small-model-design.md §5) */
    IDLETOKEN_KV_HYBRID = 4, /* linear-attention hybrid (Qwen3.5): a fixed-size
                          * recurrent state on linear layers (INDEPENDENT of
                          * ctx) + a normal GQA cache on the 1-in-N full layers.
                          * needed = state_bytes_per_layer·n_linear_on_node
                          *        + kv_bytes_per_token_layer·ctx·n_full_on_node
                          * This is why long ctx costs far less than pure GQA —
                          * a real advantage, so compute it honestly rather than
                          * charging every layer the full KV price. */
} idletoken_kv_kind;

/* One selectable precision (quant) of a model. Small models ship in several
 * quants (Q4_K_M default .. BF16) that differ only in byte size + download
 * source; the shape (layers/hidden/kv) is quant-independent and lives on the
 * spec. A model with n_variants==0 has a single implicit variant given by the
 * spec's scalar layer_weight_bytes/shared_weight_bytes/default_gguf fields.
 * (small-model-design.md §3.2). */
typedef struct {
    const char *quant;             /* "Q4_K_M" etc; matches ASSIGN_PLAN.quant */
    uint64_t layer_weight_bytes;   /* Σ blk.* at this quant */
    uint64_t shared_weight_bytes;  /* embd + output head at this quant */
    const char *gguf;              /* default filename for this quant */
} idletoken_model_variant;

typedef struct {
    const char *id;            /* stable id, e.g. "deepseek-v4-flash" */
    const char *label;         /* human name for logs/UI */
    uint8_t  backend;          /* idletoken_backend */
    uint8_t  available;        /* 0 = registered but not yet runnable */
    uint8_t  deployment;       /* idletoken_deployment — mirrors the manifest's
                                * "deployment"; 0 means somebody added a model
                                * and forgot, which model_manifest_check.py
                                * rejects and idletoken_model_may_cluster()
                                * treats as "no" */

    uint16_t n_layers;
    uint32_t n_embd;
    uint8_t  hc_streams;       /* activation streams crossing a stage boundary
                                * (DSv4 mHC = 4; plain residual models = 1) */
    uint32_t n_vocab;
    uint16_t n_expert;         /* 0 on dense models */
    uint16_t n_expert_used;    /* routed experts consulted per token */

    uint64_t layer_weight_bytes;   /* Σ all blk.* tensors at the shipped quant */
    uint64_t shared_weight_bytes;  /* embd + output head + mtp — every stage loads */
    uint32_t ctx_max;              /* trained context window */
    uint32_t ctx_yarn_max;         /* curated/validated YaRN-extended window
                                    * (Qwen: 4x the trained one); 0 = no
                                    * approved extension, ctx_max is the hard
                                    * ceiling */
    uint16_t split_boundary_multiple; /* prefer PP cuts at multiples (0/1 = none;
                                       * GLM-5.2: 4 — shared DSA indexer group) */

    uint8_t  kv_kind;              /* idletoken_kv_kind */
    uint32_t kv_bytes_per_token_layer; /* IDLETOKEN_KV_MLA / IDLETOKEN_KV_GQA */
    /* IDLETOKEN_KV_HYBRID only: fixed recurrent-state bytes per linear layer, and
     * the full-attention period (every full_attn_interval-th layer is full). */
    uint32_t state_bytes_per_layer;
    uint32_t full_attn_interval;
    /* DeepSeek4's pinned llama.cpp cache is not a linear
     * bytes-per-token-per-layer allocation: it keeps one raw K cache for every
     * layer plus CSA (1/4) and HCA (1/128) compressed caches, each rounded to
     * 256 cells, and fixed f32 compressor state. These are WHOLE-MODEL f16
     * bytes per cache cell / per sequence, read from the GGUF geometry and the
     * pinned engine allocation code. Zero for every other KV family. */
    uint64_t dsv4_raw_bytes_per_cell;
    uint64_t dsv4_csa_bytes_per_cell;
    uint64_t dsv4_hca_bytes_per_cell;
    uint64_t dsv4_fixed_bytes_per_seq;
    uint64_t overhead_base_bytes;      /* MLA/GQA: non-KV per-node overhead
                                        * (activations/workspace/comms) */
    /* MEASURED graph workspace at the two product context tiers, in bytes,
     * from llama.cpp's own no_alloc dry-run — `scripts/measure_model_memory.sh`
     * against the pinned engine. NOT estimated, and not derivable from any
     * other field here: the growth rate is set by the model's architecture
     * (results/memory-need-measured-20260901.md). 0 = not yet measured.
     *
     * Per BACKEND: the value is identical across GPUs and across every
     * quantization in the menu, but NOT across backends — GLM-5.2 at 256K
     * measures 1.50 GiB on CUDA and 33.3 GiB on Metal. Re-measure when the
     * engine pin moves. */
    uint64_t compute_bytes_256k_cuda;
    uint64_t compute_bytes_1m_cuda;
    uint64_t compute_bytes_256k_metal;
    uint64_t compute_bytes_1m_metal;

    const char *default_gguf;  /* default filename when --model-path is absent;
                                * mirrors variants[default_variant].gguf */

    /* Selectable precisions. When n_variants==0 the scalar *_weight_bytes /
     * default_gguf above ARE the single implicit variant (unchanged behaviour
     * for DSv4/GLM/Kimi). When present, variants[default_variant] mirrors the
     * scalars so quant-unaware callers keep working. */
    const idletoken_model_variant *variants;
    uint8_t  n_variants;
    uint8_t  default_variant;
} idletoken_model_spec;

/* NULL when id is unknown (callers must handle — a joiner with a newer
 * client may name a model this build has never heard of). */
const idletoken_model_spec *idletoken_model_get(const char *id);

/* The v0.1 default: DeepSeek V4 Flash. Never NULL. */
const idletoken_model_spec *idletoken_model_default(void);

/* Iterate the registry. Needed by anything that has to answer "what can this
 * machine run?" over ALL models (the capability advisor) rather than one the
 * caller already named. Index order is the table order in model.c. */
int idletoken_model_count(void);
const idletoken_model_spec *idletoken_model_at(int index);

/* Resolve a precision by quant name (e.g. "Q8_0"). Returns the matching
 * variant, or the model's default variant when `quant` is NULL/unknown, or
 * NULL only if the model has no variant table (caller falls back to the
 * spec's scalar layer_weight_bytes/shared_weight_bytes/default_gguf).
 * small-model-design.md §3.2/§3.3. */
const idletoken_model_variant *idletoken_model_variant_get(const idletoken_model_spec *m,
                                                     const char *quant);

/* Weight bytes for `quant` (variant if resolvable, else spec scalars).
 * Convenience for planners that must size the SELECTED precision. */
void idletoken_model_weight_bytes(const idletoken_model_spec *m, const char *quant,
                               uint64_t *layer_out, uint64_t *shared_out);

/* May this backend/model combination technically be served by more than one
 * node? This is not a recommendation. False for an explicitly unsplittable
 * model and for an undeclared model; NULL is false.
 *
 * `why` (optional, may be NULL) receives a user-facing sentence explaining the
 * refusal; it is left untouched when the answer is true. */
int idletoken_model_may_cluster(const idletoken_model_spec *m, char *why, size_t why_cap);

/* Per-node inference overhead (KV + activations + workspace + comms + CUDA
 * context + margin) for `layers_on_node` of this model at `ctx_size`.
 * DSv4 uses the calibrated per-tier table (layers_on_node ignored); MLA
 * models compute base + kv_bytes_per_token_layer × ctx × layers.
 * ESTIMATES -- recalibrate per model on real machines. */
uint64_t idletoken_model_overhead(const idletoken_model_spec *m, uint32_t ctx_size,
                               int layers_on_node);

#endif /* IDLETOKEN_MODEL_H */
