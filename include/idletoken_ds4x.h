/* idletoken_ds4x.h — the ds4x generic MLA-MoE backend's runtime config
 * (multi-model design §3.2, Phase B).
 *
 * Where vendor/ds4 hard-wires DSv4-Flash as compile-time DS4_N_* constants,
 * the ds4x path reads EVERY model dimension from GGUF metadata at load time,
 * dispatched on `general.architecture`. v0.1 coverage: the MLA + MoE family —
 *   deepseek2  (Kimi K2.x, DeepSeek V3/R1 lineage)
 *   glm_dsa    (GLM-5.x — DSA sparse path NOT implemented; runs as full MLA,
 *               same fallback llama.cpp mainline uses)
 *
 * Contract: the manifest (models/<id>.json + src/common/model.c) is the
 * PLANNING truth; the GGUF is the RUNTIME truth. ds4x_config_check_spec()
 * enforces they agree before any weights are touched.
 *
 * C only. No C++. Config parsing is host-only (no GPU) so it unit-tests
 * anywhere (src/tools/ds4x_config_test.c, `make ds4xtest`).
 */
#ifndef IDLETOKEN_DS4X_H
#define IDLETOKEN_DS4X_H

#include <stddef.h>
#include <stdint.h>

#include "idletoken_model.h"

typedef enum {
    DS4X_ARCH_UNKNOWN   = 0,
    DS4X_ARCH_DEEPSEEK2 = 1,   /* MLA */
    DS4X_ARCH_GLM_DSA   = 2,   /* MLA (DSA sparse unimplemented; full-MLA fallback) */
    DS4X_ARCH_QWEN3     = 3,   /* GQA dense/MoE (small-model-design.md) */
    DS4X_ARCH_LLAMA     = 4,   /* GQA dense (dispatch reserved; first batch = qwen3) */
    DS4X_ARCH_QWEN35    = 5,   /* hybrid Gated DeltaNet : Gated Attention (3:1) */
} ds4x_arch;

/* Attention family — MLA (latent KV, deepseek2/glm) vs standard GQA (full K/V
 * cache, qwen3/llama). Selects the forward branch + KV-cache shape. */
typedef enum {
    DS4X_ATTN_MLA    = 1,
    DS4X_ATTN_GQA    = 2,
    DS4X_ATTN_LINEAR = 3,   /* Gated DeltaNet (Qwen3.5/3.6, Kimi K3 KDA) */
} ds4x_attn_kind;

/* Hybrid models (Qwen3.5: 3 linear : 1 full) are NOT uniform per layer, so the
 * attention kind is per-layer. cfg.attn_kind stays the model-level default and
 * layer_types[il] overrides it. Bounded by the same 512-layer sanity cap the
 * config check enforces. docs/linear-attention-design.md §3. */
#define DS4X_MAX_LAYERS 512

/* Router gating op — matches llama.cpp expert_gating_func (1=softmax default,
 * 2=sigmoid; DeepSeek-V3 / Kimi lineage use sigmoid). */
typedef enum {
    DS4X_GATE_SOFTMAX = 1,
    DS4X_GATE_SIGMOID = 2,
} ds4x_gate;

typedef struct {
    ds4x_arch arch;
    char arch_name[32];      /* verbatim general.architecture */

    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_vocab;        /* vocab_size key, else len(tokenizer.ggml.tokens) */
    uint32_t n_ctx_train;

    /* attention */
    uint8_t  attn_kind;         /* ds4x_attn_kind (MLA or GQA) */
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t kv_lora_rank;      /* MLA compressed-KV rank (per-token KV = rank+rope) */
    uint32_t q_lora_rank;       /* 0 = no q compression */
    uint32_t qk_nope_head_dim;
    uint32_t qk_rope_head_dim;
    uint32_t v_head_dim;
    /* GQA (attn_kind == DS4X_ATTN_GQA): the per-head width; the whole head_dim
     * gets rope. For GQA the loader sets qk_nope_head_dim=0,
     * qk_rope_head_dim=head_dim, v_head_dim=head_dim so the MLA scratch layout
     * and attn_out shape carry over unchanged. */
    uint32_t head_dim;
    uint8_t  qk_norm;           /* 1 = per-head RMSNorm on Q and K (Qwen3) */
    /* qwen35: rope covers only the first rope_dim of head_dim (partial
     * rotary, 64 of 256), and the q projection carries an extra gate of
     * n_head*head_dim that multiplies the attention output. §4b. */
    uint32_t rope_dim_partial;  /* 0 = rope the whole head_dim */
    uint8_t  attn_out_gate;     /* 1 = q proj emits [q | gate] */

    /* Gated DeltaNet (attn_kind/layer_types == DS4X_ATTN_LINEAR).
     * State per layer is lin_v_heads × lin_k_dim × lin_v_dim and does NOT
     * grow with context — that is the whole point of the linear path. */
    uint32_t lin_k_heads, lin_v_heads;   /* v_heads is a multiple of k_heads */
    uint32_t lin_k_dim,   lin_v_dim;
    uint32_t conv_kernel;                /* causal depthwise conv (Qwen3.5: 4) */

    /* Per-layer attention kind; 0 = fall back to attn_kind (uniform models). */
    uint8_t  layer_types[DS4X_MAX_LAYERS];

    /* MoE */
    uint32_t n_dense_lead;      /* leading dense FFN blocks (GLM-5.2: 3) */
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_expert_shared;
    uint32_t n_ff_exp;          /* expert FFN width */
    uint32_t n_ff_dense;        /* dense-block FFN width (0 = no dense blocks) */
    /* router weighting (llama.cpp build_moe_ffn semantics) */
    uint8_t  gating_func;       /* ds4x_gate */
    uint8_t  expert_weights_norm; /* normalize selected weights by their sum */
    float    expert_weights_scale; /* routed_scaling_factor (0/1 = none) */

    /* rope */
    float    rope_theta;

    /* DSA indexer (glm_dsa only; informational in v0.1 — full-MLA fallback) */
    uint32_t index_topk;
    uint32_t index_group;       /* layers sharing one indexer (GLM-5.2: 4) */
} ds4x_config;

/* Fill cfg from the GGUF at `path`. Returns 0, or -1 with a reason in err
 * (unknown architecture, missing required keys, implausible values). */
int ds4x_config_from_gguf(const char *path, ds4x_config *cfg,
                          char *err, size_t errlen);

/* Cross-check runtime truth (GGUF) against planning truth (registry spec):
 * layer count, hidden size, vocab. Returns 0 or -1 with reason — a mismatch
 * means the user pointed the cluster at the wrong file; refuse loudly. */
int ds4x_config_check_spec(const ds4x_config *cfg, const idletoken_model_spec *spec,
                           char *err, size_t errlen);

/* MLA per-token-per-layer KV bytes at fp16: (kv_lora_rank + rope_dim) × 2.
 * The resource-model counterpart of manifest kv.bytes_per_token_per_layer. */
uint64_t ds4x_kv_bytes_per_token_layer(const ds4x_config *cfg);

/* ---- CPU reference forward (Phase B.2) ---------------------------------
 * fp32, single-threaded, correctness-first: the numeric oracle the CUDA path
 * must match (which in turn must match llama.cpp on real GLM-5.2 weights —
 * G-GLM ladder). Math spec lives in scripts/ds4x_ref.py; keep in sync.
 * All matrices are row-major [out_dim][in_dim]; y = W·x. */

/* A weight matrix, possibly still quantized. `data` points at either fp32
 * (type 0) or raw GGUF-quantized bytes (Q8_0=8, Q4_0=2, …); matvec
 * dequantizes ONE ROW at a time into scratch, so a real model stays at its
 * on-disk size in RAM instead of the 4× fp32 blowup that hangs the box.
 * Tiny fixtures use type 0 (plain fp32, bit-identical to the old path). */
typedef struct {
    const void *data;
    uint32_t    type;   /* ggml type id (0=F32) */
    /* Optional VRAM-resident mirror (ds4x_cuda_wt *). NULL = CPU-only, which
     * is always valid — the CPU path stays the numeric reference. Set by
     * ds4x_model_load on a CUDA build with a usable device. */
    void       *dev;
    /* Element offset of THIS view inside `dev`'s matrix. A stacked
     * [n_expert][n_ff][n_embd] MoE tensor is uploaded once and each expert is a
     * sub-range of it, so a slice must carry where it starts — otherwise
     * wt_slice would hand the GPU the parent handle and every expert would
     * compute the whole stack into an n_ff-sized buffer. 0 for whole tensors. */
    uint64_t    dev_elem_off;
} ds4x_wt;

typedef struct {
    const float *attn_norm;   /* [n_embd] (norms/bias stay fp32 — tiny) */
    /* q: LoRA trio (q_lora_rank>0) OR full projection q_proj (q_lora_rank==0) */
    ds4x_wt      q_a;         /* [q_lora_rank][n_embd] */
    const float *q_a_norm;    /* [q_lora_rank] */
    ds4x_wt      q_b;         /* [n_head*(nope+rope)][q_lora_rank] */
    ds4x_wt      q_proj;      /* MLA: [n_head*(nope+rope)][n_embd] (non-LoRA q);
                               * GQA: [n_head*head_dim][n_embd] */
    ds4x_wt      kv_a;        /* [kv_lora_rank+rope][n_embd] */
    const float *kv_a_norm;   /* [kv_lora_rank] */
    ds4x_wt      kv_b;        /* [n_head*(nope+v_dim)][kv_lora_rank] */
    /* GQA-only projections (attn_kind == DS4X_ATTN_GQA) */
    ds4x_wt      k_proj;      /* [n_head_kv*head_dim][n_embd] */
    ds4x_wt      v_proj;      /* [n_head_kv*head_dim][n_embd] */
    const float *q_norm;      /* [head_dim] per-head Q RMSNorm (qk_norm; else NULL) */
    const float *k_norm;      /* [head_dim] per-head K RMSNorm (qk_norm; else NULL) */
    /* Gated DeltaNet layer (DS4X_ATTN_LINEAR). Names mirror the reference in
     * docs/linear-attention-design.md §2. */
    /* Real qwen35 GGUFs keep these as SEPARATE tensors (attn_qkv / attn_gate /
     * ssm_beta / ssm_alpha) rather than one fused in_proj — see design doc
     * §4b. Mirroring that avoids copying quantized weights at load. */
    ds4x_wt      in_proj_qkv;  /* [k_dim*2 + v_dim][n_embd] → q,k,v (conv'd) */
    ds4x_wt      in_proj_z;    /* [v_dim][n_embd]           → z gate         */
    ds4x_wt      in_proj_b;    /* [lin_v_heads][n_embd]     → beta logits    */
    ds4x_wt      in_proj_a;    /* [lin_v_heads][n_embd]     → alpha logits   */
    const float *conv1d_w;     /* [conv_ch][conv_kernel] depthwise, conv_ch = k_dim*2+v_dim */
    const float *conv1d_b;     /* [conv_ch] */
    const float *A_log;        /* [lin_v_heads]  g = -exp(A_log)*softplus(a+dt_bias) */
    const float *dt_bias;      /* [lin_v_heads] */
    const float *ssm_norm;     /* [v_dim] gated RMSNorm weight */
    ds4x_wt      out_proj;     /* [n_embd][v_dim] */
    ds4x_wt      attn_out;    /* [n_embd][n_head*v_dim] */
    const float *ffn_norm;    /* [n_embd] */
    /* dense-lead layers */
    ds4x_wt      gate, up;    /* [n_ff_dense][n_embd] */
    ds4x_wt      down;        /* [n_embd][n_ff_dense] */
    /* MoE layers */
    ds4x_wt      router;                  /* [n_expert][n_embd] */
    const float *e_score_bias;            /* [n_expert] router correction bias
                                           * (DeepSeek-V3 lineage; NULL when
                                           * the GGUF has none) */
    ds4x_wt      e_gate, e_up;            /* [n_expert][n_ff_exp][n_embd] */
    ds4x_wt      e_down;                  /* [n_expert][n_embd][n_ff_exp] */
    ds4x_wt      s_gate, s_up;            /* [n_ff_exp][n_embd] */
    ds4x_wt      s_down;                  /* [n_embd][n_ff_exp] */
    /* qwen35moe only: the shared expert is SCALED by sigmoid(w·x), a single
     * scalar per token (llama.cpp: shared_expert_gate → SIGMOID → MUL). Other
     * MoE models add the shared expert ungated, so this stays NULL for them. */
    ds4x_wt      s_gate_inp;              /* [1][n_embd] */
} ds4x_layer_weights;

/* Per-layer KV cache, absolute-position indexed (the caller sizes it to the
 * context window; PP keeps each layer's cache on its own worker). MLA uses
 * latent+k_rope; GQA uses k+v. The runner allocates only the pair its
 * attn_kind needs; the other stays NULL. */
typedef struct {
    float *latent;   /* MLA: [ctx][kv_lora_rank] (rmsnormed compressed KV) */
    float *k_rope;   /* MLA: [ctx][qk_rope_head_dim] (rope applied) */
    float *k;        /* GQA: [ctx][n_head_kv*head_dim] (rope applied) */
    float *v;        /* GQA: [ctx][n_head_kv*head_dim] */
    /* LINEAR (Gated DeltaNet): fixed-size recurrent state, ctx-independent.
     * `state` is the per-v-head outer-product accumulator and `conv_win` the
     * last conv_kernel-1 raw channel rows the causal conv still needs. Carrying
     * both is what makes one-shot prefill identical to token-by-token decode —
     * the oracle proves that equality (ds4x_ref.py --arch qwen3next). */
    float *state;    /* [lin_v_heads][lin_k_dim][lin_v_dim] */
    float *conv_win; /* [conv_kernel-1][k_dim*2 + v_dim] */
    /* Optional VRAM-resident mirror of `state` (ds4x_cuda_gdn *), created by
     * ds4x_runner_create when the GPU path is on. When set it is AUTHORITATIVE:
     * the recurrence runs on the device and the host `state` above is left
     * behind (still allocated, and still advanced under
     * IDLETOKEN_DS4X_GDN_CHECK=1, which runs both and diffs them). */
    void  *dev_state;
} ds4x_kv_cache;

/* Run layer `il` over `n_tokens` starting at absolute position pos0, reading
 * hidden states x [n_tokens][n_embd] and writing out (same shape; may alias
 * x). Appends positions pos0..pos0+n_tokens-1 to the cache and attends over
 * 0..pos causally. Returns 0, -1 on bad arguments. */
int ds4x_layer_forward_cpu(const ds4x_config *cfg, uint32_t il,
                           const ds4x_layer_weights *w,
                           const float *x, uint32_t n_tokens, uint32_t pos0,
                           ds4x_kv_cache *cache, float *out);

/* Why the last ds4x_layer_forward_cpu() returned -1. Valid until the next call.
 * The distinction that matters in the field is "a cache slot / shape is wrong"
 * (config or plan bug) vs "malloc failed" (the box is out of memory and the
 * model is fine) — a bare -1 cannot tell those apart without reading source. */
const char *ds4x_forward_last_error(void);

/* ---- model loading (Phase B.3) -----------------------------------------
 * Loads config + per-layer weights from a GGUF using llama.cpp tensor
 * naming (blk.N.attn_q_a.weight, ffn_gate_exps.weight, ...). v0.1 of the
 * loader is correctness-first: F32/F16 tensors only, dequantized/converted
 * into malloc'd fp32 (quant formats + mmap reuse land with the CUDA path).
 * [layer_lo, layer_hi) selects the PP shard; layers outside are not read —
 * works on the sparse partial GGUFs from weights.c. */
typedef struct ds4x_model ds4x_model;

ds4x_model *ds4x_model_load(const char *path, uint16_t layer_lo, uint16_t layer_hi,
                            char *err, size_t errlen);
const ds4x_config *ds4x_model_config(const ds4x_model *m);
/* Layer weights for absolute layer il (must be within [layer_lo, layer_hi)). */
const ds4x_layer_weights *ds4x_model_layer(const ds4x_model *m, uint32_t il);
void ds4x_model_free(ds4x_model *m);

/* Non-layer weights are loaded only where the PP shard needs them:
 * token_embd on the first stage (layer_lo==0), output_norm+output on the last
 * (layer_hi==n_layer). Both return -1 when the shard doesn't hold them. */
int ds4x_embed_tokens(const ds4x_model *m, const int32_t *tokens, uint32_t n,
                      float *out);                       /* out: [n][n_embd] */
int ds4x_output_logits(const ds4x_model *m, const float *hidden, float *logits);
                                                          /* logits: [n_vocab] */

/* ---- PP runner (Phase B.4) ---------------------------------------------
 * The inference contact surface a worker drives: owns this stage's KV caches
 * (one per local layer, sized to ctx) and runs the whole [layer_lo,layer_hi)
 * range over hidden states. Stage 0 feeds embeddings in; the last stage reads
 * logits out; middle stages just transform hidden→hidden (the tensor that
 * crosses the PP boundary). Mirrors ds4's cur_hc handoff, generic-config. */
typedef struct ds4x_runner ds4x_runner;

ds4x_runner *ds4x_runner_create(const ds4x_model *model, uint32_t ctx_size,
                                char *err, size_t errlen);
/* Transform hidden [n_tokens][n_embd] in place through this stage's layers,
 * appending KV at absolute positions pos0..pos0+n_tokens-1. Returns 0/-1. */
/* Why the last ds4x_runner_run() returned -1 (which layer, which attention
 * family, which cache slots were NULL). Valid until the next run call. */
const char *ds4x_runner_last_error(void);

int ds4x_runner_run(ds4x_runner *r, float *hidden, uint32_t n_tokens,
                    uint32_t pos0);
/* Clear the linear-attention recurrent state (new sequence). Called
 * automatically by ds4x_runner_run() when pos0 == 0; exposed for callers that
 * need to drop a sequence without running one. No-op for MLA/GQA-only models,
 * whose caches are position-indexed and self-overwriting. */
/* IDLETOKEN_DS4X_PROF=1 only: accumulated seconds inside linear layers, split
 * four ways — projections (matvec), causal conv + gates + L2 norm, the
 * delta-rule recurrence, and the post work (gated RMSNorm + out_proj). The
 * recurrence used to be timed together with conv and out_proj and so
 * over-reported itself; keep the buckets separate. All zero when profiling is
 * off. Any argument may be NULL. */
void ds4x_prof_report(double *proj_s, double *conv_s, double *rec_s, double *post_s);
/* Same split for the full-attention (GQA) layers: rope + per-head Q/K RMSNorm,
 * the attention itself, and the sigmoid output gate. Only the middle one is on
 * the device; the other two are host loops that looked too cheap to move. */
void ds4x_prof_gqa_report(double *rope_s, double *attn_s, double *gate_s);

/* The Gated DeltaNet delta-rule recurrence for one chunk, CPU reference —
 * exposed so the CUDA kernel is gated against the code that actually SHIPS
 * rather than a second transcription of the algorithm in the test
 * (src/tools/ds4x_cuda_test.c). Same contract as ds4x_cuda_gdn_run:
 *   cnv  : [n_tokens][k_heads*k_dim*2 + v_heads*v_dim], per token [q̂|k̂|v],
 *          q and k ALREADY L2-normalised per k-head
 *   bet/dec : [n_tokens][v_heads]  sigmoid(b) / exp(g)
 *   state: [v_heads][k_dim][v_dim], advanced in place
 *   core : [n_tokens][v_heads*v_dim], scaled by 1/sqrt(v_dim)
 * memv/updv are v_dim scratch buffers; pass NULL to have them allocated. */
void ds4x_gdn_recur_cpu(uint32_t n_tokens, uint32_t k_heads, uint32_t v_heads,
                        uint32_t k_dim, uint32_t v_dim, float *state,
                        const float *cnv, const float *bet, const float *dec,
                        float *core, float *memv, float *updv);

/* IDLETOKEN_DS4X_GDN_CHECK=1 only: worst per-element gap seen between the CPU and
 * GPU delta-rule recurrences, and how many chunks were compared. (0, 0) when
 * the check is off or the GPU path never ran. */
void ds4x_gdn_check_report(double *max_abs, uint64_t *chunks);

void ds4x_runner_reset(ds4x_runner *r);

void ds4x_runner_free(ds4x_runner *r);

#endif /* IDLETOKEN_DS4X_H */
