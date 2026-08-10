/* ds4x_config.c — GGUF-metadata-driven runtime config for the ds4x backend.
 * See include/idletoken_ds4x.h for the contract.
 *
 * Key naming follows llama.cpp's GGUF conventions ({arch}.block_count,
 * {arch}.attention.kv_lora_rank, ...). Where converters have historically
 * disagreed on a key's name, we try each candidate in order — the fixture
 * test pins the canonical spelling, and the real-GGUF spellings get verified
 * against unsloth/GLM-5.2-GGUF + Kimi K2.5 GGUF on the DGX (Phase B
 * checklist; docs/multi-model-design.md §7). */
#include "idletoken_ds4x.h"
#include "idletoken_gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* First present candidate key wins. Returns 0 / -1. */
static int meta_u32_any(const idletoken_gguf_meta *m, const char *const *keys,
                        uint32_t *out) {
    for (int i = 0; keys[i]; i++)
        if (idletoken_gguf_meta_u32(m, keys[i], out) == 0) return 0;
    return -1;
}

/* Build "{arch}{suffix}" into buf. */
static const char *akey(char *buf, size_t cap, const char *arch, const char *suffix) {
    snprintf(buf, cap, "%s%s", arch, suffix);
    return buf;
}

int ds4x_config_from_gguf(const char *path, ds4x_config *cfg,
                          char *err, size_t errlen) {
    memset(cfg, 0, sizeof(*cfg));
    idletoken_gguf_meta *m = idletoken_gguf_meta_open(path, err, errlen);
    if (!m) return -1;

#define FAIL(...) do { \
        if (err) snprintf(err, errlen, __VA_ARGS__); \
        idletoken_gguf_meta_close(m); \
        return -1; \
    } while (0)

    if (idletoken_gguf_meta_str(m, "general.architecture",
                             cfg->arch_name, sizeof(cfg->arch_name)) != 0)
        FAIL("GGUF has no general.architecture");

    if      (!strcmp(cfg->arch_name, "deepseek2")) cfg->arch = DS4X_ARCH_DEEPSEEK2;
    else if (!strcmp(cfg->arch_name, "glm_dsa") ||
             !strcmp(cfg->arch_name, "glm-dsa")) cfg->arch = DS4X_ARCH_GLM_DSA;
    else if (!strcmp(cfg->arch_name, "qwen3") ||
             !strcmp(cfg->arch_name, "qwen3moe")) cfg->arch = DS4X_ARCH_QWEN3;
    else if (!strcmp(cfg->arch_name, "llama")) cfg->arch = DS4X_ARCH_LLAMA;
    else if (!strcmp(cfg->arch_name, "qwen35") ||
             !strcmp(cfg->arch_name, "qwen35moe") ||
             !strcmp(cfg->arch_name, "qwen3next")) cfg->arch = DS4X_ARCH_QWEN35;
    else
        FAIL("architecture '%s' not supported by ds4x (MLA-MoE + GQA dense/MoE)",
             cfg->arch_name);
    const int is_gqa = (cfg->arch == DS4X_ARCH_QWEN3 || cfg->arch == DS4X_ARCH_LLAMA ||
                        cfg->arch == DS4X_ARCH_QWEN35);
    cfg->attn_kind = is_gqa ? DS4X_ATTN_GQA : DS4X_ATTN_MLA;

    const char *a = cfg->arch_name;
    char k[96];

    /* ---- required core dims -------------------------------------------- */
    if (idletoken_gguf_meta_u32(m, akey(k, sizeof(k), a, ".block_count"), &cfg->n_layer) != 0)
        FAIL("missing %s.block_count", a);
    if (idletoken_gguf_meta_u32(m, akey(k, sizeof(k), a, ".embedding_length"), &cfg->n_embd) != 0)
        FAIL("missing %s.embedding_length", a);
    if (idletoken_gguf_meta_u32(m, akey(k, sizeof(k), a, ".attention.head_count"), &cfg->n_head) != 0)
        FAIL("missing %s.attention.head_count", a);
    if (idletoken_gguf_meta_u32(m, akey(k, sizeof(k), a, ".attention.head_count_kv"), &cfg->n_head_kv) != 0)
        cfg->n_head_kv = cfg->n_head;   /* absent = MHA */

    /* ---- attention geometry (branch on family) ------------------------- */
    if (cfg->attn_kind == DS4X_ATTN_GQA) {
        /* Standard GQA: one head_dim, whole head gets rope, no latent KV. Map
         * onto the MLA scratch fields so the forward/loader shapes carry over:
         * nope=0, rope=head_dim, v=head_dim, no LoRA. */
        char k1[96];
        uint32_t head_dim = 0;
        if (idletoken_gguf_meta_u32(m, akey(k1, sizeof(k1), a, ".attention.key_length"), &head_dim) != 0 ||
            head_dim == 0) {
            if (cfg->n_head == 0) FAIL("missing %s.attention.head_count", a);
            head_dim = cfg->n_embd / cfg->n_head;   /* fallback when key_length absent */
        }
        cfg->head_dim         = head_dim;
        cfg->qk_nope_head_dim = 0;
        cfg->qk_rope_head_dim = head_dim;   /* full-head rotary */
        cfg->v_head_dim       = head_dim;
        cfg->kv_lora_rank     = 0;
        cfg->q_lora_rank      = 0;
        /* Qwen3/Qwen3.5 apply per-head Q/K RMSNorm; llama does not. */
        cfg->qk_norm          = (cfg->arch == DS4X_ARCH_QWEN3 ||
                                 cfg->arch == DS4X_ARCH_QWEN35) ? 1 : 0;

        if (cfg->arch == DS4X_ARCH_QWEN35) {
            /* Hybrid stack — keys verified against a real GGUF, see
             * docs/linear-attention-design.md §4b. */
            char kq[96];
            /* full attention layers use PARTIAL rope: rope.dimension_count
             * (64) < head_dim (256). Everything above rope_dim is untouched. */
            uint32_t rdim = 0;
            (void)idletoken_gguf_meta_u32(m, akey(kq, sizeof(kq), a, ".rope.dimension_count"), &rdim);
            cfg->rope_dim_partial = (rdim && rdim < head_dim) ? rdim : 0;
            cfg->attn_out_gate = 1;         /* attn_q emits [q | gate] */

            uint32_t interval = 0, inner = 0, state = 0, groups = 0, tsr = 0, ck = 0;
            (void)idletoken_gguf_meta_u32(m, akey(kq, sizeof(kq), a, ".full_attention_interval"), &interval);
            (void)idletoken_gguf_meta_u32(m, akey(kq, sizeof(kq), a, ".ssm.inner_size"), &inner);
            (void)idletoken_gguf_meta_u32(m, akey(kq, sizeof(kq), a, ".ssm.state_size"), &state);
            (void)idletoken_gguf_meta_u32(m, akey(kq, sizeof(kq), a, ".ssm.group_count"), &groups);
            (void)idletoken_gguf_meta_u32(m, akey(kq, sizeof(kq), a, ".ssm.time_step_rank"), &tsr);
            (void)idletoken_gguf_meta_u32(m, akey(kq, sizeof(kq), a, ".ssm.conv_kernel"), &ck);
            if (!interval || !inner || !state || !groups || !tsr || !ck)
                FAIL("%s: missing ssm.* / full_attention_interval keys", a);
            cfg->lin_v_heads = tsr;                 /* ssm_a / dt_bias length   */
            cfg->lin_v_dim   = inner / tsr;         /* inner_size / v_heads     */
            cfg->lin_k_heads = groups;
            cfg->lin_k_dim   = state;               /* per-key-head state width */
            cfg->conv_kernel = ck;
            /* Layer types come from the interval, not an explicit list: the
             * LAST layer of each group is full attention (measured: full =
             * {3,7,11,...} for interval 4). */
            for (uint32_t i = 0; i < cfg->n_layer && i < DS4X_MAX_LAYERS; i++)
                cfg->layer_types[i] = ((i + 1) % interval == 0)
                                      ? DS4X_ATTN_GQA : DS4X_ATTN_LINEAR;
        }
    } else {
        if (idletoken_gguf_meta_u32(m, akey(k, sizeof(k), a, ".attention.kv_lora_rank"), &cfg->kv_lora_rank) != 0)
            FAIL("missing %s.attention.kv_lora_rank (MLA arch)", a);
        (void)idletoken_gguf_meta_u32(m, akey(k, sizeof(k), a, ".attention.q_lora_rank"), &cfg->q_lora_rank);
        /* head geometry: nope/rope split + value head */
        char k1[96], k2[96], k3[96], k4[96];
        const char *rope_keys[] = {
            akey(k1, sizeof(k1), a, ".rope.dimension_count"),
            akey(k2, sizeof(k2), a, ".attention.qk_rope_head_dim"),
            NULL,
        };
        if (meta_u32_any(m, rope_keys, &cfg->qk_rope_head_dim) != 0)
            FAIL("missing %s rope dimension", a);
        const char *nope_keys[] = {
            akey(k3, sizeof(k3), a, ".attention.qk_nope_head_dim"),
            NULL,
        };
        if (meta_u32_any(m, nope_keys, &cfg->qk_nope_head_dim) != 0) {
            /* nope+rope comes from key_length — but when the GGUF also carries
             * `key_length_mla`, THAT is the per-head decompressed width and
             * plain `key_length` describes the LATENT row instead
             * (GLM-5.2: key_length 576 = kv_lora 512 + rope 64, while
             * key_length_mla 256 = nope 192 + rope 64). Taking key_length here
             * yields nope=512 and a kv_b three times too big; the loader would
             * reject the shape, but with a message about tensor sizes rather
             * than about which key to trust — so prefer the _mla key. */
            uint32_t key_len = 0;
            if (idletoken_gguf_meta_u32(m, akey(k4, sizeof(k4), a, ".attention.key_length_mla"), &key_len) != 0)
                (void)idletoken_gguf_meta_u32(m, akey(k4, sizeof(k4), a, ".attention.key_length"), &key_len);
            if (key_len > cfg->qk_rope_head_dim)
                cfg->qk_nope_head_dim = key_len - cfg->qk_rope_head_dim;
            else
                FAIL("missing %s qk_nope_head_dim / key_length", a);
        }
        const char *v_keys[] = {
            akey(k1, sizeof(k1), a, ".attention.value_length_mla"),
            akey(k2, sizeof(k2), a, ".attention.v_head_dim"),
            akey(k3, sizeof(k3), a, ".attention.value_length"),
            NULL,
        };
        if (meta_u32_any(m, v_keys, &cfg->v_head_dim) != 0)
            FAIL("missing %s v_head_dim / value_length", a);
    }

    /* ---- FFN / MoE ------------------------------------------------------ */
    /* MLA-family GGUFs are always MoE (expert keys required). GQA models may
     * be dense (qwen3 8B/32B: no experts) or MoE (qwen3-30B-A3B). */
    (void)idletoken_gguf_meta_u32(m, akey(k, sizeof(k), a, ".expert_count"), &cfg->n_expert);
    if (cfg->attn_kind == DS4X_ATTN_MLA && cfg->n_expert == 0)
        FAIL("missing %s.expert_count (MLA arch)", a);
    if (cfg->n_expert > 0) {
        if (idletoken_gguf_meta_u32(m, akey(k, sizeof(k), a, ".expert_used_count"), &cfg->n_expert_used) != 0)
            FAIL("missing %s.expert_used_count", a);
        (void)idletoken_gguf_meta_u32(m, akey(k, sizeof(k), a, ".expert_shared_count"), &cfg->n_expert_shared);
        if (idletoken_gguf_meta_u32(m, akey(k, sizeof(k), a, ".expert_feed_forward_length"), &cfg->n_ff_exp) != 0)
            FAIL("missing %s.expert_feed_forward_length", a);
    }
    (void)idletoken_gguf_meta_u32(m, akey(k, sizeof(k), a, ".leading_dense_block_count"), &cfg->n_dense_lead);
    /* Dense FFN width is needed by the dense-lead blocks AND by every layer of
     * a fully-dense (n_expert==0) model. */
    if ((cfg->n_dense_lead > 0 || cfg->n_expert == 0) &&
        idletoken_gguf_meta_u32(m, akey(k, sizeof(k), a, ".feed_forward_length"), &cfg->n_ff_dense) != 0)
        FAIL("missing %s.feed_forward_length", a);

    /* router weighting (build_moe_ffn semantics). Defaults match llama.cpp:
     * softmax gating, no norm, no scale — but real deepseek2/glm GGUFs set
     * these keys (V3 lineage: sigmoid + norm + routed_scaling_factor). */
    uint32_t gate = DS4X_GATE_SOFTMAX;
    (void)idletoken_gguf_meta_u32(m, akey(k, sizeof(k), a, ".expert_gating_func"), &gate);
    cfg->gating_func = (uint8_t)(gate == DS4X_GATE_SIGMOID ? DS4X_GATE_SIGMOID
                                                           : DS4X_GATE_SOFTMAX);
    uint32_t norm = 0;
    (void)idletoken_gguf_meta_u32(m, akey(k, sizeof(k), a, ".expert_weights_norm"), &norm);
    cfg->expert_weights_norm = (uint8_t)(norm ? 1 : 0);
    cfg->expert_weights_scale = 1.0f;
    (void)idletoken_gguf_meta_f32(m, akey(k, sizeof(k), a, ".expert_weights_scale"), &cfg->expert_weights_scale);

    /* qwen35moe (Qwen3.5-35B-A3B): hybrid linear attention AND MoE. Two of its
     * router settings are not in the GGUF at all — llama.cpp hardcodes them per
     * architecture (build_moe_ffn with norm_w=true), so read them from the
     * architecture here rather than from a key that will never be present:
     *   - the top-k weights ARE renormalised (SUM_ROWS → CLAMP → DIV in the graph)
     *   - a shared expert exists whenever expert_shared_feed_forward_length is
     *     set; the GGUF has no expert_shared_count, and leaving it 0 would skip
     *     the shared expert silently (it contributes to every token). */
    if (cfg->arch == DS4X_ARCH_QWEN35 && cfg->n_expert > 0) {
        cfg->expert_weights_norm = 1;
        uint32_t sff = 0;
        if (idletoken_gguf_meta_u32(m, akey(k, sizeof(k), a,
                ".expert_shared_feed_forward_length"), &sff) == 0 && sff > 0) {
            if (sff % cfg->n_ff_exp)
                FAIL("%s: shared expert ff %u is not a multiple of expert ff %u",
                     a, sff, cfg->n_ff_exp);
            cfg->n_expert_shared = sff / cfg->n_ff_exp;
        }
    }

    /* ---- rope / context ------------------------------------------------- */
    if (idletoken_gguf_meta_f32(m, akey(k, sizeof(k), a, ".rope.freq_base"), &cfg->rope_theta) != 0)
        cfg->rope_theta = 10000.0f;
    (void)idletoken_gguf_meta_u32(m, akey(k, sizeof(k), a, ".context_length"), &cfg->n_ctx_train);

    /* ---- vocab ---------------------------------------------------------- */
    if (idletoken_gguf_meta_u32(m, akey(k, sizeof(k), a, ".vocab_size"), &cfg->n_vocab) != 0) {
        uint64_t n = 0;
        if (idletoken_gguf_meta_arr_len(m, "tokenizer.ggml.tokens", &n) != 0 || n == 0)
            FAIL("missing %s.vocab_size and tokenizer.ggml.tokens", a);
        cfg->n_vocab = (uint32_t)n;
    }

    /* ---- DSA indexer (glm_dsa; informational — v0.1 runs full MLA) ------ */
    if (cfg->arch == DS4X_ARCH_GLM_DSA) {
        (void)idletoken_gguf_meta_u32(m, akey(k, sizeof(k), a, ".index_topk"), &cfg->index_topk);
        (void)idletoken_gguf_meta_u32(m, akey(k, sizeof(k), a, ".index_topk_freq"), &cfg->index_group);
    }

    /* ---- sanity --------------------------------------------------------- */
    if (cfg->n_layer == 0 || cfg->n_layer > 512)
        FAIL("implausible layer count %u", cfg->n_layer);
    if (cfg->n_embd == 0 || cfg->n_embd % 64 != 0)
        FAIL("implausible embedding length %u", cfg->n_embd);
    if (cfg->n_expert > 0 &&
        (cfg->n_expert_used == 0 || cfg->n_expert_used > cfg->n_expert))
        FAIL("implausible top-k %u of %u experts", cfg->n_expert_used, cfg->n_expert);
    if (cfg->n_dense_lead >= cfg->n_layer)
        FAIL("dense lead %u >= layers %u", cfg->n_dense_lead, cfg->n_layer);
    if (cfg->attn_kind == DS4X_ATTN_GQA &&
        (cfg->head_dim == 0 || cfg->n_head_kv == 0 || cfg->n_head % cfg->n_head_kv != 0))
        FAIL("implausible GQA geometry: %u heads / %u kv-heads / head_dim %u",
             cfg->n_head, cfg->n_head_kv, cfg->head_dim);

    idletoken_gguf_meta_close(m);
    return 0;
#undef FAIL
}

int ds4x_config_check_spec(const ds4x_config *cfg, const idletoken_model_spec *spec,
                           char *err, size_t errlen) {
#define MISMATCH(field, got, want) do { \
        if (err) snprintf(err, errlen, \
                          "GGUF/manifest mismatch for %s: " field " %u vs %u — " \
                          "wrong file or stale manifest", \
                          spec->id, (unsigned)(got), (unsigned)(want)); \
        return -1; \
    } while (0)
    if (spec->n_layers && cfg->n_layer != spec->n_layers)
        MISMATCH("layers", cfg->n_layer, spec->n_layers);
    if (spec->n_embd && cfg->n_embd != spec->n_embd)
        MISMATCH("hidden", cfg->n_embd, spec->n_embd);
    if (spec->n_vocab && cfg->n_vocab != spec->n_vocab)
        MISMATCH("vocab", cfg->n_vocab, spec->n_vocab);
    return 0;
#undef MISMATCH
}

uint64_t ds4x_kv_bytes_per_token_layer(const ds4x_config *cfg) {
    if (cfg->attn_kind == DS4X_ATTN_GQA)
        /* K + V, each n_head_kv·head_dim elements, fp16 */
        return (uint64_t)2 * cfg->n_head_kv * cfg->head_dim * 2;
    return (uint64_t)(cfg->kv_lora_rank + cfg->qk_rope_head_dim) * 2;  /* MLA, fp16 */
}
