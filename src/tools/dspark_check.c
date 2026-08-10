/* dspark_check.c — G-DSPARK-LOAD oracle (docs/dspark-design.md §8).
 *
 * Validates DeepSeek-V4-Flash-DSpark-support.gguf: the speculative-decoding
 * draft module that ships with the OFFICIAL 0731 release. This is the gate
 * that must pass BEFORE any forward pass is written, because it pins down
 * what the file actually contains — every later stage builds on that.
 *
 * Checks, in order:
 *   1. arch string + the dspark.* metadata block,
 *   2. all 81 tensors present with the expected shape and quant type,
 *   3. per-stage byte accounting, self-checked against the file size.
 *
 * (3) is the one that cannot lie. Tensor sizes are not stored in GGUF — they
 * are the gaps between consecutive data offsets — so summing them and
 * requiring `data_offset + Σ == file size` catches a misparsed header, a
 * truncated download, and a silently re-quantized upload alike.
 *
 * Shapes are derived from the model registry (n_embd / n_vocab / expert count)
 * and the file's own dspark.* metadata, not from magic numbers, so a future
 * DSpark for a different model does not need a second copy of this logic.
 *
 * Contract: last line DSPARK_LOAD_OK or DSPARK_LOAD_FAIL.
 * Usage: dspark_check <path to DSpark gguf>
 */
#include "idletoken_gguf.h"
#include "idletoken_model.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* GGML type ids we expect to see in this file. */
#define TY_F32     0u
#define TY_F16     1u
#define TY_Q8_0    8u
#define TY_Q2_K   10u
#define TY_IQ2_XXS 16u

static int g_fail;

static const char *type_name(uint32_t t) {
    switch (t) {
    case TY_F32:     return "F32";
    case TY_F16:     return "F16";
    case TY_Q8_0:    return "Q8_0";
    case TY_Q2_K:    return "Q2_K";
    case TY_IQ2_XXS: return "IQ2_XXS";
    default:         return "?";
    }
}

static void fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("  FAIL: ", stdout);
    vprintf(fmt, ap);
    fputc('\n', stdout);
    va_end(ap);
    g_fail++;
}

/* Expect a tensor with the given dims (0-terminated list) and type.
 * `want_type` of UINT32_MAX means "any type" (unused today, kept because the
 * hc_* tensors moved F32 -> F16 between the MTP module and this one and may
 * move again). */
static void expect(const idletoken_gguf_meta *m, const char *name,
                   uint32_t want_type, uint32_t ndim,
                   uint64_t d0, uint64_t d1, uint64_t d2) {
    idletoken_gguf_tensor t;
    if (idletoken_gguf_tensor_find(m, name, &t) != 0) {
        fail("missing tensor %s", name);
        return;
    }
    if (t.ndim != ndim) {
        fail("%s: ndim %u, expected %u", name, t.ndim, ndim);
        return;
    }
    const uint64_t want[3] = { d0, d1, d2 };
    for (uint32_t i = 0; i < ndim; i++) {
        if (t.dims[i] != want[i]) {
            fail("%s: dim[%u] = %" PRIu64 ", expected %" PRIu64,
                 name, i, t.dims[i], want[i]);
            return;
        }
    }
    if (want_type != UINT32_MAX && t.type != want_type) {
        fail("%s: type %s, expected %s", name, type_name(t.type),
             type_name(want_type));
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <DeepSeek-V4-Flash-DSpark-support.gguf>\n", argv[0]);
        puts("DSPARK_LOAD_FAIL");
        return 2;
    }
    const char *path = argv[1];

    char err[256] = {0};
    idletoken_gguf_meta *m = idletoken_gguf_meta_open(path, err, sizeof err);
    if (!m) {
        printf("  FAIL: cannot open %s: %s\n", path, err);
        puts("DSPARK_LOAD_FAIL");
        return 1;
    }

    const idletoken_model_spec *spec = idletoken_model_get("deepseek-v4-flash");
    if (!spec) {
        printf("  FAIL: deepseek-v4-flash missing from the registry\n");
        puts("DSPARK_LOAD_FAIL");
        return 1;
    }

    /* ---- 1. architecture + dspark.* metadata --------------------------- */
    char arch[64] = {0};
    if (idletoken_gguf_meta_str(m, "general.architecture", arch, sizeof arch) != 0 ||
        strcmp(arch, "deepseek4-dspark") != 0) {
        fail("general.architecture = '%s', expected 'deepseek4-dspark'", arch);
    }

    uint32_t block_size = 0, markov_rank = 0, noise_id = 0, stage_count = 0, n_layers = 0;
    struct { const char *key; uint32_t *out; uint32_t want; } kv[] = {
        { "dspark.block_size",     &block_size,  5      },
        { "dspark.markov_rank",    &markov_rank, 256    },
        { "dspark.noise_token_id", &noise_id,    128799 },
        { "dspark.stage_count",    &stage_count, 3      },
        { "dspark.n_layers",       &n_layers,    3      },
    };
    for (size_t i = 0; i < sizeof kv / sizeof kv[0]; i++) {
        if (idletoken_gguf_meta_u32(m, kv[i].key, kv[i].out) != 0) {
            fail("missing metadata key %s", kv[i].key);
        } else if (*kv[i].out != kv[i].want) {
            fail("%s = %u, expected %u", kv[i].key, *kv[i].out, kv[i].want);
        }
    }
    /* The noise token must be a real vocab entry: the drafter embeds it at
     * every non-anchor block position, so an out-of-range id would index past
     * token_embd and read garbage rather than error. */
    if (noise_id >= spec->n_vocab) {
        fail("dspark.noise_token_id %u is outside the vocab (%u)", noise_id, spec->n_vocab);
    }

    int32_t tgt[8] = {0};
    uint64_t ntgt_u = 0;
    int ntgt = 0;
    if (idletoken_gguf_meta_arr_len(m, "dspark.target_layer_ids", &ntgt_u) != 0) {
        fail("missing metadata key dspark.target_layer_ids");
    } else {
        ntgt = (int)ntgt_u;
        for (uint64_t i = 0; i < ntgt_u && i < 8; i++) {
            if (idletoken_gguf_meta_arr_i32(m, "dspark.target_layer_ids", i, &tgt[i]) != 0) {
                fail("cannot read dspark.target_layer_ids[%" PRIu64 "]", i);
            }
        }
    }
    if (ntgt != 3) {
        fail("dspark.target_layer_ids has %d entries, expected 3", ntgt);
    } else {
        for (int i = 0; i < 3; i++) {
            const int32_t want = (int32_t)spec->n_layers - 3 + i; /* 40,41,42 */
            if (tgt[i] != want) {
                fail("dspark.target_layer_ids[%d] = %d, expected %d", i, tgt[i], want);
            }
        }
    }
    if (stage_count != n_layers) {
        fail("stage_count (%u) != n_layers (%u)", stage_count, n_layers);
    }

    /* ---- 2. tensor inventory ------------------------------------------- */
    const uint64_t n_tensors = idletoken_gguf_meta_n_tensors(m);
    const uint32_t E   = spec->n_embd;              /* 4096  */
    const uint32_t V   = spec->n_vocab;             /* 129280 */
    const uint32_t HC  = spec->hc_streams;          /* 4 */
    const uint32_t NEXP = 256, FFEXP = 2048;        /* routed experts / expert FF */
    const uint32_t KV_DIM = 512, QLORA = 1024, NHEAD = 64;
    const uint32_t HCMIX = 2 * HC + HC * HC;        /* 24 */

    for (uint32_t s = 0; s < stage_count; s++) {
        char n[160];
#define T(suffix, ty, nd, a, b, c) \
        do { snprintf(n, sizeof n, "mtp.%u." suffix, s); \
             expect(m, n, (ty), (nd), (a), (b), (c)); } while (0)

        T("attn_norm.weight",      TY_F32,  1, E, 0, 0);
        T("attn_q_a.weight",       TY_Q8_0, 2, E, QLORA, 0);
        T("attn_q_a_norm.weight",  TY_F32,  1, QLORA, 0, 0);
        T("attn_q_b.weight",       TY_Q8_0, 2, QLORA, 32768, 0);
        T("attn_kv.weight",        TY_Q8_0, 2, E, KV_DIM, 0);
        T("attn_kv_a_norm.weight", TY_F32,  1, KV_DIM, 0, 0);
        T("attn_sinks.weight",     TY_F32,  1, NHEAD, 0, 0);
        T("attn_output_a.weight",  TY_Q8_0, 2, E, 8192, 0);
        T("attn_output_b.weight",  TY_Q8_0, 2, 8192, E, 0);

        T("ffn_norm.weight",       TY_F32,  1, E, 0, 0);
        /* Q8_0 here, not F32 — the MTP module used F32 and ds4's
         * tensor_expect_plain_layout only accepts F16/F32, so the DSpark
         * binder needs its own check rather than reusing that one. */
        T("ffn_gate_inp.weight",   TY_Q8_0, 2, E, NEXP, 0);
        T("exp_probs_b.bias",      TY_F32,  1, NEXP, 0, 0);
        T("ffn_gate_exps.weight",  TY_IQ2_XXS, 3, E, FFEXP, NEXP);
        T("ffn_up_exps.weight",    TY_IQ2_XXS, 3, E, FFEXP, NEXP);
        T("ffn_down_exps.weight",  TY_Q2_K,    3, FFEXP, E, NEXP);
        T("ffn_gate_shexp.weight", TY_Q8_0, 2, E, FFEXP, 0);
        T("ffn_up_shexp.weight",   TY_Q8_0, 2, E, FFEXP, 0);
        T("ffn_down_shexp.weight", TY_Q8_0, 2, FFEXP, E, 0);

        T("hc_attn_fn.weight",     TY_F16,  2, (uint64_t)E * HC, HCMIX, 0);
        T("hc_attn_scale.weight",  TY_F32,  1, 3, 0, 0);
        T("hc_attn_base.weight",   TY_F32,  1, HCMIX, 0, 0);
        T("hc_ffn_fn.weight",      TY_F16,  2, (uint64_t)E * HC, HCMIX, 0);
        T("hc_ffn_scale.weight",   TY_F32,  1, 3, 0, 0);
        T("hc_ffn_base.weight",    TY_F32,  1, HCMIX, 0, 0);
#undef T
    }

    /* Stage-unique tensors. main_* is the entry projection on stage 0; the
     * heads live on the LAST stage. main_proj is n_target_layers * n_embd
     * wide because the drafter's context is the CONCATENATION of the target
     * model's hidden states at layers 40/41/42 (vLLM's DSv4 DSpark:
     * main_x = main_norm(main_proj(concat(aux_hidden)))). */
    const uint32_t last = stage_count - 1;
    char n[160];
    snprintf(n, sizeof n, "mtp.0.main_norm.weight");
    expect(m, n, TY_F32, 1, E, 0, 0);
    snprintf(n, sizeof n, "mtp.0.main_proj.weight");
    expect(m, n, TY_Q8_0, 2, (uint64_t)E * (uint32_t)(ntgt > 0 ? ntgt : 3), E, 0);

    snprintf(n, sizeof n, "mtp.%u.norm.weight", last);
    expect(m, n, TY_F32, 1, E, 0, 0);
    snprintf(n, sizeof n, "mtp.%u.hc_head_fn.weight", last);
    expect(m, n, TY_F16, 2, (uint64_t)E * HC, HC, 0);
    snprintf(n, sizeof n, "mtp.%u.hc_head_scale.weight", last);
    expect(m, n, TY_F32, 1, 1, 0, 0);
    snprintf(n, sizeof n, "mtp.%u.hc_head_base.weight", last);
    expect(m, n, TY_F32, 1, HC, 0, 0);

    /* Markov head: w1 = Embedding(V, rank), w2 = Linear(rank, V) — both land
     * as [rank, V] in GGUF's innermost-first dim order. This is the sequential
     * module that gives the block intra-token dependency:
     *   bias = w2(w1[prev_token]);  logits[k] += bias. */
    snprintf(n, sizeof n, "mtp.%u.markov_head.markov_w1.weight", last);
    expect(m, n, TY_Q8_0, 2, markov_rank, V, 0);
    snprintf(n, sizeof n, "mtp.%u.markov_head.markov_w2.weight", last);
    expect(m, n, TY_Q8_0, 2, markov_rank, V, 0);

    /* Present in the checkpoint but NOT used at inference — vLLM's DSv4
     * DSpark drops these weights on load ("the confidence head is not wired
     * into inference yet"). Validate the shape so a future change is a
     * deliberate one, but do not require it to be consumed.
     * 4352 = n_embd + markov_rank: AcceptRatePredictor is a Linear(d+r, 1)
     * over [hidden ; markov_w1[prev_token]]. */
    snprintf(n, sizeof n, "mtp.%u.confidence_head.proj.weight", last);
    expect(m, n, TY_Q8_0, 2, E + markov_rank, 1, 0);

    const uint64_t expect_tensors = 24u * stage_count + 9u;
    if (n_tensors != expect_tensors) {
        fail("tensor count %" PRIu64 ", expected %" PRIu64,
             n_tensors, expect_tensors);
    }

    /* ---- 3. byte accounting -------------------------------------------- */
    /* GGUF does not record tensor sizes; a tensor's size is the gap to the
     * next data offset (and the last one runs to EOF). Summing them and
     * checking against the file size is therefore a real end-to-end parse
     * check, not a restatement of the header. */
    uint64_t *offs = calloc((size_t)n_tensors, sizeof *offs);
    uint32_t *stage_of = calloc((size_t)n_tensors, sizeof *stage_of);
    if (!offs || !stage_of) {
        printf("  FAIL: out of memory\n");
        puts("DSPARK_LOAD_FAIL");
        return 1;
    }
    for (uint64_t i = 0; i < n_tensors; i++) {
        idletoken_gguf_tensor t;
        if (idletoken_gguf_tensor_info(m, i, &t) != 0) {
            fail("cannot read tensor info at index %" PRIu64, i);
            continue;
        }
        offs[i] = t.offset;
        unsigned s = 0;
        if (sscanf(t.name, "mtp.%u.", &s) == 1 && s < stage_count) stage_of[i] = s;
        else fail("tensor %s is not under a known mtp.<stage> prefix", t.name);
    }

    FILE *f = fopen(path, "rb");
    if (!f) { printf("  FAIL: cannot reopen %s\n", path); puts("DSPARK_LOAD_FAIL"); return 1; }
    if (fseek(f, 0, SEEK_END) != 0) { printf("  FAIL: seek\n"); puts("DSPARK_LOAD_FAIL"); return 1; }
    const uint64_t file_size = (uint64_t)ftell(f);
    fclose(f);
    const uint64_t data_off = idletoken_gguf_data_offset(m);

    /* Sort indices by data offset so the gap to the next tensor is its size. */
    for (uint64_t i = 1; i < n_tensors; i++) {
        for (uint64_t j = i; j > 0 && offs[j - 1] > offs[j]; j--) {
            uint64_t to = offs[j - 1]; offs[j - 1] = offs[j]; offs[j] = to;
            uint32_t ts = stage_of[j - 1]; stage_of[j - 1] = stage_of[j]; stage_of[j] = ts;
        }
    }
    uint64_t per_stage[8] = {0}, total = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        const uint64_t end = (i + 1 < n_tensors) ? offs[i + 1] : (file_size - data_off);
        if (end < offs[i]) { fail("tensor data offsets are not monotonic"); break; }
        const uint64_t sz = end - offs[i];
        per_stage[stage_of[i] & 7u] += sz;
        total += sz;
    }

    printf("  file            : %s\n", path);
    printf("  arch            : %s   tensors=%" PRIu64 "  data_off=%" PRIu64 "\n",
           arch, n_tensors, data_off);
    printf("  block_size=%u  stages=%u  markov_rank=%u  noise_token=%u  targets=[%d,%d,%d]\n",
           block_size, stage_count, markov_rank, noise_id, tgt[0], tgt[1], tgt[2]);
    for (uint32_t s = 0; s < stage_count; s++) {
        printf("  stage mtp.%u     : %13" PRIu64 " B  (%.3f GiB)\n",
               s, per_stage[s], (double)per_stage[s] / (1024.0 * 1024 * 1024));
    }
    printf("  TOTAL           : %13" PRIu64 " B  (%.3f GiB)\n",
           total, (double)total / (1024.0 * 1024 * 1024));

    if (total + data_off != file_size) {
        fail("byte accounting: data_off %" PRIu64 " + Σtensors %" PRIu64
             " = %" PRIu64 ", but the file is %" PRIu64 " B",
             data_off, total, total + data_off, file_size);
    } else {
        printf("  self-check      : data_off + Σtensors == file size (%" PRIu64 " B) OK\n",
               file_size);
    }

    free(offs);
    free(stage_of);
    idletoken_gguf_meta_close(m);

    if (g_fail) {
        printf("%d problem(s)\n", g_fail);
        puts("DSPARK_LOAD_FAIL");
        return 1;
    }
    puts("DSPARK_LOAD_OK");
    return 0;
}
