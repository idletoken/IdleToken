/* ds4x_gqa_test.c — numeric alignment of the C GQA (qwen3) forward against the
 * numpy reference (small-model-design.md §8; scripts/ds4x_ref.py --arch qwen3
 * writes the bundle + a tiny qwen3 GGUF).
 *   make ds4xtest
 * PASS = max |C − numpy| < 1e-4 over hidden + logits; PP/incremental Δ ~0.
 * This is the Mac-verifiable gate for the GQA path BEFORE any CUDA work; real
 * Qwen3-8B alignment vs llama.cpp happens on the DGX (G-SMALL ladder). */
#include "idletoken_ds4x.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- bundle loader (same DS4XVEC1 format as ds4x_forward_test) --------- */
typedef struct { char name[128]; uint32_t ndim; uint64_t dims[4]; float *data; } vrec;
static vrec *g_recs; static uint32_t g_n;

static const float *rec(const char *fmt, ...) {
    char name[128];
    va_list ap; va_start(ap, fmt);
    vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    for (uint32_t i = 0; i < g_n; i++)
        if (!strcmp(g_recs[i].name, name)) return g_recs[i].data;
    fprintf(stderr, "missing record: %s\n", name);
    exit(1);
}

static int load_bundle(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); return -1; }
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "DS4XVEC1", 8) != 0) {
        fprintf(stderr, "bad bundle magic\n"); fclose(f); return -1;
    }
    uint32_t n;
    if (fread(&n, 4, 1, f) != 1) { fclose(f); return -1; }
    g_recs = calloc(n, sizeof(vrec));
    g_n = n;
    for (uint32_t i = 0; i < n; i++) {
        uint64_t nl;
        if (fread(&nl, 8, 1, f) != 1 || nl >= sizeof(g_recs[i].name)) goto bad;
        if (fread(g_recs[i].name, 1, (size_t)nl, f) != nl) goto bad;
        if (fread(&g_recs[i].ndim, 4, 1, f) != 1 || g_recs[i].ndim > 4) goto bad;
        uint64_t total = 1;
        for (uint32_t d = 0; d < g_recs[i].ndim; d++) {
            if (fread(&g_recs[i].dims[d], 8, 1, f) != 1) goto bad;
            total *= g_recs[i].dims[d];
        }
        g_recs[i].data = malloc((size_t)total * 4);
        if (!g_recs[i].data ||
            fread(g_recs[i].data, 4, (size_t)total, f) != total) goto bad;
    }
    fclose(f);
    return 0;
bad:
    fprintf(stderr, "truncated bundle at record %u\n", g_n);
    fclose(f);
    return -1;
}

/* GQA KV cache sized to n_tok. */
static ds4x_kv_cache gqa_cache(const ds4x_config *c, uint32_t n_tok) {
    ds4x_kv_cache cache = {0};
    const size_t kv = (size_t)n_tok * c->n_head_kv * c->head_dim;
    cache.k = calloc(kv, sizeof(float));
    cache.v = calloc(kv, sizeof(float));
    return cache;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "build/fixtures/ds4x_qwen3.bin";
    if (load_bundle(path) != 0) return 1;

    /* config layout (qwen_main): n_layer, n_embd, n_head, n_head_kv, head_dim,
     * ff_dense, rope_theta, n_tokens, qk_norm, n_vocab */
    const float *cv = rec("config");
    ds4x_config cfg = {0};
    cfg.arch      = DS4X_ARCH_QWEN3;
    cfg.attn_kind = DS4X_ATTN_GQA;
    cfg.n_layer   = (uint32_t)cv[0];
    cfg.n_embd    = (uint32_t)cv[1];
    cfg.n_head    = (uint32_t)cv[2];
    cfg.n_head_kv = (uint32_t)cv[3];
    cfg.head_dim  = (uint32_t)cv[4];
    cfg.n_ff_dense = (uint32_t)cv[5];
    cfg.rope_theta = cv[6];
    const uint32_t n_tok = (uint32_t)cv[7];
    cfg.qk_norm   = (uint8_t)cv[8];
    cfg.n_vocab   = (uint32_t)cv[9];
    /* GQA maps onto the MLA scratch fields: whole head_dim gets rope, no LoRA,
     * no experts (fully dense). */
    cfg.qk_nope_head_dim = 0;
    cfg.qk_rope_head_dim = cfg.head_dim;
    cfg.v_head_dim       = cfg.head_dim;
    cfg.kv_lora_rank     = 0;
    cfg.q_lora_rank      = 0;
    cfg.n_expert         = 0;
    cfg.n_dense_lead     = 0;

    printf("ds4x GQA forward alignment: %u layers, %u tokens, embd %u, "
           "heads %u/%u kv, head_dim %u, qk_norm %u\n",
           cfg.n_layer, n_tok, cfg.n_embd, cfg.n_head, cfg.n_head_kv,
           cfg.head_dim, cfg.qk_norm);

    const size_t total = (size_t)n_tok * cfg.n_embd;
    float *h = malloc(total * sizeof(float));
    memcpy(h, rec("input"), total * sizeof(float));

    #define WT(p) ((ds4x_wt){ (p), 0 })
    for (uint32_t il = 0; il < cfg.n_layer; il++) {
        ds4x_layer_weights w = {0};
        w.attn_norm = rec("blk.%u.attn_norm", il);
        w.q_proj = WT(rec("blk.%u.q_proj", il));
        w.k_proj = WT(rec("blk.%u.k_proj", il));
        w.v_proj = WT(rec("blk.%u.v_proj", il));
        if (cfg.qk_norm) {
            w.q_norm = rec("blk.%u.q_norm", il);
            w.k_norm = rec("blk.%u.k_norm", il);
        }
        w.attn_out = WT(rec("blk.%u.attn_out", il));
        w.ffn_norm = rec("blk.%u.ffn_norm", il);
        w.gate = WT(rec("blk.%u.gate", il));
        w.up   = WT(rec("blk.%u.up", il));
        w.down = WT(rec("blk.%u.down", il));
        ds4x_kv_cache cache = gqa_cache(&cfg, n_tok);
        if (ds4x_layer_forward_cpu(&cfg, il, &w, h, n_tok, 0, &cache, h) != 0) {
            fprintf(stderr, "layer %u forward failed\n", il); return 1;
        }
        free(cache.k); free(cache.v);
    }

    const float *expect = rec("expect");
    double max_abs = 0;
    for (size_t i = 0; i < total; i++) {
        const double d = fabs((double)h[i] - (double)expect[i]);
        if (d > max_abs) max_abs = d;
    }
    printf("  bundle weights: max |Δ| = %.3e   (budget 1e-4)\n", max_abs);
    if (max_abs >= 1e-4) { printf("DS4X_GQA_TEST_FAIL\n"); return 1; }

    /* ---- second pass: load the tiny qwen3 GGUF and drive the PP runner ---- */
    double lg_max = -1, inc_max = -1, pp_max = -1, gg_max = -1;
    if (argc > 2) {
        char err[256];
        ds4x_model *mdl = ds4x_model_load(argv[2], 0, 0, err, sizeof(err));
        if (!mdl) { fprintf(stderr, "model load: %s\n", err); return 1; }
        const ds4x_config *mc = ds4x_model_config(mdl);
        if (mc->attn_kind != DS4X_ATTN_GQA || mc->n_head_kv != cfg.n_head_kv ||
            mc->head_dim != cfg.head_dim) {
            fprintf(stderr, "GGUF config diverges from bundle (attn/kv/head_dim)\n"); return 1;
        }
        /* re-run layers from GGUF weights to check the loader/naming */
        memcpy(h, rec("input"), total * sizeof(float));
        for (uint32_t il = 0; il < mc->n_layer; il++) {
            ds4x_kv_cache cache = gqa_cache(mc, n_tok);
            if (ds4x_layer_forward_cpu(mc, il, ds4x_model_layer(mdl, il),
                                       h, n_tok, 0, &cache, h) != 0) {
                fprintf(stderr, "gguf layer %u failed\n", il); return 1;
            }
            free(cache.k); free(cache.v);
        }
        gg_max = 0;
        for (size_t i = 0; i < total; i++) {
            const double d = fabs((double)h[i] - (double)expect[i]);
            if (d > gg_max) gg_max = d;
        }
        printf("  gguf-loaded:    max |Δ| = %.3e   (budget 1e-4)\n", gg_max);

        /* full pipeline: tokens → embed → runner → output head → logits */
        const float *tok_f = rec("tokens");
        int32_t toks[64];
        for (uint32_t t = 0; t < n_tok && t < 64; t++) toks[t] = (int32_t)tok_f[t];
        float *he = malloc(total * sizeof(float));
        if (ds4x_embed_tokens(mdl, toks, n_tok, he) != 0) { fprintf(stderr, "embed\n"); return 1; }
        ds4x_runner *run = ds4x_runner_create(mdl, n_tok, err, sizeof(err));
        if (!run) { fprintf(stderr, "runner: %s\n", err); return 1; }
        if (ds4x_runner_run(run, he, n_tok, 0) != 0) { fprintf(stderr, "run\n"); return 1; }
        float *logits = malloc((size_t)mc->n_vocab * sizeof(float));
        if (ds4x_output_logits(mdl, he + (size_t)(n_tok - 1) * mc->n_embd, logits) != 0) {
            fprintf(stderr, "logits\n"); return 1;
        }
        const float *exp_lg = rec("expect_logits");
        lg_max = 0;
        for (uint32_t o = 0; o < mc->n_vocab; o++) {
            const double d = fabs((double)logits[o] - (double)exp_lg[o]);
            if (d > lg_max) lg_max = d;
        }
        printf("  pipeline logits: max |Δ| = %.3e over %u vocab (budget 1e-4)\n", lg_max, mc->n_vocab);
        ds4x_runner_free(run);

        /* incremental decode must be bit-identical to all-at-once prefill */
        if (n_tok >= 3) {
            const uint32_t pref = n_tok - 2;
            float *hi2 = malloc(total * sizeof(float));
            if (ds4x_embed_tokens(mdl, toks, n_tok, hi2) != 0) return 1;
            ds4x_runner *ri = ds4x_runner_create(mdl, n_tok, err, sizeof(err));
            if (!ri) { fprintf(stderr, "inc runner: %s\n", err); return 1; }
            if (ds4x_runner_run(ri, hi2, pref, 0) != 0) return 1;
            for (uint32_t pp = pref; pp < n_tok; pp++)
                if (ds4x_runner_run(ri, hi2 + (size_t)pp * mc->n_embd, 1, pp) != 0) return 1;
            float *ilg = malloc((size_t)mc->n_vocab * sizeof(float));
            if (ds4x_output_logits(mdl, hi2 + (size_t)(n_tok - 1) * mc->n_embd, ilg) != 0) return 1;
            inc_max = 0;
            for (uint32_t o = 0; o < mc->n_vocab; o++) {
                const double d = fabs((double)ilg[o] - (double)logits[o]);
                if (d > inc_max) inc_max = d;
            }
            printf("  incr decode:    max |Δ| vs prefill = %.3e (must be ~0)\n", inc_max);
            free(hi2); free(ilg); ds4x_runner_free(ri);
        }

        /* PP 2-stage split — only the hidden tensor crosses the seam */
        if (mc->n_layer >= 2) {
            const uint16_t mid = 1;
            ds4x_model *a = ds4x_model_load(argv[2], 0, mid, err, sizeof(err));
            ds4x_model *b = ds4x_model_load(argv[2], mid, (uint16_t)mc->n_layer, err, sizeof(err));
            if (!a || !b) { fprintf(stderr, "PP load: %s\n", err); return 1; }
            float *hp = malloc(total * sizeof(float));
            if (ds4x_embed_tokens(a, toks, n_tok, hp) != 0) return 1;
            ds4x_runner *ra = ds4x_runner_create(a, n_tok, err, sizeof(err));
            ds4x_runner *rb = ds4x_runner_create(b, n_tok, err, sizeof(err));
            if (!ra || !rb) { fprintf(stderr, "PP runner\n"); return 1; }
            if (ds4x_runner_run(ra, hp, n_tok, 0) != 0 ||
                ds4x_runner_run(rb, hp, n_tok, 0) != 0) return 1;
            float *lg2 = malloc((size_t)mc->n_vocab * sizeof(float));
            if (ds4x_output_logits(b, hp + (size_t)(n_tok - 1) * mc->n_embd, lg2) != 0) return 1;
            pp_max = 0;
            for (uint32_t o = 0; o < mc->n_vocab; o++) {
                const double d = fabs((double)lg2[o] - (double)logits[o]);
                if (d > pp_max) pp_max = d;
            }
            printf("  PP 2-stage:     max |Δ| vs single = %.3e (must be ~0)\n", pp_max);
            free(hp); free(lg2);
            ds4x_runner_free(ra); ds4x_runner_free(rb);
            ds4x_model_free(a); ds4x_model_free(b);
        }
        free(he); free(logits);
        ds4x_model_free(mdl);
        if (gg_max >= 1e-4 || lg_max >= 1e-4 || pp_max >= 1e-5 || inc_max >= 1e-5) {
            printf("DS4X_GQA_TEST_FAIL\n"); return 1;
        }
    }

    printf("DS4X_GQA_TEST_OK\n");
    return 0;
}
