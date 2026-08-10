/* ds4x_forward_test.c — numeric alignment of the C CPU forward against the
 * numpy reference (Phase B.2 oracle; scripts/ds4x_ref.py writes the bundle).
 *   make ds4xtest
 * PASS = max |C − numpy| over all hidden elements < 1e-4 (both sides fp32,
 * same op order). This is the gate BEFORE any CUDA work. */
#include "idletoken_ds4x.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- bundle loader ------------------------------------------------------ */
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

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "build/fixtures/ds4x_vectors.bin";
    if (load_bundle(path) != 0) return 1;

    const float *cv = rec("config");
    ds4x_config cfg = {0};
    cfg.arch = DS4X_ARCH_GLM_DSA;    /* shape family only; arch not used here */
    cfg.n_layer          = (uint32_t)cv[0];
    cfg.n_embd           = (uint32_t)cv[1];
    cfg.n_head           = (uint32_t)cv[2];
    cfg.kv_lora_rank     = (uint32_t)cv[3];
    cfg.q_lora_rank      = (uint32_t)cv[4];
    cfg.qk_nope_head_dim = (uint32_t)cv[5];
    cfg.qk_rope_head_dim = (uint32_t)cv[6];
    cfg.v_head_dim       = (uint32_t)cv[7];
    cfg.n_expert         = (uint32_t)cv[8];
    cfg.n_expert_used    = (uint32_t)cv[9];
    cfg.n_expert_shared  = (uint32_t)cv[10];
    cfg.n_ff_exp         = (uint32_t)cv[11];
    cfg.n_ff_dense       = (uint32_t)cv[12];
    cfg.n_dense_lead     = (uint32_t)cv[13];
    cfg.rope_theta       = cv[14];
    const uint32_t n_tok = (uint32_t)cv[15];
    cfg.gating_func         = (uint8_t)cv[16];
    cfg.expert_weights_norm = (uint8_t)cv[17];
    cfg.expert_weights_scale = cv[18];

    printf("ds4x forward alignment: %u layers (%u dense lead), %u tokens, "
           "embd %u, heads %u, kv_rank %u\n",
           cfg.n_layer, cfg.n_dense_lead, n_tok, cfg.n_embd, cfg.n_head,
           cfg.kv_lora_rank);

    float *h = malloc((size_t)n_tok * cfg.n_embd * sizeof(float));
    memcpy(h, rec("input"), (size_t)n_tok * cfg.n_embd * sizeof(float));

    /* fixtures are plain fp32 → wrap matrix records as ds4x_wt{ptr, type 0} */
    #define WT(p) ((ds4x_wt){ (p), 0 })
    for (uint32_t il = 0; il < cfg.n_layer; il++) {
        ds4x_layer_weights w = {0};
        w.attn_norm = rec("blk.%u.attn_norm", il);
        if (cfg.q_lora_rank > 0) {
            w.q_a       = WT(rec("blk.%u.q_a", il));
            w.q_a_norm  = rec("blk.%u.q_a_norm", il);
            w.q_b       = WT(rec("blk.%u.q_b", il));
        } else {
            w.q_proj    = WT(rec("blk.%u.q_proj", il));
        }
        w.kv_a      = WT(rec("blk.%u.kv_a", il));
        w.kv_a_norm = rec("blk.%u.kv_a_norm", il);
        w.kv_b      = WT(rec("blk.%u.kv_b", il));
        w.attn_out  = WT(rec("blk.%u.attn_out", il));
        w.ffn_norm  = rec("blk.%u.ffn_norm", il);
        if (il < cfg.n_dense_lead) {
            w.gate = WT(rec("blk.%u.gate", il));
            w.up   = WT(rec("blk.%u.up", il));
            w.down = WT(rec("blk.%u.down", il));
        } else {
            w.router = WT(rec("blk.%u.router", il));
            w.e_score_bias = rec("blk.%u.e_score_bias", il);
            w.e_gate = WT(rec("blk.%u.e_gate", il));
            w.e_up   = WT(rec("blk.%u.e_up", il));
            w.e_down = WT(rec("blk.%u.e_down", il));
            /* A MoE need not have a shared expert (Qwen3-MoE has none); the
             * bundle then carries no s_* records at all. */
            if (cfg.n_expert_shared > 0) {
                w.s_gate = WT(rec("blk.%u.s_gate", il));
                w.s_up   = WT(rec("blk.%u.s_up", il));
                w.s_down = WT(rec("blk.%u.s_down", il));
            }
        }
        ds4x_kv_cache cache;
        cache.latent = calloc((size_t)n_tok * cfg.kv_lora_rank, sizeof(float));
        cache.k_rope = calloc((size_t)n_tok * cfg.qk_rope_head_dim, sizeof(float));
        if (ds4x_layer_forward_cpu(&cfg, il, &w, h, n_tok, 0, &cache, h) != 0) {
            fprintf(stderr, "layer %u forward failed\n", il);
            return 1;
        }
        free(cache.latent);
        free(cache.k_rope);
    }

    const float *expect = rec("expect");
    const size_t total = (size_t)n_tok * cfg.n_embd;
    double max_abs = 0, mean_abs = 0;
    for (size_t i = 0; i < total; i++) {
        const double d = fabs((double)h[i] - (double)expect[i]);
        if (d > max_abs) max_abs = d;
        mean_abs += d;
    }
    mean_abs /= (double)total;
    printf("  bundle weights: max |Δ| = %.3e   mean |Δ| = %.3e   (budget 1e-4)\n",
           max_abs, mean_abs);
    if (max_abs >= 1e-4) { printf("DS4X_FORWARD_TEST_FAIL\n"); return 1; }

    /* Optional second pass: same forward but with weights loaded from the
     * tiny GGUF via ds4x_model_load — exercises config-from-file, tensor
     * directory, llama.cpp naming and the f32 read path end-to-end. */
    if (argc > 2) {
        char err[256];
        ds4x_model *mdl = ds4x_model_load(argv[2], 0, 0, err, sizeof(err));
        if (!mdl) { fprintf(stderr, "model load: %s\n", err); return 1; }
        const ds4x_config *mc = ds4x_model_config(mdl);
        if (mc->n_layer != cfg.n_layer || mc->n_embd != cfg.n_embd) {
            fprintf(stderr, "GGUF config diverges from bundle config\n");
            return 1;
        }
        memcpy(h, rec("input"), total * sizeof(float));
        for (uint32_t il = 0; il < mc->n_layer; il++) {
            ds4x_kv_cache cache;
            cache.latent = calloc((size_t)n_tok * mc->kv_lora_rank, sizeof(float));
            cache.k_rope = calloc((size_t)n_tok * mc->qk_rope_head_dim, sizeof(float));
            if (ds4x_layer_forward_cpu(mc, il, ds4x_model_layer(mdl, il),
                                       h, n_tok, 0, &cache, h) != 0) {
                fprintf(stderr, "gguf-loaded layer %u forward failed\n", il);
                return 1;
            }
            free(cache.latent);
            free(cache.k_rope);
        }
        max_abs = 0;
        for (size_t i = 0; i < total; i++) {
            const double d = fabs((double)h[i] - (double)expect[i]);
            if (d > max_abs) max_abs = d;
        }
        printf("  gguf-loaded:    max |Δ| = %.3e   (budget 1e-4)\n", max_abs);

        /* ---- full pipeline: tokens → embed → runner → output → logits ----
         * Single stage holds the whole layer range; exercises the PP runner
         * (owned KV caches), the embedding gather and the output head — the
         * exact contact surface a worker drives. */
        const float *tok_f = rec("tokens");
        int32_t toks[64];
        for (uint32_t t = 0; t < n_tok && t < 64; t++) toks[t] = (int32_t)tok_f[t];

        float *he = malloc(total * sizeof(float));
        if (ds4x_embed_tokens(mdl, toks, n_tok, he) != 0) {
            fprintf(stderr, "embed_tokens failed\n"); return 1;
        }
        ds4x_runner *run = ds4x_runner_create(mdl, n_tok, err, sizeof(err));
        if (!run) { fprintf(stderr, "runner: %s\n", err); return 1; }
        if (ds4x_runner_run(run, he, n_tok, 0) != 0) {
            fprintf(stderr, "runner_run failed\n"); return 1;
        }
        float *logits = malloc((size_t)mc->n_vocab * sizeof(float));
        if (ds4x_output_logits(mdl, he + (size_t)(n_tok - 1) * mc->n_embd, logits) != 0) {
            fprintf(stderr, "output_logits failed\n"); return 1;
        }
        const float *exp_lg = rec("expect_logits");
        double lg_max = 0;
        for (uint32_t o = 0; o < mc->n_vocab; o++) {
            const double d = fabs((double)logits[o] - (double)exp_lg[o]);
            if (d > lg_max) lg_max = d;
        }
        printf("  pipeline logits: max |Δ| = %.3e over %u vocab (budget 1e-4)\n",
               lg_max, mc->n_vocab);
        ds4x_runner_free(run);

        /* ---- incremental decode: prefill a prefix, then decode the rest one
         * token at a time. Real generation drives the runner with n_tokens=1
         * at growing pos; the last token's logits must be BIT-IDENTICAL to the
         * all-at-once prefill (same KV cache contents, same math). This is the
         * KV-cache incremental-append correctness proof. */
        double inc_max = -1;
        if (n_tok >= 3) {
            const uint32_t pref = n_tok - 2;
            float *hi2 = malloc(total * sizeof(float));
            if (ds4x_embed_tokens(mdl, toks, n_tok, hi2) != 0) { fprintf(stderr, "inc embed\n"); return 1; }
            ds4x_runner *ri = ds4x_runner_create(mdl, n_tok, err, sizeof(err));
            if (!ri) { fprintf(stderr, "inc runner: %s\n", err); return 1; }
            /* prefill [0,pref) in one shot */
            if (ds4x_runner_run(ri, hi2, pref, 0) != 0) { fprintf(stderr, "inc prefill\n"); return 1; }
            /* decode remaining tokens one at a time */
            for (uint32_t p = pref; p < n_tok; p++) {
                if (ds4x_runner_run(ri, hi2 + (size_t)p * mc->n_embd, 1, p) != 0) {
                    fprintf(stderr, "inc decode at %u\n", p); return 1;
                }
            }
            float *ilg = malloc((size_t)mc->n_vocab * sizeof(float));
            if (ds4x_output_logits(mdl, hi2 + (size_t)(n_tok - 1) * mc->n_embd, ilg) != 0) {
                fprintf(stderr, "inc output\n"); return 1;
            }
            inc_max = 0;
            for (uint32_t o = 0; o < mc->n_vocab; o++) {
                const double d = fabs((double)ilg[o] - (double)logits[o]);
                if (d > inc_max) inc_max = d;
            }
            printf("  incr decode:    max |Δ| vs prefill = %.3e (must be ~0)\n", inc_max);
            free(hi2); free(ilg);
            ds4x_runner_free(ri);
        }

        /* ---- PP split: same model as two stages, hidden crosses the seam --
         * Stage A holds [0,mid) + the embedding; stage B holds [mid,n) + the
         * output head. The only thing that crosses is the hidden tensor —
         * exactly what the worker sends over TCP. Result must be identical. */
        double pp_max = -1;
        if (mc->n_layer >= 2) {
            const uint16_t mid = 1;
            ds4x_model *a = ds4x_model_load(argv[2], 0, mid, err, sizeof(err));
            ds4x_model *b = ds4x_model_load(argv[2], mid, (uint16_t)mc->n_layer, err, sizeof(err));
            if (!a || !b) { fprintf(stderr, "PP-split load failed: %s\n", err); return 1; }
            float *hp = malloc(total * sizeof(float));
            if (ds4x_embed_tokens(a, toks, n_tok, hp) != 0) { fprintf(stderr, "stageA embed\n"); return 1; }
            ds4x_runner *ra = ds4x_runner_create(a, n_tok, err, sizeof(err));
            ds4x_runner *rb = ds4x_runner_create(b, n_tok, err, sizeof(err));
            if (!ra || !rb) { fprintf(stderr, "PP runner: %s\n", err); return 1; }
            if (ds4x_runner_run(ra, hp, n_tok, 0) != 0 ||   /* stage A layers */
                ds4x_runner_run(rb, hp, n_tok, 0) != 0) {   /* stage B layers */
                fprintf(stderr, "PP-split run failed\n"); return 1;
            }
            float *lg2 = malloc((size_t)mc->n_vocab * sizeof(float));
            if (ds4x_output_logits(b, hp + (size_t)(n_tok - 1) * mc->n_embd, lg2) != 0) {
                fprintf(stderr, "stageB output\n"); return 1;
            }
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
        if (max_abs >= 1e-4 || lg_max >= 1e-4 || pp_max >= 1e-5 || inc_max >= 1e-5) {
            printf("DS4X_FORWARD_TEST_FAIL\n"); return 1;
        }
    }

    printf("DS4X_FORWARD_TEST_OK\n");
    return 0;
}
