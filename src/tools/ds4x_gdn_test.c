/* ds4x_gdn_test.c — numeric alignment of the C Gated DeltaNet (Qwen3.5-style
 * hybrid) forward against the numpy reference.
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


/* GDN/hybrid cache sized for one layer. Linear layers need the fixed-size
 * recurrent state + conv window; full layers need the usual K/V. */
/* Zero a hybrid cache in place — same shapes as hybrid_cache() allocates. */
static void clear_cache(const ds4x_config *c, ds4x_kv_cache *cache, uint32_t n_tok) {
    const uint32_t conv_ch = c->lin_k_heads * c->lin_k_dim * 2
                           + c->lin_v_heads * c->lin_v_dim;
    if (cache->state)
        memset(cache->state, 0,
               (size_t)c->lin_v_heads * c->lin_k_dim * c->lin_v_dim * sizeof(float));
    if (cache->conv_win)
        memset(cache->conv_win, 0, (size_t)(c->conv_kernel - 1) * conv_ch * sizeof(float));
    const size_t kv = (size_t)n_tok * c->n_head_kv * c->head_dim;
    if (cache->k) memset(cache->k, 0, kv * sizeof(float));
    if (cache->v) memset(cache->v, 0, kv * sizeof(float));
}

static ds4x_kv_cache hybrid_cache(const ds4x_config *c, uint32_t n_tok, int linear) {
    ds4x_kv_cache cache = {0};
    if (linear) {
        const uint32_t conv_ch = c->lin_k_heads * c->lin_k_dim * 2
                               + c->lin_v_heads * c->lin_v_dim;
        cache.state    = calloc((size_t)c->lin_v_heads * c->lin_k_dim * c->lin_v_dim, sizeof(float));
        cache.conv_win = calloc((size_t)(c->conv_kernel - 1) * conv_ch, sizeof(float));
    } else {
        const size_t kv = (size_t)n_tok * c->n_head_kv * c->head_dim;
        cache.k = calloc(kv, sizeof(float));
        cache.v = calloc(kv, sizeof(float));
    }
    return cache;
}
static void free_cache(ds4x_kv_cache *c) {
    free(c->state); free(c->conv_win); free(c->k); free(c->v);
    memset(c, 0, sizeof(*c));
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "build/fixtures/ds4x_gdn.bin";
    if (load_bundle(path) != 0) return 1;

    /* config: n_layer, n_embd, n_head, n_head_kv, head_dim, lin_k_heads,
     * lin_v_heads, lin_k_dim, lin_v_dim, conv_kernel, ff_dense, rope_theta,
     * n_tokens, qk_norm, n_vocab, attn_out_gate, rope_dim_partial */
    const float *cv = rec("config");
    ds4x_config cfg = {0};
    cfg.arch        = DS4X_ARCH_QWEN3;
    cfg.attn_kind   = DS4X_ATTN_GQA;          /* default; layer_types overrides */
    cfg.n_layer     = (uint32_t)cv[0];
    cfg.n_embd      = (uint32_t)cv[1];
    cfg.n_head      = (uint32_t)cv[2];
    cfg.n_head_kv   = (uint32_t)cv[3];
    cfg.head_dim    = (uint32_t)cv[4];
    cfg.lin_k_heads = (uint32_t)cv[5];
    cfg.lin_v_heads = (uint32_t)cv[6];
    cfg.lin_k_dim   = (uint32_t)cv[7];
    cfg.lin_v_dim   = (uint32_t)cv[8];
    cfg.conv_kernel = (uint32_t)cv[9];
    cfg.n_ff_dense  = (uint32_t)cv[10];
    cfg.rope_theta  = cv[11];
    const uint32_t n_tok = (uint32_t)cv[12];
    cfg.qk_norm     = (uint8_t)cv[13];
    cfg.n_vocab     = (uint32_t)cv[14];
    /* qwen35 full-layer features: interleaved [q|gate] + partial rope. Without
     * these the fixture never exercised the pair that broke on real weights. */
    cfg.attn_out_gate    = (uint8_t)cv[15];
    cfg.rope_dim_partial = (uint32_t)cv[16];
    cfg.qk_nope_head_dim = 0;
    cfg.qk_rope_head_dim = cfg.head_dim;
    cfg.v_head_dim       = cfg.head_dim;
    cfg.n_expert = 0; cfg.n_dense_lead = 0;

    const float *lt = rec("layer_types");     /* 0 = linear, 1 = full */
    for (uint32_t i = 0; i < cfg.n_layer; i++)
        cfg.layer_types[i] = (lt[i] == 0.0f) ? DS4X_ATTN_LINEAR : DS4X_ATTN_GQA;

    printf("ds4x GDN hybrid alignment: %u layers (", cfg.n_layer);
    for (uint32_t i = 0; i < cfg.n_layer; i++)
        printf("%s%s", cfg.layer_types[i] == DS4X_ATTN_LINEAR ? "L" : "F",
               i + 1 < cfg.n_layer ? "" : "), ");
    printf("%u tokens, embd %u, lin %ux%u/%ux%u, conv %u\n",
           n_tok, cfg.n_embd, cfg.lin_k_heads, cfg.lin_k_dim,
           cfg.lin_v_heads, cfg.lin_v_dim, cfg.conv_kernel);

    const size_t total = (size_t)n_tok * cfg.n_embd;
    float *h = malloc(total * sizeof(float));
    memcpy(h, rec("input"), total * sizeof(float));

    #define WT(p) ((ds4x_wt){ (p), 0, NULL })
    for (uint32_t il = 0; il < cfg.n_layer; il++) {
        const int linear = cfg.layer_types[il] == DS4X_ATTN_LINEAR;
        ds4x_layer_weights w = {0};
        w.attn_norm = rec("blk.%u.attn_norm", il);
        if (linear) {
            w.in_proj_qkv = WT(rec("blk.%u.in_proj_qkv", il));
            w.in_proj_z   = WT(rec("blk.%u.in_proj_z", il));
            w.in_proj_b   = WT(rec("blk.%u.in_proj_b", il));
            w.in_proj_a   = WT(rec("blk.%u.in_proj_a", il));
            w.conv1d_w     = rec("blk.%u.conv1d_w", il);
            w.conv1d_b     = rec("blk.%u.conv1d_b", il);
            w.A_log        = rec("blk.%u.A_log", il);
            w.dt_bias      = rec("blk.%u.dt_bias", il);
            w.ssm_norm     = rec("blk.%u.ssm_norm", il);
            w.out_proj     = WT(rec("blk.%u.out_proj", il));
        } else {
            w.q_proj = WT(rec("blk.%u.q_proj", il));
            w.k_proj = WT(rec("blk.%u.k_proj", il));
            w.v_proj = WT(rec("blk.%u.v_proj", il));
            if (cfg.qk_norm) { w.q_norm = rec("blk.%u.q_norm", il); w.k_norm = rec("blk.%u.k_norm", il); }
            w.attn_out = WT(rec("blk.%u.attn_out", il));
        }
        w.ffn_norm = rec("blk.%u.ffn_norm", il);
        w.gate = WT(rec("blk.%u.gate", il));
        w.up   = WT(rec("blk.%u.up", il));
        w.down = WT(rec("blk.%u.down", il));

        ds4x_kv_cache cache = hybrid_cache(&cfg, n_tok, linear);
        if (ds4x_layer_forward_cpu(&cfg, il, &w, h, n_tok, 0, &cache, h) != 0) {
            fprintf(stderr, "layer %u (%s) forward failed\n", il, linear ? "linear" : "full");
            return 1;
        }
        free_cache(&cache);
    }

    const float *expect = rec("expect");
    double max_abs = 0;
    for (size_t i = 0; i < total; i++) {
        const double d = fabs((double)h[i] - (double)expect[i]);
        if (d > max_abs) max_abs = d;
    }
    printf("  one-shot prefill: max |Δ| = %.3e   (budget 1e-4)\n", max_abs);

    /* Incremental: feed the SAME layers one token at a time, carrying state.
     * This is the property the recurrence exists for — the oracle proves it
     * holds in numpy (1.9e-08); the C side must match it too. */
    float *hi = malloc(total * sizeof(float));
    memcpy(hi, rec("input"), total * sizeof(float));
    ds4x_kv_cache *caches = calloc(cfg.n_layer, sizeof(ds4x_kv_cache));
    for (uint32_t il = 0; il < cfg.n_layer; il++)
        caches[il] = hybrid_cache(&cfg, n_tok, cfg.layer_types[il] == DS4X_ATTN_LINEAR);
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t il = 0; il < cfg.n_layer; il++) {
            const int linear = cfg.layer_types[il] == DS4X_ATTN_LINEAR;
            ds4x_layer_weights w = {0};
            w.attn_norm = rec("blk.%u.attn_norm", il);
            if (linear) {
                w.in_proj_qkv = WT(rec("blk.%u.in_proj_qkv", il));
                w.in_proj_z   = WT(rec("blk.%u.in_proj_z", il));
                w.in_proj_b   = WT(rec("blk.%u.in_proj_b", il));
                w.in_proj_a   = WT(rec("blk.%u.in_proj_a", il));
                w.conv1d_w     = rec("blk.%u.conv1d_w", il);
                w.conv1d_b     = rec("blk.%u.conv1d_b", il);
                w.A_log        = rec("blk.%u.A_log", il);
                w.dt_bias      = rec("blk.%u.dt_bias", il);
                w.ssm_norm     = rec("blk.%u.ssm_norm", il);
                w.out_proj     = WT(rec("blk.%u.out_proj", il));
            } else {
                w.q_proj = WT(rec("blk.%u.q_proj", il));
                w.k_proj = WT(rec("blk.%u.k_proj", il));
                w.v_proj = WT(rec("blk.%u.v_proj", il));
                if (cfg.qk_norm) { w.q_norm = rec("blk.%u.q_norm", il); w.k_norm = rec("blk.%u.k_norm", il); }
                w.attn_out = WT(rec("blk.%u.attn_out", il));
            }
            w.ffn_norm = rec("blk.%u.ffn_norm", il);
            w.gate = WT(rec("blk.%u.gate", il));
            w.up   = WT(rec("blk.%u.up", il));
            w.down = WT(rec("blk.%u.down", il));
            if (ds4x_layer_forward_cpu(&cfg, il, &w, hi + (size_t)t * cfg.n_embd,
                                       1, t, &caches[il],
                                       hi + (size_t)t * cfg.n_embd) != 0) {
                fprintf(stderr, "incremental layer %u at t=%u failed\n", il, t);
                return 1;
            }
        }
    }
    double inc_max = 0;
    for (size_t i = 0; i < total; i++) {
        const double d = fabs((double)hi[i] - (double)h[i]);
        if (d > inc_max) inc_max = d;
    }
    printf("  incremental:      max |Δ| vs prefill = %.3e (state-carry proof)\n", inc_max);

    /* Channel 3: a SECOND sequence must not inherit the first one's state.
     * The recurrent state has no position index (unlike a KV cache, which a new
     * sequence simply overwrites), so replaying the same tokens on the dirty
     * caches gives DIFFERENT output — that is what ds4x_runner_reset() exists
     * for. Asserts both halves: dirty MUST differ, cleared MUST be identical.
     * Real symptom before the fix (3-node cluster, 2026-07-28): the same prompt
     * at temperature 0 returned different tokens on the 2nd request. */
    double dirty_max = 0, clean_max = 0;
    float *h2 = (float *)malloc(total * sizeof(float));
    for (int pass = 0; pass < 2; pass++) {
        if (pass == 1)
            for (uint32_t il = 0; il < cfg.n_layer; il++)
                clear_cache(&cfg, &caches[il], n_tok);
        memcpy(h2, rec("input"), total * sizeof(float));
        for (uint32_t il = 0; il < cfg.n_layer; il++) {
            const int linear = cfg.layer_types[il] == DS4X_ATTN_LINEAR;
            ds4x_layer_weights w = {0};
            w.attn_norm = rec("blk.%u.attn_norm", il);
            if (linear) {
                w.in_proj_qkv = WT(rec("blk.%u.in_proj_qkv", il));
                w.in_proj_z   = WT(rec("blk.%u.in_proj_z", il));
                w.in_proj_b   = WT(rec("blk.%u.in_proj_b", il));
                w.in_proj_a   = WT(rec("blk.%u.in_proj_a", il));
                w.conv1d_w    = rec("blk.%u.conv1d_w", il);
                w.conv1d_b    = rec("blk.%u.conv1d_b", il);
                w.A_log       = rec("blk.%u.A_log", il);
                w.dt_bias     = rec("blk.%u.dt_bias", il);
                w.ssm_norm    = rec("blk.%u.ssm_norm", il);
                w.out_proj    = WT(rec("blk.%u.out_proj", il));
            } else {
                w.q_proj = WT(rec("blk.%u.q_proj", il));
                w.k_proj = WT(rec("blk.%u.k_proj", il));
                w.v_proj = WT(rec("blk.%u.v_proj", il));
                if (cfg.qk_norm) { w.q_norm = rec("blk.%u.q_norm", il); w.k_norm = rec("blk.%u.k_norm", il); }
                w.attn_out = WT(rec("blk.%u.attn_out", il));
            }
            w.ffn_norm = rec("blk.%u.ffn_norm", il);
            w.gate = WT(rec("blk.%u.gate", il));
            w.up   = WT(rec("blk.%u.up", il));
            w.down = WT(rec("blk.%u.down", il));
            if (ds4x_layer_forward_cpu(&cfg, il, &w, h2, n_tok, 0, &caches[il], h2) != 0) {
                fprintf(stderr, "replay layer %u failed\n", il);
                return 1;
            }
        }
        double m = 0;
        for (size_t i = 0; i < total; i++) {
            const double d = fabs((double)h2[i] - (double)h[i]);
            if (d > m) m = d;
        }
        if (pass == 0) dirty_max = m; else clean_max = m;
    }
    printf("  replay on dirty state: max |Δ| = %.3e (MUST be non-zero)\n", dirty_max);
    printf("  replay after reset:    max |Δ| = %.3e (must be 0)\n", clean_max);

    /* ---- the failure path must SAY something -------------------------------
     * A forward that returns a bare -1 costs a debugging round in the field
     * (the still-unexplained win-a transient, design doc §4e). Break a linear
     * layer's cache deliberately and require the reason to (a) come back
     * non-empty, (b) actually name the broken slot. Testing the happy path
     * only would let the diagnostic rot without anyone noticing. */
    int diag_ok = 0;
    {
        uint32_t lin = cfg.n_layer;
        for (uint32_t il = 0; il < cfg.n_layer; il++)
            if (cfg.layer_types[il] == DS4X_ATTN_LINEAR) { lin = il; break; }
        if (lin == cfg.n_layer) {
            printf("  diagnostic:     NO LINEAR LAYER — cannot check\n");
        } else {
            ds4x_kv_cache broken;
            memset(&broken, 0, sizeof broken);       /* every slot NULL */
            ds4x_layer_weights w;
            memset(&w, 0, sizeof w);
            w.attn_norm = rec("blk.%u.attn_norm", lin);
            float *tmp = calloc(total, sizeof(float));
            const int rc = ds4x_layer_forward_cpu(&cfg, lin, &w, tmp, n_tok, 0,
                                                  &broken, tmp);
            const char *why = ds4x_forward_last_error();
            diag_ok = (rc != 0) && strstr(why, "state=NULL") != NULL;
            printf("  diagnostic on broken cache: rc=%d, reason=\"%s\"%s\n",
                   rc, why, diag_ok ? "" : "   <-- must be non-zero and name the slot");
            free(tmp);
        }
    }
    free(h2);
    for (uint32_t il = 0; il < cfg.n_layer; il++) free_cache(&caches[il]);
    free(caches); free(hi); free(h);

    if (max_abs >= 1e-4 || inc_max >= 1e-4) { printf("DS4X_GDN_TEST_FAIL\n"); return 1; }
    /* dirty_max == 0 would mean the state does not actually influence the
     * output, i.e. this channel proves nothing — treat that as a failure too. */
    if (dirty_max == 0.0 || clean_max != 0.0) { printf("DS4X_GDN_TEST_FAIL\n"); return 1; }
    if (!diag_ok) { printf("DS4X_GDN_TEST_FAIL\n"); return 1; }
    printf("DS4X_GDN_TEST_OK\n");
    return 0;
}
