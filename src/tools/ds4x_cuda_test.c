/* ds4x_cuda_test.c — CUDA vs CPU matvec parity gate (small-model-design §S-C).
 *
 * The CPU dequant+matvec in ds4x_quant.c / ds4x_forward.c is the numeric
 * REFERENCE. This test builds random quantized weight blocks of every
 * supported type, runs the same matvec through both paths, and requires them
 * to agree. If CUDA is unavailable the test SKIPS loudly (exit 0) rather than
 * pretending to pass — a machine without a GPU must not look green.
 *
 * Tolerance: both sides accumulate fp32 in a different order (CPU is a serial
 * row dot; GPU is a strided block sum), so exact equality is not expected.
 * Budget is relative: max|Δ| / max|ref| < 1e-5.
 */
#include "idletoken_ds4x.h"
#include "idletoken_ds4x_cuda.h"
#include "idletoken_ds4x_quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks = 0, failures = 0;
static void ok(int cond, const char *what, double rel) {
    checks++;
    if (cond) printf("  [ok] %-28s rel|Δ| = %.3e\n", what, rel);
    else { failures++; printf("  [FAIL] %-26s rel|Δ| = %.3e\n", what, rel); }
}

static uint32_t rng_state = 12345u;
static uint32_t rnd(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return rng_state;
}
static float rndf(void) { return (float)((int)(rnd() % 2001) - 1000) / 1000.0f; }

/* CPU reference matvec over raw quantized bytes (same math as ds4x_forward). */
static void cpu_matvec(const uint8_t *W, uint32_t type, uint32_t n_out, uint32_t n_in,
                       const float *x, float *y, float *rowbuf) {
    const uint32_t bc = ds4x_type_block_count(type);
    const uint64_t bb = ds4x_type_block_bytes(type);
    for (uint32_t o = 0; o < n_out; o++) {
        const uint8_t *p = W + (size_t)o * (n_in / bc) * bb;
        ds4x_dequant_row(type, p, rowbuf, n_in, NULL, 0);
        double acc = 0.0;
        for (uint32_t i = 0; i < n_in; i++) acc += (double)rowbuf[i] * (double)x[i];
        y[o] = (float)acc;
    }
}

static void run_type_tol(uint32_t type, const char *name, uint32_t n_out, uint32_t n_in,
                         double tol) {
    const uint32_t bc = ds4x_type_block_count(type);
    const uint64_t bb = ds4x_type_block_bytes(type);
    if (bc == 0 || n_in % bc) { printf("  [skip] %s (geometry)\n", name); return; }

    const size_t wbytes = (size_t)n_out * (n_in / bc) * bb;
    uint8_t *W = (uint8_t *)malloc(wbytes);
    float *x = (float *)malloc((size_t)n_in * sizeof(float));
    float *y_cpu = (float *)malloc((size_t)n_out * sizeof(float));
    float *y_gpu = (float *)malloc((size_t)n_out * sizeof(float));
    float *rowbuf = (float *)malloc((size_t)n_in * sizeof(float));
    if (!W || !x || !y_cpu || !y_gpu || !rowbuf) { printf("  [FAIL] %s oom\n", name); failures++; checks++; goto done; }

    /* Random bytes make valid (if meaningless) quant blocks for every layout
     * here — exactly what we want: it exercises the full bit-unpacking space,
     * including the packed 6-bit scale/min splits real weights hit. */
    for (size_t i = 0; i < wbytes; i++) W[i] = (uint8_t)(rnd() & 0xFF);
    /* F32 random bytes could be NaN/Inf; fill those types as real floats. */
    if (type == 0) { float *f = (float *)W; for (size_t i = 0; i < wbytes / 4; i++) f[i] = rndf(); }
    if (type == 1) { /* F16: clamp exponent so we never make NaN/Inf */
        uint16_t *h = (uint16_t *)W;
        for (size_t i = 0; i < wbytes / 2; i++) h[i] = (uint16_t)(h[i] & 0x7BFFu);
    }
    for (uint32_t i = 0; i < n_in; i++) x[i] = rndf();

    cpu_matvec(W, type, n_out, n_in, x, y_cpu, rowbuf);

    ds4x_cuda_wt *dw = ds4x_cuda_upload(W, type, n_out, n_in);
    if (!dw) { printf("  [FAIL] %s upload: %s\n", name, ds4x_cuda_last_error()); failures++; checks++; goto done; }
    int rc = ds4x_cuda_matvec(dw, x, y_gpu);
    if (rc != 0) { printf("  [FAIL] %s matvec: %s\n", name, ds4x_cuda_last_error()); failures++; checks++; ds4x_cuda_free(dw); goto done; }
    ds4x_cuda_free(dw);

    double maxd = 0.0, maxref = 0.0;
    for (uint32_t o = 0; o < n_out; o++) {
        const double d = fabs((double)y_gpu[o] - (double)y_cpu[o]);
        if (d > maxd) maxd = d;
        if (fabs((double)y_cpu[o]) > maxref) maxref = fabs((double)y_cpu[o]);
    }
    const double rel = maxref > 0 ? maxd / maxref : maxd;
    ok(rel < tol, name, rel);

done:
    free(W); free(x); free(y_cpu); free(y_gpu); free(rowbuf);
}

static void run_type(uint32_t type, const char *name, uint32_t n_out, uint32_t n_in) {
    run_type_tol(type, name, n_out, n_in, 1e-5);
}

/* ---- batched matmul parity ----------------------------------------------
 * Two claims, and the second is the one that bites if it is wrong:
 *   1. the chunk result matches the CPU row-dot reference;
 *   2. a token's result does NOT depend on how many tokens shared the launch
 *      — one call with T tokens must be BIT-IDENTICAL to T calls of one token.
 * Without (2), one-shot prefill and token-by-token decode silently disagree,
 * and the disagreement only shows up as different generated text later. */
/* tol: allowed relative deviation from the CPU reference.
 * exact: whether chunk-vs-one-at-a-time must match BIT for bit. The tensor-core
 * path cannot (bf16 inputs, and cuBLAS picks an algorithm per M), so it is checked
 * against tol instead -- see ds4x_tc_wanted in idletoken_ds4x_cuda.h. */
static void run_matmul_tol(uint32_t type, const char *name, uint32_t n_out,
                           uint32_t n_in, uint32_t n_tok, double tol, int exact) {
    const uint32_t bc = ds4x_type_block_count(type);
    const uint64_t bb = ds4x_type_block_bytes(type);
    if (bc == 0 || n_in % bc) { printf("  [skip] %s (geometry)\n", name); return; }
    const size_t wbytes = (size_t)n_out * (n_in / bc) * bb;
    uint8_t *W = (uint8_t *)malloc(wbytes);
    float *X = (float *)malloc((size_t)n_tok * n_in * sizeof(float));
    float *Y_cpu = (float *)malloc((size_t)n_tok * n_out * sizeof(float));
    float *Y_gpu = (float *)malloc((size_t)n_tok * n_out * sizeof(float));
    float *Y_one = (float *)malloc((size_t)n_tok * n_out * sizeof(float));
    float *rowbuf = (float *)malloc((size_t)n_in * sizeof(float));
    ds4x_cuda_wt *dw = NULL;
    if (!W || !X || !Y_cpu || !Y_gpu || !Y_one || !rowbuf) {
        printf("  [FAIL] %s oom\n", name); failures++; checks++; goto done;
    }
    for (size_t i = 0; i < wbytes; i++) W[i] = (uint8_t)(rnd() & 0xFF);
    if (type == 0) { float *f = (float *)W; for (size_t i = 0; i < wbytes / 4; i++) f[i] = rndf(); }
    if (type == 1) { uint16_t *h = (uint16_t *)W;
                     for (size_t i = 0; i < wbytes / 2; i++) h[i] = (uint16_t)(h[i] & 0x7BFFu); }
    /* BF16 is [sign|8-bit exp|7-bit mantissa]; random bytes hit exp == 0xFF, i.e.
     * NaN/Inf, and every comparison downstream then reads "nan". Clearing bit 14
     * caps the exponent below 0xFF while leaving the full mantissa space. */
    if (type == 30) { uint16_t *b = (uint16_t *)W;
                      for (size_t i = 0; i < wbytes / 2; i++) b[i] = (uint16_t)(b[i] & 0xBFFFu); }
    for (size_t i = 0; i < (size_t)n_tok * n_in; i++) X[i] = rndf();

    for (uint32_t t = 0; t < n_tok; t++)
        cpu_matvec(W, type, n_out, n_in, X + (size_t)t * n_in,
                   Y_cpu + (size_t)t * n_out, rowbuf);

    dw = ds4x_cuda_upload(W, type, n_out, n_in);
    if (!dw) { printf("  [FAIL] %s upload: %s\n", name, ds4x_cuda_last_error()); failures++; checks++; goto done; }
    if (ds4x_cuda_matmul(dw, X, Y_gpu, n_tok) != 0) {
        printf("  [FAIL] %s matmul: %s\n", name, ds4x_cuda_last_error()); failures++; checks++; goto done;
    }
    for (uint32_t t = 0; t < n_tok; t++)
        if (ds4x_cuda_matmul(dw, X + (size_t)t * n_in, Y_one + (size_t)t * n_out, 1) != 0) {
            printf("  [FAIL] %s matmul(1): %s\n", name, ds4x_cuda_last_error()); failures++; checks++; goto done;
        }

    double md = 0, mr = 0;
    for (size_t i = 0; i < (size_t)n_tok * n_out; i++) {
        const double d = fabs((double)Y_gpu[i] - (double)Y_cpu[i]);
        if (d > md) md = d;
        if (fabs((double)Y_cpu[i]) > mr) mr = fabs((double)Y_cpu[i]);
    }
    { char t[80]; snprintf(t, sizeof t, "%s vs cpu", name);
      ok(mr > 0 ? md / mr < tol : md < tol, t, mr > 0 ? md / mr : md); }

    md = 0;
    size_t worst = 0;
    for (size_t i = 0; i < (size_t)n_tok * n_out; i++) {
        const double d = fabs((double)Y_gpu[i] - (double)Y_one[i]);
        if (d > md) { md = d; worst = i; }
    }
    if (getenv("DS4X_TEST_LOCATE") && md > 0)
        printf("      locate %s: worst at token %zu of %u (elem %zu): gpu=%g one=%g cpu=%g\n",
               name, worst / n_out, n_tok, worst % n_out,
               (double)Y_gpu[worst], (double)Y_one[worst], (double)Y_cpu[worst]);
    { char t[80]; snprintf(t, sizeof t, "%s chunk==1-by-1", name);
      if (exact) ok(md == 0.0, t, md);
      else ok(mr > 0 ? md / mr < tol : md < tol, t, mr > 0 ? md / mr : md); }

done:
    ds4x_cuda_free(dw);
    free(W); free(X); free(Y_cpu); free(Y_gpu); free(Y_one); free(rowbuf);
}

static void run_matmul(uint32_t type, const char *name, uint32_t n_out,
                       uint32_t n_in, uint32_t n_tok) {
    run_matmul_tol(type, name, n_out, n_in, n_tok, 1e-5, 1);
}

/* ---- Gated DeltaNet recurrence parity ----------------------------------
 * Reference is ds4x_gdn_recur_cpu() — the function the CPU forward itself
 * calls, not a copy of it — so a change to one side without the other cannot
 * pass. Three things are checked, and the last two matter most:
 *   1. one-shot chunk: GPU core output vs CPU;
 *   2. the resulting STATE, downloaded from VRAM — a kernel can produce the
 *      right outputs for one chunk and still corrupt the state it carries;
 *   3. token-by-token on the GPU == one-shot on the GPU. That is the decode ==
 *      prefill equality; without it, streaming diverges from the batch path
 *      only after the first token, where nothing is looking. */
/* ---- GQA attention parity ------------------------------------------------
 * Reference is the HOST path transcribed from ds4x_forward.c: scan for the max,
 * exponentiate, weight. The kernel uses online softmax instead, so this cannot
 * be an equality assertion -- but it is exactly the comparison that matters,
 * because the host path is what the model was validated against.
 *
 * The shapes below are Qwen3.5-4B's real ones (16 heads over 4 KV heads,
 * head_dim 256) plus a 1-token case, because decode and prefill take the same
 * entry point and a kernel that only works when n_tok > 1 would pass a
 * prefill-shaped gate and then produce garbage on the first generated token. */
static void attn_cpu_ref(const float *q, const float *k, const float *v, float *o,
                         uint32_t n_tok, uint32_t pos0, uint32_t n_head,
                         uint32_t n_head_kv, uint32_t hdim, uint32_t qstride,
                         float scale, float *scores, int gated) {
    const uint32_t kv_dim = n_head_kv * hdim, grp = n_head / n_head_kv;
    for (uint32_t t = 0; t < n_tok; t++) {
        const uint32_t pos = pos0 + t;
        const float *qt = q + (size_t)t * n_head * qstride;
        float *ht = o + (size_t)t * n_head * hdim;
        for (uint32_t hd = 0; hd < n_head; hd++) {
            const uint32_t kvh = hd / grp;
            const float *q_h = qt + (size_t)hd * qstride;
            float smax = -1e30f;
            for (uint32_t u = 0; u <= pos; u++) {
                const float *kk = k + (size_t)u * kv_dim + (size_t)kvh * hdim;
                double s = 0.0;
                for (uint32_t i = 0; i < hdim; i++) s += (double)q_h[i] * (double)kk[i];
                scores[u] = (float)(s * (double)scale);
                if (scores[u] > smax) smax = scores[u];
            }
            float ssum = 0.0f;
            for (uint32_t u = 0; u <= pos; u++) { scores[u] = expf(scores[u] - smax); ssum += scores[u]; }
            float *o_h = ht + (size_t)hd * hdim;
            memset(o_h, 0, hdim * sizeof(float));
            for (uint32_t u = 0; u <= pos; u++) {
                const float p = scores[u] / ssum;
                const float *vv = v + (size_t)u * kv_dim + (size_t)kvh * hdim;
                for (uint32_t i = 0; i < hdim; i++) o_h[i] += p * vv[i];
            }
            /* Gated Attention, transcribed from the host path in
             * ds4x_forward.c. The kernel applies it too, so leaving it out here
             * would make the gate assert an ungated result. */
            if (gated) {
                const float *g_h = q_h + hdim;
                for (uint32_t i = 0; i < hdim; i++)
                    o_h[i] *= 1.0f / (1.0f + expf(-g_h[i]));
            }
        }
    }
}

static void run_attn(uint32_t n_tok, uint32_t pos0, uint32_t n_head,
                     uint32_t n_head_kv, uint32_t hdim, int gated,
                     const char *name) {
    const uint32_t qstride = gated ? hdim * 2u : hdim;
    const uint32_t n_kv = pos0 + n_tok, kv_dim = n_head_kv * hdim;
    const float scale = 1.0f / sqrtf((float)hdim);
    float *q = (float *)malloc((size_t)n_tok * n_head * qstride * sizeof(float));
    float *k = (float *)malloc((size_t)n_kv * kv_dim * sizeof(float));
    float *v = (float *)malloc((size_t)n_kv * kv_dim * sizeof(float));
    float *o_cpu = (float *)malloc((size_t)n_tok * n_head * hdim * sizeof(float));
    float *o_gpu = (float *)malloc((size_t)n_tok * n_head * hdim * sizeof(float));
    float *scores = (float *)malloc((size_t)n_kv * sizeof(float));
    if (!q || !k || !v || !o_cpu || !o_gpu || !scores) {
        printf("  [FAIL] %s oom\n", name); failures++; checks++; goto done;
    }
    for (size_t i = 0; i < (size_t)n_tok * n_head * qstride; i++) q[i] = rndf();
    for (size_t i = 0; i < (size_t)n_kv * kv_dim; i++) { k[i] = rndf(); v[i] = rndf(); }

    attn_cpu_ref(q, k, v, o_cpu, n_tok, pos0, n_head, n_head_kv, hdim, qstride,
                 scale, scores, gated);
    if (ds4x_cuda_attn_gqa(q, k, v, o_gpu, n_tok, pos0, n_head, n_head_kv,
                           hdim, qstride, scale, gated) != 0) {
        printf("  [FAIL] %s run: %s\n", name, ds4x_cuda_last_error()); failures++; checks++; goto done;
    }
    double md = 0, mr = 0;
    for (size_t i = 0; i < (size_t)n_tok * n_head * hdim; i++) {
        const double d = fabs((double)o_gpu[i] - (double)o_cpu[i]);
        if (d > md) md = d;
        if (fabs((double)o_cpu[i]) > mr) mr = fabs((double)o_cpu[i]);
    }
    ok(mr > 0 ? md / mr < 1e-4 : md < 1e-4, name, mr > 0 ? md / mr : md);
done:
    free(q); free(k); free(v); free(o_cpu); free(o_gpu); free(scores);
}

static void run_gdn(uint32_t kh, uint32_t vh, uint32_t kdim, uint32_t vdim,
                    uint32_t n_tok, const char *name) {
    const uint32_t kd = kh * kdim, vd = vh * vdim, stride = kd * 2u + vd;
    float *cnv  = (float *)malloc((size_t)n_tok * stride * sizeof(float));
    float *bet  = (float *)malloc((size_t)n_tok * vh * sizeof(float));
    float *dec  = (float *)malloc((size_t)n_tok * vh * sizeof(float));
    float *c_cpu = (float *)malloc((size_t)n_tok * vd * sizeof(float));
    float *c_gpu = (float *)malloc((size_t)n_tok * vd * sizeof(float));
    float *c_inc = (float *)malloc((size_t)n_tok * vd * sizeof(float));
    const size_t sn = (size_t)vh * kdim * vdim;
    float *s_cpu = (float *)calloc(sn, sizeof(float));
    float *s_gpu = (float *)malloc(sn * sizeof(float));
    ds4x_cuda_gdn *g = NULL, *g2 = NULL;
    if (!cnv || !bet || !dec || !c_cpu || !c_gpu || !c_inc || !s_cpu || !s_gpu) {
        printf("  [FAIL] %s oom\n", name); failures++; checks++; goto done;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        float *base = cnv + (size_t)t * stride;
        for (uint32_t i = 0; i < stride; i++) base[i] = rndf();
        /* q,k arrive L2-normalised per head from the caller — mirror that so
         * the magnitudes (and therefore the fp32 cancellation) are realistic. */
        for (uint32_t h = 0; h < kh * 2; h++) {
            float *p = base + (size_t)h * kdim, ss = 0.0f;
            for (uint32_t i = 0; i < kdim; i++) ss += p[i] * p[i];
            const float inv = 1.0f / sqrtf(ss + 1e-6f);
            for (uint32_t i = 0; i < kdim; i++) p[i] *= inv;
        }
        for (uint32_t h = 0; h < vh; h++) {
            bet[(size_t)t * vh + h] = 0.5f + 0.5f * rndf();          /* sigmoid range */
            dec[(size_t)t * vh + h] = 0.90f + 0.05f * (rndf() + 1.0f); /* exp(g) < 1   */
        }
    }

    ds4x_gdn_recur_cpu(n_tok, kh, vh, kdim, vdim, s_cpu, cnv, bet, dec, c_cpu, NULL, NULL);

    g = ds4x_cuda_gdn_create(kh, vh, kdim, vdim);
    if (!g) { printf("  [FAIL] %s create: %s\n", name, ds4x_cuda_last_error()); failures++; checks++; goto done; }
    if (ds4x_cuda_gdn_run(g, n_tok, cnv, bet, dec, c_gpu) != 0 ||
        ds4x_cuda_gdn_get_state(g, s_gpu) != 0) {
        printf("  [FAIL] %s run: %s\n", name, ds4x_cuda_last_error()); failures++; checks++; goto done;
    }

    double md = 0, mr = 0;
    for (size_t i = 0; i < (size_t)n_tok * vd; i++) {
        const double d = fabs((double)c_gpu[i] - (double)c_cpu[i]);
        if (d > md) md = d;
        if (fabs((double)c_cpu[i]) > mr) mr = fabs((double)c_cpu[i]);
    }
    { char t[64]; snprintf(t, sizeof t, "%s out", name);
      ok(mr > 0 ? md / mr < 1e-5 : md < 1e-5, t, mr > 0 ? md / mr : md); }

    md = mr = 0;
    for (size_t i = 0; i < sn; i++) {
        const double d = fabs((double)s_gpu[i] - (double)s_cpu[i]);
        if (d > md) md = d;
        if (fabs((double)s_cpu[i]) > mr) mr = fabs((double)s_cpu[i]);
    }
    { char t[64]; snprintf(t, sizeof t, "%s state", name);
      ok(mr > 0 ? md / mr < 1e-5 : md < 1e-5, t, mr > 0 ? md / mr : md); }

    /* token-by-token on a fresh state must reproduce the one-shot chunk */
    g2 = ds4x_cuda_gdn_create(kh, vh, kdim, vdim);
    if (!g2) { printf("  [FAIL] %s create2: %s\n", name, ds4x_cuda_last_error()); failures++; checks++; goto done; }
    for (uint32_t t = 0; t < n_tok; t++)
        if (ds4x_cuda_gdn_run(g2, 1, cnv + (size_t)t * stride,
                              bet + (size_t)t * vh, dec + (size_t)t * vh,
                              c_inc + (size_t)t * vd) != 0) {
            printf("  [FAIL] %s incr: %s\n", name, ds4x_cuda_last_error()); failures++; checks++; goto done;
        }
    md = mr = 0;
    for (size_t i = 0; i < (size_t)n_tok * vd; i++) {
        const double d = fabs((double)c_inc[i] - (double)c_gpu[i]);
        if (d > md) md = d;
        if (fabs((double)c_gpu[i]) > mr) mr = fabs((double)c_gpu[i]);
    }
    { char t[64]; snprintf(t, sizeof t, "%s decode==prefill", name);
      ok(md == 0.0, t, mr > 0 ? md / mr : md); }

done:
    ds4x_cuda_gdn_free(g); ds4x_cuda_gdn_free(g2);
    free(cnv); free(bet); free(dec); free(c_cpu); free(c_gpu); free(c_inc);
    free(s_cpu); free(s_gpu);
}

int main(void) {
    if (!ds4x_cuda_available()) {
        printf("ds4x CUDA parity: NO CUDA DEVICE — skipping (not a pass)\n");
        printf("  reason: %s\n", ds4x_cuda_last_error());
        printf("DS4X_CUDA_TEST_SKIP\n");
        return 0;
    }
    printf("ds4x CUDA parity vs CPU reference on %s\n", ds4x_cuda_device_name());

    /* Shapes mirror a real Qwen3-8B layer (n_embd 4096, ff 12288, kv 1024). */
    run_type(0,  "F32  4096x4096",  512, 4096);
    run_type(1,  "F16  4096x4096",  512, 4096);
    run_type(8,  "Q8_0 4096x4096",  512, 4096);
    run_type(2,  "Q4_0 4096x4096",  512, 4096);
    /* Q4_K / Q5_K / Q6_K all take the integer-domain (DP4A) matvec by default, so
     * the exact assertions need the escape hatch; the shipped default is asserted
     * against a tolerance further down. Both halves, always — asserting only the
     * exact path leaves whichever one actually ships unmeasured. */
    setenv("IDLETOKEN_DS4X_NO_DP4A", "1", 1);
    run_type(12, "Q4_K 4096x4096",  512, 4096);
    run_type(13, "Q5_K 4096x4096",  512, 4096);
    run_type(14, "Q6_K 4096x4096",  512, 4096);
    unsetenv("IDLETOKEN_DS4X_NO_DP4A");
    /* sub-4-bit: what the big MoE builds are made of (GLM-5.2) */
    run_type(10, "Q2_K 4096x4096",  512, 4096);
    run_type(16, "IQ2_XXS 4096x4096", 512, 4096);
    /* non-square (FFN down: n_embd x ff) */
    setenv("IDLETOKEN_DS4X_NO_DP4A", "1", 1);
    run_type(12, "Q4_K 4096x12288", 256, 12288);
    unsetenv("IDLETOKEN_DS4X_NO_DP4A");

    /* NARROW rows (n_in <= 2048) select a different Q4_K kernel — one warp per
     * row instead of one block per row, with a different summation order. Every
     * case above has n_in 4096 or 12288, so without these two the kernel that
     * 1024-wide models (Qwen3.5-0.8B and friends) actually run would be
     * completely uncovered while the gate reported all green.
     *
     * That is not hypothetical: a kernel added one commit earlier passed this
     * gate 35/35 while never executing, because its shape predicate never
     * matched anything the gate builds. A gate that cannot fail is not a gate.
     * n_out = 2050 is deliberately not a multiple of MV_WARPS (4), so the
     * last block's masked-off warps are exercised too. */
    run_type(12, "Q4_K narrow 2050x1024", 2050, 1024);
    run_type(12, "Q4_K narrow  512x512",   512,  512);

    /* The integer-domain matvec (DP4A) is on by default and quantizes the
     * ACTIVATION to int8, so it cannot meet the 1e-5 budget the fp32 kernels do.
     * Same split as the batched cases: the exact kernels are asserted with the
     * escape hatch set, then the shipped default is asserted against a tolerance.
     * Everything above ran with DP4A available -- shapes with n_in <= 2048 take
     * the narrow kernel instead, which is why those still read 1e-7. */
    run_type_tol(12, "Q4_K 4096x4096 [dp4a]",  512, 4096, 2e-2);
    run_type_tol(12, "Q4_K 4096x12288 [dp4a]", 256, 12288, 2e-2);
    run_type_tol(13, "Q5_K 4096x4096 [dp4a]",  512, 4096, 2e-2);
    run_type_tol(14, "Q6_K 4096x4096 [dp4a]",  512, 4096, 2e-2);
    /* 2560 is the real width of the Qwen3.5-4B projections these two kernels were
     * written for (attn_qkv/ssm_out at Q5_K, the tied output head at Q6_K), and it
     * is deliberately NOT a power of two: nblk = 10, so the word-per-thread loop
     * ends mid-block for a 128-thread block and the tail predicate gets exercised.
     *
     * The odd-nblk Q6_K case is the one that matters most. A Q6_K quant block is
     * 210 B, so 32-bit loads into it fault unless every offset is even-aligned
     * only — with nblk = 3 the row stride is 630 and EVERY OTHER ROW starts 2 mod 4,
     * which is the alignment the 4096-wide cases (stride 3360, always 0 mod 4)
     * never produce. Without this case the kernel would be exercised at only one
     * of the two alignments it has to survive. */
    run_type_tol(13, "Q5_K 2560x2560 [dp4a]",  512, 2560, 2e-2);
    run_type_tol(14, "Q6_K 2560x2560 [dp4a]",  512, 2560, 2e-2);
    run_type_tol(14, "Q6_K odd-nblk 512x768 [dp4a]", 512, 768, 2e-2);

    /* Batched matmul. 13 tokens is deliberately not a multiple of the token
     * tile, so the masked tail lanes are exercised; 300 crosses the internal
     * per-launch token block (256).
     *
     * These run with the tensor-core path DISABLED, because what they assert is
     * bit-identity between a chunk and one-token-at-a-time, and only the
     * hand-written kernels can offer that. Leaving it on silently retargeted
     * every n_tok >= TC_MIN_TOKENS case at cuBLAS and turned the bit-identity
     * assertion into a 4.2e6 failure (2026-08-13) -- the assertion was right, the
     * path underneath it had changed. The tensor-core path is covered separately
     * below, against a tolerance. */
    setenv("IDLETOKEN_DS4X_NO_TC", "1", 1);
    setenv("IDLETOKEN_DS4X_NO_MMQ", "1", 1);
    run_matmul(12, "MM Q4_K 1024x1024 x13", 1024, 1024, 13);
    run_matmul(12, "MM Q4_K 1024x1024 x300", 1024, 1024, 300);
    run_matmul(14, "MM Q6_K  512x1024 x13",  512, 1024, 13);
    run_matmul( 8, "MM Q8_0  512x1024 x13",  512, 1024, 13);
    run_matmul( 0, "MM F32   512x1024 x5",   512, 1024,  5);
    run_matmul(12, "MM Q4_K  512x1024 x1",   512, 1024,  1);
    run_matmul(10, "MM Q2_K 1024x1024 x13", 1024, 1024, 13);
    run_matmul(16, "MM IQ2_XXS 1024x1024 x13", 1024, 1024, 13);
    /* Q5_K was missing here until 2026-08-13, and it is not an exotic type: in a
     * Q4_K_M Qwen3.5 checkpoint the 48 largest projections (attn_qkv and ssm_out
     * of every GDN layer) are Q5_K, and nsys put most of prefill's kernel time in
     * the generic path that serves them. The batched matmul for the model's
     * hottest quant had no parity assertion at all. */
    run_matmul(13, "MM Q5_K 1024x1024 x13", 1024, 1024, 13);
    run_matmul(13, "MM Q5_K 1024x1024 x300", 1024, 1024, 300);

    /* BF16 is the type that reaches the cuBLAS tensor-core path, which is ON by
     * default. Both halves need covering, and they need DIFFERENT assertions:
     * with the escape hatch set the hand-written kernel must still be bit-exact;
     * with tensor cores the inputs carry 8 mantissa bits and cuBLAS varies its
     * algorithm with M, so only a tolerance is meaningful. Testing just one of
     * them would leave the shipped default unmeasured. */
    run_matmul(30, "MM BF16 1024x1024 x13 [exact]", 1024, 1024, 13);
    run_matmul(30, "MM BF16 1024x1024 x300 [exact]", 1024, 1024, 300);

    /* Now the shipped default: cuBLAS on tensor cores. Inputs carry 8 mantissa
     * bits and cuBLAS varies its algorithm with M, so neither assertion can be
     * exact -- but both still have to hold to a tolerance, and the path has to be
     * exercised at all. Quantized types reach it only above TC_MIN_TOKENS (the
     * dequant-to-bf16 pass has to pay for itself), hence 300 rather than 13. */
    unsetenv("IDLETOKEN_DS4X_NO_TC");
    run_matmul_tol(30, "MM BF16 1024x1024 x13 [tc]", 1024, 1024, 13, 5e-2, 0);
    run_matmul_tol(30, "MM BF16 1024x1024 x300 [tc]", 1024, 1024, 300, 5e-2, 0);
    /* NO_MMQ is still set here on purpose. MMQ runs BEFORE the cuBLAS route and
     * takes every K-quant, so without it these three would exercise MMQ under a
     * name that says [tc] and the cuBLAS-for-quantized-weights path would have no
     * assertion at all — green, and testing something else. */
    run_matmul_tol(12, "MM Q4_K 1024x1024 x300 [tc]", 1024, 1024, 300, 5e-2, 0);
    run_matmul_tol(13, "MM Q5_K 1024x1024 x300 [tc]", 1024, 1024, 300, 5e-2, 0);
    run_matmul_tol(14, "MM Q6_K  512x1024 x300 [tc]",  512, 1024, 300, 5e-2, 0);

    /* MMQ: the shipped default for K-quant prefill. Two assertions, and the
     * SECOND one is the interesting one. Against the CPU it can only hold to a
     * tolerance (int8 activations, ~7 bits) like every other quantized path — but
     * unlike the cuBLAS route it must still be BIT-IDENTICAL between a chunk and
     * one token at a time, because k advances in a fixed order, each output's sum
     * lives in one thread, and a token's activation scale is computed from that
     * token's own 32 values. If that ever stops holding, prefill and decode have
     * begun to disagree and the only symptom would be different generated text.
     *
     * 13 tokens is smaller than one M tile (128) and 300 spans three of them;
     * 2560 is the real projection width of Qwen3.5-4B and gives an odd number of
     * K steps per row. Q6_K is here in its own right: its scale changes every 16
     * elements rather than every 32, which is the reason the kernel's flush is
     * half-unit sized at all. */
    unsetenv("IDLETOKEN_DS4X_NO_MMQ");
    /* MMQ has TWO inner loops over the same shared tiles -- DP4A on the ALU and
     * int8 tensor cores -- and both ship: the tensor-core one is the default,
     * the DP4A one is what runs when IDLETOKEN_DS4X_NO_MMA=1. Asserting only the
     * default would leave the fallback untested, which is the same trap the
     * [tc] block above documents. Both are run over the full case list.
     *
     * Both must ALSO stay bit-identical between a chunk and one token at a time:
     * the mma fragment layout partitions k the same fixed way for every M, so
     * that property survives the switch to tensor cores. If it ever stops
     * holding, prefill and decode have begun to disagree. */
    setenv("IDLETOKEN_DS4X_NO_MMA", "1", 1);
    run_matmul_tol(12, "MM Q4_K 1024x1024 x13 [mmq/dp4a]", 1024, 1024, 13, 5e-2, 1);
    run_matmul_tol(12, "MM Q4_K 1024x1024 x300 [mmq/dp4a]", 1024, 1024, 300, 5e-2, 1);
    run_matmul_tol(13, "MM Q5_K 1024x1024 x300 [mmq/dp4a]", 1024, 1024, 300, 5e-2, 1);
    run_matmul_tol(14, "MM Q6_K  512x1024 x300 [mmq/dp4a]",  512, 1024, 300, 5e-2, 1);
    run_matmul_tol(13, "MM Q5_K  512x2560 x300 [mmq/dp4a]",  512, 2560, 300, 5e-2, 1);
    run_matmul_tol(14, "MM Q6_K  512x2560 x130 [mmq/dp4a]",  512, 2560, 130, 5e-2, 1);
    unsetenv("IDLETOKEN_DS4X_NO_MMA");
    run_matmul_tol(12, "MM Q4_K 1024x1024 x13 [mmq/mma]", 1024, 1024, 13, 5e-2, 1);
    run_matmul_tol(12, "MM Q4_K 1024x1024 x300 [mmq/mma]", 1024, 1024, 300, 5e-2, 1);
    run_matmul_tol(13, "MM Q5_K 1024x1024 x300 [mmq/mma]", 1024, 1024, 300, 5e-2, 1);
    run_matmul_tol(14, "MM Q6_K  512x1024 x300 [mmq/mma]",  512, 1024, 300, 5e-2, 1);
    run_matmul_tol(13, "MM Q5_K  512x2560 x300 [mmq/mma]",  512, 2560, 300, 5e-2, 1);
    run_matmul_tol(14, "MM Q6_K  512x2560 x130 [mmq/mma]",  512, 2560, 130, 5e-2, 1);

    /* The K-slab Q4_K kernel is opt-in (it is scaffolding, and slower on its
     * own), so without this block it would be code nothing runs. It is
     * BIT-IDENTICAL to the kernel above by construction -- same integer dot
     * order, and folding the min's negation into the scale turns `a - b` into
     * `a + (-b)`, which is exact in IEEE -- so it is asserted exactly, not to a
     * tolerance. If that ever stops holding, the restructure changed arithmetic
     * it was not supposed to. */
    setenv("IDLETOKEN_DS4X_K256", "1", 1);
    run_matmul_tol(12, "MM Q4_K 1024x1024 x13 [k256]", 1024, 1024, 13, 5e-2, 1);
    run_matmul_tol(12, "MM Q4_K 1024x1024 x300 [k256]", 1024, 1024, 300, 5e-2, 1);
    run_matmul_tol(12, "MM Q4_K  512x2560 x130 [k256]",  512, 2560, 130, 5e-2, 1);
    unsetenv("IDLETOKEN_DS4X_K256");

    /* Gated DeltaNet recurrence. First shape is Qwen3.5-0.8B exactly
     * (16 k-heads = 16 v-heads, 128/128); the second has v_heads > k_heads,
     * the linear-path GQA sharing that Qwen3.5-27B uses (16 k : 32 v) and that
     * the 0.8B shape would never exercise. */
    /* GQA attention. First shape is Qwen3.5-4B exactly (16 heads over 4 KV
     * heads, head_dim 256, Gated Attention so qstride is 2*hdim); the rest cover
     * what the first one cannot. pos0 > 0 is the second and later prefill chunks
     * AND every decode step — token t must attend to history it did not write.
     * n_tok == 1 is decode itself. MHA (grp == 1) and a non-256 head_dim keep
     * the lane-per-dim-group indexing honest for other models. */
    run_attn(130,   0, 16, 4, 256, 1, "ATTN 16h/4kv d256 x130");
    run_attn( 64, 200, 16, 4, 256, 1, "ATTN 16h/4kv d256 x64 pos0=200");
    run_attn(  1, 511, 16, 4, 256, 1, "ATTN 16h/4kv d256 x1 pos0=511 [decode]");
    run_attn( 33,   0,  8, 8, 128, 0, "ATTN 8h/8kv d128 x33 [mha, ungated]");
    run_attn( 17,   5,  4, 2,  64, 0, "ATTN 4h/2kv d64 x17 pos0=5");
    /* A long history, because that is where this kernel's cost actually lives:
     * at 3210 tokens it is 34% of all GPU time (nsys, 2026-08-13). Whatever
     * replaces it has to be correct here, not only at gate-sized lengths. */
    run_attn( 64, 3146, 16, 4, 256, 1, "ATTN 16h/4kv d256 x64 pos0=3146");

    run_gdn(16, 16, 128, 128, 24, "GDN 16k/16v 128x128");
    run_gdn(16, 32, 128, 128,  7, "GDN 16k/32v 128x128");
    run_gdn( 4,  4,  64, 192,  3, "GDN 4k/4v 64x192");

    printf("\n%d checks, %d failures (%.1f MB resident at peak)\n",
           checks, failures, (double)ds4x_cuda_bytes_resident() / 1048576.0);
    if (failures) { printf("DS4X_CUDA_TEST_FAIL\n"); return 1; }
    printf("DS4X_CUDA_TEST_OK\n");
    return 0;
}
