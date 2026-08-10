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

static void run_type(uint32_t type, const char *name, uint32_t n_out, uint32_t n_in) {
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
    ok(rel < 1e-5, name, rel);

done:
    free(W); free(x); free(y_cpu); free(y_gpu); free(rowbuf);
}

/* ---- batched matmul parity ----------------------------------------------
 * Two claims, and the second is the one that bites if it is wrong:
 *   1. the chunk result matches the CPU row-dot reference;
 *   2. a token's result does NOT depend on how many tokens shared the launch
 *      — one call with T tokens must be BIT-IDENTICAL to T calls of one token.
 * Without (2), one-shot prefill and token-by-token decode silently disagree,
 * and the disagreement only shows up as different generated text later. */
static void run_matmul(uint32_t type, const char *name, uint32_t n_out,
                       uint32_t n_in, uint32_t n_tok) {
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
      ok(mr > 0 ? md / mr < 1e-5 : md < 1e-5, t, mr > 0 ? md / mr : md); }

    md = 0;
    for (size_t i = 0; i < (size_t)n_tok * n_out; i++) {
        const double d = fabs((double)Y_gpu[i] - (double)Y_one[i]);
        if (d > md) md = d;
    }
    { char t[80]; snprintf(t, sizeof t, "%s chunk==1-by-1", name);
      ok(md == 0.0, t, md); }

done:
    ds4x_cuda_free(dw);
    free(W); free(X); free(Y_cpu); free(Y_gpu); free(Y_one); free(rowbuf);
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
    run_type(12, "Q4_K 4096x4096",  512, 4096);
    run_type(13, "Q5_K 4096x4096",  512, 4096);
    run_type(14, "Q6_K 4096x4096",  512, 4096);
    /* sub-4-bit: what the big MoE builds are made of (GLM-5.2) */
    run_type(10, "Q2_K 4096x4096",  512, 4096);
    run_type(16, "IQ2_XXS 4096x4096", 512, 4096);
    /* non-square (FFN down: n_embd x ff) */
    run_type(12, "Q4_K 4096x12288", 256, 12288);

    /* Batched matmul. 13 tokens is deliberately not a multiple of the token
     * tile (8), so the masked tail lanes are exercised; 300 crosses the
     * internal per-launch token block (256). */
    run_matmul(12, "MM Q4_K 1024x1024 x13", 1024, 1024, 13);
    run_matmul(12, "MM Q4_K 1024x1024 x300", 1024, 1024, 300);
    run_matmul(14, "MM Q6_K  512x1024 x13",  512, 1024, 13);
    run_matmul( 8, "MM Q8_0  512x1024 x13",  512, 1024, 13);
    run_matmul( 0, "MM F32   512x1024 x5",   512, 1024,  5);
    run_matmul(12, "MM Q4_K  512x1024 x1",   512, 1024,  1);
    run_matmul(10, "MM Q2_K 1024x1024 x13", 1024, 1024, 13);
    run_matmul(16, "MM IQ2_XXS 1024x1024 x13", 1024, 1024, 13);

    /* Gated DeltaNet recurrence. First shape is Qwen3.5-0.8B exactly
     * (16 k-heads = 16 v-heads, 128/128); the second has v_heads > k_heads,
     * the linear-path GQA sharing that Qwen3.5-27B uses (16 k : 32 v) and that
     * the 0.8B shape would never exercise. */
    run_gdn(16, 16, 128, 128, 24, "GDN 16k/16v 128x128");
    run_gdn(16, 32, 128, 128,  7, "GDN 16k/32v 128x128");
    run_gdn( 4,  4,  64, 192,  3, "GDN 4k/4v 64x192");

    printf("\n%d checks, %d failures (%.1f MB resident at peak)\n",
           checks, failures, (double)ds4x_cuda_bytes_resident() / 1048576.0);
    if (failures) { printf("DS4X_CUDA_TEST_FAIL\n"); return 1; }
    printf("DS4X_CUDA_TEST_OK\n");
    return 0;
}
