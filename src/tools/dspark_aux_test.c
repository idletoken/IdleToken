/* dspark_aux_test.c — is the DSpark aux-hidden capture actually right?
 *
 * The drafter's context is main_norm(main_proj(concat(aux[40], aux[41],
 * aux[42]))), where aux[L] is the target model's hidden state at layer L
 * MEAN-REDUCED over the hc streams. Getting that reduction wrong (or capturing
 * at the wrong layer) yields a drafter that runs and drafts badly — the
 * failure mode this whole gate series exists to avoid.
 *
 * The oracle is a real cross-check, not a plausibility check: the capture runs
 * on the GPU as a uniform-weight hc_weighted_sum, while the PP boundary already
 * exposes cur_hc to the host. After encoding layers [0, 43), cur_hc IS layer
 * 42's output, so the host can average it independently and the two numbers
 * must agree. Two implementations, one answer.
 *
 * Also asserts the three captures differ from one another — capturing the same
 * layer three times would pass an "is it finite" check and nothing else.
 *
 * Usage: dspark_aux_test <main gguf> <dspark gguf>
 * Contract: last line DSPARK_AUX_OK or DSPARK_AUX_FAIL.
 */
#include "ds4.h"
#include "idletoken_gguf.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define N_EMBD 4096u
#define N_HC   4u

static int fails;
static void check(int cond, const char *what) {
    printf("  [%s] %s\n", cond ? "ok" : "FAIL", what);
    if (!cond) fails++;
}

static float half_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    const uint32_t man  = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {
            uint32_t ee = 127 - 15 + 1, mm = man;
            while (!(mm & 0x400u)) { mm <<= 1; ee--; }
            mm &= 0x3ffu;
            bits = sign | (ee << 23) | (mm << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f; memcpy(&f, &bits, sizeof f); return f;
}

/* Dequantize row `row` of a Q8_0 [row_len, n_rows] tensor from raw bytes. */
static void deq_row_q8(const unsigned char *base, uint64_t row_len, uint64_t row, double *out) {
    const uint64_t blocks = row_len / 32;
    for (uint64_t b = 0; b < blocks; b++) {
        const unsigned char *blk = base + (row * blocks + b) * 34;
        uint16_t h; memcpy(&h, blk, sizeof h);
        const double sc = (double)half_to_f32(h);
        for (uint64_t j = 0; j < 32; j++) out[b * 32 + j] = sc * (double)(int8_t)blk[2 + j];
    }
}

static float l2(const float *v, uint32_t n) {
    double s = 0.0;
    for (uint32_t i = 0; i < n; i++) s += (double)v[i] * (double)v[i];
    return (float)sqrt(s);
}

static float maxabs_diff(const float *a, const float *b, uint32_t n) {
    float m = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        const float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <main gguf> <dspark gguf>\n", argv[0]);
        puts("DSPARK_AUX_FAIL");
        return 2;
    }
    ds4_engine_options eo;
    memset(&eo, 0, sizeof eo);
    eo.model_path  = argv[1];
    eo.dspark_path = argv[2];
    eo.backend     = DS4_BACKEND_CUDA;

    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &eo) != 0 || !e) {
        printf("  engine open failed\n"); puts("DSPARK_AUX_FAIL"); return 1;
    }
    check(ds4_engine_has_dspark(e), "engine reports a DSpark module");
    check(ds4_engine_dspark_block_size(e) == 5, "block size is 5");

    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, 2048) != 0 || !s) {
        printf("  session create failed\n"); puts("DSPARK_AUX_FAIL"); return 1;
    }

    /* One decode step through every layer. Any token works: the point is that
     * the capture fires and matches, not what the model predicts. */
    if (!ds4_session_encode_layer_range(s, 1, 0, 0, 43)) {
        printf("  encode_layer_range failed\n"); puts("DSPARK_AUX_FAIL"); return 1;
    }

    float *aux[3];
    for (int k = 0; k < 3; k++) {
        aux[k] = malloc(N_EMBD * sizeof(float));
        if (!ds4_session_dspark_aux_read(s, (uint32_t)k, aux[k], N_EMBD)) {
            printf("  aux_read(%d) failed\n", k); puts("DSPARK_AUX_FAIL"); return 1;
        }
    }

    int finite_ok = 1, nonzero_ok = 1;
    for (int k = 0; k < 3; k++) {
        const float n = l2(aux[k], N_EMBD);
        printf("        aux[%d] (layer %d)  L2 = %.4f\n", k, 40 + k, (double)n);
        if (!isfinite(n)) finite_ok = 0;
        if (!(n > 0.0f)) nonzero_ok = 0;
    }
    check(finite_ok,  "all three captures are finite");
    check(nonzero_ok, "all three captures are non-zero");

    /* Distinct layers must give distinct vectors. */
    const float d01 = maxabs_diff(aux[0], aux[1], N_EMBD);
    const float d12 = maxabs_diff(aux[1], aux[2], N_EMBD);
    printf("        |aux0-aux1| = %.4g   |aux1-aux2| = %.4g\n", (double)d01, (double)d12);
    check(d01 > 1e-6f && d12 > 1e-6f, "the three captures differ from each other");

    /* The cross-check: cur_hc after layer 42 is what aux[2] was reduced from. */
    const uint64_t hc_bytes = ds4_session_hc_tensor_bytes(s);
    check(hc_bytes == (uint64_t)N_HC * N_EMBD * sizeof(float), "cur_hc is [N_HC, N_EMBD] f32");
    float *hc = malloc(hc_bytes);
    if (!ds4_session_hc_tensor_read(s, hc, hc_bytes)) {
        printf("  hc_tensor_read failed\n"); puts("DSPARK_AUX_FAIL"); return 1;
    }
    float *ref = malloc(N_EMBD * sizeof(float));
    for (uint32_t i = 0; i < N_EMBD; i++) {
        double acc = 0.0;
        for (uint32_t h = 0; h < N_HC; h++) acc += (double)hc[(uint64_t)h * N_EMBD + i];
        ref[i] = (float)(acc / (double)N_HC);
    }
    const float d = maxabs_diff(ref, aux[2], N_EMBD);
    const float scale = l2(ref, N_EMBD) / sqrtf((float)N_EMBD);
    printf("        host mean vs GPU capture: max|Δ| = %.4g  (rms scale %.4g)\n",
           (double)d, (double)scale);
    check(d <= 1e-4f * (scale > 1.0f ? scale : 1.0f),
          "GPU capture equals an independent host mean over the hc streams");

    /* ---- main_x = main_norm(main_proj(concat(aux))) --------------------
     * Same two-implementation rule as above, and again with a DIFFERENT GGUF
     * parser on the host side (IdleToken's, not ds4's) so a shared misreading of
     * Q8_0 cannot make both sides agree. */
    float *gpu_mx = malloc(N_EMBD * sizeof(float));
    check(ds4_engine_dspark_main_x(e, s, gpu_mx, N_EMBD) != 0, "main_x computed");

    char gerr[256] = {0};
    idletoken_gguf_meta *gm = idletoken_gguf_meta_open(argv[2], gerr, sizeof gerr);
    idletoken_gguf_tensor tp, tn;
    if (!gm || idletoken_gguf_tensor_find(gm, "mtp.0.main_proj.weight", &tp) != 0 ||
               idletoken_gguf_tensor_find(gm, "mtp.0.main_norm.weight", &tn) != 0) {
        printf("  cannot locate main_proj/main_norm: %s\n", gerr);
        puts("DSPARK_AUX_FAIL"); return 1;
    }
    check(tp.dims[0] == (uint64_t)3 * N_EMBD,
          "main_proj input width is n_targets * n_embd (a concat, not a mean)");

    const uint64_t doff  = idletoken_gguf_data_offset(gm);
    const uint64_t in_d  = tp.dims[0];              /* 12288 */
    const uint64_t blks  = in_d / 32;
    const uint64_t wsz   = (uint64_t)N_EMBD * blks * 34ull;
    FILE *wf = fopen(argv[2], "rb");
    unsigned char *wp = malloc(wsz);
    float *wn = malloc(N_EMBD * sizeof(float));
    if (!wf || !wp || !wn ||
        fseek(wf, (long)(doff + tp.offset), SEEK_SET) != 0 || fread(wp, 1, wsz, wf) != wsz ||
        fseek(wf, (long)(doff + tn.offset), SEEK_SET) != 0 ||
        fread(wn, sizeof(float), N_EMBD, wf) != N_EMBD) {
        printf("  cannot read main_proj/main_norm\n"); puts("DSPARK_AUX_FAIL"); return 1;
    }
    fclose(wf);

    /* ds4's Q8_0 matmul quantizes the ACTIVATION to int8 per 32-wide block
     * before multiplying (quantize_q8_0_f32_kernel: d = max|x|/127,
     * q = clamp(lrintf(x/d), -128, 127)). A host reference that skips that
     * step is not a stricter oracle, it is a different function — it disagrees
     * by ~0.5%, which is the quantization error and not a bug. Mirroring it
     * here keeps the tolerance tight AND checks that we read ds4's matmul
     * correctly, which the drafter depends on. */
    double *cat = malloc(in_d * sizeof(double));
    for (int k = 0; k < 3; k++)
        for (uint32_t i = 0; i < N_EMBD; i++) cat[(uint64_t)k * N_EMBD + i] = aux[k][i];
    for (uint64_t b = 0; b < blks; b++) {
        float amax = 0.0f;
        for (uint64_t j = 0; j < 32; j++) {
            const float a = fabsf((float)cat[b * 32 + j]);
            if (a > amax) amax = a;
        }
        const float dq = amax / 127.0f;
        const float idq = dq != 0.0f ? 1.0f / dq : 0.0f;
        for (uint64_t j = 0; j < 32; j++) {
            int v = (int)lrintf((float)cat[b * 32 + j] * idq);
            v = v > 127 ? 127 : (v < -128 ? -128 : v);
            cat[b * 32 + j] = (double)dq * (double)v;
        }
    }

    double *proj = malloc(N_EMBD * sizeof(double));
    double *rowb = malloc(in_d * sizeof(double));
    for (uint32_t o = 0; o < N_EMBD; o++) {
        deq_row_q8(wp, in_d, o, rowb);
        double acc = 0.0;
        for (uint64_t i = 0; i < in_d; i++) acc += rowb[i] * cat[i];
        proj[o] = acc;
    }
    double ss = 0.0;
    for (uint32_t i = 0; i < N_EMBD; i++) ss += proj[i] * proj[i];
    const double inv = 1.0 / sqrt(ss / (double)N_EMBD + 1e-6);
    float mxd = 0.0f, mxs = 0.0f;
    for (uint32_t i = 0; i < N_EMBD; i++) {
        const float ref_i = (float)(proj[i] * inv * (double)wn[i]);
        const float d = fabsf(ref_i - gpu_mx[i]);
        if (d > mxd) mxd = d;
        if (fabsf(ref_i) > mxs) mxs = fabsf(ref_i);
    }
    printf("        main_x host vs GPU: max|Δ| = %.4g   (peak |main_x| = %.4g)\n",
           (double)mxd, (double)mxs);
    check(mxd <= 1e-3f * (mxs > 1.0f ? mxs : 1.0f),
          "main_x equals an independent host projection + norm");

    /* ---- §4② context KV: attn_kv_a_norm(attn_kv(main_x)) --------------
     * Same rule again. attn_kv is Q8_0 [n_embd, head_dim] and the activation
     * (main_x) is quantized per 32-block by ds4's matmul, so the host side
     * mirrors that too. Checks stage 0 and stage 2 so a binder that pointed
     * every stage at the same weights would be caught. */
    for (uint32_t st = 0; st < 3; st += 2) {
        float gpu_kv[512];
        if (!ds4_engine_dspark_context_kv(e, s, st, gpu_kv, 512)) {
            printf("  context_kv(stage %u) failed\n", st);
            puts("DSPARK_AUX_FAIL"); return 1;
        }
        char nm[96];
        idletoken_gguf_tensor tk, tkn;
        snprintf(nm, sizeof nm, "mtp.%u.attn_kv.weight", st);
        if (idletoken_gguf_tensor_find(gm, nm, &tk) != 0) { printf("  no %s\n", nm); puts("DSPARK_AUX_FAIL"); return 1; }
        snprintf(nm, sizeof nm, "mtp.%u.attn_kv_a_norm.weight", st);
        if (idletoken_gguf_tensor_find(gm, nm, &tkn) != 0) { printf("  no %s\n", nm); puts("DSPARK_AUX_FAIL"); return 1; }

        const uint64_t kblks = (uint64_t)N_EMBD / 32;
        const uint64_t ksz   = 512ull * kblks * 34ull;
        FILE *kf = fopen(argv[2], "rb");
        unsigned char *kw = malloc(ksz);
        float *kn = malloc(512 * sizeof(float));
        if (!kf || !kw || !kn ||
            fseek(kf, (long)(doff + tk.offset), SEEK_SET) != 0 || fread(kw, 1, ksz, kf) != ksz ||
            fseek(kf, (long)(doff + tkn.offset), SEEK_SET) != 0 ||
            fread(kn, sizeof(float), 512, kf) != 512) {
            printf("  cannot read stage %u kv weights\n", st); puts("DSPARK_AUX_FAIL"); return 1;
        }
        fclose(kf);

        /* quantize main_x the way ds4's matmul does */
        double *mx = malloc(N_EMBD * sizeof(double));
        for (uint32_t i = 0; i < N_EMBD; i++) mx[i] = gpu_mx[i];
        for (uint64_t b = 0; b < N_EMBD / 32; b++) {
            float amax = 0.0f;
            for (uint64_t j = 0; j < 32; j++) {
                const float a = fabsf((float)mx[b * 32 + j]);
                if (a > amax) amax = a;
            }
            const float dq = amax / 127.0f, idq = dq != 0.0f ? 1.0f / dq : 0.0f;
            for (uint64_t j = 0; j < 32; j++) {
                int v = (int)lrintf((float)mx[b * 32 + j] * idq);
                v = v > 127 ? 127 : (v < -128 ? -128 : v);
                mx[b * 32 + j] = (double)dq * (double)v;
            }
        }
        double *kraw = malloc(512 * sizeof(double));
        double *krow = malloc(N_EMBD * sizeof(double));
        for (uint32_t o = 0; o < 512; o++) {
            deq_row_q8(kw, N_EMBD, o, krow);
            double acc = 0.0;
            for (uint32_t i = 0; i < N_EMBD; i++) acc += krow[i] * mx[i];
            kraw[o] = acc;
        }
        double kss = 0.0;
        for (uint32_t i = 0; i < 512; i++) kss += kraw[i] * kraw[i];
        const double kinv = 1.0 / sqrt(kss / 512.0 + 1e-6);
        float kd = 0.0f, kpk = 0.0f;
        for (uint32_t i = 0; i < 512; i++) {
            const float r = (float)(kraw[i] * kinv * (double)kn[i]);
            const float dd = fabsf(r - gpu_kv[i]);
            if (dd > kd) kd = dd;
            if (fabsf(r) > kpk) kpk = fabsf(r);
        }
        printf("        stage %u context KV: max|Δ| = %.4g   (peak %.4g)\n",
               st, (double)kd, (double)kpk);
        char lbl[80];
        snprintf(lbl, sizeof lbl, "stage %u context KV matches an independent host projection", st);
        check(kd <= 1e-3f * (kpk > 1.0f ? kpk : 1.0f), lbl);
        free(kw); free(kn); free(mx); free(kraw); free(krow);
    }
    check(ds4_engine_dspark_push_context(e, s, 0) != 0, "context KV pushed into every stage cache");

    idletoken_gguf_meta_close(gm);
    free(gpu_mx); free(wp); free(wn); free(cat); free(proj); free(rowb);
    for (int k = 0; k < 3; k++) free(aux[k]);
    free(hc); free(ref);
    ds4_session_free(s);
    ds4_engine_close(e);

    if (fails) { printf("%d failure(s)\n", fails); puts("DSPARK_AUX_FAIL"); return 1; }
    puts("DSPARK_AUX_OK");
    return 0;
}
