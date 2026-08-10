/* dspark_markov_test.c — is the DSpark markov bias computed correctly?
 *
 *     bias = markov_w2( markov_w1[prev_token] )
 *
 * This is the whole of §4④ and the only source of intra-block dependency in an
 * otherwise-parallel drafter. Both matrices are Q8_0, and ds4's only embedding
 * op reads the F16 token_embd — pointing that at Q8_0 bytes returns garbage
 * rather than failing, so "it ran" proves nothing here.
 *
 * The oracle is a second, independent implementation: this test opens the same
 * GGUF with IdleToken's own parser (src/common/gguf.c, not ds4's), reads the raw
 * bytes, dequantizes them on the host in double precision, and does the matmul
 * itself. Different parser, different dequant, different arithmetic order. The
 * two answers have to agree.
 *
 * Also checks two tokens give different biases — a lookup that ignores its
 * index would otherwise sail through.
 *
 * Usage: dspark_markov_test <main gguf> <dspark gguf>
 * Contract: last line DSPARK_MARKOV_OK or DSPARK_MARKOV_FAIL.
 */
#include "ds4.h"
#include "idletoken_gguf.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_VOCAB 129280u
#define RANK    256u
#define TOK_A   1234
#define TOK_B   9876

static int fails;
static void check(int cond, const char *what) {
    printf("  [%s] %s\n", cond ? "ok" : "FAIL", what);
    if (!cond) fails++;
}

/* GGUF Q8_0: 32 values per block, each block = f16 scale + 32 int8. */
static float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    const uint32_t man  = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {
            uint32_t e = 127 - 15 + 1, m = man;
            while (!(m & 0x400u)) { m <<= 1; e--; }
            m &= 0x3ffu;
            bits = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* Dequantize row `row` of a Q8_0 [row_len, n_rows] tensor from raw file bytes. */
static void deq_row(const unsigned char *base, uint64_t row_len, uint64_t row, double *out) {
    const uint64_t blocks = row_len / 32;
    for (uint64_t b = 0; b < blocks; b++) {
        const unsigned char *blk = base + (row * blocks + b) * 34;
        uint16_t h;
        memcpy(&h, blk, sizeof h);
        const double scale = (double)half_to_float(h);
        for (uint64_t j = 0; j < 32; j++) {
            out[b * 32 + j] = scale * (double)(int8_t)blk[2 + j];
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <main gguf> <dspark gguf>\n", argv[0]);
        puts("DSPARK_MARKOV_FAIL");
        return 2;
    }

    /* ---- side A: the engine ------------------------------------------- */
    ds4_engine_options eo;
    memset(&eo, 0, sizeof eo);
    eo.model_path  = argv[1];
    eo.dspark_path = argv[2];
    eo.backend     = DS4_BACKEND_CUDA;
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &eo) != 0 || !e) {
        printf("  engine open failed\n"); puts("DSPARK_MARKOV_FAIL"); return 1;
    }
    float *gpu_a = malloc((size_t)N_VOCAB * sizeof(float));
    float *gpu_b = malloc((size_t)N_VOCAB * sizeof(float));
    check(ds4_engine_dspark_markov_bias(e, TOK_A, gpu_a, N_VOCAB) != 0, "markov bias computed for token A");
    check(ds4_engine_dspark_markov_bias(e, TOK_B, gpu_b, N_VOCAB) != 0, "markov bias computed for token B");

    /* ---- side B: an independent host computation ----------------------- */
    char err[256] = {0};
    idletoken_gguf_meta *m = idletoken_gguf_meta_open(argv[2], err, sizeof err);
    idletoken_gguf_tensor t1, t2;
    if (!m || idletoken_gguf_tensor_find(m, "mtp.2.markov_head.markov_w1.weight", &t1) != 0 ||
              idletoken_gguf_tensor_find(m, "mtp.2.markov_head.markov_w2.weight", &t2) != 0) {
        printf("  cannot locate the markov tensors: %s\n", err); puts("DSPARK_MARKOV_FAIL"); return 1;
    }
    const uint64_t data_off = idletoken_gguf_data_offset(m);
    const uint64_t blocks   = RANK / 32;
    const uint64_t row_bytes = blocks * 34;
    const uint64_t w_bytes   = (uint64_t)N_VOCAB * row_bytes;

    FILE *f = fopen(argv[2], "rb");
    unsigned char *w1 = malloc(w_bytes);
    unsigned char *w2 = malloc(w_bytes);
    if (!f || !w1 || !w2 ||
        fseek(f, (long)(data_off + t1.offset), SEEK_SET) != 0 || fread(w1, 1, w_bytes, f) != w_bytes ||
        fseek(f, (long)(data_off + t2.offset), SEEK_SET) != 0 || fread(w2, 1, w_bytes, f) != w_bytes) {
        printf("  cannot read the markov weights\n"); puts("DSPARK_MARKOV_FAIL"); return 1;
    }
    fclose(f);

    double *emb = malloc(RANK * sizeof(double));
    deq_row(w1, RANK, (uint64_t)TOK_A, emb);
    /* w2 is Linear(rank, vocab): row v of [rank, vocab] dotted with the embedding. */
    double *row = malloc(RANK * sizeof(double));
    float worst = 0.0f; double scale_acc = 0.0;
    for (uint64_t v = 0; v < N_VOCAB; v++) {
        deq_row(w2, RANK, v, row);
        double acc = 0.0;
        for (uint32_t i = 0; i < RANK; i++) acc += row[i] * emb[i];
        const float d = fabsf((float)acc - gpu_a[v]);
        if (d > worst) worst = d;
        scale_acc += acc * acc;
    }
    const double rms = sqrt(scale_acc / (double)N_VOCAB);
    printf("        host vs GPU: max|Δ| = %.4g   bias rms = %.4g\n", (double)worst, rms);
    check(worst <= 2e-3f * (float)(rms > 1.0 ? rms : 1.0),
          "GPU bias equals an independent host dequant + matmul");

    /* A lookup that ignored its index would give the same bias for both. */
    float diff = 0.0f;
    for (uint64_t v = 0; v < N_VOCAB; v++) {
        const float d = fabsf(gpu_a[v] - gpu_b[v]);
        if (d > diff) diff = d;
    }
    printf("        |bias(A) - bias(B)| = %.4g\n", (double)diff);
    check(diff > 1e-4f, "different tokens give different biases");

    idletoken_gguf_meta_close(m);
    free(gpu_a); free(gpu_b); free(w1); free(w2); free(emb); free(row);
    ds4_engine_close(e);

    if (fails) { printf("%d failure(s)\n", fails); puts("DSPARK_MARKOV_FAIL"); return 1; }
    puts("DSPARK_MARKOV_OK");
    return 0;
}
