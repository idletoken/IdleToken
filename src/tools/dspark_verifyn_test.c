/* dspark_verifyn_test.c — is the N-position verifier the same function?
 *
 * SHELVED 2026-08-03; the ds4 line it tests was shelved 2026-08-16 and is no
 * longer compiled into any binary (src/common/ds4_stub.c stands in for it).
 * Nothing here is built by default, nothing calls it, and the G_DSPARK gate
 * SKIPs unless IDLETOKEN_DSPARK_GATE=1. Kept for the record, not for use.
 *
 * DSpark needs N = block_size + 1 verification positions; ds4 shipped a
 * hardcoded N=2 verifier for MTP. Generalising it touches the path that
 * produces the target model's committed tokens, so the generalisation needs
 * its own oracle BEFORE it is wired into decoding.
 *
 * The oracle: with n_tokens == 2 the generalised verifier must reproduce the
 * hardcoded one exactly — same argmax at both positions, same logits.
 *
 * They run in SEPARATE sessions warmed identically. Back to back in one
 * session the first run advances the compressed attention state, so the second
 * would start from a different place and differ for reasons unrelated to the
 * code under test. (Learned the hard way elsewhere in this series: a reference
 * that does not reproduce the callee's state is not a stricter check, it is a
 * different function.)
 */
#include "ds4.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB 129280u

static int fails;
static void check(int c, const char *w) {
    printf("  [%s] %s\n", c ? "ok" : "FAIL", w);
    if (!c) fails++;
}

/* Identical warm-up so both sessions start from the same state. */
static int warm(ds4_session *s, uint32_t n) {
    static const int W[] = { 1, 100, 3838, 275, 481, 2153, 19, 4210, 62, 913 };
    for (uint32_t p = 0; p < n; p++) {
        if (!ds4_session_encode_layer_range(s, W[p % 10], p, 0, 43)) return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <main gguf>\n", argv[0]);
                    puts("DSPARK_VERIFYN_FAIL"); return 2; }
    ds4_engine_options eo;
    memset(&eo, 0, sizeof eo);
    eo.model_path = argv[1];
    eo.backend = DS4_BACKEND_CUDA;
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &eo) != 0 || !e) { printf("  engine open failed\n"); puts("DSPARK_VERIFYN_FAIL"); return 1; }

    const uint32_t WARM = 32;
    const int toks[2] = { 913, 275 };
    int tops2[2] = {-1,-1}, topsN[2] = {-1,-1};
    float *l2 = malloc((size_t)2 * VOCAB * sizeof(float));
    float *lN = malloc((size_t)2 * VOCAB * sizeof(float));

    ds4_session *a = NULL, *b = NULL;
    if (ds4_session_create(&a, e, 2048) != 0 || !warm(a, WARM) ||
        !ds4_session_verify_probe(a, 0, toks, 2, WARM, tops2, l2)) {
        printf("  hardcoded N=2 verifier failed\n"); puts("DSPARK_VERIFYN_FAIL"); return 1;
    }
    ds4_session_free(a);

    if (ds4_session_create(&b, e, 2048) != 0 || !warm(b, WARM) ||
        !ds4_session_verify_probe(b, 1, toks, 2, WARM, topsN, lN)) {
        printf("  generalised N verifier failed\n"); puts("DSPARK_VERIFYN_FAIL"); return 1;
    }
    ds4_session_free(b);

    printf("        argmax  N=2 path: %d %d   generalised: %d %d\n",
           tops2[0], tops2[1], topsN[0], topsN[1]);
    check(tops2[0] == topsN[0] && tops2[1] == topsN[1],
          "both verifiers pick the same token at both positions");

    float worst = 0.0f, peak = 0.0f;
    for (size_t i = 0; i < (size_t)2 * VOCAB; i++) {
        const float d = fabsf(l2[i] - lN[i]);
        if (d > worst) worst = d;
        if (fabsf(l2[i]) > peak) peak = fabsf(l2[i]);
    }
    printf("        logits max|Δ| = %.4g   (peak |logit| = %.4g)\n", (double)worst, (double)peak);
    check(worst == 0.0f, "logits are bit-identical");

    free(l2); free(lN);
    ds4_engine_close(e);
    if (fails) { printf("%d failure(s)\n", fails); puts("DSPARK_VERIFYN_FAIL"); return 1; }
    puts("DSPARK_VERIFYN_OK");
    return 0;
}
