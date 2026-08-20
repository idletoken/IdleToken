/* dspark_verify_test.c — G-DSPARK-VERIFY: does speculation change the output?
 *
 * SHELVED 2026-08-03; the ds4 line it tests was shelved 2026-08-16 and is no
 * longer compiled into any binary (src/common/ds4_stub.c stands in for it).
 * Nothing here is built by default, nothing calls it, and the G_DSPARK gate
 * SKIPs unless IDLETOKEN_DSPARK_GATE=1. Kept for the record, not for use.
 *
 * The whole point of speculative decoding is that it must not. Greedy decoding
 * with the drafter ON has to produce exactly the same token stream as greedy
 * decoding with it OFF — the drafter only proposes, the target model decides.
 * That makes this the first check in the whole DSpark series that gives a
 * binary right/wrong answer instead of a numeric agreement.
 *
 * Two engines are opened from the same weights: one with the DSpark module,
 * one without. Each generates greedily from the same prompt tokens. Any
 * divergence is a bug in drafting, verification or the bookkeeping that keeps
 * checkpoint.len / the frontier snapshots / s->logits in agreement.
 *
 * Usage: dspark_verify_test <main gguf> <dspark gguf> [n_tokens]
 * Contract: last line DSPARK_VERIFY_OK or DSPARK_VERIFY_FAIL.
 */
#include "ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB 129280

static const int PROMPT[] = { 1, 100, 3838, 275, 481, 2153, 19, 4210 };
#define PROMPT_N ((int)(sizeof(PROMPT) / sizeof(PROMPT[0])))

/* Plain greedy: one target decode per token. The reference stream. */
static int gen_plain(ds4_session *s, int n, int *out) {
    char err[256];
    int cur = PROMPT[0];
    for (int i = 1; i < PROMPT_N; i++) {
        if (ds4_session_eval(s, PROMPT[i - 1], err, sizeof err) != 0) return -1;
        cur = PROMPT[i];
    }
    int got = 0;
    while (got < n) {
        if (ds4_session_eval(s, cur, err, sizeof err) != 0) return -1;
        cur = ds4_session_argmax(s);
        out[got++] = cur;
    }
    return got;
}

/* Same stream, but each step may commit a whole verified block. */
static int gen_spec(ds4_session *s, int n, int *out) {
    char err[256];
    int cur = PROMPT[0];
    for (int i = 1; i < PROMPT_N; i++) {
        if (ds4_session_eval(s, PROMPT[i - 1], err, sizeof err) != 0) return -1;
        cur = PROMPT[i];
    }
    int got = 0, blocks = 0, committed_multi = 0;
    while (got < n) {
        int acc[32];
        const int k = ds4_session_eval_dspark_argmax(s, cur, 24, -1, acc, 32, err, sizeof err);
        if (k <= 0) return -1;
        blocks++;
        if (k > 1) committed_multi++;
        /* acc[0] is `cur` itself (already committed); the rest are new. */
        for (int j = 1; j < k && got < n; j++) out[got++] = acc[j];
        if (k > 1) cur = acc[k - 1];
        else {
            cur = ds4_session_argmax(s);
            if (got < n) out[got++] = cur;
        }
    }
    printf("        spec: %d cycles, %d committed a block\n", blocks, committed_multi);
    return got;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <main gguf> <dspark gguf> [n]\n", argv[0]);
                    puts("DSPARK_VERIFY_FAIL"); return 2; }
    const int N = argc > 3 ? atoi(argv[3]) : 24;

    ds4_engine_options eo;
    memset(&eo, 0, sizeof eo);
    eo.model_path = argv[1];
    eo.backend = DS4_BACKEND_CUDA;
    ds4_engine *plain = NULL;
    if (ds4_engine_open(&plain, &eo) != 0 || !plain) {
        printf("  plain engine open failed\n"); puts("DSPARK_VERIFY_FAIL"); return 1; }

    ds4_session *a = NULL;
    int *ref = malloc((size_t)N * sizeof(int));
    if (ds4_session_create(&a, plain, 2048) != 0 || gen_plain(a, N, ref) != N) {
        printf("  reference generation failed\n"); puts("DSPARK_VERIFY_FAIL"); return 1; }
    ds4_session_free(a);
    ds4_engine_close(plain);

    eo.dspark_path = argv[2];
    ds4_engine *spec = NULL;
    if (ds4_engine_open(&spec, &eo) != 0 || !spec) {
        printf("  dspark engine open failed\n"); puts("DSPARK_VERIFY_FAIL"); return 1; }
    ds4_session *b = NULL;
    int *got = malloc((size_t)N * sizeof(int));
    if (ds4_session_create(&b, spec, 2048) != 0 || gen_spec(b, N, got) != N) {
        printf("  speculative generation failed\n"); puts("DSPARK_VERIFY_FAIL"); return 1; }
    ds4_session_free(b);
    ds4_engine_close(spec);

    printf("        plain: "); for (int i = 0; i < N; i++) printf("%d ", ref[i]); printf("\n");
    printf("        spec : "); for (int i = 0; i < N; i++) printf("%d ", got[i]); printf("\n");

    int first_bad = -1;
    for (int i = 0; i < N; i++) if (ref[i] != got[i]) { first_bad = i; break; }
    if (first_bad >= 0) {
        printf("  [FAIL] streams diverge at token %d (%d vs %d)\n",
               first_bad, ref[first_bad], got[first_bad]);
        puts("DSPARK_VERIFY_FAIL"); return 1;
    }
    printf("  [ok] %d tokens identical with DSpark on and off\n", N);
    free(ref); free(got);
    puts("DSPARK_VERIFY_OK");
    return 0;
}
