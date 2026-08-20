/* dspark_draft_test.c — G-DSPARK-DRAFT: does the drafter produce a block?
 *
 * SHELVED 2026-08-03; the ds4 line it tests was shelved 2026-08-16 and is no
 * longer compiled into any binary (src/common/ds4_stub.c stands in for it).
 * Nothing here is built by default, nothing calls it, and the G_DSPARK gate
 * SKIPs unless IDLETOKEN_DSPARK_GATE=1. Kept for the record, not for use.
 *
 * The gate criterion at this stage is deliberately modest: the block forward
 * runs, returns n candidate tokens, they are in range, and the pass leaves the
 * target model's own state alone. Whether the candidates are GOOD is not
 * measurable here — that is G-DSPARK-VERIFY (greedy equality with DSpark off)
 * and G-DSPARK-PERF (accept rate).
 *
 * What IS worth asserting now: the drafter must not silently degenerate. A
 * block of five identical tokens, or five copies of the noise token, is what a
 * broken context or a dead markov head looks like.
 */
#include "ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
static void check(int c, const char *w) {
    printf("  [%s] %s\n", c ? "ok" : "FAIL", w);
    if (!c) fails++;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <main gguf> <dspark gguf>\n", argv[0]);
                    puts("DSPARK_DRAFT_FAIL"); return 2; }
    ds4_engine_options eo;
    memset(&eo, 0, sizeof eo);
    eo.model_path = argv[1]; eo.dspark_path = argv[2]; eo.backend = DS4_BACKEND_CUDA;
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &eo) != 0 || !e) { printf("  engine open failed\n"); puts("DSPARK_DRAFT_FAIL"); return 1; }
    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, 2048) != 0 || !s) { printf("  session failed\n"); puts("DSPARK_DRAFT_FAIL"); return 1; }

    /* Fill the drafter's context window. The count matters: with only a
     * couple of positions the block attends to almost nothing, and a drafter
     * starved of context looks exactly like a broken one (flat, nearly
     * position-independent logits). Default 64; argv[3] overrides. */
    uint32_t warm = argc > 3 ? (uint32_t)atoi(argv[3]) : 64u;
    static const int WORDS[] = { 1, 100, 3838, 275, 481, 2153, 19, 4210, 62, 913 };
    for (uint32_t p = 0; p < warm; p++) {
        const int t = WORDS[p % (sizeof(WORDS) / sizeof(WORDS[0]))];
        if (!ds4_session_encode_layer_range(s, t, p, 0, 43)) {
            printf("  decode step %u failed\n", p); puts("DSPARK_DRAFT_FAIL"); return 1;
        }
        float mx[4096];
        if (!ds4_engine_dspark_main_x(e, s, mx, 4096) ||
            !ds4_engine_dspark_push_context(e, s, p)) {
            printf("  context push %u failed\n", p); puts("DSPARK_DRAFT_FAIL"); return 1;
        }
    }
    printf("        warmed %u context positions\n", warm);

    const int B = ds4_engine_dspark_block_size(e);
    check(B == 5, "block size is 5");
    int toks[8];
    memset(toks, -1, sizeof toks);
    check(ds4_engine_dspark_draft(e, s, 913, warm, toks, (uint32_t)B) != 0,
          "draft block produced");
    printf("        draft tokens:");
    for (int k = 0; k < B; k++) printf(" %d", toks[k]);
    printf("\n");

    int in_range = 1, all_same = 1, any_noise = 0;
    for (int k = 0; k < B; k++) {
        if (toks[k] < 0 || toks[k] >= 129280) in_range = 0;
        if (k && toks[k] != toks[0]) all_same = 0;
        if (toks[k] == 128799) any_noise = 1;
    }
    check(in_range, "every candidate is a valid token id");
    check(!all_same, "the block is not five copies of one token");
    check(!any_noise, "no candidate is the noise/mask token");

    ds4_session_free(s);
    ds4_engine_close(e);
    if (fails) { printf("%d failure(s)\n", fails); puts("DSPARK_DRAFT_FAIL"); return 1; }
    puts("DSPARK_DRAFT_OK");
    return 0;
}
