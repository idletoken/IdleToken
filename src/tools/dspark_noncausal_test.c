/* dspark_noncausal_test.c — does the non-causal mask actually do anything?
 *
 * SHELVED 2026-08-03; the ds4 line it tests was shelved 2026-08-16 and is no
 * longer compiled into any binary (src/common/ds4_stub.c stands in for it).
 * Nothing here is built by default, nothing calls it, and the G_DSPARK gate
 * SKIPs unless IDLETOKEN_DSPARK_GATE=1. Kept for the record, not for use.
 *
 * The DSpark drafter proposes a whole block in one pass, and every position in
 * that block must see every other one, later ones included. The batched
 * attention kernel was causal (`hi = min(qpos, raw_last_pos)`), so a new
 * non-causal entry point was added. This test exists because the failure mode
 * of getting that wrong is invisible: a causal block still runs, still returns
 * plausible numbers, and only shows up as a mysteriously low accept rate.
 *
 * The oracle needs no reference values. Feed the SAME query vector at every
 * block position:
 *   - non-causal: every position sees the same key set, so every output row
 *     must be identical;
 *   - causal: the key set grows with t, so the rows must differ;
 *   - the LAST row must agree between the two modes, because the last position
 *     already saw everything under causal masking.
 * All three have to hold. The first alone would pass if the kernel ignored the
 * keys entirely; the second alone would pass if non-causal were broken in some
 * other way; the third pins the two modes to the same arithmetic.
 *
 * Contract: last line DSPARK_NONCAUSAL_OK or DSPARK_NONCAUSAL_FAIL.
 */
#include "ds4_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NH      1u      /* heads */
#define HD      512u    /* head dim (DSv4) */
#define NT      5u      /* block size = dspark.block_size */
#define NRAW    8u      /* 3 context rows + the 5 block rows */
#define RCAP    16u
#define RSTART  0u
#define WINDOW  128u    /* SWA window of a compress_ratio-0 layer */
#define POS0    100u    /* first block position; first_raw_pos = POS0+NT-NRAW = 97 */

static int fails;
static void check(int cond, const char *what) {
    printf("  [%s] %s\n", cond ? "ok" : "FAIL", what);
    if (!cond) fails++;
}

static float maxdiff(const float *a, const float *b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(void) {
    /* Lets the kernel take a plain device pointer as the "model map" so the
     * test does not need a real GGUF just to supply n_head attention sinks. */
    setenv("DS4_CUDA_DIRECT_MODEL", "1", 1);
    if (!ds4_gpu_init()) { printf("  no GPU backend\n"); puts("DSPARK_NONCAUSAL_FAIL"); return 1; }

    const uint64_t q_n   = (uint64_t)NT * NH * HD;
    const uint64_t kv_n  = (uint64_t)RCAP * HD;
    ds4_gpu_tensor *sinks = ds4_gpu_tensor_alloc((uint64_t)NH * sizeof(float));
    ds4_gpu_tensor *q     = ds4_gpu_tensor_alloc(q_n * sizeof(float));
    ds4_gpu_tensor *kv    = ds4_gpu_tensor_alloc(kv_n * sizeof(float));
    ds4_gpu_tensor *out_c = ds4_gpu_tensor_alloc(q_n * sizeof(float));
    ds4_gpu_tensor *out_n = ds4_gpu_tensor_alloc(q_n * sizeof(float));
    if (!sinks || !q || !kv || !out_c || !out_n) {
        printf("  allocation failed\n"); puts("DSPARK_NONCAUSAL_FAIL"); return 1;
    }

    float *h = malloc(kv_n * sizeof(float));
    /* sinks = 0: same constant in both modes, so it cannot mask a difference. */
    for (uint32_t i = 0; i < NH; i++) h[i] = 0.0f;
    ds4_gpu_tensor_write(sinks, 0, h, (uint64_t)NH * sizeof(float));

    /* Same query vector at every block position — the whole point of the test. */
    for (uint32_t d = 0; d < HD; d++) h[d] = sinf((float)d * 0.01f);
    for (uint32_t t = 0; t < NT; t++)
        for (uint32_t hh = 0; hh < NH; hh++)
            ds4_gpu_tensor_write(q, ((uint64_t)t * NH + hh) * HD * sizeof(float),
                                 h, (uint64_t)HD * sizeof(float));

    /* Distinct KV rows, so "sees more keys" changes the result. */
    for (uint32_t r = 0; r < RCAP; r++)
        for (uint32_t d = 0; d < HD; d++)
            h[(uint64_t)r * HD + d] = cosf((float)(r * 31 + d) * 0.017f);
    ds4_gpu_tensor_write(kv, 0, h, kv_n * sizeof(float));
    free(h);

    void *map = ds4_gpu_tensor_contents(sinks);

    if (!ds4_gpu_begin_commands()) { puts("DSPARK_NONCAUSAL_FAIL"); return 1; }
    int rc1 = ds4_gpu_attention_decode_raw_batch_heads_tensor(
            out_c, map, (uint64_t)NH * sizeof(float), 0, q, kv,
            NT, POS0, NRAW, RCAP, RSTART, WINDOW, NH, HD);
    int rc2 = ds4_gpu_attention_decode_raw_batch_noncausal_heads_tensor(
            out_n, map, (uint64_t)NH * sizeof(float), 0, q, kv,
            NT, POS0, NRAW, RCAP, RSTART, WINDOW, NH, HD);
    if (!ds4_gpu_end_commands() || !ds4_gpu_synchronize()) {
        puts("DSPARK_NONCAUSAL_FAIL"); return 1;
    }
    check(rc1 != 0, "causal op accepted the batch");
    check(rc2 != 0, "non-causal op accepted the batch");

    float *c = malloc(q_n * sizeof(float));
    float *n = malloc(q_n * sizeof(float));
    ds4_gpu_tensor_read(out_c, 0, c, q_n * sizeof(float));
    ds4_gpu_tensor_read(out_n, 0, n, q_n * sizeof(float));

    const size_t row = (size_t)NH * HD;
    /* 1. non-causal: identical q + identical key set => identical rows. */
    float worst_n = 0.0f;
    for (uint32_t t = 1; t < NT; t++) {
        const float d = maxdiff(n, n + (size_t)t * row, row);
        if (d > worst_n) worst_n = d;
    }
    check(worst_n < 1e-5f, "non-causal: every block row identical");
    printf("        max row-to-row delta = %.3g\n", (double)worst_n);

    /* 2. causal: the key set grows with t, so rows must differ. */
    const float spread_c = maxdiff(c, c + (size_t)(NT - 1) * row, row);
    check(spread_c > 1e-4f, "causal: first and last rows differ");
    printf("        first-vs-last delta  = %.3g\n", (double)spread_c);

    /* 3. the last position sees everything under either rule. */
    const float last = maxdiff(c + (size_t)(NT - 1) * row,
                               n + (size_t)(NT - 1) * row, row);
    check(last < 1e-5f, "last row agrees between the two modes");
    printf("        last-row delta       = %.3g\n", (double)last);

    /* 4. and the earlier rows must actually have changed. */
    const float first = maxdiff(c, n, row);
    check(first > 1e-4f, "first row changed when the mask was lifted");
    printf("        first-row delta      = %.3g\n", (double)first);

    free(c); free(n);
    if (fails) { printf("%d failure(s)\n", fails); puts("DSPARK_NONCAUSAL_FAIL"); return 1; }
    puts("DSPARK_NONCAUSAL_OK");
    return 0;
}
