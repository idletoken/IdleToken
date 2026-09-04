/* GGUF_GEOM vs the engine — the table that sizes every tensor we ship.
 *
 * WHY THIS EXISTS
 *
 * `GGUF_GEOM` in src/common/weights.c is a copy of ggml's (blck_size,
 * type_size) pairs. Nothing at runtime can notice a wrong row: the index it
 * produces is internally consistent, so the coordinator hands a worker byte
 * ranges that describe a file nobody has. On 2026-09-02 that cost a real
 * cluster — IQ1_S carried IQ3_S's 110 bytes per block, so every IQ1_S tensor in
 * an Unsloth UD quant was indexed at 2.2x its size, the worker cached that many
 * bytes under the tensor's name, and the coordinator's engine aborted with
 * "required RPC model cache miss": a message about a symptom, pointing at
 * neither the size nor the type.
 *
 * WHAT MAKES THIS AN ORACLE
 *
 * It does NOT restate the numbers. It includes the pinned engine's own
 * ggml-common.h and derives every row from `sizeof(block_*)` and the QK_*
 * constants, so the check fails when our copy drifts from the engine — which is
 * the only way this table can be wrong. Restating the constants here would test
 * that two copies of the same mistake agree -- an oracle that shares the
 * assumption under test is not an oracle.
 *
 * Bumping the engine pin therefore has to run this: a type whose block layout
 * changed upstream, or a new type id inserted mid-enum, lands here first.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

/* THE table — the same array the fetcher indexes with, not a copy of it.
 * A copy here would test that two copies of the same mistake agree. */
#include "idletoken_gguf_geom.h"
#define GGUF_GEOM   IDLETOKEN_GGUF_GEOM
#define GGUF_GEOM_N IDLETOKEN_GGUF_GEOM_N

static int failures;
static int checks;

static void expect(int id, const char *name, unsigned be, unsigned bb) {
    checks++;
    if ((unsigned)id >= GGUF_GEOM_N) {
        printf("  [BAD] %s (id %d) is past the end of GGUF_GEOM (%u rows)\n",
               name, id, (unsigned)GGUF_GEOM_N);
        failures++;
        return;
    }
    if (GGUF_GEOM[id].be != be || GGUF_GEOM[id].bb != bb) {
        printf("  [BAD] %-8s (id %2d): table says {%u,%u}, the engine says "
               "{%u,%u}\n", name, id, GGUF_GEOM[id].be, GGUF_GEOM[id].bb,
               be, bb);
        failures++;
        return;
    }
    printf("  [ok]  %-8s (id %2d) {%u,%u}\n", name, id, be, bb);
}

/* A row of zeroes means "no such type"; the fetcher refuses those by name. A
 * type the engine HAS must never be marked absent, and a gap in the enum must
 * never grow a size. */
static void expect_absent(int id, const char *why) {
    checks++;
    if ((unsigned)id >= GGUF_GEOM_N) return;   /* past the table is also absent */
    if (GGUF_GEOM[id].be != 0 || GGUF_GEOM[id].bb != 0) {
        printf("  [BAD] id %2d (%s) has geometry {%u,%u} — a removed or unused "
               "id must stay {0,0} so a file naming it is refused\n",
               id, why, GGUF_GEOM[id].be, GGUF_GEOM[id].bb);
        failures++;
        return;
    }
    printf("  [ok]  id %2d is absent (%s)\n", id, why);
}

int main(void) {
    printf("== GGUF_GEOM vs the pinned engine's ggml-common.h ==\n");

    /* Every row derived from the engine, never typed out. */
    expect(0,  "F32",     1, sizeof(float));
    expect(1,  "F16",     1, sizeof(ggml_half));
    expect(2,  "Q4_0",    QK4_0, sizeof(block_q4_0));
    expect(3,  "Q4_1",    QK4_1, sizeof(block_q4_1));
    expect_absent(4, "Q4_2, removed upstream");
    expect_absent(5, "Q4_3, removed upstream");
    expect(6,  "Q5_0",    QK5_0, sizeof(block_q5_0));
    expect(7,  "Q5_1",    QK5_1, sizeof(block_q5_1));
    expect(8,  "Q8_0",    QK8_0, sizeof(block_q8_0));
    expect(9,  "Q8_1",    QK8_1, sizeof(block_q8_1));
    expect(10, "Q2_K",    QK_K,  sizeof(block_q2_K));
    expect(11, "Q3_K",    QK_K,  sizeof(block_q3_K));
    expect(12, "Q4_K",    QK_K,  sizeof(block_q4_K));
    expect(13, "Q5_K",    QK_K,  sizeof(block_q5_K));
    expect(14, "Q6_K",    QK_K,  sizeof(block_q6_K));
    expect(15, "Q8_K",    QK_K,  sizeof(block_q8_K));
    expect(16, "IQ2_XXS", QK_K,  sizeof(block_iq2_xxs));
    expect(17, "IQ2_XS",  QK_K,  sizeof(block_iq2_xs));
    expect(18, "IQ3_XXS", QK_K,  sizeof(block_iq3_xxs));
    expect(19, "IQ1_S",   QK_K,  sizeof(block_iq1_s));
    expect(20, "IQ4_NL",  QK4_NL, sizeof(block_iq4_nl));
    expect(21, "IQ3_S",   QK_K,  sizeof(block_iq3_s));
    expect(22, "IQ2_S",   QK_K,  sizeof(block_iq2_s));
    expect(23, "IQ4_XS",  QK_K,  sizeof(block_iq4_xs));
    expect(24, "I8",      1, sizeof(int8_t));
    expect(25, "I16",     1, sizeof(int16_t));
    expect(26, "I32",     1, sizeof(int32_t));
    expect(27, "I64",     1, sizeof(int64_t));
    expect(28, "F64",     1, sizeof(double));
    expect(29, "IQ1_M",   QK_K,  sizeof(block_iq1_m));
    /* ggml_bf16_t lives in ggml.h, which drags in the whole API; the
     * engine defines it as a 16-bit container and nothing else. */
    expect(30, "BF16",    1, sizeof(uint16_t));
    expect_absent(31, "Q4_0_4_4, removed upstream");
    expect_absent(32, "Q4_0_4_8, removed upstream");
    expect_absent(33, "Q4_0_8_8, removed upstream");
    expect(34, "TQ1_0",   QK_K,  sizeof(block_tq1_0));
    expect(35, "TQ2_0",   QK_K,  sizeof(block_tq2_0));
    expect_absent(36, "IQ4_NL_4_4, removed upstream");
    expect_absent(37, "IQ4_NL_4_8, removed upstream");
    expect_absent(38, "IQ4_NL_8_8, removed upstream");
    expect(39, "MXFP4",   QK_MXFP4, sizeof(block_mxfp4));
    expect(40, "NVFP4",   QK_NVFP4, sizeof(block_nvfp4));
    expect(41, "Q1_0",    QK1_0, sizeof(block_q1_0));
    expect(42, "Q2_0",    QK2_0, sizeof(block_q2_0));

    /* The table must cover the whole enum. A type id past its end is refused
     * by the fetcher, which is safe, but it means a model we could serve is
     * reported as "unsupported" for a reason nobody will guess. */
    checks++;
    if (GGUF_GEOM_N < 43) {
        printf("  [BAD] GGUF_GEOM has %u rows; the pinned engine has type ids "
               "up to 42\n", (unsigned)GGUF_GEOM_N);
        failures++;
    } else {
        printf("  [ok]  the table covers every type id the engine defines\n");
    }

    /* Control: the comparison must be able to fail. Without this a typo that
     * made `expect` compare a value to itself would report a clean run. */
    checks++;
    {
        const unsigned real_bb = GGUF_GEOM[19].bb;
        const unsigned wrong_bb = (unsigned)sizeof(block_iq3_s);   /* the 2026-09-02 bug */
        if (real_bb == wrong_bb) {
            printf("  [BAD] CONTROL: IQ1_S and IQ3_S have the same block size, "
                   "so this test cannot distinguish the bug it was written for\n");
            failures++;
        } else {
            printf("  [ok]  control: the known-bad IQ1_S value (%u) differs "
                   "from the correct one (%u)\n", wrong_bb, real_bb);
        }
    }

    printf("\n%d checks, %d failures\n", checks, failures);
    if (failures) {
        printf("GGUF_GEOM_TEST_FAIL\n");
        return 1;
    }
    printf("GGUF_GEOM_TEST_OK\n");
    return 0;
}
