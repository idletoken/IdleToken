/* Unit gate for the local admission capability (threat register PROV-28 /
 * PRIV-05 / CHAIN-05).
 *
 * Runs anywhere a C compiler does — no GGUF, no engine, no platform. The
 * coordinator's --selftest calls the same function; this target exists so the
 * capability can be judged on a machine that cannot start a coordinator at all,
 * which is where most of the review of it will happen.
 *
 *   make admissiontest
 *
 * Contract: prints one PASS/FAIL line per assertion and exits non-zero if any
 * assertion failed. */

#include "idletoken_admission.h"
#include <stdio.h>

int main(void) {
    int fails = idletoken_admission_selftest();
    if (fails) {
        fprintf(stderr, "ADMISSION_TEST_FAIL: %d assertion(s)\n", fails);
        return 1;
    }
    printf("ADMISSION_TEST_OK\n");
    return 0;
}
