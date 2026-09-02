/* Spend one admission capability from the command line.
 *
 * The coordinator is the only thing that consumes these in production. This
 * tool exists so a SHELL gate can do it too — specifically so
 * scripts/admission_origin_gate.sh can take a capability that the REAL platform
 * agent binary minted, in its own process, and prove that the coordinator's own
 * verifier accepts it exactly once (threat register PROV-28).
 *
 * That two-process shape is the point. Every assertion inside a single process
 * can hold while the shipping pair does not — the repo has lost gates to
 * precisely that (docs/: an oracle that shares the assumption under test is not
 * an oracle). Here the minter is a separate binary, started separately, and the
 * only thing they share is the 0600 channel key the coordinator published.
 *
 *   admission_verify <channel-key-file> <capability> <body-file> [--twice]
 *
 * Prints one line: "ADMIT_OK <job-id>" or "ADMIT_REFUSED <reason>", and exits
 * 0 only on the first. With --twice it spends the SAME capability a second time
 * and requires that attempt to be refused as already spent; that is the
 * single-use half of the contract, and it is not observable from one call.
 *
 * Deliberately NOT a wrapper that re-implements anything: it links the same
 * src/common/admission.c the coordinator links. What it adds is the ability to
 * run that code against bytes produced elsewhere. */

#include "idletoken_admission.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int read_all(const char *path, unsigned char **out, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    size_t cap = 65536, len = 0;
    unsigned char *buf;
    if (!f) return -1;
    buf = (unsigned char *)malloc(cap);
    if (!buf) { fclose(f); return -1; }
    for (;;) {
        size_t n = fread(buf + len, 1, cap - len, f);
        len += n;
        if (len < cap) break;
        cap *= 2;
        {
            unsigned char *nb = (unsigned char *)realloc(buf, cap);
            if (!nb) { free(buf); fclose(f); return -1; }
            buf = nb;
        }
    }
    fclose(f);
    *out = buf;
    *out_len = len;
    return 0;
}

int main(int argc, char **argv) {
    char err[240] = "", job[IDLETOKEN_ADM_JOB_CAP] = "";
    unsigned char *body = NULL;
    size_t body_len = 0;
    unsigned char hash[32];
    idletoken_adm_rc rc;
    int twice = 0, i;

    if (argc < 4) {
        fprintf(stderr,
                "usage: %s <channel-key-file> <capability> <body-file> [--twice]\n",
                argv[0]);
        return 2;
    }
    for (i = 4; i < argc; i++)
        if (!strcmp(argv[i], "--twice")) twice = 1;

    /* Attach, do not init: init would MINT a fresh channel and then verify
     * against itself, which would pass no matter what the agent produced. */
    if (idletoken_admission_attach_file(argv[1], err, sizeof err) != 0) {
        printf("ADMIT_REFUSED could not load the channel key: %s\n",
               err[0] ? err : "unknown error");
        return 1;
    }
    if (read_all(argv[3], &body, &body_len) != 0) {
        printf("ADMIT_REFUSED could not read the body file %s\n", argv[3]);
        return 1;
    }
    idletoken_admission_body_hash(body, body_len, hash);

    rc = idletoken_admission_consume(argv[2], hash, (long long)time(NULL),
                                     job, sizeof job);
    if (rc != IDLETOKEN_ADM_OK) {
        printf("ADMIT_REFUSED %s\n", idletoken_admission_rc_str(rc));
        free(body);
        return 1;
    }
    idletoken_admission_request_end();

    if (twice) {
        rc = idletoken_admission_consume(argv[2], hash, (long long)time(NULL),
                                         NULL, 0);
        if (rc != IDLETOKEN_ADM_REPLAYED) {
            printf("ADMIT_REFUSED the same capability was spendable twice (%s)\n",
                   idletoken_admission_rc_str(rc));
            free(body);
            return 1;
        }
    }
    printf("ADMIT_OK %s\n", job);
    free(body);
    return 0;
}
