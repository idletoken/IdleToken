/* model_auto_test.c — unit tests for the open-model intake (WS-B4):
 * GGUF header -> runtime idletoken_model_spec. Runs against the metadata-only
 * fixtures from scripts/make_test_gguf.py:
 *   make autotest
 * Prints MODEL_AUTO_TEST_OK on success. */
#include "idletoken_model_auto.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int checks = 0, failures = 0;
static void ok(int cond, const char *what) {
    checks++;
    if (cond) { printf("  [ok] %s\n", what); }
    else      { failures++; printf("  [FAIL] %s\n", what); }
}

static int copy_file(const char *src, const char *dst) {
    FILE *in = fopen(src, "rb");
    if (!in) return -1;
    FILE *out = fopen(dst, "wb");
    if (!out) { fclose(in); return -1; }
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), in)) > 0)
        if (fwrite(buf, 1, n, out) != n) { fclose(in); fclose(out); return -1; }
    fclose(in);
    fclose(out);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <fixtures-dir>\n", argv[0]);
        return 2;
    }
    const char *dir = argv[1];
    char path[1024], err[256];
    idletoken_auto_model am;

    /* ---- glm_dsa fixture: explicit vocab_size, no general.name ----------- */
    snprintf(path, sizeof(path), "%s/glm_dsa.gguf", dir);
    ok(idletoken_model_from_gguf(path, &am, err, sizeof(err)) == 0,
       "glm_dsa fixture parses");
    ok(strcmp(am.arch, "glm_dsa") == 0, "architecture read from the header");
    ok(am.spec.n_layers == 78, "layer count = <arch>.block_count");
    ok(am.spec.ctx_max == 1048576, "ctx max = <arch>.context_length");
    ok(am.spec.n_vocab == 154880, "vocab from <arch>.vocab_size");
    ok(strcmp(am.id, "glm_dsa") == 0,
       "no general.name -> id sanitized from the filename stem");
    {
        struct stat st;
        ok(stat(path, &st) == 0 && am.file_bytes == (uint64_t)st.st_size,
           "file_bytes equals the on-disk size");
        /* Metadata-only fixture: everything resident is the header itself. */
        ok(am.spec.layer_weight_bytes + am.spec.shared_weight_bytes >=
               am.file_bytes,
           "layer+shared bytes cover the whole file");
    }
    /* glm_dsa: scalar head_count_kv=1, key_length missing -> derived from
     * embedding_length(6144)/head_count(64) = 96; kv/token =
     * 78 * 1 * (96+96) * 2 = 29952. */
    ok(am.kv_bytes_per_token == 78ull * 1 * (96 + 96) * 2,
       "kv bytes/token derived from kv heads x head dim x layers");
    ok(am.spec.backend == IDLETOKEN_BACKEND_LLAMACPP && am.spec.available == 1,
       "auto spec is a runnable llamacpp entry");

    /* ---- deepseek2 fixture: vocab via tokenizer token count, explicit
     * key/value lengths ---------------------------------------------------- */
    snprintf(path, sizeof(path), "%s/deepseek2.gguf", dir);
    ok(idletoken_model_from_gguf(path, &am, err, sizeof(err)) == 0,
       "deepseek2 fixture parses");
    ok(am.spec.n_layers == 61, "deepseek2 layer count");
    ok(am.spec.n_vocab == 1000, "vocab falls back to len(tokenizer.ggml.tokens)");
    ok(am.kv_bytes_per_token == 61ull * 1 * (192 + 128) * 2,
       "kv bytes/token uses explicit key_length + value_length");

    /* ---- split (multi-file) GGUFs -----------------------------------------
     * Supported since 2026-08-16 (they used to be refused with "merge it
     * first"). Every model past ~50 GB ships split, so refusing them meant the
     * only path that measures the REAL file was unusable for exactly the
     * models whose size matters most. Three behaviours to hold down: a missing
     * part is named, a non-first part is redirected, and a complete set is
     * sized by the SUM — not by part 1, which is a few MiB of header. */
    {
        char src[1024], p2[1024], p3[1024];
        snprintf(src, sizeof(src), "%s/glm_dsa.gguf", dir);
        snprintf(path, sizeof(path), "%s/fake-00001-of-00003.gguf", dir);
        snprintf(p2, sizeof(p2), "%s/fake-00002-of-00003.gguf", dir);
        snprintf(p3, sizeof(p3), "%s/fake-00003-of-00003.gguf", dir);
        ok(copy_file(src, path) == 0, "split-name fixture staged");

        /* (a) part 1 alone: the siblings are missing, and saying which one is
         * missing is the difference between "your download stopped" and a
         * loader crash three minutes later. */
        err[0] = '\0';
        ok(idletoken_model_from_gguf(path, &am, err, sizeof(err)) == -1,
           "split part 1 without its siblings is refused");
        ok(strstr(err, "missing") != NULL && strstr(err, "incomplete") != NULL,
           "the refusal names the missing part, not a generic failure");

        /* (b) pointing at a middle part: redirect to part 1, which is the only
         * one carrying the header. */
        ok(copy_file(src, p3) == 0, "third part staged");
        err[0] = '\0';
        ok(idletoken_model_from_gguf(p3, &am, err, sizeof(err)) == -1 &&
               strstr(err, "part 1") != NULL,
           "a non-first part is refused by pointing at part 1");

        /* (c) complete set: accepted, and sized as the SUM of the parts. */
        ok(copy_file(src, p2) == 0, "second part staged");
        err[0] = '\0';
        const int rc = idletoken_model_from_gguf(path, &am, err, sizeof(err));
        ok(rc == 0, "a complete split set is accepted");
        if (rc == 0) {
            struct stat s1;
            ok(stat(path, &s1) == 0 && am.file_bytes == (uint64_t)s1.st_size * 3,
               "split model size is the sum of every part");
        }
        remove(path); remove(p2); remove(p3);
    }

    /* ---- garbage / missing files fail with a reason ---------------------- */
    snprintf(path, sizeof(path), "%s/not-a.gguf", dir);
    {
        FILE *f = fopen(path, "wb");
        if (f) { fputs("plainly not a gguf", f); fclose(f); }
    }
    err[0] = '\0';
    ok(idletoken_model_from_gguf(path, &am, err, sizeof(err)) == -1 &&
           err[0] != '\0',
       "non-GGUF file is refused with a reason");
    remove(path);
    ok(idletoken_model_from_gguf("/no/such/file.gguf", &am, err, sizeof(err)) == -1,
       "missing file is refused");

    printf("\n%d checks, %d failures\n", checks, failures);
    if (failures) { printf("MODEL_AUTO_TEST_FAIL\n"); return 1; }
    printf("MODEL_AUTO_TEST_OK\n");
    return 0;
}
