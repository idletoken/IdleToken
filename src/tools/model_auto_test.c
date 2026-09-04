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
    /* MLA stores the latent row plus its RoPE component, not a conventional
     * K+V pair: 78 * (kv_lora_rank 512 + rope 64) * f16. */
    ok(am.kv_bytes_per_token == 78ull * (512 + 64) * 2,
       "MLA bytes/token derived from latent rank plus RoPE dims");
    ok(am.spec.backend == IDLETOKEN_BACKEND_LLAMACPP && am.spec.available == 1,
       "auto spec is a runnable llamacpp entry");

    /* ---- measured workspace comes from the registry when the id matches ----
     *
     * `plan.c` refuses any model whose compute_bytes are zero, and the client
     * starts every engine through this function (`--llama-gguf`). So if an auto
     * manifest for a CURATED model reports zero here, the product cannot start
     * an engine at all -- which is exactly what shipped until 2026-09-02.
     *
     * Both directions, because copying unconditionally would be as wrong as
     * copying never: an id the registry does not know must keep its zeros and
     * stay refusable. */
    {
        const idletoken_model_spec *reg = idletoken_model_get("glm-5.2");
        char curated[1024];
        snprintf(path, sizeof(path), "%s/glm_dsa.gguf", dir);
        snprintf(curated, sizeof(curated), "%s/glm-5.2.gguf", dir);
        ok(reg != NULL && reg->compute_bytes_256k_cuda > 0,
           "the registry itself carries a measured workspace for glm-5.2");
        ok(copy_file(path, curated) == 0, "curated-name fixture staged");
        ok(idletoken_model_from_gguf(curated, &am, err, sizeof(err)) == 0,
           "a GGUF named after a curated model parses");
        ok(strcmp(am.id, "glm-5.2") == 0, "id resolves to the registry id");
        /* memcmp over the whole KV-tier array, not `==`: these are arrays
         * since 2026-09-02, and `==` compares the two addresses — which is
         * never equal, so the check would fail for the wrong reason. The
         * negative control below had the mirror bug: an array's address is
         * never 0, so `== 0` could never hold either. */
        ok(reg &&
           !memcmp(am.spec.compute_bytes_256k_cuda, reg->compute_bytes_256k_cuda,
                   sizeof reg->compute_bytes_256k_cuda) &&
           !memcmp(am.spec.compute_bytes_1m_cuda, reg->compute_bytes_1m_cuda,
                   sizeof reg->compute_bytes_1m_cuda) &&
           !memcmp(am.spec.compute_bytes_256k_metal, reg->compute_bytes_256k_metal,
                   sizeof reg->compute_bytes_256k_metal) &&
           !memcmp(am.spec.compute_bytes_1m_metal, reg->compute_bytes_1m_metal,
                   sizeof reg->compute_bytes_1m_metal),
           "all four measured workspaces are adopted from the registry, every KV tier");
        remove(curated);

        /* The negative control: the gate must still bite for a GGUF nobody
         * measured. `glm_dsa` is a fixture name, not a registry id. EVERY tier
         * has to stay zero — one populated entry is enough to let a launch
         * through on the precision that selects it. */
        int all_zero = 1;
        if (idletoken_model_from_gguf(path, &am, err, sizeof(err)) != 0 ||
            idletoken_model_get(am.id) != NULL)
            all_zero = 0;
        else
            for (int t = 0; t < IDLETOKEN_KV_TIER_COUNT; t++)
                if (am.spec.compute_bytes_256k_cuda[t] ||
                    am.spec.compute_bytes_1m_cuda[t] ||
                    am.spec.compute_bytes_256k_metal[t] ||
                    am.spec.compute_bytes_1m_metal[t]) all_zero = 0;
        ok(all_zero,
           "an unregistered GGUF keeps zeros in every KV tier, so the planner still refuses it");
    }

    /* ---- the precision an auto-manifest is serving -------------------------
     * An auto-generated manifest describes ONE file, so it carries no variant
     * table (model_auto.c sets n_variants = 0) -- and resolving the served
     * precision against that empty table always answered "". That is not a
     * harmless blank: the marketplace reads an empty quant as "any precision"
     * (providerServesModel in shared/models.ts), so every coordinator the
     * client launches was published as willing to serve every precision of its
     * model. Measured on both Windows nodes 2026-09-03
     * (results/agent-stale-registration-20260903.md).
     *
     * The fix borrows the registry's table for the same id, so this asserts on
     * BOTH: a registered id resolves, an unregistered one still cannot. */
    {
        const idletoken_model_spec *reg = idletoken_model_get("qwen3.8-27b");
        ok(reg != NULL && reg->n_variants > 1,
           "the registry carries a variant table for qwen3.8-27b");

        /* Stand in for what model_auto.c produces: the registry's identity,
         * none of its menu. */
        idletoken_model_spec auto_spec;
        memset(&auto_spec, 0, sizeof(auto_spec));
        auto_spec.id = "qwen3.8-27b";
        auto_spec.n_variants = 0;
        auto_spec.variants = NULL;

        ok(strcmp(idletoken_model_quant_from_gguf(&auto_spec,
                      "D:\\gguf\\Qwen3.8-27B-UD-Q2_K_XL.gguf"), "Q2_K_XL") == 0,
           "a variant-less manifest still names its precision (Windows path)");
        ok(strcmp(idletoken_model_quant_from_gguf(&auto_spec,
                      "/mnt/gguf/Qwen3.8-27B-UD-IQ2_XXS.gguf"), "IQ2_XXS") == 0,
           "...and with a POSIX path and a different variant");
        ok(strcmp(idletoken_model_quant_from_gguf(&auto_spec,
                      "Qwen3.8-27B-UD-Q2_K_XL.gguf"), "Q2_K_XL") == 0,
           "...and with no directory at all");

        /* The registry's OWN spec must keep answering identically -- this is
         * the path the curated cluster launch takes, and it was already right. */
        ok(strcmp(idletoken_model_quant_from_gguf(reg,
                      "D:\\gguf\\Qwen3.8-27B-UD-Q2_K_XL.gguf"), "Q2_K_XL") == 0,
           "a registry spec resolves the same precision it always did");

        /* Negative controls: a name nothing matches, and an id the registry
         * has never heard of. Both must stay blank rather than borrow from
         * some other model's menu -- a wrong precision on the wire is worse
         * than an absent one, because it is believed. */
        ok(idletoken_model_quant_from_gguf(&auto_spec,
               "Qwen3.8-27B-UD-NOT_A_QUANT.gguf")[0] == '\0',
           "an unrecognised file name resolves to no precision");
        idletoken_model_spec unknown = auto_spec;
        unknown.id = "not-a-registered-model";
        ok(idletoken_model_quant_from_gguf(&unknown,
               "Qwen3.8-27B-UD-Q2_K_XL.gguf")[0] == '\0',
           "an unregistered id borrows no other model's variant table");
        ok(idletoken_model_quant_from_gguf(&auto_spec, "")[0] == '\0' &&
           idletoken_model_quant_from_gguf(NULL, "x.gguf")[0] == '\0',
           "empty and NULL inputs answer blank instead of crashing");
    }

    /* ---- deepseek2 fixture: vocab via tokenizer token count, explicit
     * key/value lengths ---------------------------------------------------- */
    snprintf(path, sizeof(path), "%s/deepseek2.gguf", dir);
    ok(idletoken_model_from_gguf(path, &am, err, sizeof(err)) == 0,
       "deepseek2 fixture parses");
    ok(am.spec.n_layers == 61, "deepseek2 layer count");
    ok(am.spec.n_vocab == 1000, "vocab falls back to len(tokenizer.ggml.tokens)");
    ok(am.kv_bytes_per_token == 61ull * (512 + 64) * 2,
       "DeepSeek2 MLA uses latent rank, not generic key/value lengths");

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
