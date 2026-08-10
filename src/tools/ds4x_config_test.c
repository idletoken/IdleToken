/* ds4x_config_test.c — unit tests for the ds4x runtime config (Phase B.0).
 * Fixtures come from scripts/make_test_gguf.py (metadata-only GGUFs).
 *   make ds4xtest
 * Pure C, runs on the mac control machine and Linux alike. */
#include "idletoken_ds4x.h"
#include "idletoken_gguf.h"

#include <stdio.h>
#include <string.h>

static int checks = 0, failures = 0;
static void ok(int cond, const char *what) {
    checks++;
    if (cond) { printf("  [ok] %s\n", what); }
    else      { failures++; printf("  [FAIL] %s\n", what); }
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "build/fixtures";
    char path[512], err[256];
    ds4x_config c;

    /* ---- GLM-5.2 shaped fixture (explicit nope/vocab keys) ------------- */
    snprintf(path, sizeof(path), "%s/glm_dsa.gguf", dir);
    ok(ds4x_config_from_gguf(path, &c, err, sizeof(err)) == 0, "glm_dsa parses");
    if (failures) { printf("  err: %s\n", err); return 1; }
    ok(c.arch == DS4X_ARCH_GLM_DSA, "arch = GLM_DSA");
    ok(c.n_layer == 78 && c.n_embd == 6144, "layers 78 / hidden 6144");
    ok(c.n_head == 64 && c.n_head_kv == 1, "heads 64 / kv heads 1");
    ok(c.kv_lora_rank == 512 && c.q_lora_rank == 2048, "MLA ranks 512/2048");
    ok(c.qk_nope_head_dim == 192 && c.qk_rope_head_dim == 64 && c.v_head_dim == 256,
       "head geometry 192+64 / v 256");
    ok(c.n_expert == 256 && c.n_expert_used == 8 && c.n_expert_shared == 1,
       "MoE 256 top-8 +1 shared");
    ok(c.n_dense_lead == 3 && c.n_ff_dense == 12288 && c.n_ff_exp == 2048,
       "3 dense lead / widths 12288+2048");
    ok(c.rope_theta > 7.9e6f && c.rope_theta < 8.1e6f, "rope theta 8e6");
    ok(c.n_vocab == 154880, "vocab from vocab_size key");
    ok(c.index_topk == 2048 && c.index_group == 4, "DSA indexer topk 2048 / group 4");
    ok(ds4x_kv_bytes_per_token_layer(&c) == 1152,
       "MLA KV = (512+64)*2 = 1152 B/token/layer (matches manifest)");

    /* runtime truth vs planning truth */
    ok(ds4x_config_check_spec(&c, idletoken_model_get("glm-5.2"), err, sizeof(err)) == 0,
       "matches the glm-5.2 registry spec");
    ok(ds4x_config_check_spec(&c, idletoken_model_get("deepseek-v4-flash"),
                              err, sizeof(err)) == -1,
       "refuses the DSv4 spec (wrong-file guard)");
    ok(strstr(err, "mismatch") != NULL, "mismatch reason is explicit");

    /* ---- deepseek2 fixture (key_length variant + vocab via tokens) ----- */
    snprintf(path, sizeof(path), "%s/deepseek2.gguf", dir);
    ok(ds4x_config_from_gguf(path, &c, err, sizeof(err)) == 0, "deepseek2 parses");
    if (failures) { printf("  err: %s\n", err); return 1; }
    ok(c.arch == DS4X_ARCH_DEEPSEEK2, "arch = DEEPSEEK2");
    ok(c.n_layer == 61 && c.n_embd == 7168, "layers 61 / hidden 7168");
    ok(c.qk_nope_head_dim == 128 && c.qk_rope_head_dim == 64,
       "nope derived from key_length (192-64)");
    ok(c.v_head_dim == 128, "v from value_length");
    ok(c.n_expert == 384 && c.n_expert_used == 8, "MoE 384 top-8");
    ok(c.n_vocab == 1000, "vocab falls back to len(tokenizer.ggml.tokens)");
    /* deepseek2 fixture omits router-weighting keys → llama.cpp defaults */
    ok(c.gating_func == DS4X_GATE_SOFTMAX, "gating defaults to softmax when absent");
    ok(c.expert_weights_norm == 0 && c.expert_weights_scale == 1.0f,
       "weights norm/scale default off");

    /* ---- refusal of non-MLA architectures ------------------------------ */
    snprintf(path, sizeof(path), "%s/unknown_arch.gguf", dir);
    ok(ds4x_config_from_gguf(path, &c, err, sizeof(err)) == -1,
       "unknown arch refused");
    ok(strstr(err, "not supported") != NULL, "refusal names the reason");

    /* ---- missing file is an error, not a crash ------------------------- */
    ok(ds4x_config_from_gguf("/no/such/file.gguf", &c, err, sizeof(err)) == -1,
       "missing file → error");

    /* ---- model identity (idletoken_gguf_identity) -------------------------
     * The coordinator sends this in ASSIGN_PLAN and every worker compares it
     * against its own copy, so the two properties that matter are: same file →
     * same digest, and different model → different digest. Without the second
     * one a "verification" that always matches would look just as green. */
    {
        uint8_t h1[32], h2[32], hd[32];
        char ie[256] = "";
        snprintf(path, sizeof(path), "%s/deepseek2.gguf", dir);
        ok(idletoken_gguf_identity(path, h1, ie, sizeof ie) == 0, "identity: hashes a GGUF");
        ok(idletoken_gguf_identity(path, h2, ie, sizeof ie) == 0, "identity: hashes it again");
        ok(memcmp(h1, h2, 32) == 0, "identity: same file → same digest");
        int nonzero = 0;
        for (int i = 0; i < 32; i++) if (h1[i]) { nonzero = 1; break; }
        ok(nonzero, "identity: digest is not all-zero (all-zero means 'no identity')");
        snprintf(path, sizeof(path), "%s/unknown_arch.gguf", dir);
        ok(idletoken_gguf_identity(path, hd, ie, sizeof ie) == 0, "identity: hashes another GGUF");
        ok(memcmp(h1, hd, 32) != 0, "identity: different model → different digest");
        ok(idletoken_gguf_identity("/no/such/file.gguf", hd, ie, sizeof ie) == -1,
           "identity: missing file → error, not a bogus digest");
    }

    printf("\n%d checks, %d failures\n", checks, failures);
    if (failures) { printf("DS4X_CONFIG_TEST_FAIL\n"); return 1; }
    printf("DS4X_CONFIG_TEST_OK\n");
    return 0;
}
