/* model_registry_dump.c — print the compiled model registry as JSON.
 *
 * Exists so `models/<id>.json` (the planning-side manifest, read by the
 * platform and the client) and `src/common/model.c` (the engine-side registry,
 * compiled into coord/worker) can be DIFFED rather than trusted. The two are
 * hand-maintained copies of the same numbers — layer bytes, KV geometry,
 * context limits — and nothing checked they agreed. A silent divergence does
 * not crash anything; it just makes the resource planner size a cluster from
 * numbers the engine disagrees with, which is the worst kind of wrong.
 *
 * scripts/model_manifest_check.py consumes this. Registry-only models (ones
 * with no manifest yet) are printed too; the checker decides what to require.
 */
#include "idletoken_model.h"

#include <stdio.h>
#include <string.h>

/* The registry has no public iterator (callers look models up by id), so the
 * ids live here. Keep in sync with src/common/model.c — the checker reports a
 * manifest with no registry entry, which is what a forgotten id looks like. */
static const char *IDS[] = {
    "deepseek-v4-flash", "qwen3.5-0.8b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b", "qwen3.5-35b-a3b", "qwen3-8b",
    "glm-5.2", "kimi-k2.5", "kimi-k3",
};

static void jstr(const char *k, const char *v, int last) {
    printf("    \"%s\": \"%s\"%s\n", k, v ? v : "", last ? "" : ",");
}
static void jnum(const char *k, unsigned long long v, int last) {
    printf("    \"%s\": %llu%s\n", k, v, last ? "" : ",");
}

int main(void) {
    printf("{\n");
    int first = 1;
    for (size_t i = 0; i < sizeof(IDS) / sizeof(IDS[0]); i++) {
        const idletoken_model_spec *m = idletoken_model_get(IDS[i]);
        if (!m) continue;
        if (!first) printf("  },\n");
        printf("%s  \"%s\": {\n", first ? "" : "", m->id);
        first = 0;
        jstr("label", m->label, 0);
        jnum("available", m->available, 0);
        jnum("n_layers", m->n_layers, 0);
        jnum("n_embd", m->n_embd, 0);
        jnum("hc_streams", m->hc_streams, 0);
        jnum("n_vocab", m->n_vocab, 0);
        jnum("layer_weight_bytes", m->layer_weight_bytes, 0);
        jnum("shared_weight_bytes", m->shared_weight_bytes, 0);
        jnum("context_max", m->ctx_max, 0);
        jnum("kv_bytes_per_token_per_layer", m->kv_bytes_per_token_layer, 0);
        jnum("state_bytes_per_layer", m->state_bytes_per_layer, 0);
        jnum("full_attention_interval", m->full_attn_interval, 0);
        jnum("overhead_base_bytes", m->overhead_base_bytes, 0);
        jstr("default_gguf", m->default_gguf, 0);
        printf("    \"variants\": [");
        for (uint8_t v = 0; v < m->n_variants; v++)
            printf("%s{\"quant\": \"%s\", \"layer_weight_bytes\": %llu, "
                   "\"shared_weight_bytes\": %llu, \"gguf\": \"%s\"}",
                   v ? ", " : "", m->variants[v].quant,
                   (unsigned long long)m->variants[v].layer_weight_bytes,
                   (unsigned long long)m->variants[v].shared_weight_bytes,
                   m->variants[v].gguf);
        printf("]\n");
    }
    if (!first) printf("  }\n");
    printf("}\n");
    return 0;
}
