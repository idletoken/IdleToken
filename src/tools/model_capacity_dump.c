/* Emit byte-exact resource-planner cases for the client parity gate.
 *
 * This is deliberately the compiled C planner, not a parser over model.c.
 * scripts/resource_estimate_gate.sh compares these results with the TypeScript
 * client for every curated model, precision, context tier and node count.
 */
#include "idletoken_model.h"
#include "idletoken_modelsize.h"
#include "idletoken_plan.h"

#include <stdint.h>
#include <stdio.h>

static void scale_for_quant(idletoken_llm_model_size *size, const char *quant) {
    const int bits = idletoken_quant_weight_bits(quant);
    if (bits >= 1 && bits <= 2)
        idletoken_llama_model_kv_scale(size, 18.0 / 64.0);
    else if (bits >= 3 && bits <= 4)
        idletoken_llama_model_kv_scale(size, 34.0 / 64.0);
}

int main(void) {
    static const uint32_t contexts[] = {262144, 1048576};
    static const int node_counts[] = {1, 2, 4};
    static const uint8_t backends[] = { IDLETOKEN_NODE_BACKEND_CUDA,
                                        IDLETOKEN_NODE_BACKEND_METAL };
    static const char *const backend_names[] = { "cuda", "metal" };
    int first = 1;
    printf("{\"cases\":[\n");
    for (int mi = 0; mi < idletoken_model_count(); mi++) {
        const idletoken_model_spec *m = idletoken_model_at(mi);
        if (!m) continue;
        const int nq = m->n_variants ? m->n_variants : 1;
        for (int qi = 0; qi < nq; qi++) {
            const char *quant = m->n_variants ? m->variants[qi].quant : NULL;
            idletoken_llm_model_size size;
            if (idletoken_model_size_resolve(m, quant, NULL, &size, NULL, 0) != 0)
                return 2;
            scale_for_quant(&size, quant);
            const uint32_t cap = idletoken_llama_product_ctx_ceiling(m);
            for (size_t ci = 0; ci < sizeof(contexts) / sizeof(contexts[0]); ci++) {
                const uint32_t ctx = contexts[ci];
                if (ctx > cap) continue;
                const uint64_t kv = idletoken_llama_kv_bytes(&size, ctx);
                for (size_t ni = 0; ni < sizeof(node_counts) / sizeof(node_counts[0]); ni++) {
                    const int nodes = node_counts[ni];
                    /* Both backends, because the workspace differs between them
                     * by up to 22x (GLM-5.2) and a gate that checked only one
                     * would leave the other free to drift. */
                    for (size_t bi = 0; bi < sizeof(backends) / sizeof(backends[0]); bi++) {
                    const uint8_t backend = backends[bi];
                    const uint64_t hard =
                        idletoken_llama_hard_need(&size, ctx, nodes, backend);
                    const uint64_t overhead = hard - kv;
                    const uint64_t need = size.total_bytes + hard;
                    printf("%s{\"model\":\"%s\",\"quant\":\"%s\","
                           "\"ctx\":%u,\"nodes\":%d,\"backend\":\"%s\","
                           "\"weights\":%llu,"
                           "\"kv\":%llu,\"overhead\":%llu,\"need\":%llu}",
                           first ? "" : ",\n", m->id, quant ? quant : "",
                           ctx, nodes, backend_names[bi],
                           (unsigned long long)size.total_bytes,
                           (unsigned long long)kv,
                           (unsigned long long)overhead,
                           (unsigned long long)need);
                    first = 0;
                    }
                }
            }
        }
    }
    printf("\n]}\n");
    return 0;
}
