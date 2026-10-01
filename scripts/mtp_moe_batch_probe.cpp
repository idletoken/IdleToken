#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <set>
#include <vector>

// A real three-weight MoE graph, with host-resident experts and owner-GPU
// routing. No model, network, tokenizer or generated-text oracle is involved.
// Build against the same GGML libraries as llama-server. Set a cache cap of 40
// to cover both a 32-expert verification union and a 64-expert staging union.
int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: moe_batch_probe DEVICE [cpu-ids|strided-ids|multi-only]\n"); return 2; }
    const bool cpu_ids = argc > 2 && std::strcmp(argv[2], "cpu-ids") == 0;
    const bool strided = argc > 2 && std::strcmp(argv[2], "strided-ids") == 0;
    const bool multi_only = argc > 2 && std::strcmp(argv[2], "multi-only") == 0;
    ggml_backend_load_all();
    auto dev = ggml_backend_dev_by_name(argv[1]);
    if (!dev) { std::fprintf(stderr, "missing backend device\n"); return 2; }
    auto gpu = ggml_backend_dev_init(dev, nullptr);
    auto cpu = ggml_backend_cpu_init();
    ggml_backend_t backends[] = { gpu, cpu };
    auto sched = ggml_backend_sched_new(backends, nullptr, 2, 1024, false, true);
    constexpr int width = 32, experts = 64, topk = 8;
    ggml_init_params meta = { 4 * 1024 * 1024, nullptr, true };
    auto wctx = ggml_init(meta);
    ggml_tensor * weights[3];
    const char * names[] = { "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight" };
    std::vector<float> weight_values[3];
    for (int p = 0; p < 3; ++p) {
        weights[p] = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, width, width, experts);
        ggml_set_name(weights[p], names[p]);
        weight_values[p].resize(width * width * experts);
        for (int e = 0; e < experts; ++e)
            for (int row = 0; row < width; ++row)
                for (int col = 0; col < width; ++col)
                    weight_values[p][(e * width + row) * width + col] =
                        float((p + 1) * 7 + (e * 7 + row * 3 + col) % 17) / 64.0f;
    }
    auto wbuf = ggml_backend_alloc_ctx_tensors(wctx, cpu);
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (int p = 0; p < 3; ++p)
        ggml_backend_tensor_set(weights[p], weight_values[p].data(), 0, ggml_nbytes(weights[p]));

    const int shapes[] = { 1, 2, 4, 65, 4, 128, 129, 4, 1 };
    uint64_t digest = UINT64_C(1469598103934665603);
    int cases = 0;
    double worst = 0;
    for (int cycle = 0; cycle < 4; ++cycle) for (int shape : shapes) {
        const int nt = multi_only && shape < 4 ? 4 : shape;
        auto inputs = ggml_init(meta);
        auto idctx = ggml_init(meta);
        auto graphctx = ggml_init(meta);
        auto x = ggml_new_tensor_3d(inputs, GGML_TYPE_F32, width, topk, nt);
        const int id_stride = topk + (strided ? 4 : 0);
        auto id_storage = ggml_new_tensor_2d(idctx, GGML_TYPE_I32, id_stride, nt);
        auto ids = strided ? ggml_view_2d(idctx, id_storage, topk, nt, id_storage->nb[1], 0) : id_storage;
        ggml_set_name(x, "owner_input"); ggml_set_input(x);
        ggml_set_name(ids, "routed_ids"); ggml_set_input(ids);
        auto xbuf = ggml_backend_alloc_ctx_tensors(inputs, gpu);
        auto idbuf = ggml_backend_alloc_ctx_tensors(idctx, cpu_ids ? cpu : gpu);
        std::vector<float> values(width * topk * nt);
        std::vector<int32_t> route(topk * nt);
        std::vector<int32_t> stored_route(id_stride * nt, 63);
        std::set<int> selected;
        for (size_t i = 0; i < values.size(); ++i) values[i] = float(int(i % 11) - 5) / 32.0f;
        for (int t = 0; t < nt; ++t) for (int k = 0; k < topk; ++k) {
            const int e = (cases * 11 + t * topk + k) % experts;
            route[t * topk + k] = e; stored_route[t * id_stride + k] = e; selected.insert(e);
        }
        auto graph = ggml_new_graph_custom(graphctx, 1024, false);
        ggml_tensor * outputs[3];
        for (int p = 0; p < 3; ++p) {
            outputs[p] = ggml_mul_mat_id(graphctx, weights[p], x, ids);
            ggml_set_output(outputs[p]);
            ggml_backend_sched_set_tensor_backend(sched, outputs[p], gpu);
            ggml_build_forward_expand(graph, outputs[p]);
        }
        if (!ggml_backend_sched_alloc_graph(sched, graph)) return 3;
        ggml_backend_tensor_set(x, values.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(id_storage, stored_route.data(), 0, ggml_nbytes(id_storage));
        if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) return 4;
        double max_abs = 0;
        for (int p = 0; p < 3; ++p) {
            std::vector<float> actual(width * topk * nt);
            ggml_backend_tensor_get(outputs[p], actual.data(), 0, ggml_nbytes(outputs[p]));
            for (int t = 0; t < nt; ++t) for (int k = 0; k < topk; ++k) for (int row = 0; row < width; ++row) {
                const int e = route[t * topk + k];
                double expected = 0;
                for (int col = 0; col < width; ++col)
                    expected += double(weight_values[p][(e * width + row) * width + col]) *
                                values[(t * topk + k) * width + col];
                const float got = actual[(t * topk + k) * width + row];
                if (!std::isfinite(got)) return 5;
                max_abs = std::max(max_abs, std::abs(double(got) - expected));
            }
            const auto * bytes = reinterpret_cast<const unsigned char *>(actual.data());
            for (size_t i = 0; i < actual.size() * sizeof(float); ++i)
                digest = (digest ^ bytes[i]) * UINT64_C(1099511628211);
        }
        if (max_abs > 1e-5) { std::fprintf(stderr, "NUMERIC_FAIL nt=%d error=%.9g\n", nt, max_abs); return 6; }
        worst = std::max(worst, max_abs);
        std::printf("CASE index=%d nt=%d union=%zu ids_bytes=%zu max_abs=%.9g\n",
                    cases, nt, selected.size(), ggml_nbytes(ids), max_abs);
        ggml_backend_sched_reset(sched);
        ggml_free(graphctx);
        ggml_backend_buffer_free(xbuf); ggml_backend_buffer_free(idbuf);
        ggml_free(inputs); ggml_free(idctx);
        ++cases;
    }
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(wbuf); ggml_free(wctx);
    ggml_backend_free(gpu); ggml_backend_free(cpu);
    std::printf("MOE_BATCH_PROBE_OK cases=%d cpu_ids=%d strided=%d max_abs=%.9g digest=%016llx\n",
                cases, int(cpu_ids), int(strided), worst, (unsigned long long) digest);
    return 0;
}
