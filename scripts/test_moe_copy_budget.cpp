// No GPU or model dependency: independently exercise exact byte admission.
#include "../vendor/llama.cpp/ggml/src/ggml-moe-copy-budget.h"
#include <cassert>
#include <cstdio>
#include <limits>

int main() {
    using V = std::vector<uint8_t>;
    assert(ggml_moe_copy_budget_pick({3, 3, 3, 3}, 0) == V({0, 0, 0, 0}));
    assert(ggml_moe_copy_budget_pick({3, 3, 3, 3}, 8) == V({1, 1, 0, 0}));
    assert(ggml_moe_copy_budget_pick({3, 3, 3, 3}, 9) == V({1, 1, 1, 0}));
    assert(ggml_moe_copy_budget_pick({0, 3, 0, 3}, 6) == V({0, 1, 0, 1}));
    assert(ggml_moe_copy_budget_pick({5, 2, 3, 1}, 4) == V({0, 1, 0, 1}));
    assert(ggml_moe_copy_budget_pick({}, 8).empty());
    const size_t max = std::numeric_limits<size_t>::max();
    assert(ggml_moe_copy_budget_pick({max, 1}, max) == V({1, 0}));
    // Exhaustively check small budgets against direct cumulative accounting.
    for (size_t budget = 0; budget < 100; budget++) {
        const std::vector<size_t> bytes = {0, 4, 6, 2, 4, 0, 5, 3, 2, 1};
        const auto selected = ggml_moe_copy_budget_pick(bytes, budget);
        size_t used = 0;
        for (size_t i = 0; i < bytes.size(); i++) {
            const bool expect = bytes[i] != 0 && used + bytes[i] <= budget;
            assert(bool(selected[i]) == expect);
            if (selected[i]) used += bytes[i];
        }
        assert(used <= budget);
    }
    std::puts("MOE_COPY_BUDGET_OK");
}
