#include "../vendor/llama.cpp/ggml/src/ggml-moe-inline-ids.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

int main() {
    using namespace ggml_moe_inline_ids;
    static_assert(std::is_trivially_copyable<payload>::value, "CUDA launch payload must copy by value");
    static_assert(sizeof(payload) <= 4096 - 2 * sizeof(size_t), "portable CUDA parameter budget");
    std::mt19937 random(3911);
    payload packed{};
    assert(!pack(packed, nullptr, 4));
    int32_t one = -1;
    assert(!pack(packed, &one, 0));
    assert(!pack(packed, &one, 3));
    assert(!pack(packed, &one, sizeof(payload) + 4));
    for (size_t count = 1; count <= capacity; ++count) {
        std::vector<int32_t> source(count);
        for (auto & value : source) value = (int32_t) random();
        source.front() = std::numeric_limits<int32_t>::min();
        if (count > 1) source.back() = std::numeric_limits<int32_t>::max();
        const auto expected = source;
        // Unaligned host input is valid: pack uses memcpy, not typed loads.
        std::vector<uint8_t> unaligned(count * sizeof(int32_t) + 1);
        std::memcpy(unaligned.data() + 1, source.data(), count * sizeof(int32_t));
        assert(pack(packed, unaligned.data() + 1, count * sizeof(int32_t)));
        std::fill(source.begin(), source.end(), 0);
        std::fill(unaligned.begin(), unaligned.end(), 0);
        for (size_t i = 0; i < count; ++i) assert(packed.values[i] == expected[i]);
        for (size_t i = count; i < capacity; ++i) assert(packed.values[i] == 0);
    }
    std::puts("MOE_INLINE_IDS_PACK_OK: all sizes, invalid bounds, signed bits and owned launch payload");
}
