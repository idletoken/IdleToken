#include "../vendor/llama.cpp/ggml/src/ggml-rpc/rpc-input-cache.h"
#include <cassert>
#include <cstdio>
#include <limits>

int main() {
    rpc_input_cache cache;
    const uint8_t a[] = {1, 2, 3, 4}, b[] = {1, 2, 3, 5};
    assert(!cache.matches(100, a, 4));
    cache.remember(100, a, 4);
    assert(cache.matches(100, a, 4));
    assert(!cache.matches(100, b, 4));
    assert(!cache.matches(100, a, 3));
    assert(!cache.matches(101, a, 4));
    cache.invalidate(104, 4); // exactly adjacent: no overlap
    cache.invalidate(96, 4);
    assert(cache.matches(100, a, 4));
    cache.invalidate(102, 1); // write through a differently named view
    assert(!cache.matches(100, a, 4));
    cache.remember(100, a, 4);
    cache.remember(200, a, 4);
    cache.invalidate(99, 2);
    assert(!cache.matches(100, a, 4));
    assert(cache.matches(200, a, 4));
    cache.clear(); // buffer clear and allocator reset use the same operation
    assert(!cache.matches(200, a, 4));
    const auto end = std::numeric_limits<uint64_t>::max();
    cache.remember(end - 4, a, 4);
    cache.invalidate(end - 1, 1);
    assert(!cache.matches(end - 4, a, 4));
    cache.remember(100, a, 4);
    cache.remember(102, b, 4);
    assert(!cache.matches(100, a, 4));
    assert(cache.matches(102, b, 4));
    cache.invalidate(102, 0);
    assert(cache.matches(102, b, 4));
    std::puts("RPC_INPUT_CACHE_OK");
}
