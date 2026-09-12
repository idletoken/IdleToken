#include "../vendor/llama.cpp/ggml/src/ggml-rpc/rpc-graph-capacity.h"
#include <cassert>
#include <cstdio>
#include <limits>

int main() {
    assert(rpc_scheduler_graph_capacity(0) == 32768);
    assert(rpc_scheduler_graph_capacity(1) == 32768);
    assert(rpc_scheduler_graph_capacity(4096) == 32768);
    assert(rpc_scheduler_graph_capacity(7168) == 32768);
    assert(rpc_scheduler_graph_capacity(7169) == 65536);
    assert(rpc_scheduler_graph_capacity(15360) == 65536);
    assert(rpc_scheduler_graph_capacity(15361) == 131072);
    assert(rpc_scheduler_graph_capacity(std::numeric_limits<size_t>::max()) == 0);
    assert(rpc_scheduler_graph_capacity((std::numeric_limits<size_t>::max() - 4096) / 4) == 0);
    size_t previous = 0;
    int allocations = 0;
    for (size_t nodes = 1; nodes <= 20000; ++nodes) {
        const size_t capacity = rpc_scheduler_graph_capacity(nodes);
        assert(capacity >= nodes * 4 + 4096);
        assert(capacity >= previous);
        assert((capacity & (capacity - 1)) == 0);
        if (capacity != previous) ++allocations;
        previous = capacity;
    }
    assert(allocations == 3);
    // A tiny auxiliary graph followed by changing DSv4 graph shapes must
    // keep one scheduler, rather than repeatedly discarding a hot MoE pool.
    const size_t shapes[] = {3, 5650, 5651, 6035, 5646, 6036};
    for (size_t nodes : shapes) assert(rpc_scheduler_graph_capacity(nodes) == 32768);
    std::puts("RPC_GRAPH_CAPACITY_OK");
}
