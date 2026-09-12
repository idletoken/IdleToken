#include "../vendor/llama.cpp/ggml/src/ggml-moe-cache-pin.h"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <set>

struct pool {
    std::vector<int32_t> slots;
    std::vector<uint64_t> last, pins;
    uint64_t tick = 8;

    size_t misses(const std::set<int32_t> & selected) const {
        size_t n = 0;
        for (int32_t e : selected) n += std::find(slots.begin(), slots.end(), e) == slots.end();
        return n;
    }

    size_t serve(const std::set<int32_t> & selected, bool protect_first) {
        ++tick;
        if (protect_first) ggml_moe_cache::pin_selected(slots, pins, tick,
            [&](int32_t e) { return selected.count(e) != 0; });
        size_t copies = 0;
        for (int32_t e : selected) {
            auto found = std::find(slots.begin(), slots.end(), e);
            size_t slot = slots.size();
            if (found != slots.end()) slot = size_t(found - slots.begin());
            else {
                uint64_t oldest = UINT64_MAX;
                for (size_t s = 0; s < slots.size(); ++s) {
                    if (pins[s] != tick && last[s] < oldest) { oldest = last[s]; slot = s; }
                }
                assert(slot < slots.size());
                slots[slot] = e;
                ++copies;
            }
            pins[slot] = last[slot] = tick;
        }
        assert(misses(selected) == 0);
        return copies;
    }
};

int main() {
    pool old{{7, 9}, {1, 2}, {0, 0}};
    pool fixed = old;
    const std::set<int32_t> selected{0, 7};
    assert(old.serve(selected, false) == 2); // positive control: needless reload
    assert(fixed.serve(selected, true) == 1);
    assert(fixed.serve(selected, true) == 0);
    std::mt19937 random(117);
    size_t routes = 0;
    for (size_t cap = 1; cap <= 256; ++cap) {
        pool p;
        p.slots.assign(cap, -1);
        p.last.assign(cap, 0);
        p.pins.assign(cap, 0);
        std::vector<int32_t> ids(256);
        std::iota(ids.begin(), ids.end(), 0);
        for (int step = 0; step < 64; ++step) {
            std::shuffle(ids.begin(), ids.end(), random);
            const size_t used = 1 + random() % std::min<size_t>(cap, 32);
            const std::set<int32_t> route(ids.begin(), ids.begin() + used);
            const size_t needed = p.misses(route);
            assert(p.serve(route, true) == needed);
            ++routes;
        }
    }
    std::printf("MOE_CACHE_PIN_TEST_OK: positive control and %zu exact-route transitions\n", routes);
}
