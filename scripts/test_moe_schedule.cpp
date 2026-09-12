#include "../vendor/llama.cpp/ggml/src/ggml-moe-schedule.h"

#include <cassert>
#include <cstdio>
#include <random>

using namespace ggml_moe_schedule;

static layer fixture(int id, size_t experts = 64, size_t active = 4) {
    layer l;
    l.id = id;
    l.experts = experts;
    l.active = active;
    l.window_ms = 2;
    l.parts = {{100, .1}, {200, .2}, {300, .3}};
    return l;
}

static void memory_and_time() {
    const auto l = fixture(0);
    auto p = make_plan({l}, 30000, 1000, 1, 0, 1);
    assert(p.valid && p.layers[0].rolling == 4);
    assert(p.bank_bytes == 2400 && p.staging_bytes == 4800);
    assert(p.layers[0].resident == 42 && p.total_bytes == 30000);
    p = make_plan({l}, 30000, 6000, 1, 0, 1);
    assert(p.valid && p.staging_bytes == 6000 && p.layers[0].resident == 40);
    p = make_plan({l}, 3400, 1000, 1, 0, 1);
    assert(p.valid && p.layers[0].resident == 4 && p.layers[0].rolling == 0);
    assert(!make_plan({l}, 3399, 1000, 1, 0).valid);

    auto fast = l;
    for (auto & part : fast.parts) part.copy_ms *= .5;
    p = make_plan({fast}, 30000, 1000, 1, 0, 1);
    assert(p.valid && p.layers[0].rolling == 7 && p.layers[0].resident == 36);
    auto unknown = l;
    unknown.parts[0].copy_ms = -1;
    p = make_plan({unknown}, 30000, 1000, 1, 0);
    assert(p.valid && p.layers[0].rolling == 0 && p.layers[0].resident == 48);
    unknown = l;
    unknown.window_ms = .01;
    p = make_plan({unknown}, 30000, 1000, 1, 0);
    assert(p.valid && p.layers[0].rolling == 0);
    auto partial = l;
    partial.window_ms = .25; // Less than one expert, more than its first part.
    p = make_plan({partial}, 30000, 6000, 1, 0, 1);
    assert(p.valid && p.layers[0].rolling == 1);
    auto picked = admit(partial, p.layers[0], {{0, false}, {1, false}, {2, false}});
    assert(picked.slots == std::vector<int>({0, -1, -1}));
    assert(picked.copy_ms == .1 && picked.bytes == 100);
    partial.window_ms = .65; // Residual .05 cannot justify another slot.
    p = make_plan({partial}, 30000, 6000, 1, 0, 1);
    assert(p.layers[0].rolling == 1);

    auto mixed = fixture(7, 128, 8);
    mixed.parts = {{800, .05}, {33, .025}}; // Fused or asymmetric expert tensors.
    p = make_plan({l, mixed}, 90000, 4096, 256, 512);
    assert(p.valid && p.total_bytes <= 90000);
    assert(p.layers[1].rolling > p.layers[0].rolling);
    assert(p.layers[0].resident >= 4 && p.layers[1].resident >= 8);
    assert(p.bank_bytes % 256 == 0);
    assert(p.staging_bytes >= 2 * p.bank_bytes && p.staging_bytes >= 4096);
}

static void invalid_and_overflow() {
    auto l = fixture(0);
    assert(!make_plan({}, 100000, 0, 256, 512).valid);
    assert(!make_plan({l, l}, 100000, 0, 256, 512).valid);
    assert(!make_plan({l}, 100000, 0, 0, 512).valid);
    assert(!make_plan({l}, 100000, 0, 1, 0, 0).valid);
    assert(!make_plan({l}, 100000, 0, 1, 0, NAN).valid);
    l.active = 65;
    assert(!make_plan({l}, 100000, 0, 1, 0).valid);
    l = fixture(0);
    l.parts[0].bytes = SIZE_MAX;
    assert(!make_plan({l}, SIZE_MAX, 0, 256, 512).valid);
    l = fixture(0);
    l.parts[0].bytes = 0;
    assert(!make_plan({l}, SIZE_MAX, 0, 1, 0).valid);
    size_t out = 0;
    assert(!add(SIZE_MAX, 1, out));
    assert(!mul(SIZE_MAX, 2, out));
    assert(!align(SIZE_MAX, 256, out));
    assert(align(777, 3, out) && out == 777);
    assert(align(778, 3, out) && out == 780);
}

static void integer_remainder() {
    std::vector<layer> layers;
    for (int i = 0; i < 4; ++i) {
        auto l = fixture(i, 6, 2);
        l.window_ms = .25;
        l.parts = {{524288, .05}, {524288, .05}, {524288, .05}};
        layers.push_back(l);
    }
    const auto p = make_plan(layers, 28318208, 3146240, 256, 512);
    assert(p.valid && p.bank_bytes == 3147264 && p.staging_bytes == 6294528);
    assert(p.layers[0].resident == 4 && p.layers[1].resident == 3);
    assert(p.layers[2].resident == 3 && p.layers[3].resident == 3);
    assert(p.total_bytes == 26747904); // Common rounding alone strands one usable resident slot.
    assert(28318208 - p.total_bytes < 3 * 524288);
}

static void ranked_admission() {
    const auto l = fixture(0);
    allocation a;
    a.rolling = 2;
    a.copy_budget_ms = .75;
    const auto picked = admit(l, a, {{0, true}, {1, false}, {2, false},
                                     {0, false}, {1, false}, {0, false}});
    assert(picked.slots == std::vector<int>({-1, 0, 0, 0, -1, 1}));
    assert(picked.bytes == 700 && picked.copy_ms <= .75);
    a.copy_budget_ms = 0;
    assert(admit(l, a, {{0, false}}).bytes == 0);
    a.copy_budget_ms = NAN;
    assert(admit(l, a, {{0, false}}).bytes == 0);
    a.copy_budget_ms = 100;
    a.rolling = 1;
    assert(admit(l, a, {{0, false}, {0, false}, {99, false}}).slots ==
           std::vector<int>({0, -1, -1}));
}

static void completion_and_reuse() {
    bank b[2];
    assert(b[0].begin(3));
    const auto first = b[0].generation;
    assert(!b[0].consume(3, first)); // Submission is not a ready hit.
    assert(!b[0].begin(4));
    assert(b[1].begin(4));          // Only the other bank can be filled.
    assert(!b[0].complete(first + 1));
    assert(b[0].complete(first));
    assert(!b[0].consume(4, first));
    assert(b[0].consume(3, first));
    assert(!b[0].begin(5));
    assert(b[0].retire());
    assert(!b[0].release(first, false));
    assert(!b[0].release(first + 1, true));
    assert(b[0].release(first, true));
    assert(b[0].begin(5));
    assert(!b[0].complete(first));  // A late event from the previous owner.
    assert(b[0].retire());         // Wrong predictions also need a completion fence.
    assert(!b[0].release(b[0].generation, false));
    assert(b[0].release(b[0].generation, true));
}

static void measurements() {
    samples s;
    assert(s.quantile(.9) < 0);
    assert(!s.push(NAN) && !s.push(INFINITY) && !s.push(-1));
    for (int i = 1; i <= 8; ++i) assert(s.push(i));
    assert(s.quantile(.1) == 1 && s.quantile(.9) == 8);
    assert(s.quantile(0) < 0 && s.quantile(1.1) < 0);
    for (int i = 0; i < 32; ++i) s.push(2);
    assert(s.values.size() == 32 && s.quantile(.9) == 2);
}

static void randomized() {
    std::mt19937 random(92841);
    for (int trial = 0; trial < 4000; ++trial) {
        std::vector<layer> layers;
        for (unsigned i = 0, n = 1 + random() % 15; i < n; ++i) {
            auto l = fixture((int) i, 8 + random() % 505, 1 + random() % 8);
            l.window_ms = (random() % 400) * .01;
            l.parts.clear();
            for (unsigned j = 0, count = 1 + random() % 4; j < count; ++j) {
                l.parts.push_back({1 + random() % 12345, .01 + (random() % 20) * .01});
            }
            layers.push_back(l);
        }
        const size_t budget = random() % 8000000;
        const size_t prefill = random() % 100000;
        const size_t alignment = 1 + random() % 512;
        const auto p = make_plan(layers, budget, prefill, alignment, 512);
        if (!p.valid) continue;
        size_t actual_resident = 0, actual_bank = 0;
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto & a = p.layers[i];
            const auto & l = layers[i];
            assert(a.resident >= l.active && a.resident <= l.experts);
            assert(a.rolling <= l.experts);
            size_t resident = 0, rolling = 0;
            // Independent byte oracle, without the planner's checked helpers.
            for (const auto & part : l.parts) {
                resident = ((resident + alignment - 1) / alignment) * alignment;
                resident += a.resident * part.bytes + 512;
                if (a.rolling) {
                    rolling = ((rolling + alignment - 1) / alignment) * alignment;
                    rolling += a.rolling * part.bytes + 512;
                }
            }
            actual_resident += ((resident + alignment - 1) / alignment) * alignment;
            actual_bank = std::max(actual_bank, ((rolling + alignment - 1) / alignment) * alignment);
            // Storage may include a partial final expert; independently check
            // the actual selected parts rather than assuming all slots copy.
            std::vector<candidate> candidates;
            for (size_t expert = 0; expert < l.experts; ++expert) {
                for (size_t part = 0; part < l.parts.size(); ++part) candidates.push_back({part, (random() % 3) == 0});
            }
            const auto picked = admit(l, a, candidates);
            double copy_ms = 0;
            size_t bytes = 0;
            for (size_t j = 0; j < candidates.size(); ++j) {
                if (picked.slots[j] < 0) continue;
                assert(!candidates[j].resident && (size_t) picked.slots[j] < a.rolling);
                copy_ms += l.parts[candidates[j].part_index].copy_ms;
                bytes += l.parts[candidates[j].part_index].bytes;
            }
            assert(copy_ms <= a.copy_budget_ms + 1e-9 && bytes == picked.bytes);
            assert(bytes <= a.rolling_bytes);
        }
        assert(actual_bank == p.bank_bytes && actual_resident == p.resident_bytes);
        assert(actual_resident + std::max(prefill, 2 * actual_bank) == p.total_bytes);
        assert(p.total_bytes <= budget);
        for (size_t i = 0; i < layers.size(); ++i) {
            if (p.layers[i].resident == layers[i].experts) continue;
            size_t current = 0, next = 0;
            for (const auto & part : layers[i].parts) {
                current = ((current + alignment - 1) / alignment) * alignment;
                next = ((next + alignment - 1) / alignment) * alignment;
                current += p.layers[i].resident * part.bytes + 512;
                next += (p.layers[i].resident + 1) * part.bytes + 512;
            }
            current = ((current + alignment - 1) / alignment) * alignment;
            next = ((next + alignment - 1) / alignment) * alignment;
            assert(next - current > budget - p.total_bytes);
        }
    }
}

int main() {
    memory_and_time();
    invalid_and_overflow();
    integer_remainder();
    ranked_admission();
    completion_and_reuse();
    measurements();
    randomized();
    std::puts("MOE_SCHEDULE_OK deterministic cases + 4000 heterogeneous layouts");
}
