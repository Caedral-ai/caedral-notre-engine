// MemoryBudget: headroom math, clamping, regime classification.
#include "cne/memory_budget.h"

#include <cassert>
#include <cstdio>

using cne::classify;
using cne::MemoryBudget;
using cne::Regime;
using cne::regime_name;

constexpr size_t G = 1024ull * 1024 * 1024;

int main() {
    // detection returns sane values on this machine
    MemoryBudget real = MemoryBudget::detect();
    assert(real.mem_total > 0);
    assert(real.mem_available > 0 && real.mem_available <= real.mem_total);
    printf("detected: total=%.1f GiB available=%.1f GiB\n",
           real.mem_total / (double)G, real.mem_available / (double)G);

    // synthetic budget: 16 GiB available, no fixed anon yet
    MemoryBudget b;
    b.mem_total = 19 * G;
    b.mem_available = 16 * G;
    b.base_reserve = 512u << 20;

    // headroom = 16G - 12% - 0.5G = 13.58G
    size_t head = b.cache_headroom();
    assert(head > 13 * G && head < 14 * G);

    // clamp honors smaller requests, caps larger ones
    assert(b.clamp_cache_cap(8 * G) == 8 * G);
    assert(b.clamp_cache_cap(64 * G) == head);

    // fixed anonymous consumption reduces headroom one-for-one
    b.kv = 1 * G;
    assert(b.cache_headroom() == head - 1 * G);
    b.kv = 0;

    // fits() reflects the invariant
    b.expert_cache = b.cache_headroom();
    assert(b.fits());
    b.expert_cache = b.cache_headroom() + 1;
    assert(!b.fits());

    // regime classification monotonic in model size
    uint64_t avail = 10 * G;
    assert(classify(5 * G, avail) == Regime::R0_RESIDENT);
    assert(classify(12 * G, avail) == Regime::R1_NEAR_LIMIT);
    assert(classify(25 * G, avail) == Regime::R2_ABOVE_RAM);
    assert(classify(60 * G, avail) == Regime::R3_FAR_ABOVE);
    assert(classify(200 * G, avail) == Regime::R4_EXTREME_EDGE);
    printf("regimes: %s %s %s %s %s\n", regime_name(Regime::R0_RESIDENT),
           regime_name(Regime::R1_NEAR_LIMIT), regime_name(Regime::R2_ABOVE_RAM),
           regime_name(Regime::R3_FAR_ABOVE), regime_name(Regime::R4_EXTREME_EDGE));

    printf("ALL MEMORY BUDGET TESTS PASSED\n");
    return 0;
}
