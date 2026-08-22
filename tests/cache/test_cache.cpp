// SliceCache unit tests: LRU order, recency refresh, dedupe, cap enforcement,
// madvise eviction path on real anonymous memory.
#include "soe/cache.h"

#include <sys/mman.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using soe::CacheLimits;
using soe::SliceCache;

namespace {

constexpr size_t SLAB = 8192;

struct Win {
    void* p;
    explicit Win(size_t bytes) {
        p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(p != MAP_FAILED);
        memset(p, 0xAB, bytes);
    }
};

} // namespace

int main() {
    const size_t CAP = 3 * SLAB;
    Win wa(16 * SLAB), wb(16 * SLAB), wc(16 * SLAB);
    SliceCache c(CacheLimits{CAP});

    // Fill A0, A1, B0 -> full.
    assert(!c.touch("A", 0, wa.p, wa.p, SLAB));
    assert(!c.touch("A", 1, (char*)wa.p + SLAB, (char*)wa.p + SLAB, SLAB));
    assert(!c.touch("B", 0, wb.p, wb.p, SLAB));
    assert(c.used_bytes() == CAP);
    assert(c.stats().misses == 3 && c.stats().hits == 0);

    // Hit refreshes recency: A0 becomes MRU, A1 is now LRU.
    assert(c.touch("A", 0, wa.p, wa.p, SLAB));
    assert(c.stats().hits == 1);

    // C0 must evict LRU = A1 (madvise path on real anon pages).
    assert(!c.touch("C", 0, wc.p, wc.p, SLAB));
    assert(c.stats().evictions == 1);
    assert(c.used_bytes() == CAP);
    // A1 gone, A0 and B0 still resident.
    assert(c.touch("A", 0, wa.p, wa.p, SLAB));
    assert(c.touch("B", 0, wb.p, wb.p, SLAB));
    assert(!c.touch("A", 1, (char*)wa.p + SLAB, (char*)wa.p + SLAB, SLAB));

    // Batch dedupe: {5,5,7} -> one dedup, two fills (5 hits? no: fresh).
    auto before = c.stats();
    size_t misses = c.touch_batch("A", (const int[]){7, 7, 9}, 3, wa.p, wa.p, SLAB);
    assert(misses == 2);
    assert(c.stats().dedup_requests == before.dedup_requests + 1);
    assert(c.stats().misses == before.misses + 2);

    // Oversized slab: served but not cached (no infinite evict loop).
    SliceCache tiny(CacheLimits{SLAB});
    assert(!tiny.touch("X", 0, wa.p, wa.p, 4 * SLAB));
    assert(tiny.used_bytes() == 0);
    assert(tiny.stats().misses == 1);

    // Verify mode exercises the memcmp guard on the next fill (fresh slice
    // => miss; fill succeeds and memcmp passes because dest mirrors src).
    c.set_verify_next(1);
    assert(!c.touch("B", 3, (char*)wb.p + 3 * SLAB, (char*)wb.p + 3 * SLAB, SLAB));

    printf("ALL CACHE TESTS PASSED (hits=%llu misses=%llu evictions=%llu)\n",
           (unsigned long long)c.stats().hits, (unsigned long long)c.stats().misses,
           (unsigned long long)c.stats().evictions);
    return 0;
}
