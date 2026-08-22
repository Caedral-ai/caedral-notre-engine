#pragma once
// Expert residency cache. LRU by bytes, inflight dedupe, hard cap invariant:
// engine_budget <= RAM_budget.
#include "soe/expert_layout.h"
#include <cstddef>
#include <cstdint>

namespace soe {

enum class EntryState { ABSENT, READING, READY, EVICTING };

struct CacheEntry {
    ExpertKey key;
    void *ptr = nullptr;
    size_t bytes = 0;
    EntryState state = EntryState::ABSENT;
    int refcount = 0;
    uint64_t last_use = 0;
    uint64_t load_generation = 0;
};

// Watermarks: eviction is async above soft cap; hard cap is never exceeded.
struct CacheLimits {
    size_t soft_cap_bytes = 0;
    size_t hard_cap_bytes = 0;
};

class ExpertCache {
public:
    explicit ExpertCache(CacheLimits limits);
};

} // namespace soe
