#pragma once
// Slice-level expert residency cache (the heart of the engine).
// Unit: (fused tensor, expert id). Storage: one virtual full-size window
// per fused tensor; slices are filled on demand at their natural offsets,
// so a single t->data repoint serves any routing (Strategy A).
// Physical RAM counts only written pages; eviction returns pages to the OS
// via madvise(MADV_DONTNEED). Plan ref: cache v1.
#include <cstddef>
#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <mutex>

namespace soe {

struct CacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t bytes_loaded = 0;      // committed bytes written
    uint64_t evictions = 0;
    uint64_t dedup_requests = 0;    // repeat (tensor,id) within one batch
};

struct CacheLimits {
    size_t hard_cap_bytes = 2ull << 30;   // never exceed; evict to stay under
};

class SliceCache {
public:
    SliceCache(CacheLimits limits);
    ~SliceCache();

    // Ensure slice (tensor, expert) is resident at
    //   dest = dest_window_base + expert * bytes
    // copying from src (original mapping) on miss. Returns true on hit.
    // Caller owns window lifetime and guarantees dest/src stability.
    bool touch(const std::string& tensor, int expert, void* dest, const void* src,
               size_t bytes);

    // Batch convenience: dedupes repeated (tensor,expert) pairs, then fills.
    // Returns number of misses serviced.
    size_t touch_batch(const std::string& tensor, const int* experts, int n,
                       void* dest_window_base, const void* src_base, size_t bytes);

    // Debug aid: verify last-filled region matches source (offset-bug guard).
    void set_verify_next(size_t n_fills);

    const CacheStats& stats() const { return stats_; }
    size_t used_bytes() const { return used_; }

private:
    struct Entry {
        std::string tensor;
        int expert;
        void* slice_addr;           // exact slice address (= victim range)
        uint64_t bytes;
        std::list<uint64_t>::iterator lru_it;
    };
    uint64_t key(const std::string& t, int e) const;

    bool touch_internal(uint64_t k, const std::string& tensor, int expert,
                        void* dest, const void* src, size_t bytes);
    bool touch_locked(uint64_t k, const std::string& tensor, int expert,
                      void* dest, const void* src, size_t bytes);

    // Callbacks arrive on multiple OpenMP worker threads (E13): all
    // mutations are serialized internally.
    mutable std::mutex mu_;

    CacheLimits limits_;
    CacheStats stats_;

    std::unordered_map<uint64_t, Entry> map_;
    std::list<uint64_t> lru_;                 // front = MRU
    size_t used_ = 0;

    // Recency shield: the newest N touched keys are immune to eviction.
    // Approximates per-decode-step pinning without lifecycle plumbing
    // (a step touches ~960-1920 slices).
    static constexpr size_t kShieldEntries = 2048;
    std::unordered_set<uint64_t> shield_;

    size_t verify_remaining_ = 0;
};

} // namespace soe
