#pragma once
// Slice-level expert residency cache (the heart of the engine).
// Unit: (fused tensor, expert id). Storage: one virtual full-size window
// per fused tensor; slices are filled on demand at their natural offsets
// using the original tensor ids, so a single t->data repoint serves any
// routing without remapping anything.
// Physical RAM counts only written pages; eviction returns pages to the OS
// via madvise(MADV_DONTNEED).
#include <cstddef>
#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <mutex>

namespace soe {

class IoScheduler;

struct CacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t bytes_loaded = 0;      // committed bytes written
    uint64_t evictions = 0;
    uint64_t dedup_requests = 0;    // repeat (tensor,id) within one batch
    uint64_t prefetched = 0;        // slices filled by prefetch_batch_at
};

struct CacheLimits {
    size_t hard_cap_bytes = 2ull << 30;   // never exceed; evict to stay under
};

class SliceCache {
public:
    // Fill backend for misses. When unset, fills memcpy from a caller-provided
    // pointer (mmap mode). When set, fills call read(dest, src_offset, bytes, ud)
    // - e.g. O_DIRECT pread from a prepared GGUF (offsets 4096-aligned).
    struct Source {
        bool (*read)(void *dest, uint64_t src_offset, size_t bytes, void *ud) = nullptr;
        void *ud = nullptr;
    };

    SliceCache(CacheLimits limits);
    ~SliceCache();

    void set_source(Source s) { src_ = s; }

    // Optional lane pool: when set, touch_batch_at fills its misses across
    // the scheduler's lanes (parallel preads under the cache lock - same
    // serialization as inline fills, no new races). Without it, misses run
    // inline on the calling thread.
    void set_scheduler(IoScheduler *s) { sched_ = s; }

    // Ensure slice (tensor, expert) is resident at
    //   dest = dest_window_base + expert * bytes
    // copying from src (original mapping) on miss. Returns true on hit.
    // Caller owns window lifetime and guarantees dest/src stability.
    bool touch(const std::string& tensor, int expert, void* dest, const void* src,
               size_t bytes);

    // Offset-source variant: on miss the configured Source reads
    // src_offset + expert*bytes into dest.
    bool touch_at(const std::string& tensor, int expert, void* dest,
                  uint64_t src_offset, size_t bytes);

    // Batch convenience: dedupes repeated (tensor,expert) pairs, then fills.
    // Returns number of misses serviced.
    size_t touch_batch(const std::string& tensor, const int* experts, int n,
                       void* dest_window_base, const void* src_base, size_t bytes);

    // Batch variant for the offset-source backend: slices live at
    // src_base_offset + expert*bytes in the backing store.
    size_t touch_batch_at(const std::string& tensor, const int* experts, int n,
                          void* dest_window_base, uint64_t src_base_offset,
                          size_t bytes);

    // Background prefetch: fill absent slices via the given
    // scheduler with the cache lock held only for filtering/committing -
    // never during I/O. Safe to race with demand fills from another thread:
    // double-fill is idempotent (same bytes, same destination) and insertion
    // is re-checked under the lock. Evictions it triggers respect the shield
    // and never straddle page boundaries (slices are 4096-multiples).
    void prefetch_batch_at(const std::string& tensor, const int* experts, int n,
                           void* dest_window_base, uint64_t src_base_offset,
                           size_t bytes, IoScheduler& sched);

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

    // Unified miss fill: pointer mode when src != nullptr, backend mode when
    // src_offset is used (src == nullptr). Returns false on backend failure
    // (fail-closed: caller aborts; a failed fill must never look resident).
    bool fill_slice(void* dest, const void* src, uint64_t src_off, size_t bytes,
                    const std::string& tensor, int expert);

    // Evict LRU (shield-aware) until bytes fit under the hard cap.
    void make_room_locked(size_t bytes);

    // Insert a filled slice into the map/LRU and account it.
    void commit_locked(uint64_t k, const std::string& tensor, int expert,
                       void* dest, size_t bytes);

    bool touch_internal(uint64_t k, const std::string& tensor, int expert,
                        void* dest, const void* src, size_t bytes);
    bool touch_internal_at(uint64_t k, const std::string& tensor, int expert,
                           void* dest, uint64_t src_offset, size_t bytes);
    bool touch_locked(uint64_t k, const std::string& tensor, int expert,
                      void* dest, const void* src, uint64_t src_off, size_t bytes);

    // Callbacks may arrive on multiple worker threads: all mutations are
    // serialized internally.
    mutable std::mutex mu_;

    CacheLimits limits_;
    CacheStats stats_;
    Source src_{};
    IoScheduler *sched_ = nullptr;

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
