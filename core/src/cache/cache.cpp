#include "soe/cache.h"

#include <sys/mman.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace soe {

namespace {

constexpr size_t kPage = 4096;

size_t page_floor(uintptr_t a) { return a & ~(kPage - 1); }
size_t span_pages(uintptr_t from, size_t bytes) {
    uintptr_t start = page_floor(from);
    uintptr_t end   = (from + bytes + kPage - 1) & ~(kPage - 1);
    return (size_t)(end - start);
}

uint64_t fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

} // namespace

SliceCache::SliceCache(CacheLimits limits) : limits_(limits) {}

SliceCache::~SliceCache() = default;

uint64_t SliceCache::key(const std::string& t, int e) const {
    return fnv1a(t) * 1000003ull ^ (uint64_t)(uint32_t)e;
}

void SliceCache::set_verify_next(size_t n_fills) { verify_remaining_ = n_fills; }

bool SliceCache::fill_slice(void* dest, const void* src, uint64_t src_off,
                            size_t bytes, const std::string& tensor, int expert) {
    if (src) {
        std::memcpy(dest, src, bytes);
    } else if (src_.read) {
        if (!src_.read(dest, src_off, bytes, src_.ud)) {
            fprintf(stderr, "[slice-cache] FILL FAILED (backend) %s#%d\n",
                    tensor.c_str(), expert);
            return false;
        }
    } else {
        fprintf(stderr, "[slice-cache] no fill source configured\n");
        return false;
    }
    if (verify_remaining_ && src) {
        // memcmp guard only exists in pointer mode; backend reads are trusted
        // upstack (DirectFile fails closed on short/misaligned reads).
        if (std::memcmp(dest, src, bytes) != 0) {
            fprintf(stderr, "[slice-cache] VERIFY MISMATCH %s#%d\n", tensor.c_str(),
                    expert);
            abort();
        }
        verify_remaining_--;
    }
    return true;
}

bool SliceCache::touch_internal(uint64_t k, const std::string& tensor, int expert,
                                void* dest, const void* src, size_t bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    return touch_locked(k, tensor, expert, dest, src, 0, bytes);
}

bool SliceCache::touch_internal_at(uint64_t k, const std::string& tensor, int expert,
                                   void* dest, uint64_t src_offset, size_t bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    return touch_locked(k, tensor, expert, dest, nullptr, src_offset, bytes);
}

bool SliceCache::touch_locked(uint64_t k, const std::string& tensor, int expert,
                              void* dest, const void* src, uint64_t src_off,
                              size_t bytes) {
    auto it = map_.find(k);
    if (it != map_.end()) {
        // Hit: refresh recency.
        lru_.erase(it->second.lru_it);
        lru_.push_front(k);
        it->second.lru_it = lru_.begin();
        shield_.insert(k);
        stats_.hits++;
        return true;
    }

    // Miss: make room, then fill. Eviction walks LRU order and prefers
    // unshielded keys (the newest touches — approximates current-step
    // pinning), but the hard cap always wins: if every entry is shielded,
    // fall back to plain LRU rather than overshoot the cap.
    while (used_ + bytes > limits_.hard_cap_bytes) {
        uint64_t victim_key = 0;
        bool found = false;
        for (auto rit = lru_.rbegin(); rit != lru_.rend(); ++rit) {
            if (!shield_.count(*rit)) { victim_key = *rit; found = true; break; }
        }
        if (!found && !lru_.empty()) {
            victim_key = lru_.back();
            found = true;
        }
        if (!found) break;   // empty cache and still over cap: oversized slab
        lru_.remove(victim_key);
        auto vit = map_.find(victim_key);
        Entry& e = vit->second;
        if (madvise((void*)page_floor((uintptr_t)e.slice_addr),
                    span_pages((uintptr_t)e.slice_addr, e.bytes),
                    MADV_DONTNEED) != 0)
            perror("madvise");
        used_ -= e.bytes;
        stats_.evictions++;
        map_.erase(vit);
    }

    if (!fill_slice(dest, src, src_off, bytes, tensor, expert))
        abort();
    stats_.misses++;
    stats_.bytes_loaded += bytes;

    if (bytes <= limits_.hard_cap_bytes) {
        lru_.push_front(k);
        map_.emplace(k, Entry{tensor, expert, dest, bytes, lru_.begin()});
        used_ += bytes;
    }
    shield_.insert(k);
    if (shield_.size() > kShieldEntries) {
        // Drop the oldest half of the shield when it grows past capacity.
        for (auto rit = lru_.rbegin(); rit != lru_.rend() && shield_.size() > kShieldEntries / 2; ++rit)
            shield_.erase(*rit);
    }
    return false;
}

bool SliceCache::touch(const std::string& tensor, int expert, void* dest,
                       const void* src, size_t bytes) {
    return touch_internal(key(tensor, expert), tensor, expert, dest, src, bytes);
}

bool SliceCache::touch_at(const std::string& tensor, int expert, void* dest,
                          uint64_t src_offset, size_t bytes) {
    return touch_internal_at(key(tensor, expert), tensor, expert, dest, src_offset,
                             bytes);
}

size_t SliceCache::touch_batch(const std::string& tensor, const int* experts, int n,
                               void* dest_window_base, const void* src_base,
                               size_t bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    std::unordered_set<int> uniq;
    for (int i = 0; i < n; i++) uniq.insert(experts[i]);
    stats_.dedup_requests += (size_t)(n - (int)uniq.size());
    size_t misses = 0;
    for (int e : uniq) {
        void*       d = (char*)dest_window_base + (size_t)e * bytes;
        const void* s = (const char*)src_base + (size_t)e * bytes;
        if (!touch_locked(key(tensor, e), tensor, e, d, s, 0, bytes)) misses++;
    }
    return misses;
}

size_t SliceCache::touch_batch_at(const std::string& tensor, const int* experts, int n,
                                  void* dest_window_base, uint64_t src_base_offset,
                                  size_t bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    std::unordered_set<int> uniq;
    for (int i = 0; i < n; i++) uniq.insert(experts[i]);
    stats_.dedup_requests += (size_t)(n - (int)uniq.size());
    size_t misses = 0;
    for (int e : uniq) {
        void* d = (char*)dest_window_base + (size_t)e * bytes;
        uint64_t off = src_base_offset + (uint64_t)e * bytes;
        if (!touch_locked(key(tensor, e), tensor, e, d, nullptr, off, bytes)) misses++;
    }
    return misses;
}

} // namespace soe
