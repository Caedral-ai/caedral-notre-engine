#pragma once
// Lane pool for parallel expert-slice reads (P3). Workers execute caller-
// supplied read jobs across N lanes; the calling thread submits a batch and
// blocks until every job finished. Concurrent preads on a shared fd are safe
// and build device queue depth; jobs must be independent (SliceCache window
// slices target disjoint destinations).
//
// Fail-closed: any failed job marks the whole run failed - callers must
// treat the batch as unserved (SliceCache aborts).
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>

namespace soe {

class IoScheduler {
public:
    explicit IoScheduler(int lanes);
    ~IoScheduler();

    IoScheduler(const IoScheduler &) = delete;
    IoScheduler &operator=(const IoScheduler &) = delete;

    // Executes fn(i) for i in [0,n) across the lanes; returns false if any
    // invocation returned false. Blocking.
    bool run(size_t n, const std::function<bool(size_t)> &fn);

    int lanes() const { return (int)workers_.size(); }

private:
    void worker_loop();

    std::function<bool(size_t)> fn_;
    size_t count_ = 0;
    size_t next_ = 0;                    // next job index to claim
    std::atomic<uint32_t> pending_{0};
    std::atomic<bool> failed_{false};
    bool shutdown_ = false;

    std::mutex mu_;
    std::condition_variable cv_work_;  // wakes workers (jobs available)
    std::condition_variable cv_done_;  // wakes caller (pending == 0)
    std::vector<std::thread> workers_;
};

} // namespace soe
