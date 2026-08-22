#include "soe/io_scheduler.h"

#include <cstdio>

namespace soe {

IoScheduler::IoScheduler(int lanes) {
    int n = lanes < 1 ? 1 : lanes;
    workers_.reserve((size_t)n);
    for (int i = 0; i < n; i++)
        workers_.emplace_back([this] { worker_loop(); });
}

IoScheduler::~IoScheduler() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        shutdown_ = true;
    }
    cv_work_.notify_all();
    for (auto &t : workers_)
        t.join();
}

bool IoScheduler::run(size_t n, const std::function<bool(size_t)> &fn) {
    if (!n)
        return true;
    {
        std::lock_guard<std::mutex> lk(mu_);
        fn_ = fn;
        count_ = n;
        next_ = 0;
        pending_.store((uint32_t)n);
        failed_.store(false);
    }
    cv_work_.notify_all();

    // Workers own the reads; the caller just blocks until all complete.
    std::unique_lock<std::mutex> lk(mu_);
    cv_done_.wait(lk, [&] { return pending_.load() == 0; });
    return !failed_.load();
}

void IoScheduler::worker_loop() {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        try {
            cv_work_.wait(lk, [&] { return shutdown_ || next_ < count_; });
            if (shutdown_)
                return;
            // claim under lock, execute unlocked (jobs are independent)
            size_t i = next_++;
            lk.unlock();
            bool ok = false;
            try {
                ok = fn_(i);
            } catch (const std::exception &e) {
                fprintf(stderr, "[io-sched] job %zu threw: %s\n", i, e.what());
                ok = false;
            }
            lk.lock();
            if (!ok)
                failed_.store(true);
            if (pending_.fetch_sub(1) == 1)
                cv_done_.notify_all();
        } catch (const std::exception &e) {
            fprintf(stderr, "[io-sched] worker op threw: %s\n", e.what());
            throw;
        }
    }
}

} // namespace soe
