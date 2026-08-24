// IoScheduler: parallel job execution, completion semantics, failure
// propagation. Jobs write distinct regions of a scratch buffer.
#include "cne/io_scheduler.h"

#include <atomic>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <vector>

using cne::IoScheduler;

int main() {
    constexpr size_t N = 1000;
    std::vector<int> out(N, 0);

    // all jobs run exactly once
    {
        std::atomic<int> calls{0};
        IoScheduler s(4);
        assert(s.run(N, [&](size_t i) {
            calls++;
            out[i] = (int)i * 2;
            return true;
        }));
        assert(calls == (int)N);
        for (size_t i = 0; i < N; i++)
            assert(out[i] == (int)i * 2);
    }

    // empty batch: trivially succeeds
    {
        IoScheduler s(2);
        assert(s.run(0, [](size_t) { return true; }));
    }

    // failure propagates and every job still runs (fail-closed caller aborts)
    {
        std::atomic<int> calls{0};
        IoScheduler s(3);
        bool ok = s.run(N, [&](size_t i) {
            calls++;
            return i != 777;
        });
        assert(!ok && calls == (int)N);
    }

    // repeated runs on the same pool
    {
        IoScheduler s(5);
        for (int round = 0; round < 10; round++) {
            assert(s.run(N, [&](size_t i) {
                out[i] = round;
                return true;
            }));
            for (size_t i = 0; i < N; i++)
                assert(out[i] == round);
        }
    }

    printf("ALL IO SCHEDULER TESTS PASSED\n");
    return 0;
}
