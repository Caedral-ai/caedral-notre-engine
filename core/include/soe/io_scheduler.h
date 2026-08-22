#pragma once
// Async expert I/O pipeline: reserve -> submit -> completion -> publish READY.
// O_DIRECT main path on Linux; bounce buffers for misaligned slices.
#include "soe/expert_layout.h"
#include <cstdint>

namespace soe {

struct IoStats {
    uint64_t submissions = 0;
    uint64_t bytes_read = 0;
    uint64_t short_reads = 0;
};

class IoScheduler {
public:
    // lanes must be autotuned per device; candidates {2,4,8}.
    explicit IoScheduler(int lanes);
};

} // namespace soe
