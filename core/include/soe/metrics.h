#pragma once
// Telemetry: explain every second spent in compute / io_wait / flash / cache.
// Decision metrics are end-to-end tok/s and p95 inter-token latency.
#include <cstdint>

namespace soe {

struct Metrics {
    double tok_s_decode = 0;
    double inter_token_ms_p50 = 0;
    double inter_token_ms_p95 = 0;
    uint64_t flash_bytes_per_token = 0;
    double cache_hit_rate = 0;
    double cache_evictions_per_s = 0;
    double io_wait_ms_per_token = 0;
    double compute_ms_per_token = 0;
    uint64_t rss_anon_steady = 0;
    double inflight_dedup_rate = 0;
};

} // namespace soe
