#pragma once
// Serving KV footprint estimates for ctx / session_max planning (T3).
#include "cne/memory_budget.h"

#include <cstddef>
#include <cstdint>

namespace cne {

// LFM2-24B hybrid measured tax (~20 KiB/token); override via CNE_KV_BPT.
constexpr size_t kDefaultKvBytesPerToken = 20480;

struct ServingKvEstimate {
    int    n_ctx         = 1024;
    int    n_seq_max     = 1;
    int    n_ctx_per_seq = 1024;
    size_t bytes_per_token = kDefaultKvBytesPerToken;

    size_t kv_bytes() const {
        return (size_t) n_ctx * bytes_per_token;
    }
};

// Suggest total n_ctx and session_max for chat serving (MTP off).
ServingKvEstimate suggest_serving_kv(const MemoryBudget& budget,
                                     size_t model_bytes, Regime regime,
                                     bool mtp_enabled);

// True when projected KV exceeds anonymous headroom (loud warning candidate).
bool serving_kv_exceeds_headroom(const ServingKvEstimate& est,
                                 const MemoryBudget& budget,
                                 size_t model_resident_bytes);

} // namespace cne
