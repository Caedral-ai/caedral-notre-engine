#include "cne/kv_budget.h"

#include <algorithm>
#include <cmath>

namespace cne {

namespace {

size_t anon_headroom(const MemoryBudget& budget, size_t model_resident) {
    const uint64_t margin =
        (uint64_t) ((double) budget.mem_available * budget.safety_margin_frac);
    uint64_t avail =
        budget.mem_available > margin ? budget.mem_available - margin : 0;
    avail = avail > budget.base_reserve ? avail - budget.base_reserve : 0;
    const size_t fixed =
        budget.fixed_anon() + model_resident + (512u << 20);
    return avail > fixed ? (size_t) (avail - fixed) : 0;
}

} // namespace

ServingKvEstimate suggest_serving_kv(const MemoryBudget& budget,
                                     size_t model_bytes, Regime regime,
                                     bool mtp_enabled) {
    ServingKvEstimate e;
    const size_t model_resident =
        regime == Regime::R0_RESIDENT ? model_bytes : 0;
    const size_t headroom = anon_headroom(budget, model_resident);

    if (mtp_enabled) {
        e.n_seq_max     = 1;
        e.n_ctx_per_seq = 1024;
        e.n_ctx         = e.n_ctx_per_seq;
        return e;
    }

    e.n_seq_max = 2;
    if (regime == Regime::R0_RESIDENT &&
        budget.mem_available > model_bytes + (2ull << 30))
        e.n_seq_max = 3;
    if (model_bytes > (size_t) (budget.mem_available * 0.85))
        e.n_seq_max = 1;

    e.n_ctx_per_seq = 2048;
    e.n_ctx         = e.n_ctx_per_seq * e.n_seq_max;

    while (e.n_ctx > 512 && e.kv_bytes() > headroom) {
        e.n_ctx_per_seq = std::max(512, e.n_ctx_per_seq / 2);
        e.n_ctx         = e.n_ctx_per_seq * e.n_seq_max;
    }

    return e;
}

bool serving_kv_exceeds_headroom(const ServingKvEstimate& est,
                                 const MemoryBudget& budget,
                                 size_t model_resident_bytes) {
    return est.kv_bytes() > anon_headroom(budget, model_resident_bytes);
}

} // namespace cne
