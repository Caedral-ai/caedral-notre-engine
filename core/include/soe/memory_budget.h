#pragma once
// Memory regimes R0-R4 and budget formula. Plan ref: §6.
#include <cstddef>

namespace soe {

enum class Regime { R0_RESIDENT, R1_NEAR_LIMIT, R2_ABOVE_RAM, R3_FAR_ABOVE, R4_EXTREME_EDGE };

struct MemoryBudget {
    // engine_budget = dense + expert_cache + KV + staging + runtime_base
    // invariant: total <= RAM_usable - safety_margin
    size_t ram_usable = 0;
    double safety_margin_frac = 0.12;
    size_t dense = 0;
    size_t expert_cache = 0;
    size_t kv = 0;
    size_t staging = 0;
    size_t runtime_base = 0;

    size_t total() const {
        return dense + expert_cache + kv + staging + runtime_base;
    }
};

Regime classify(size_t model_bytes, size_t ram_usable);

} // namespace soe
