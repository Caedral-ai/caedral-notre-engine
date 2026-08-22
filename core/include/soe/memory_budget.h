#pragma once
// Memory regimes R0-R4 and the engine budget formula.
//
// Amended budget (P2/P5 notes):
//   engine_budget = expert_cache + shared_experts + KV + recurrent
//                 + staging + runtime_base  <=  usable()
//
// Dense weights live in mmap (file-backed, kernel-reclaimable) and are NOT
// part of the anonymous sum - but the KERNEL still needs reclaimable headroom,
// which usable() encodes via the safety margin taken from MemAvailable.
#include <cstddef>
#include <cstdint>
#include <string>

namespace soe {

enum class Regime {
    R0_RESIDENT,   // whole model fits comfortably in RAM
    R1_NEAR_LIMIT, // model ~= RAM; cache small but useful
    R2_ABOVE_RAM,  // model > RAM; cache holds hot set
    R3_FAR_ABOVE,  // model >> RAM; streaming is the product
    R4_EXTREME_EDGE,
};

struct MemoryBudget {
    // Detected at construction from the OS.
    uint64_t mem_total = 0;      // physical RAM
    uint64_t mem_available = 0;  // MemAvailable at startup (kernel estimate)

    double safety_margin_frac = 0.12;   // kept away from the OOM edge
    size_t base_reserve = 512u << 20;   // runtime growth headroom (allocs, heap)

    size_t expert_cache = 0;     // SliceCache hard cap (the knob we clamp)
    size_t shared_experts = 0;   // mandatory-resident bytes
    size_t kv = 0;               // attention KV + recurrent state
    size_t staging = 0;          // quantize/compute scratch
    size_t runtime_base = 0;     // everything else anonymous we account

    // Anonymous bytes we intend to pin, excluding the cache.
    size_t fixed_anon() const { return shared_experts + kv + staging + runtime_base; }

    // Bytes the engine may commit anonymously: available minus margin minus
    // base reserve minus everything already accounted besides the cache.
    size_t cache_headroom() const;

    // Clamp a requested cache cap to what the machine can hold.
    size_t clamp_cache_cap(size_t requested) const;

    bool fits() const { return expert_cache + fixed_anon() <= cache_headroom(); }

    // Detect total/available memory from the OS (sysinfo /proc fallback).
    static MemoryBudget detect();
};

// Regime classification per doc section 14.
Regime classify(size_t model_bytes, uint64_t ram_available);

const char *regime_name(Regime r);

} // namespace soe
