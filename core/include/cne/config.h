#pragma once
// Engine configuration: profiles (SAFE-8G, BALANCED-16G, FAST-32G, EDGE-12G,
// DEBUG), runtime knobs, lossless/accelerated contract.
#include <cstddef>
#include <string>

namespace cne {

struct Config {
    // dense weights policy: "mmap" | "warm" | "anon"
    std::string dense_policy = "anon";
    bool odirect = true;
    bool overlap = true;
    int io_lanes = 2;
    size_t expert_cache_bytes = 2ull << 30;
    bool lossless = true; // accelerated/lossy modes are opt-in only, never silent
    bool verify = true;
};

}

namespace cne {

// Runtime knob lookup: reads "CNE_<name>" first and falls back to the
// legacy "SOE_<name>" spelling, so pre-existing scripts keep working.
// Returned pointer has the lifetime of a static buffer per name slot.
const char* env(const char* name);

} // namespace cne
