#pragma once
// Engine configuration types and runtime knob lookup. Lossless behavior is
// the default; anything that changes model math is an explicit opt-in.
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
// Returned pointers remain valid for the process lifetime.
const char* env(const char* name);

} // namespace cne
