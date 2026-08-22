#include "soe/memory_budget.h"

#include <fstream>

#include <sys/sysinfo.h>

namespace soe {

namespace {

// MemAvailable accounts reclaimable page cache - the honest number for
// "how much can this process allocate before the OOM killer looks".
uint64_t read_memavailable() {
    std::ifstream f("/proc/meminfo");
    std::string label;
    uint64_t value = 0;
    std::string unit;
    while (f >> label >> value >> unit)
        if (label == "MemAvailable:")
            return value * 1024ull;   // meminfo reports KiB
    return 0;
}

} // namespace

MemoryBudget MemoryBudget::detect() {
    MemoryBudget b;
    struct sysinfo si;
    if (sysinfo(&si) == 0)
        b.mem_total = (uint64_t)si.totalram * si.mem_unit;
    b.mem_available = read_memavailable();
    return b;
}

size_t MemoryBudget::cache_headroom() const {
    uint64_t margin = (uint64_t)((double)mem_available * safety_margin_frac);
    uint64_t avail = mem_available > margin ? mem_available - margin : 0;
    avail = avail > base_reserve ? avail - base_reserve : 0;
    size_t fixed = fixed_anon();
    return avail > fixed ? (size_t)(avail - fixed) : 0;
}

size_t MemoryBudget::clamp_cache_cap(size_t requested) const {
    size_t head = cache_headroom();
    return requested < head ? requested : head;
}

Regime classify(size_t model_bytes, uint64_t ram_available) {
    double ratio = ram_available ? (double)model_bytes / (double)ram_available : 10.0;
    if (ratio < 1.0)
        return Regime::R0_RESIDENT;
    if (ratio < 1.5)
        return Regime::R1_NEAR_LIMIT;
    if (ratio < 4.0)
        return Regime::R2_ABOVE_RAM;
    if (ratio < 8.0)
        return Regime::R3_FAR_ABOVE;
    return Regime::R4_EXTREME_EDGE;
}

const char *regime_name(Regime r) {
    switch (r) {
    case Regime::R0_RESIDENT: return "R0_RESIDENT";
    case Regime::R1_NEAR_LIMIT: return "R1_NEAR_LIMIT";
    case Regime::R2_ABOVE_RAM: return "R2_ABOVE_RAM";
    case Regime::R3_FAR_ABOVE: return "R3_FAR_ABOVE";
    case Regime::R4_EXTREME_EDGE: return "R4_EXTREME_EDGE";
    }
    return "?";
}

} // namespace soe
