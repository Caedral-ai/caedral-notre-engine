#include "cne/config.h"

#include <cstdlib>
#include <map>
#include <string>

namespace cne {

// Runtime knob lookup: reads "CNE_<name>" first and falls back to the
// legacy "SOE_<name>" spelling, so pre-existing scripts keep working.
// Resolved values are cached per name; returned pointers stay valid.
const char* env(const char* name) {
    static std::map<std::string, std::string> resolved;

    std::string key = std::string("CNE_") + name;
    auto it = resolved.find(key);
    if (it != resolved.end())
        return it->second.empty() ? nullptr : it->second.c_str();

    const char* v = getenv(key.c_str());
    if (!v) {
        std::string legacy = std::string("SOE_") + name;
        v = getenv(legacy.c_str());
        key = legacy;
    }
    if (!v)
        return nullptr;
    resolved[key] = v;
    return resolved[key].c_str();
}

} // namespace cne
