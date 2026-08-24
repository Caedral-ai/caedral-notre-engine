#include "cne/config.h"

#include <cstdio>
#include <cstring>

namespace cne {

const char* env(const char* name) {
    static char buf[2][128];   // two slots: callers may hold two names at once
    static int slot = 0;
    slot ^= 1;
    snprintf(buf[slot], sizeof(buf[slot]), "CNE_%s", name);
    const char* v = getenv(buf[slot]);
    if (v)
        return v;
    snprintf(buf[slot], sizeof(buf[slot]), "SOE_%s", name);
    return getenv(buf[slot]);
}

} // namespace cne
