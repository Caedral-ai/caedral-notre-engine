// lru-sim: replays a touch-recorder trace against exact-LRU caches of
// several capacities and reports hit-rate / flash-bytes-per-step.
// Slice-level units: (fused-tensor name, expert id). Shared experts are
// NOT simulated — they are mandatory-resident and never miss.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

struct Key {
    uint32_t t; // tensor index
    int32_t e;  // expert id
    bool operator==(const Key &o) const { return t == o.t && e == o.e; }
};
struct KeyHash {
    size_t operator()(const Key &k) const { return (uint64_t)k.t << 32 ^ (uint32_t)k.e; }
};

struct Touch {
    long step;
    uint32_t t;
    std::vector<int32_t> ids;
};

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <trace.out>\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "r");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }

    std::vector<std::string> names;
    std::vector<uint64_t> slice_bytes;
    uint64_t shared_bytes = 0, dense_other = 0;
    std::vector<Touch> touches;
    char line[8192];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') {
            if (line[1] == 'T') {
                char name[256];
                unsigned long long b = 0;
                if (sscanf(line + 2, "%255s %llu", name, &b) == 2) {
                    names.push_back(name);
                    slice_bytes.push_back(b);
                }
            } else if (line[1] == 'M') {
                sscanf(line + 2, "%llu %llu", (unsigned long long *)&shared_bytes, (unsigned long long *)&dense_other);
            }
            continue;
        }
        if (line[0] == 'S') {
            Touch tc{};
            char name[256];
            if (sscanf(line + 1, "%ld %255s", &tc.step, name) != 2)
                continue;
            auto it = std::find(names.begin(), names.end(), name);
            if (it == names.end())
                continue;
            tc.t = (uint32_t)(it - names.begin());
            const char *p = strchr(line, ' ');
            p = strchr(p + 1, ' '); // past name
            while (p && *p) {
                char *end = nullptr;
                long v = strtol(p, &end, 10);
                if (end == p) {
                    p++;
                    continue;
                }
                tc.ids.push_back((int32_t)v);
                p = end;
            }
            touches.push_back(std::move(tc));
        }
    }
    fclose(f);

    long steps = touches.empty() ? 0 : touches.back().step + 1;
    printf("trace: %zu touches over %ld steps | tensors=%zu | "
           "shared=%.1f MiB dense_other=%.1f MiB\n",
           touches.size(), steps, names.size(), shared_bytes / 1048576.0, dense_other / 1048576.0);

    const uint64_t sizes_mib[] = {512, 1024, 2048, 4096, 8192};
    for (uint64_t cap_mib : sizes_mib) {
        uint64_t cap = cap_mib << 20;
        std::list<Key> order; // front = MRU
        std::unordered_map<Key, std::pair<uint64_t, std::list<Key>::iterator>,
                           KeyHash> cache; // key -> bytes,it
        uint64_t used = 0;
        uint64_t hits = 0, misses = 0, miss_bytes = 0, evictions = 0;

        auto touch = [&](const Key &k, uint64_t b) {
            auto it = cache.find(k);
            if (it != cache.end()) {
                hits++;
                order.splice(order.begin(), order, it->second.second);
                it->second.second = order.begin();
                return;
            }
            misses++;
            miss_bytes += b;
            while (used + b > cap && !order.empty()) {
                Key victim = order.back();
                order.pop_back();
                auto vit = cache.find(victim);
                used -= vit->second.first;
                cache.erase(vit);
                evictions++;
            }
            if (b <= cap) {
                order.push_front(k);
                cache[k] = {b, order.begin()};
                used += b;
            }
        };

        for (auto &tc : touches)
            for (int32_t e : tc.ids)
                touch({tc.t, e}, slice_bytes[tc.t]);

        double hr = hits + misses ? 100.0 * hits / (hits + misses) : 0;
        printf("cap %5llu MiB | hit %6.2f%% | cold %.1f MiB/step | "
               "evictions %llu | final entries %zu\n",
               (unsigned long long)cap_mib, hr, steps ? miss_bytes / (double)steps / 1048576.0 : 0.0,
               (unsigned long long)evictions, cache.size());
    }
    printf("(shared %.1f MiB + dense_other %.1f MiB sit OUTSIDE these caps)\n", shared_bytes / 1048576.0,
           dense_other / 1048576.0);
    return 0;
}
