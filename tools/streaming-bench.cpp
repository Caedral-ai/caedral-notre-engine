// streaming-bench: end-to-end measurement of demand-served expert streaming.
// Full greedy generation with SliceCache filling slices at the callback
// ask-point into full-size windows per fused tensor (original tensor ids kept,
// rebinding on first touch). Reports tok/s, hit-rate, cold MiB/step,
// evictions - comparable against offline LRU simulation curves.
#include "soe/cache.h"
#include "soe/direct_io.h"
#include "soe/io_scheduler.h"
#include "soe/memory_budget.h"
#include "soe/model.h"
#include "soe/model_registry.h"
#include "soe/tensor_classify.h"

#include "llama.h"

#include <sys/mman.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <unordered_set>
#include <memory>
#include <mutex>
static std::mutex g_cb_mu;
#include <map>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <unistd.h>
#include <sys/resource.h>

namespace {

using Clock = std::chrono::steady_clock;

struct Window {
    void* base = nullptr;
    void* orig = nullptr;      // original llama mapping (copy source; memcpy mode)
    uint64_t file_off = 0;     // tensor start in backing store (odirect mode)
    bool  rebound = false;
    const soe::TensorInfo* ti = nullptr;
};

static soe::DirectFile g_direct;
static bool g_use_odirect = false;
static std::unique_ptr<soe::IoScheduler> g_sched;
static double g_fill_ns = 0;   // time inside batch fills (I/O + memcpy)
static uint64_t g_fill_calls = 0;
static size_t g_use_cap = 0;

// ---- Dense residency policies --------------------------------------------
enum class DensePolicy { MMAP, WARM, ANON };
static DensePolicy g_dense = DensePolicy::MMAP;
static std::unordered_map<std::string, void*> g_dense_bind;   // name -> anon copy
static std::unordered_set<std::string> g_dense_names;         // manifest non-routed names
static size_t g_dense_anon_bytes = 0;
static bool g_anon_scan = false;                              // warmup pass observes all nodes

// bind one weight tensor to an anonymous copy (ANON policy)
static void try_bind_dense(const ggml_tensor* s) {
    if (!s || !s->name || !s->data || s->type == GGML_TYPE_I32) return;
    std::string nm(s->name);
    if (!g_dense_names.count(nm) || g_dense_bind.count(nm)) return;
    size_t bytes = ggml_nbytes(s);
    if (bytes == 0 || bytes > (2ull << 30)) return;
    void* buf = nullptr;
    if (posix_memalign(&buf, 4096, bytes) != 0) return;
    memcpy(buf, s->data, bytes);
    ((ggml_tensor*)s)->data = buf;
    g_dense_bind[nm] = buf;
    g_dense_anon_bytes += bytes;
}

static long dbg_scan_nodes = 0;
static void scan_dense_srcs(ggml_tensor* t) {
    if (!t) return;
    if (++dbg_scan_nodes % 500 == 1)
        fprintf(stderr, "[anon-scan] node #%ld op=%d name='%s'\n",
                dbg_scan_nodes, (int)t->op, t->name ? t->name : "(null)");
    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; i++) {
        const ggml_tensor* s = t->src[i];
        if (dbg_scan_nodes <= 3 && i < 4 && s && s->name)
            fprintf(stderr, "[anon-scan]   src[%d] type=%d name='%s'\n",
                    i, (int)s->type, s->name);
        try_bind_dense(s);
    }
}

// RSS / fault telemetry
struct MemSnap { long minflt=0, majflt=0, rss_kib=0, hwm_kib=0; };
static MemSnap mem_snap() {
    MemSnap m;
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    m.minflt = ru.ru_minflt; m.majflt = ru.ru_majflt;
    std::ifstream st("/proc/self/status");
    std::string line;
    while (std::getline(st, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0)
            m.rss_kib = atol(line.c_str() + 6);
        else if (line.compare(0, 6, "VmHWM:") == 0)
            m.hwm_kib = atol(line.c_str() + 6);
    }
    return m;
}

static bool od_read(void* dest, uint64_t off, size_t bytes, void* ud) {
    return ((soe::DirectFile*)ud)->read_aligned(dest, bytes, off);
}

struct State {
    const soe::ModelManifest* manifest = nullptr;
    soe::SliceCache* cache = nullptr;
    std::map<std::string, Window> windows;
    long step = -1;
    // per-step stats snapshot
    uint64_t h0 = 0, m0 = 0;
    double cold_bytes_step_accum = 0;
    long steps_counted = 0;
};

struct AuditRec {
    std::string tname;
    std::vector<int32_t> ids;
    long step;
    size_t bpe;
};
static std::map<ggml_tensor*, AuditRec> g_audit;

// Router-ids tensors are NOT guaranteed contiguous (prefill rows are strided
// in the graph pool): always gather via nb[] strides, never flat memcpy.
static void read_ids_strided(const ggml_tensor* ids, std::vector<int32_t>& out) {
    out.clear();
    for (int64_t i1 = 0; i1 < ids->ne[1]; i1++)
        for (int64_t i0 = 0; i0 < ids->ne[0]; i0++) {
            int32_t e;
            memcpy(&e, (const char*)ids->data + i1 * ids->nb[1] + i0 * ids->nb[0], 4);
            out.push_back(e);
        }
}

State g;

// ---- Overlap: speculative prefetch (last-token routing reuse) -------------

enum class PfMode { OFF, FULL, LOOKAHEAD };
static PfMode g_pf_mode = PfMode::OFF;

struct PrefetchItem {
    std::string name;
    std::vector<int32_t> ids;
};
static std::mutex g_pi_mu;
static std::condition_variable g_pi_cv;
static std::deque<PrefetchItem> g_pi_q;      // latest-wins
static bool g_pi_stop = false;
static std::thread g_pi_thread;
static std::unique_ptr<soe::IoScheduler> g_psched;

static void prefetch_worker() {
    std::unique_ptr<soe::IoScheduler> psched = std::make_unique<soe::IoScheduler>(4);
    for (;;) {
        PrefetchItem it;
        {
            std::unique_lock<std::mutex> lk(g_pi_mu);
            g_pi_cv.wait(lk, [] { return g_pi_stop || !g_pi_q.empty(); });
            if (g_pi_stop && g_pi_q.empty())
                return;
            it = std::move(g_pi_q.front());
            g_pi_q.pop_front();
        }
        auto wit = g.windows.find(it.name);
        if (wit == g.windows.end() || !wit->second.base || !wit->second.ti)
            continue;   // window not created yet: demand path will fill
        g.cache->prefetch_batch_at(it.name, it.ids.data(), (int)it.ids.size(),
                                   wit->second.base, wit->second.file_off,
                                   wit->second.ti->bytes_per_expert, *psched);
    }
}

// Lookahead: while computing node rank r (using fresh routing), speculate
// nodes r+1..r+W from THEIR previous-step routing. Two id slots: the one
// being written this step, and the read-only previous one.
static std::unordered_map<std::string, std::vector<int32_t>> g_step_ids[2];
static int g_id_slot = 0;
static std::unordered_set<std::string> g_looked;
static long g_looked_step = -1;
static constexpr int kLookaheadNodes = 6;

static int node_rank(const char* name) {
    int L = soe::parse_layer_index(name);
    const char* k = strstr(name, "ffn_gate_exps");
    int kind = k ? 0 : (strstr(name, "ffn_up_exps") ? 1 : 2);
    return L >= 0 ? L * 3 + kind : -1;
}

static std::string rank_name(int rank) {
    // inverse of node_rank for our fixed naming scheme
    static const char* kinds[] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
    char buf[128];
    snprintf(buf, sizeof(buf), "blk.%d.%s.weight", rank / 3, kinds[rank % 3]);
    return buf;
}

// called from cb_eval after a node's demand fill
static void lookahead_push(const char* name) {
    if (g_pf_mode != PfMode::LOOKAHEAD)
        return;
    int rank = node_rank(name);
    if (rank < 0)
        return;
    if (g_looked_step != g.step) {   // new step: reset per-step dedup
        g_looked_step = g.step;
        g_looked.clear();
    }
    const auto& prev = g_step_ids[g_id_slot ^ 1];   // previous step's routing
    for (int a = 1; a <= kLookaheadNodes; a++) {
        int r = rank + a;
        std::string rn = rank_name(r);
        auto pit = prev.find(rn);
        if (pit == prev.end() || pit->second.empty())
            continue;
        if (g_looked.count(rn))
            continue;   // already speculated this step
        g_looked.insert(rn);
        {
            std::lock_guard<std::mutex> lk(g_pi_mu);
            g_pi_q.push_back({rn, pit->second});
        }
        g_pi_cv.notify_one();
    }
}



static bool g_step_fills = false;
static long g_audit_checks = 0;

static void maybe_dump_dst(ggml_tensor* t, long step) {
    const char* dir = getenv("SOE_DUMP_DST");
    if (!dir || !*dir || !t || !t->data || step < -1) return;
    char path[512];
    std::string clean = t->name;
    for (auto& c : clean) if (c == '.') c = '_';
    if (step < 0)
        snprintf(path, sizeof(path), "%s/prefill_%s.f32", dir, clean.c_str());
    else
        snprintf(path, sizeof(path), "%s/s%03ld_%s.f32", dir, step, clean.c_str());
    FILE* f = fopen(path, "wb");
    if (!f) return;
    size_t n = 1;
    for (int i = 0; i < GGML_MAX_DIMS; i++) n *= (size_t)t->ne[i];
    fwrite(t->data, sizeof(float), n, f);
    fclose(f);
    // also dump the ids tensor and src1 activations as seen at post-compute
    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; i++) {
        ggml_tensor* s = t->src[i];
        if (!s || !s->data || !s->name) continue;
        std::string sc = clean + std::string("_src") + std::to_string(i) +
                         (s->type == GGML_TYPE_I32 ? "_i32" : "");
        if (step < 0)
            snprintf(path, sizeof(path), "%s/prefill_%s.bin", dir, sc.c_str());
        else
            snprintf(path, sizeof(path), "%s/s%03ld_%s.bin", dir, step, sc.c_str());
        FILE* g2 = fopen(path, "wb");
        if (!g2) continue;
        size_t bn = ggml_nbytes(s);
        if (bn > 64ull << 20) { fclose(g2); remove(path); continue; }   // skip fused expert weights
        fwrite(s->data, 1, bn, g2);
        fclose(g2);
    }
}
static bool g_rebind = true;
static std::atomic<bool> g_tid_warned{false};
static std::map<long,int> g_tids;
static std::atomic<long> g_ask_n{0};
static void note_tid() {
    long tid = (long)gettid();
    static std::mutex m;
    std::lock_guard<std::mutex> lk(m);
    if (++g_tids[tid] == 1)
        fprintf(stderr, "[bench] callback on NEW tid=%ld\n", tid);
}
static bool g_full_fill = false;   // debug: fill whole windows once at creation

void ensure_window(const char* name, ggml_tensor* w) {
    if (!g_rebind) return;
    static int layer_limit = getenv("SOE_LAYER_LIMIT") ? atoi(getenv("SOE_LAYER_LIMIT")) : -1;
    if (layer_limit >= 0) {
        int L = soe::parse_layer_index(name);
        if (L < 0 || L >= layer_limit) return;
    }
    auto it = g.windows.find(name);
    if (it != g.windows.end()) return;
    for (const auto& ti : g.manifest->tensors)
        if (ti.name == name && ti.kind == soe::TensorKind::ROUTED_EXPERT) {
            Window win;
            win.ti   = &ti;
            win.file_off = ti.abs_offset;
            win.orig = g_use_odirect ? nullptr : w->data;
            win.base = mmap(nullptr, ti.bytes_total, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (win.base == MAP_FAILED) {
                fprintf(stderr, "[streaming-bench] window mmap FAILED for %s\n", name);
                exit(1);
            }
            g.windows[name] = win;
            fprintf(stderr,
                    "[geom] %s manifest bpe=%zu total=%zu | runtime ne=[%lld,%lld,%lld] "
                    "nb=[%zu,%zu,%zu,%zu] -> nb2=%zu type=%d size=%zu\n",
                    name, (size_t)ti.bytes_per_expert, (size_t)ti.bytes_total,
                    (long long)w->ne[0], (long long)w->ne[1], (long long)w->ne[2],
                    (size_t)w->nb[0], (size_t)w->nb[1], (size_t)w->nb[2], (size_t)w->nb[3],
                    (size_t)w->nb[2], (int)w->type, ggml_type_size(w->type));
            if (g_full_fill) {
                memcpy(win.base, win.orig, ti.bytes_total);
                win.rebound = true;
                w->data     = win.base;
            }
            return;
        }
}

bool cb_eval(ggml_tensor* t, bool ask, void*) {
    // Flow: isolate only ROUTERS (ffn_moe_topk) and MUL_MAT_IDs - 160/step
    // instead of ~7800. Fresh ids are harvested at the router's ask=false and
    // stashed per layer; MMID asks serve from that stash, so the notorious
    // ids-tensor staleness inside batched regions stops mattering.
    if (!t)
        return true;
    if (g_anon_scan) {   // ANON policy: one warmup pass observes every node
        scan_dense_srcs(t);
        return true;
    }
    const char* nm = t->name ? t->name : "";
    bool is_router = strstr(nm, "ffn_moe_topk") != nullptr;
    bool is_mmid = t->op == GGML_OP_MUL_MAT_ID;
    if (!is_router && !is_mmid)
        return ask ? false : true;   // batch everything else

    static std::unordered_map<int, std::vector<int32_t>> g_fresh;  // layer -> ids

    // ---- ROUTER ----
    if (is_router) {
        if (ask)
            return true;   // isolate: it computes alone, then we harvest
        int L = soe::parse_layer_index(nm);
        if (L < 0 || !t->data || t->type != GGML_TYPE_I32)
            return true;
        std::vector<int32_t> v;
        read_ids_strided(t, v);
        if (v.empty())
            return true;
        g_fresh[L] = std::move(v);
        // opportunistic fill: if this layer's windows already exist (they do
        // from the previous evaluation onward), serve them NOW - off the
        // critical path of anything that matters.
        if (!g_rebind || !g_use_odirect || g_full_fill)
            return true;
        char buf[96];
        for (int k = 0; k < 3; k++) {
            static const char* kinds[] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
            snprintf(buf, sizeof(buf), "blk.%d.%s.weight", L, kinds[k]);
            auto wit = g.windows.find(buf);
            if (wit == g.windows.end() || !wit->second.base || !wit->second.ti)
                continue;   // first evaluation: MMID asks will create + serve
            auto t0 = Clock::now();
            g.cache->touch_batch_at(buf, g_fresh[L].data(), (int)g_fresh[L].size(),
                                    wit->second.base, wit->second.file_off,
                                    wit->second.ti->bytes_per_expert);
            g_fill_ns += std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
            g_fill_calls++;
        }
        return true;
    }

    // ---- MUL_MAT_ID ----
    if (!ask) {
        // post-compute audit / dumps
        AuditRec rec;
        bool have = false;
        for (int i = 0; i < GGML_MAX_SRC && t->src[i]; i++)
            if (t->src[i] && g_audit.count(t->src[i])) {
                rec = g_audit[t->src[i]];
                g_audit.erase(t->src[i]);
                have = true;
                break;
            }
        maybe_dump_dst(t, g.step);
        if (have && rec.step == g.step && !g_use_odirect) {
            auto wit = g.windows.find(rec.tname);
            if (wit != g.windows.end()) {
                Window& win = wit->second;
                for (int e : rec.ids) {
                    if (e < 0 || e >= 256) continue;
                    if (memcmp((const char*)win.base + (size_t)e * rec.bpe,
                               (const char*)win.orig + (size_t)e * rec.bpe,
                               rec.bpe) != 0) {
                        fprintf(stderr, "[audit] step %ld %s: SLICE %d CORRUPT\n",
                                g.step, rec.tname.c_str(), e);
                        break;
                    }
                }
            }
        }
        return true;
    }

    // ask: discover weight + bind window + serve from FRESH router ids
    ggml_tensor* w = nullptr;
    ggml_tensor* ids = nullptr;
    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; i++) {
        auto* s = t->src[i];
        if (!ids && s->type == GGML_TYPE_I32) ids = s;
        else if (!w && s->name) {
            auto it = g.windows.find(s->name);
            if (it != g.windows.end() ||
                [&] {
                    for (const auto& ti : g.manifest->tensors)
                        if (ti.name == s->name)
                            return ti.kind == soe::TensorKind::ROUTED_EXPERT;
                    return false;
                }())
                w = s;
        }
    }
    if (!w || !ids || !ids->data) return true;

    ensure_window(w->name, w);
    auto wit = g.windows.find(w->name);
    if (wit == g.windows.end()) return true;   // layer-limited out: stays mmap
    Window& win = wit->second;
    if (!win.base || !win.ti || !w->data) return true;
    if (!g_use_odirect && !win.orig) return true;   // memcpy mode needs mirror

    int n = (int)ggml_nelements(ids);
    if (n <= 0 || n > 4096) return true;
    static thread_local std::vector<int32_t> v;
    int L = soe::parse_layer_index(w->name);
    auto fit = g_fresh.find(L);
    if (fit != g_fresh.end() && (int)fit->second.size() == n &&
        fit->second.size() == (size_t)ggml_nelements(ids)) {
        v = fit->second;   // fresh routing harvested from this eval's router
    } else {
        read_ids_strided(ids, v);   // fallback (should not happen once warm)
        if ((int)v.size() != n) return true;
    }

    auto t0 = Clock::now();
    if (!g_full_fill) {
        if (g_use_odirect)
            g.cache->touch_batch_at(w->name, v.data(), n, win.base,
                                    win.file_off, win.ti->bytes_per_expert);
        else
            g.cache->touch_batch(w->name, v.data(), n, win.base, win.orig,
                                 win.ti->bytes_per_expert);
        g_fill_ns += std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
        g_fill_calls++;
    }
    if (g_pf_mode != PfMode::OFF)
        g_step_ids[g_id_slot][w->name] = v;
    lookahead_push(w->name);

    if (!win.rebound && w->data != win.base) {
        w->data = win.base;
        win.rebound = true;
        fprintf(stderr, "[bench] rebound %s\n", w->name);
    }

    g_audit[w] = {w->name, v, g.step, win.ti->bytes_per_expert};
    if (getenv("SOE_STEP_FILLS")) {
        char buf2[4096];
        int off = 0;
        for (int i = 0; i < (int)v.size() && off < 4000; i++)
            off += snprintf(buf2 + off, sizeof(buf2) - off, "%d,", v[i]);
        fprintf(stderr, "[fills] step %ld %s n=%d [%s]\n", g.step, w->name, n, buf2);
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <model.gguf> [cap_gib=2] [n_gen=64] [verify_n=64] "
                "[rebind=1]\n",
                argv[0]);
        return 2;
    }
    size_t cap_gib = argc > 2 ? (size_t)atoll(argv[2]) : 8;   // budget-clamped later
    int n_gen      = argc > 3 ? atoi(argv[3]) : 64;
    size_t verify_n = argc > 4 ? (size_t)atoll(argv[4]) : 64;
    g_rebind        = argc > 5 ? atoi(argv[5]) != 0 : true;
    if (getenv("SOE_FULL_FILL")) g_full_fill = true;
    if (getenv("SOE_STEP_FILLS")) g_step_fills = true;

    soe::ModelRegistry reg;
    soe::ModelManifest manifest;
    if (!reg.build(argv[1], manifest)) {
        fprintf(stderr, "[streaming-bench] manifest FAILED: %s\n", reg.error().c_str());
        return 1;
    }

    // Budget manager: clamp the cache cap to what this machine can hold.
    // The 12G-cap OOM experiment is impossible by construction now.
    soe::MemoryBudget budget = soe::MemoryBudget::detect();
    budget.kv = 64u << 20;         // measured: llama compute buffer + KV/S-state
    budget.staging = 64u << 20;
    budget.runtime_base = 512u << 20;
    // ANON policy moves dense weights from reclaimable page cache into the
    // anonymous sum - the cache clamp must shrink by the same amount.
    if (g_dense == DensePolicy::ANON)
        for (const auto& ti : manifest.tensors)
            if (ti.kind != soe::TensorKind::ROUTED_EXPERT)
                budget.runtime_base += ti.bytes_total;
    size_t requested = cap_gib << 30;
    size_t effective = budget.clamp_cache_cap(requested);
    if (effective != requested)
        fprintf(stderr, "[budget] cache cap clamped: %zu GiB -> %zu GiB "
                        "(available %.1f GiB)\n",
                (size_t)cap_gib, effective >> 30,
                budget.mem_available / 1073741824.0);
    {
        auto regime = soe::classify((size_t)manifest.file_size, budget.mem_available);
        printf("budget: available=%.1f GiB model=%.1f GiB regime=%s cap=%zu MiB\n",
               budget.mem_available / 1073741824.0,
               manifest.file_size / 1073741824.0,
               soe::regime_name(regime), effective >> 20);
    }
    g_use_cap = effective;

    // O_DIRECT mode auto-selects on prepared inputs (io_alignment >= 4096,
    // every routed slice aligned). Misses then read straight from this file.
    int lanes = getenv("SOE_LANES") ? atoi(getenv("SOE_LANES")) : 4;   // device-saturated beyond 4
    if (manifest.io_alignment >= 4096 &&
        manifest.all_slices_aligned == manifest.routed_expert_tensors) {
        if (g_direct.open_read(argv[1]) && g_direct.valid()) {
            g_use_odirect = true;
            if (lanes > 1)
                g_sched = std::make_unique<soe::IoScheduler>(lanes);
            printf("fill backend: O_DIRECT (%s), lanes=%d\n",
                   g_direct.direct() ? "direct" : "buffered fallback", lanes);
        }
    } else {
        printf("fill backend: memcpy from mmap\n");
    }

    // Dense policy: explicit env wins; otherwise AUTO from detected regime -
    // R2+ (model well above available RAM) prefers anon-dense (fault-free),
    // smaller regimes stay mmap (nothing to protect from reclaim).
    {
        const char* dp = getenv("SOE_DENSE");
        std::string s = dp ? dp : "";
        soe::MemoryBudget pre = soe::MemoryBudget::detect();
        uint64_t dense_bytes = 0;
        for (const auto& ti : manifest.tensors)
            if (ti.kind != soe::TensorKind::ROUTED_EXPERT) {
                g_dense_names.insert(ti.name);
                dense_bytes += ti.bytes_total;
            }
        soe::Regime reg = soe::classify((size_t)manifest.file_size, pre.mem_available);
        printf("regime=%s (available %.1f GiB, dense %.2f GiB)\n",
               soe::regime_name(reg), pre.mem_available / 1073741824.0,
               dense_bytes / 1073741824.0);
        if (s.empty())
            g_dense = (reg != soe::Regime::R0_RESIDENT &&
                       pre.mem_available > dense_bytes * 2)
                          ? DensePolicy::ANON
                          : DensePolicy::MMAP;
        else if (s == "warm") g_dense = DensePolicy::WARM;
        else if (s == "anon") g_dense = DensePolicy::ANON;
        printf("dense policy: %s\n",
               g_dense == DensePolicy::ANON ? "anon"
               : g_dense == DensePolicy::WARM ? "warm" : "mmap");
    }

    // WARM policy: page in all dense spans before context creation.
    if (g_dense == DensePolicy::WARM) {
        auto t0 = Clock::now();
        FILE* f = fopen(argv[1], "rb");
        if (!f) { fprintf(stderr, "[dense] warm open failed\n"); return 1; }
        char wbuf[1 << 20];
        size_t warmed = 0;
        for (const auto& ti : manifest.tensors) {
            if (ti.kind == soe::TensorKind::ROUTED_EXPERT) continue;
            if (fseeko(f, (off_t)ti.abs_offset, SEEK_SET) != 0) continue;
            uint64_t left = ti.bytes_total;
            while (left) {
                size_t c = left < sizeof(wbuf) ? (size_t)left : sizeof(wbuf);
                if (fread(wbuf, 1, c, f) != c) break;
                left -= c; warmed += c;
            }
        }
        fclose(f);
        double secs = std::chrono::duration<double>(Clock::now() - t0).count();
        printf("dense=warm: %.1f MiB paged in (%.2fs)\n", warmed / 1048576.0, secs);
    }

    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers    = 0;
    mparams.use_extra_bufts = false;
    llama_model* model = llama_model_load_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "[streaming-bench] LOAD FAILED\n"); return 1; }

    soe::SliceCache cache(soe::CacheLimits{g_use_cap});
    cache.set_verify_next(verify_n);
    if (g_use_odirect) {
        cache.set_source({od_read, &g_direct});
        cache.set_scheduler(g_sched.get());   // null -> inline fills
        const char* pf = getenv("SOE_PREFETCH");
        std::string pfs = pf ? pf : "";
        if (pfs == "1" || pfs == "lookahead")
            g_pf_mode = PfMode::LOOKAHEAD;
        else if (pfs == "full")
            g_pf_mode = PfMode::FULL;
        if (g_pf_mode != PfMode::OFF) {
            g_pi_thread = std::thread(prefetch_worker);
            printf("prefetch: %s mode\n",
                   g_pf_mode == PfMode::FULL ? "FULL" : "LOOKAHEAD");
        }
    }
    g.manifest = &manifest;
    g.cache    = &cache;

    auto cparams             = llama_context_default_params();
    cparams.n_ctx            = 256;
    cparams.n_batch          = cparams.n_ctx;
    cparams.n_ubatch         = 64;
    cparams.n_threads        = 8;
    cparams.n_threads_batch  = 8;
    cparams.cb_eval          = cb_eval;
    cparams.cb_eval_user_data = nullptr;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "[streaming-bench] CONTEXT FAILED\n"); return 1; }
    llama_set_warmup(ctx, false);

    // ANON policy: one scan decode binds every dense weight to anon memory,
    // then memory is cleared so generation state is pristine.
    if (g_dense == DensePolicy::ANON) {
        g_anon_scan = true;
        llama_token b = llama_vocab_bos(llama_model_get_vocab(model));
        llama_batch wb = llama_batch_get_one(&b, 1);
        if (llama_decode(ctx, wb))
            fprintf(stderr, "[dense] anon scan decode failed (continuing)\n");
        g_anon_scan = false;
        llama_memory_clear(llama_get_memory(ctx), true);
        printf("dense=anon: %zu tensors bound, %zu MiB anonymous\n",
               g_dense_bind.size(), g_dense_anon_bytes >> 20);
    }

    const char* prompt = getenv("SOE_PROMPT") ? getenv("SOE_PROMPT")
                                             : "The capital of France is";
    const auto* vocab  = llama_model_get_vocab(model);
    std::vector<llama_token> toks(32);
    int n_tok = -1;
    // llama_tokenize contract: negative return = required token count.
    while (n_tok < 0) {
        if ((size_t)(-n_tok) > toks.size())
            toks.resize((size_t)(-n_tok));
        n_tok = llama_tokenize(vocab, prompt, (int)strlen(prompt),
                               toks.data(), (int)toks.size(), true, false);
        if (n_tok < 0 && (size_t)(-n_tok) == toks.size())
            toks.resize(toks.size() * 2);
    }
    toks.resize(n_tok);
    printf("prefill tokens: %d (n_ctx %d)\n", n_tok, (int)cparams.n_ctx);
    if (llama_decode(ctx, llama_batch_get_one(toks.data(), n_tok))) {
        fprintf(stderr, "[streaming-bench] PREFILL FAILED\n");
        return 1;
    }

    printf("windows created: %zu | cap: %zu MiB\n", g.windows.size(), g_use_cap >> 20);

    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    uint64_t misses_before = cache.stats().misses;
    MemSnap snap0 = mem_snap();
    auto t0 = Clock::now();
    printf("tokens:");
    int produced = 0;
    auto dump_logits = [&](int tag) {
        const char* ldir = getenv("SOE_DUMP_LOGITS");
        if (!ldir || !*ldir) return;
        float* lg = llama_get_logits_ith(ctx, 0);
        int nv = llama_vocab_n_tokens(vocab);
        char lp[512];
        snprintf(lp, sizeof(lp), "%s/s%03d.f32", ldir, tag);
        FILE* lf = fopen(lp, "wb");
        if (lf) { fwrite(lg, sizeof(float), nv, lf); fclose(lf); }
    };
    for (int i = 0; i < n_gen; i++) {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;
        printf(" %d", id);
        fflush(stdout);
        produced++;
        fprintf(stderr, "[bench] step %d\n", i);
        g.step = i;
        if (llama_decode(ctx, llama_batch_get_one(&id, 1))) {
            fprintf(stderr, "\n[streaming-bench] DECODE FAILED\n");
            return 1;
        }
        dump_logits(i);   // logits of THIS step's forward (produces token i+1)
        // FULL-mode overlap: speculate the ENTIRE next step from this step's
        // routing. No-op when the cache already retains the working set;
        // decisive when capacity pressure evicts slices between steps.
        if (g_pf_mode == PfMode::FULL) {
            std::lock_guard<std::mutex> lk(g_pi_mu);
            g_pi_q.clear();
            for (auto& [nm, ids] : g_step_ids[g_id_slot])
                if (!ids.empty())
                    g_pi_q.push_back({nm, ids});
            g_pi_cv.notify_one();
        }
        g_id_slot ^= 1;   // step boundary: previous becomes read-only source
        g_step_ids[g_id_slot].clear();
        g_looked.clear();
        g_looked_step = -1;
    }

    // stop the prefetcher before tearing down cache/windows
    {
        std::lock_guard<std::mutex> lk(g_pi_mu);
        g_pi_stop = true;
    }
    g_pi_cv.notify_all();
    if (g_pi_thread.joinable())
        g_pi_thread.join();
    double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    MemSnap snap1 = mem_snap();
    printf("\n[mem] minflt+%ld majflt+%ld rss=%.2f GiB hwm=%.2f GiB\n",
           snap1.minflt - snap0.minflt, snap1.majflt - snap0.majflt,
           snap1.rss_kib / 1048576.0, snap1.hwm_kib / 1048576.0);

    auto st = cache.stats();
    uint64_t gen_misses = st.misses - misses_before;

    // full-window integrity check: every expert slice in every window must
    // match the original mapping byte-for-byte (zeros where never touched
    // would mean the kernel read unfilled data; foreign writes reveal overlap).
    // Skipped in O_DIRECT mode: there is no mmap mirror to compare against.
    if (g_rebind && !g_full_fill && !g_use_odirect) {
        for (auto& [name, win] : g.windows) {
            if (!win.base || !win.orig || !win.ti) continue;
            size_t bpe = win.ti->bytes_per_expert;
            for (int e = 0; e < (int)(win.ti->bytes_total / bpe); e++) {
                if (memcmp((const char*)win.base + (size_t)e * bpe,
                           (const char*)win.orig + (size_t)e * bpe, bpe) != 0) {
                    size_t first = 0;
                    const char* a = (const char*)win.base + (size_t)e * bpe;
                    const char* b = (const char*)win.orig + (size_t)e * bpe;
                    while (first < bpe && a[first] == b[first]) first++;
                    fprintf(stderr,
                            "[integrity] %s expert %d DIFFERS at byte %zu "
                            "(win=0x%02x orig=0x%02x)\n",
                            name.c_str(), e, first,
                            (unsigned char)a[first], (unsigned char)b[first]);
                }
            }
        }
        fprintf(stderr, "[integrity] window check done\n");
    }
    double hit = st.hits + st.misses ? 100.0 * st.hits / (st.hits + st.misses) : 0;
    printf("=== streaming-bench summary ===\n");
    printf("tok/s              : %.2f\n", produced / secs);
    printf("produced           : %d in %.1fs\n", produced, secs);
    printf("cache hit rate     : %.2f%% (hits=%llu misses=%llu)\n", hit,
           (unsigned long long)st.hits, (unsigned long long)st.misses);
    printf("gen-phase misses   : %llu (%.1f MiB/step)\n",
           (unsigned long long)gen_misses,
           produced ? gen_misses * (double)manifest.tensors.front().bytes_per_expert /
                          produced / 1048576.0
                    : 0.0);
    printf("bytes loaded       : %.2f MiB | evictions: %llu | used: %.2f / %zu MiB\n",
           st.bytes_loaded / 1048576.0, (unsigned long long)st.evictions,
           cache.used_bytes() / 1048576.0, g_use_cap >> 10);
    printf("dedup requests     : %llu\n", (unsigned long long)st.dedup_requests);
    printf("audit checks run   : %ld | pending records: %zu\n", g_audit_checks, g_audit.size());
    double fill_s = g_fill_ns / 1e9;
    double fill_frac = secs > 0 ? 100.0 * fill_s / secs : 0;
    printf("fill time          : %.2fs (%.1f%% of gen wall) over %llu batch fills\n",
           fill_s, fill_frac, (unsigned long long)g_fill_calls);
    printf("prefetched slices  : %llu\n", (unsigned long long)st.prefetched);

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
