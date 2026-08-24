// Demand-serving runtime implementation. Extracted from
// tools/cne-bench.cpp: router-harvest callback, expert
// windows, slice fills, expert-mass gating, prefetch overlap,
// anon-dense binding, audit/dump instrumentation. Single-decode assumption.
#include "cne_stream_cb.h"

#include "cne/config.h"
#include "cne/direct_io.h"
#include "cne/io_scheduler.h"
#include "cne/tensor_classify.h"

#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cne {

struct Window {
    void* base = nullptr;
    void* orig = nullptr;      // original llama mapping (copy source; memcpy mode)
    uint64_t file_off = 0;     // tensor start in backing store (odirect mode)
    bool  rebound = false;
    const cne::TensorInfo* ti = nullptr;
};

namespace {

using Clock2 = std::chrono::steady_clock;

cne::DirectFile g_direct;
bool g_use_odirect = false;
std::unique_ptr<cne::IoScheduler> g_sched;
double g_fill_ns = 0;   // time inside batch fills (I/O + memcpy)
uint64_t g_fill_calls = 0;

// ---- Dense residency (ANON binding observed via callback edges) ----------
std::unordered_map<std::string, void*> g_dense_bind;   // name -> anon copy
std::unordered_set<std::string> g_dense_names;         // manifest non-routed names
size_t g_dense_anon_bytes = 0;
bool g_anon_scan = false;                              // scan pass observes all nodes

// bind one weight tensor to an anonymous copy (ANON policy)
void try_bind_dense(const ggml_tensor* s) {
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

void scan_dense_srcs(ggml_tensor* t) {
    if (!t) return;
    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; i++)
        try_bind_dense(t->src[i]);
}

bool od_read(void* dest, uint64_t off, size_t bytes, void* ud) {
    return ((cne::DirectFile*)ud)->read_aligned(dest, bytes, off);
}

struct State {
    const cne::ModelManifest* manifest = nullptr;
    cne::SliceCache* cache = nullptr;
    std::map<std::string, Window> windows;
    long step = -1;
};

struct AuditRec {
    std::string tname;
    std::vector<int32_t> ids;
    long step;
    size_t bpe;
};
std::map<ggml_tensor*, AuditRec> g_audit;

// Router-ids tensors are NOT guaranteed contiguous (prefill rows are strided
// in the graph pool): always gather via nb[] strides, never flat memcpy.
void read_ids_strided(const ggml_tensor* ids, std::vector<int32_t>& out) {
    out.clear();
    for (int64_t i1 = 0; i1 < ids->ne[1]; i1++)
        for (int64_t i0 = 0; i0 < ids->ne[0]; i0++) {
            int32_t e;
            memcpy(&e, (const char*)ids->data + i1 * ids->nb[1] + i0 * ids->nb[0], 4);
            out.push_back(e);
        }
}

State g;

// ---- Expert-mass gating (conditional expert execution) --------------------
float g_l2_mass = 0.0f;      // 0 = disabled (lossless)
int g_l2_min_k = 2;
long g_l2_dropped_slices = 0;

// ---- Overlap: speculative prefetch (last-token routing reuse) -------------
PfMode g_pf_mode = PfMode::OFF;

struct PrefetchItem {
    std::string name;
    std::vector<int32_t> ids;
};
std::mutex g_pi_mu;
std::condition_variable g_pi_cv;
std::deque<PrefetchItem> g_pi_q;      // latest-wins
bool g_pi_stop = false;
std::thread g_pi_thread;
std::unique_ptr<cne::IoScheduler> g_psched;

void prefetch_worker() {
    std::unique_ptr<cne::IoScheduler> psched = std::make_unique<cne::IoScheduler>(4);
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
std::unordered_map<std::string, std::vector<int32_t>> g_step_ids[2];
int g_id_slot = 0;
std::unordered_set<std::string> g_looked;
long g_looked_step = -1;
constexpr int kLookaheadNodes = 6;

int node_rank(const char* name) {
    int L = parse_layer_index(name);
    const char* k = strstr(name, "ffn_gate_exps");
    int kind = k ? 0 : (strstr(name, "ffn_up_exps") ? 1 : 2);
    return L >= 0 ? L * 3 + kind : -1;
}

std::string rank_name(int rank) {
    // inverse of node_rank for our fixed naming scheme
    static const char* kinds[] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
    char buf[128];
    snprintf(buf, sizeof(buf), "blk.%d.%s.weight", rank / 3, kinds[rank % 3]);
    return buf;
}

// called from cb_eval after a node's demand fill
void lookahead_push(const char* name) {
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

bool g_step_fills = false;
long g_audit_checks = 0;

void maybe_dump_dst(ggml_tensor* t, long step) {
    const char* dir = cne::env("DUMP_DST");
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
bool g_rebind = true;
bool g_full_fill = false;   // debug: fill whole windows once at creation

void ensure_window(const char* name, ggml_tensor* w) {
    if (!g_rebind) return;
    static int layer_limit = cne::env("LAYER_LIMIT") ? atoi(cne::env("LAYER_LIMIT")) : -1;
    if (layer_limit >= 0) {
        int L = parse_layer_index(name);
        if (L < 0 || L >= layer_limit) return;
    }
    auto it = g.windows.find(name);
    if (it != g.windows.end()) return;
    for (const auto& ti : g.manifest->tensors)
        if (ti.name == name && ti.kind == TensorKind::ROUTED_EXPERT) {
            Window win;
            win.ti   = &ti;
            win.file_off = ti.abs_offset;
            win.orig = g_use_odirect ? nullptr : w->data;
            win.base = mmap(nullptr, ti.bytes_total, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (win.base == MAP_FAILED) {
                fprintf(stderr, "[cne-bench] window mmap FAILED for %s\n", name);
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

} // namespace

ggml_backend_sched_eval_callback stream_cb_eval() {
    return [](ggml_tensor* t, bool ask, void*) -> bool {
        // Flow: isolate only the MoE routing argsort nodes and the expert
        // matmuls (MUL_MAT_ID) - 160/step instead of ~7800. At each routing
        // node's post-compute callback the engine harvests that layer's kept
        // expert ids and fills the matching window slices; the expert matmul
        // asks then serve from that stash. Everything else batches normally.
        if (!t)
            return true;
        static long cb_vis = 0;
        if (++cb_vis <= 15 && t->name)
            fprintf(stderr, "[cb] #%ld name='%s' op=%d\n",
                    cb_vis, t->name, (int)t->op);
        if (g_anon_scan) {   // ANON policy: one warmup pass observes every node
            scan_dense_srcs(t);
            return true;
        }
        // Router node = the MoE routing argsort over ALL expert probabilities,
        // named 'ffn_moe_argsort-<layer>' holding ranked ids [n_exp, n_tokens].
        // ggml_top_k compiles to argsort + view: there is no TOP_K op node and
        // most nodes carry generic pool names - match op AND name prefix.
        const char* nm = t->name ? t->name : "";
        const bool is_router = ask == false || true ? (t->op == GGML_OP_ARGSORT &&
                               strncmp(nm, "ffn_moe_argsort", 15) == 0) : false;
        bool is_mmid = t->op == GGML_OP_MUL_MAT_ID;
        if (!is_router && !is_mmid)
            return ask ? false : true;   // batch everything else

        static std::unordered_map<int, std::vector<int32_t>> g_fresh_kept; // layer -> kept ids

        // ---- ROUTER (argsort over all expert probabilities) ----
        if (is_router) {
            if (ask)
                return true;   // isolate: it computes alone, then we harvest
            int L = parse_layer_index(t->name ? t->name : "");
            if (L < 0 || !t->data || t->type != GGML_TYPE_I32)
                return true;
            std::vector<int32_t> v;
            read_ids_strided(t, v);
            if (v.empty())
                return true;

            // Rank-aware conditional execution: walk the ranked ids
            // ascending, accumulate normalized probability mass over the USED
            // experts, drop every rank past the mass threshold (min floor kept).
            // Dropped ids are excluded from the fill list so their window slices
            // stay zero - dropped experts contribute exactly nothing. No writes
            // into graph-pool memory happen anywhere in this flow: pool slots
            // are reused across nodes, so callback-side writes corrupt whichever
            // tensor next occupies the slot.
            const ggml_tensor* probs = t->src[0] ? t->src[0] : nullptr;
            const int64_t n_exp = t->ne[0], ntok = t->ne[1];
            const int64_t used = g.manifest ? (int64_t)g.manifest->n_experts_used : 0;
            bool can_gate = g_l2_mass > 0.0f && !g_anon_scan &&
                            probs && probs->type == GGML_TYPE_F32 && probs->data &&
                            probs->ne[0] == n_exp && probs->ne[1] == ntok &&
                            used > 0 && used <= n_exp;
            std::vector<int32_t> kept_ids;
            if (!can_gate) {
                for (int64_t col = 0; col < ntok; col++)
                    for (int64_t r = 0; r < used; r++) {
                        int32_t e;
                        memcpy(&e, (const char*)t->data +
                                    r * t->nb[0] + col * t->nb[1], 4);
                        kept_ids.push_back(e);
                    }
            } else {
                for (int64_t col = 0; col < ntok; col++) {
                    double p[4096];
                    double sel = 0.0;
                    for (int64_t r = 0; r < used && r < 4096; r++) {
                        memcpy(&p[r], (const char*)probs->data +
                                   r * probs->nb[0] + col * probs->nb[1], 4);
                        sel += p[r];
                    }
                    if (!(sel > 0)) continue;
                    // ranks arrive sorted desc by probability (argsort contract)
                    double cum = 0.0;
                    int64_t m = used;
                    for (int64_t r = 0; r < used && r < 4096; r++) {
                        cum += p[r] / sel;
                        if (cum >= (double)g_l2_mass) { m = r + 1; break; }
                    }
                    if (m < g_l2_min_k) m = g_l2_min_k;
                    for (int64_t r = m; r < used; r++) g_l2_dropped_slices++;
                    for (int64_t r = 0; r < m; r++) {
                        int32_t e;
                        memcpy(&e, (const char*)t->data +
                                    r * t->nb[0] + col * t->nb[1], 4);
                        kept_ids.push_back(e);
                    }
                }
            }
            g_fresh_kept[L] = std::move(kept_ids);
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
                auto& kept = g_fresh_kept[L];
                if (kept.empty()) { wit = g.windows.end(); continue; }
                auto t0 = Clock2::now();
                g.cache->touch_batch_at(buf, kept.data(), (int)kept.size(),
                                        wit->second.base, wit->second.file_off,
                                        wit->second.ti->bytes_per_expert);
                g_fill_ns += std::chrono::duration<double, std::nano>(Clock2::now() - t0).count();
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
                                return ti.kind == TensorKind::ROUTED_EXPERT;
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

        std::vector<int32_t> v;
        int L = parse_layer_index(w->name);
        {
            auto kit = g_fresh_kept.find(L);
            if (kit != g_fresh_kept.end() && !kit->second.empty())
                v = kit->second;   // fresh + mass-filtered routing from this eval's router
            else {
                read_ids_strided(ids, v);   // lossless fallback (gating disabled)
            }
        }
        const int n = (int)v.size();
        if (n <= 0 || n > 8192) return true;


        auto t0 = Clock2::now();
        if (!g_full_fill) {
            if (g_use_odirect)
                g.cache->touch_batch_at(w->name, v.data(), (int)v.size(),
                                        win.base, win.file_off,
                                        win.ti->bytes_per_expert);
            else
                g.cache->touch_batch(w->name, v.data(), (int)v.size(),
                                     win.base, win.orig,
                                     win.ti->bytes_per_expert);
            g_fill_ns += std::chrono::duration<double, std::nano>(Clock2::now() - t0).count();
            g_fill_calls++;
        }
        if (g_pf_mode != PfMode::OFF)
            g_step_ids[g_id_slot][w->name] = v;
        lookahead_push(w->name);   // speculates kept routing only

        if (!win.rebound && w->data != win.base) {
            w->data = win.base;
            win.rebound = true;
            fprintf(stderr, "[cne] rebound %s\n", w->name);
        }

        g_audit[w] = {w->name, v, g.step, win.ti->bytes_per_expert};
        if (g_step_fills) {
            char buf2[4096];
            int off = 0;
            for (int i = 0; i < (int)v.size() && off < 4000; i++)
                off += snprintf(buf2 + off, sizeof(buf2) - off, "%d,", v[i]);
            fprintf(stderr, "[fills] step %ld %s n=%d [%s]\n", g.step, w->name, n, buf2);
        }
        return true;
    };
}

// ---- Public API ------------------------------------------------------------

void stream_init(const ModelManifest& manifest, SliceCache& cache,
                 const StreamConfig& cfg) {
    g.manifest = &manifest;
    g.cache    = &cache;
    g_rebind     = cfg.rebind;
    g_full_fill  = cfg.full_fill;
    g_step_fills = cfg.step_fills;
    g_l2_mass    = cfg.l2_mass;
    g_l2_min_k   = cfg.l2_min_k;
    g_pf_mode    = cfg.pf_mode;
    for (const auto& ti : manifest.tensors)
        if (ti.kind != TensorKind::ROUTED_EXPERT)
            g_dense_names.insert(ti.name);
}

bool stream_open_fill_backend(const char* model_path, int lanes) {
    // O_DIRECT mode auto-selects on prepared inputs (io_alignment >= 4096,
    // every routed slice aligned). Misses then read straight from this file.
    if (g.manifest->io_alignment >= 4096 &&
        g.manifest->all_slices_aligned == g.manifest->routed_expert_tensors) {
        if (g_direct.open_read(model_path) && g_direct.valid()) {
            g_use_odirect = true;
            if (lanes > 1)
                g_sched = std::make_unique<IoScheduler>(lanes);
            g.cache->set_source({od_read, &g_direct});
            g.cache->set_scheduler(g_sched.get());   // null -> inline fills
            printf("fill backend: O_DIRECT (%s), lanes=%d\n",
                   g_direct.direct() ? "direct" : "buffered fallback", lanes);
            return true;
        }
    }
    printf("fill backend: memcpy from mmap\n");
    return false;
}

bool stream_use_odirect() { return g_use_odirect; }

void stream_anon_scan_begin() { g_anon_scan = true; }
void stream_anon_scan_end()   { g_anon_scan = false; }
size_t stream_dense_bound_count() { return g_dense_bind.size(); }
size_t stream_dense_anon_bytes()  { return g_dense_anon_bytes; }

void stream_prefetch_start() {
    if (g_pf_mode == PfMode::OFF) return;
    g_pi_thread = std::thread(prefetch_worker);
    printf("prefetch: %s mode\n",
           g_pf_mode == PfMode::FULL ? "FULL" : "LOOKAHEAD");
}

void stream_prefetch_stop() {
    {
        std::lock_guard<std::mutex> lk(g_pi_mu);
        g_pi_stop = true;
    }
    g_pi_cv.notify_all();
    if (g_pi_thread.joinable())
        g_pi_thread.join();
}

void stream_prefetch_kick_full() {
    if (g_pf_mode != PfMode::FULL)
        return;
    {
        std::lock_guard<std::mutex> lk(g_pi_mu);
        g_pi_q.clear();
        for (auto& [nm, ids] : g_step_ids[g_id_slot])
            if (!ids.empty())
                g_pi_q.push_back({nm, ids});
        g_pi_cv.notify_one();
    }
}

void stream_set_step(long step) { g.step = step; }

void stream_step_boundary() {
    g_id_slot ^= 1;   // step boundary: previous becomes read-only source
    g_step_ids[g_id_slot].clear();
    g_looked.clear();
    g_looked_step = -1;
}

StreamTelemetry stream_telemetry() {
    StreamTelemetry t;
    t.fill_s        = g_fill_ns / 1e9;
    t.fill_calls    = g_fill_calls;
    t.audit_checks  = g_audit_checks;
    t.audit_pending = g_audit.size();
    t.l2_dropped    = g_l2_dropped_slices;
    t.l2_mass       = g_l2_mass;
    t.l2_min_k      = g_l2_min_k;
    return t;
}

void stream_check_windows() {
    // full-window integrity check: every expert slice in every window must
    // match the original mapping byte-for-byte (zeros where never touched
    // would mean the kernel read unfilled data; foreign writes reveal overlap).
    // Skipped in O_DIRECT mode: there is no mmap mirror to compare against.
    if (!g_rebind || g_full_fill || g_use_odirect)
        return;
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

} // namespace cne
