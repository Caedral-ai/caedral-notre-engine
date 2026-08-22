// streaming-bench: P2 exit measurement. Full greedy generation with the
// SliceCache demand-serving slices at the callback ask-point (E12), Strategy-A
// windows per fused tensor, rebinding on first touch. Reports tok/s,
// hit-rate, cold MiB/step, evictions — to be compared against lru-sim curves.
#include "soe/cache.h"
#include "soe/direct_io.h"
#include "soe/io_scheduler.h"
#include "soe/model.h"
#include "soe/model_registry.h"
#include "soe/tensor_classify.h"

#include "llama.h"

#include <sys/mman.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
static std::mutex g_cb_mu;
#include <map>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <unistd.h>

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
static bool g_full_fill = false;   // debug: fill whole windows once (E9 style)

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
    if (!g_rebind) {
        if (!ask && t && t->op == GGML_OP_MUL_MAT_ID) maybe_dump_dst(t, g.step);
        return true;
    }
    static bool dump_all = getenv("SOE_DUMP_ALL") != nullptr;
    if (!ask && dump_all && t && t->op == GGML_OP_MUL_MAT_ID && !g_audit.count(t->src[0] ? t->src[0] : t))
        maybe_dump_dst(t, g.step);
    long n = ++g_ask_n;
    if (n <= 100) {
        long tid = (long)gettid();
        static std::mutex m2;
        std::lock_guard<std::mutex> lk(m2);
        if (++g_tids[tid] == 2) fprintf(stderr, "[bench] !!! SECOND THREAD in callback: tid=%ld (ask #%ld)\n", tid, n);
    }
    if (!t || t->op != GGML_OP_MUL_MAT_ID) return true;
    if (ask) {
        ggml_tensor* w   = nullptr;
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
        read_ids_strided(ids, v);
        if ((int)v.size() != n) return true;

        if (!g_full_fill) {
            if (g_use_odirect)
                g.cache->touch_batch_at(w->name, v.data(), n, win.base,
                                        win.file_off, win.ti->bytes_per_expert);
            else
                g.cache->touch_batch(w->name, v.data(), n, win.base, win.orig,
                                     win.ti->bytes_per_expert);
        }
        if (!win.rebound && w->data != win.base) {
            w->data = win.base;
            win.rebound = true;
            fprintf(stderr, "[bench] rebound %s\n", w->name);
        }

        // post-compute audit state: remember ids + node for the ask=false pass
        g_audit[w] = {v, g.step, win.ti->bytes_per_expert};
        if (g_step_fills) {
            char buf[4096]; int off = 0;
            for (int i = 0; i < (int)v.size() && off < 4000; i++)
                off += snprintf(buf + off, sizeof(buf) - off, "%d,", v[i]);
            fprintf(stderr, "[fills] step %ld %s n=%d [%s]\n", g.step, w->name, n, buf);
        }
    } else {
        // ask=false: this MUL_MAT_ID just computed. Audit pending record.
        ggml_tensor* wten = nullptr;
        for (int i = 0; i < GGML_MAX_SRC && t->src[i]; i++)
            if (t->src[i] && g_audit.count(t->src[i])) { wten = t->src[i]; break; }
        if (!wten) return true;
        AuditRec rec = g_audit[wten];
        g_audit.erase(wten);
        maybe_dump_dst(t, g.step);
        if (rec.step != g.step) return true;
        auto wit = g.windows.find(wten->name);
        if (wit == g.windows.end()) return true;
        Window& win = wit->second;
        ggml_tensor* ids_now = nullptr;
        for (int i = 0; i < GGML_MAX_SRC && t->src[i]; i++)
            if (t->src[i]->type == GGML_TYPE_I32) { ids_now = t->src[i]; break; }
        if (!ids_now || !ids_now->data) return true;
        int nn = (int)ggml_nelements(ids_now);
        static thread_local std::vector<int32_t> v_post;
        read_ids_strided(ids_now, v_post);
        if ((int)v_post.size() != nn || nn != (int)rec.ids.size() ||
            memcmp(rec.ids.data(), v_post.data(), nn * 4) != 0) {
            fprintf(stderr, "[audit] step %ld %s: IDS CHANGED between ask and post\n",
                    g.step, wten->name);
            return true;
        }
        g_audit_checks++;
        bool odirect = g_use_odirect;
        if (!odirect) {
            for (int i = 0; i < nn; i++) {
                int e = rec.ids[i];
                if (e < 0 || e >= 256) continue;
                const void* d = (const char*)win.base + (size_t)e * rec.bpe;
                const void* s = (const char*)win.orig + (size_t)e * rec.bpe;
                if (memcmp(d, s, rec.bpe) != 0) {
                    fprintf(stderr, "[audit] step %ld %s: SLICE %d CORRUPT post-compute\n",
                            g.step, wten->name, e);
                    break;
                }
            }
        }
        // In odirect mode there is no mmap mirror to compare against;
        // DirectFile already fails closed on short/misaligned reads.
        if (wten->data != win.base)
            fprintf(stderr, "[audit] step %ld %s: REPOINT LOST post-compute\n",
                    g.step, wten->name);
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
    size_t cap_gib = argc > 2 ? (size_t)atoll(argv[2]) : 2;
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

    // O_DIRECT mode auto-selects on prepared inputs (io_alignment >= 4096,
    // every routed slice aligned). Misses then read straight from this file.
    int lanes = getenv("SOE_LANES") ? atoi(getenv("SOE_LANES")) : 1;
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

    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers    = 0;
    mparams.use_extra_bufts = false;
    llama_model* model = llama_model_load_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "[streaming-bench] LOAD FAILED\n"); return 1; }

    soe::SliceCache cache(soe::CacheLimits{cap_gib << 30});
    cache.set_verify_next(verify_n);
    if (g_use_odirect) {
        cache.set_source({od_read, &g_direct});
        cache.set_scheduler(g_sched.get());   // null -> inline fills
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

    const char* prompt = "The capital of France is";
    const auto* vocab  = llama_model_get_vocab(model);
    std::vector<llama_token> toks(32);
    int n_tok = llama_tokenize(vocab, prompt, (int)strlen(prompt),
                               toks.data(), (int)toks.size(), true, false);
    toks.resize(n_tok);
    if (llama_decode(ctx, llama_batch_get_one(toks.data(), n_tok))) {
        fprintf(stderr, "[streaming-bench] PREFILL FAILED\n");
        return 1;
    }

    printf("windows created: %zu | cap: %zu GiB\n", g.windows.size(), cap_gib);

    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    uint64_t misses_before = cache.stats().misses;
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
    }
    double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    printf("\n");

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
           cache.used_bytes() / 1048576.0, cap_gib << 10);
    printf("dedup requests     : %llu\n", (unsigned long long)st.dedup_requests);
    printf("audit checks run   : %ld | pending records: %zu\n", g_audit_checks, g_audit.size());

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
