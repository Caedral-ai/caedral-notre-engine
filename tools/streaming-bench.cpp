// cne-bench: end-to-end measurement of demand-served expert streaming.
// Full greedy generation with SliceCache filling slices at the callback
// ask-point. Reports tok/s, hit-rate, cold MiB/step, evictions - comparable
// against offline LRU simulation curves.
//
// The demand-serving runtime lives in adapters/stream_cb.cpp behind the
// cne_stream_cb.h API; this file is only a measurement driver.
#include "cne/cache.h"
#include "cne/io_scheduler.h"
#include "cne/memory_budget.h"
#include "cne/model.h"
#include "cne/config.h"
#include "cne/model_registry.h"

#include "cne_stream_cb.h"
#include "cne_stream_spec.h"

#include "llama.h"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

// RSS / fault telemetry
struct MemSnap { long minflt=0, majflt=0, rss_kib=0, hwm_kib=0; };
MemSnap mem_snap() {
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

// ---- Dense residency policies --------------------------------------------
enum class DensePolicy { MMAP, WARM, ANON };

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
    bool rebind     = argc > 5 ? atoi(argv[5]) != 0 : true;

    cne::ModelRegistry reg;
    cne::ModelManifest manifest;
    if (!reg.build(argv[1], manifest)) {
        fprintf(stderr, "[streaming-bench] manifest FAILED: %s\n", reg.error().c_str());
        return 1;
    }

    // Dense policy: explicit env wins; otherwise AUTO from the detected
    // memory regime - when the model is well above available RAM, anon-dense
    // residency avoids page-fault storms; smaller models stay on mmap.
    DensePolicy g_dense = DensePolicy::MMAP;
    {
        const char* dp = cne::env("DENSE");
        std::string s = dp ? dp : "";
        cne::MemoryBudget pre = cne::MemoryBudget::detect();
        uint64_t dense_bytes = 0;
        for (const auto& ti : manifest.tensors)
            if (ti.kind != cne::TensorKind::ROUTED_EXPERT)
                dense_bytes += ti.bytes_total;
        cne::Regime rg = cne::classify((size_t)manifest.file_size, pre.mem_available);
        printf("regime=%s (available %.1f GiB, dense %.2f GiB)\n",
               cne::regime_name(rg), pre.mem_available / 1073741824.0,
               dense_bytes / 1073741824.0);
        if (s.empty())
            g_dense = (rg != cne::Regime::R0_RESIDENT &&
                       pre.mem_available > dense_bytes * 2)
                          ? DensePolicy::ANON
                          : DensePolicy::MMAP;
        else if (s == "warm") g_dense = DensePolicy::WARM;
        else if (s == "anon") g_dense = DensePolicy::ANON;
        printf("dense policy: %s\n",
               g_dense == DensePolicy::ANON ? "anon"
               : g_dense == DensePolicy::WARM ? "warm" : "mmap");
    }

    // Budget manager: clamp the cache cap to what this machine can hold.
    // The 12G-cap OOM experiment is impossible by construction now.
    cne::MemoryBudget budget = cne::MemoryBudget::detect();
    budget.kv = 64u << 20;         // measured: llama compute buffer + KV/S-state
    budget.staging = 64u << 20;
    budget.runtime_base = 512u << 20;
    // ANON policy moves dense weights from reclaimable page cache into the
    // anonymous sum - the cache clamp must shrink by the same amount.
    if (g_dense == DensePolicy::ANON)
        for (const auto& ti : manifest.tensors)
            if (ti.kind != cne::TensorKind::ROUTED_EXPERT)
                budget.runtime_base += ti.bytes_total;
    size_t requested = cap_gib << 30;
    size_t effective = budget.clamp_cache_cap(requested);
    if (effective != requested)
        fprintf(stderr, "[budget] cache cap clamped: %zu GiB -> %zu GiB "
                        "(available %.1f GiB)\n",
                (size_t)cap_gib, effective >> 30,
                budget.mem_available / 1073741824.0);
    {
        auto regime = cne::classify((size_t)manifest.file_size, budget.mem_available);
        printf("budget: available=%.1f GiB model=%.1f GiB regime=%s cap=%zu MiB\n",
               budget.mem_available / 1073741824.0,
               manifest.file_size / 1073741824.0,
               cne::regime_name(regime), effective >> 20);
    }

    // L2 knob: DEFAULT UNSET = lossless full-k. Loud telemetry when active.
    float l2_mass = 0.0f;
    int l2_min_k = 2;
    if (cne::env("EXPERT_MASS")) {
        l2_mass = (float)atof(cne::env("EXPERT_MASS"));
        l2_min_k = cne::env("EXPERT_MIN_K") ? atoi(cne::env("EXPERT_MIN_K")) : 2;
        if (l2_mass <= 0.0f || l2_mass > 1.0f) {
            fprintf(stderr, "[L2] invalid CNE_EXPERT_MASS (or legacy SOE_EXPERT_MASS) - knob disabled\n");
            l2_mass = 0.0f;
        } else {
            printf("*** [L2] ACTIVE: expert-mass=%.2f min_k=%d "
                   "(LOSSY profile - quality degradation expected) ***\n",
                   l2_mass, l2_min_k);
        }
    }

    // Prefetch overlap: measured as a no-op or regression on this
    // hardware/model (adequate caching already captures cross-token routing
    // locality, and the prefetcher contends with demand fills on the cache
    // mutex); kept flag-gated for future capacity-pressure regimes.
    cne::PfMode pf_mode = cne::PfMode::OFF;
    {
        const char* pf = cne::env("PREFETCH");
        std::string pfs = pf ? pf : "";
        if (pfs == "1" || pfs == "lookahead")
            pf_mode = cne::PfMode::LOOKAHEAD;
        else if (pfs == "full")
            pf_mode = cne::PfMode::FULL;
    }

    cne::SliceCache cache(cne::CacheLimits{effective});
    cache.set_verify_next(verify_n);

    // Runtime wiring: callback machinery + fill backend.
    cne::StreamConfig cfg;
    cfg.rebind     = rebind;
    cfg.full_fill  = cne::env("FULL_FILL") != nullptr;
    cfg.step_fills = cne::env("STEP_FILLS") != nullptr;
    cfg.l2_mass    = l2_mass;
    cfg.l2_min_k   = l2_min_k;
    cfg.pf_mode    = pf_mode;
    cne::stream_init(manifest, cache, cfg);

    bool use_odirect = cne::stream_open_fill_backend(argv[1],
                         cne::env("LANES") ? atoi(cne::env("LANES")) : 4);
    cne::stream_prefetch_start();

    // WARM policy: page in all dense spans before context creation.
    if (g_dense == DensePolicy::WARM) {
        auto t0 = Clock::now();
        FILE* f = fopen(argv[1], "rb");
        if (!f) { fprintf(stderr, "[dense] warm open failed\n"); return 1; }
        char wbuf[1 << 20];
        size_t warmed = 0;
        for (const auto& ti : manifest.tensors) {
            if (ti.kind == cne::TensorKind::ROUTED_EXPERT) continue;
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
    if (cne::env("MTP")) mparams.load_mtp = true;
    llama_model* model = llama_model_load_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "[streaming-bench] LOAD FAILED\n"); return 1; }

    auto cparams             = llama_context_default_params();
    int ctx_size             = cne::env("CTX") ? atoi(cne::env("CTX")) : 256;
    if (ctx_size < 64) ctx_size = 64;
    cparams.n_ctx            = ctx_size;
    cparams.n_batch          = ctx_size < 512 ? ctx_size : 512;
    cparams.n_ubatch         = 64;
    cparams.n_threads        = 8;
    cparams.n_threads_batch  = 8;
    cparams.cb_eval          = cne::stream_cb_eval();
    cparams.cb_eval_user_data = nullptr;
    // MTP speculative decoding (CNE_MTP): 1 = default draft depth, N = depth.
    // The target context stays LLAMA_CONTEXT_TYPE_DEFAULT - the MTP context
    // type builds graph_mtp (nextn block alone) and belongs on the DRAFT
    // context, which common_speculative_init_from_params creates for us.
    int mtp_k = 0;
    if (cne::env("MTP")) {
        mtp_k = atoi(cne::env("MTP"));
        if (mtp_k == 1) mtp_k = 4;   // sane CPU default depth
        if (mtp_k < 0) mtp_k = 0;
    }
    if (mtp_k > 0) {
        cne::spec_mtp_size_outputs(cparams, mtp_k, (int)cparams.n_batch);
        fprintf(stderr, "[mtp] draft-mtp enabled: n_max=%d n_outputs_max=%u\n",
                mtp_k, cparams.n_outputs_max);
    }

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "[streaming-bench] CONTEXT FAILED\n"); return 1; }
    // Warmup runs must stay off for models near or above RAM size: the
    // dummy-shape decode can page in large weight spans for no benefit.
    // Exception: MTP probing keeps upstream's warmup-on behavior, since the
    // nextn-augmented graph may need shaped buffers before real decodes.
    if (!cne::env("MTP"))
        llama_set_warmup(ctx, false);

    // ANON policy: one scan decode binds every dense weight to anon memory,
    // then memory is cleared so generation state is pristine.
    if (g_dense == DensePolicy::ANON) {
        cne::stream_anon_scan_begin();
        llama_token b = llama_vocab_bos(llama_model_get_vocab(model));
        llama_batch wb = llama_batch_get_one(&b, 1);
        if (llama_decode(ctx, wb))
            fprintf(stderr, "[dense] anon scan decode failed (continuing)\n");
        cne::stream_anon_scan_end();
        llama_memory_clear(llama_get_memory(ctx), true);
        printf("dense=anon: %zu tensors bound, %zu MiB anonymous\n",
               cne::stream_dense_bound_count(),
               cne::stream_dense_anon_bytes() >> 20);
    }

    const char* prompt = cne::env("PROMPT") ? cne::env("PROMPT")
                                             : "The capital of France is";
    const auto* vocab  = llama_model_get_vocab(model);

    // ---- PL whole-model PPL mode (policy-aware: L2/anon apply) ----
    // Chunked CE over a plain-text corpus: windows of ctx tokens, non-
    // overlapping, memory cleared between windows. This is the canonical
    // whole-model gate because external llama-perplexity cannot apply our
    // callback policies (L2 knob etc).
    if (cne::env("PPL_FILE")) {
        const char* pf = cne::env("PPL_FILE");
        std::ifstream cf(pf);
        if (!cf) { fprintf(stderr, "[ppl] cannot open %s\n", pf); return 1; }
        std::string text((std::istreambuf_iterator<char>(cf)),
                         std::istreambuf_iterator<char>());
        std::vector<llama_token> seq(64);
        int n_seq = -1;
        while (n_seq < 0) {
            if ((size_t)(-n_seq) > seq.size()) seq.resize((size_t)(-n_seq));
            n_seq = llama_tokenize(vocab, text.c_str(), (int)text.size(),
                                   seq.data(), (int)seq.size(), true, false);
            if (n_seq < 0 && (size_t)(-n_seq) == seq.size())
                seq.resize(seq.size() * 2);
        }
        seq.resize(n_seq);
        const int nv = llama_vocab_n_tokens(vocab);
        double nll = 0.0; long long ntok = 0;
        int wins = 0;
        for (size_t off = 0; off + 1 < (size_t)n_seq; off += (size_t)ctx_size) {
            const int n = (int)std::min((size_t)ctx_size, (size_t)n_seq - off);
            if (n < 2) break;
            llama_batch wb = llama_batch_init(n, 0, 1);
            for (int i = 0; i < n; i++) {
                wb.token[i]  = seq[off + i];
                wb.pos[i]    = (llama_pos)i;
                wb.n_seq_id[i] = 1;
                wb.seq_id[i][0] = 0;
                wb.logits[i] = true;   // need CE at every position
            }
            wb.n_tokens = n;
            if (llama_decode(ctx, wb)) {
                llama_batch_free(wb);
                fprintf(stderr, "[ppl] decode failed at offset %zu\n", off);
                return 1;
            }
            llama_batch_free(wb);
            for (int i = 0; i + 1 < (int)n; i++) {
                const float* lg = llama_get_logits_ith(ctx, i);
                float mx = lg[0];
                for (int v2 = 1; v2 < nv; v2++) mx = std::max(mx, lg[v2]);
                double lse = mx;
                double s = 0.0;
                for (int v2 = 0; v2 < nv; v2++) s += std::exp((double)lg[v2] - mx);
                lse += std::log(s);
                nll += lse - (double)lg[seq[off + i + 1]];
                ntok++;
            }
            llama_memory_clear(llama_get_memory(ctx), true);
            wins++;
            fprintf(stderr, "[ppl] window %d done (%lld tokens evaluated)\n",
                    wins, ntok);
        }
        double ppl = ntok ? std::exp(nll / (double)ntok) : 0.0;
        printf("engine-ppl: %.4f (ntok %lld, windows %d, n_ctx %d)\n",
               ppl, ntok, wins, ctx_size);
        if (l2_mass > 0.0f)
            printf("[L2] dropped slices total: %ld\n",
                   cne::stream_telemetry().l2_dropped);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

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
    if (mtp_k > 0) {
        // MTP path below does its own two-phase prefill (n-1 tokens, then
        // the last token together with the first verify batch).
    } else if (cne::env("SPLIT_PREFILL")) {
        // Bisect knob: same two-phase prefill shape as the MTP path
        // (n-1 tokens, then 1) WITHOUT any speculation.
        llama_batch bp = llama_batch_init(n_tok > 1 ? (size_t)(n_tok - 1) : 1, 0, 1);
        for (int i = 0; i + 1 < n_tok; i++) {
            bp.token[i]  = toks[i];
            bp.pos[i]    = (llama_pos)i;
            bp.n_seq_id[i] = 1;
            bp.seq_id[i][0] = 0;
            bp.logits[i] = false;
        }
        bp.n_tokens = n_tok > 1 ? n_tok - 1 : 1;
        if (llama_decode(ctx, bp)) { fprintf(stderr, "[streaming-bench] PREFILL FAILED\n"); return 1; }
        llama_batch_free(bp);
        llama_token il = toks.back();
        if (llama_decode(ctx, llama_batch_get_one(&il, 1))) {
            fprintf(stderr, "[streaming-bench] PREFILL FAILED\n");
            return 1;
        }
    } else if (llama_decode(ctx, llama_batch_get_one(toks.data(), n_tok))) {
        fprintf(stderr, "[streaming-bench] PREFILL FAILED\n");
        return 1;
    }

    printf("windows created: (see [geom] lines above) | cap: %zu MiB\n", effective >> 20);

    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    uint64_t misses_before = cache.stats().misses;
    MemSnap snap0 = mem_snap();
    auto t0 = Clock::now();
    printf("tokens:");
    int produced = 0;
    long mtp_drafted = 0, mtp_accepted = 0;
    const int dump_every =
        cne::env("DUMP_LOGITS_EVERY") ? atoi(cne::env("DUMP_LOGITS_EVERY")) : 1;
    auto dump_logits = [&](int tag) {
        const char* ldir = cne::env("DUMP_LOGITS");
        if (!ldir || !*ldir) return;
        float* lg = llama_get_logits_ith(ctx, 0);
        int nv = llama_vocab_n_tokens(vocab);
        char lp[512];
        snprintf(lp, sizeof(lp), "%s/s%03d.f32", ldir, tag);
        FILE* lf = fopen(lp, "wb");
        if (lf) { fwrite(lg, sizeof(float), nv, lf); fclose(lf); }
    };
    for (int i = 0; i < n_gen; i++) {
        if (mtp_k > 0) break;   // MTP path below owns generation
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;
        printf(" %d", id);
        fflush(stdout);
        produced++;
        fprintf(stderr, "[bench] step %d\n", i);
        cne::stream_set_step(i);
        if (llama_decode(ctx, llama_batch_get_one(&id, 1))) {
            fprintf(stderr, "\n[streaming-bench] DECODE FAILED\n");
            return 1;
        }
        // stride limits tmpfs use on long canary runs (default: every token)
        if (!dump_every || ((long)i % dump_every) == 0)
            dump_logits(i);   // logits of THIS step's forward (produces token i+1)
        cne::stream_prefetch_kick_full();
        cne::stream_step_boundary();
    }

    // ---- MTP speculative decoding path (draft-mtp, greedy) ----
    if (mtp_k > 0 && produced < n_gen) {
        struct Ctx {
            FILE* out;
        } ctx_cb{stdout};
        auto stats = cne::spec_mtp_generate(
                model, ctx, toks, mtp_k, n_gen - produced,
                cne::stream_cb_eval(),
                [](void* ud, llama_token id) {
                    fprintf((FILE*)ud, " %d", id);
                    fflush((FILE*)ud);
                },
                &ctx_cb);
        produced += stats.produced;
        fprintf(stderr, "[mtp] drafted=%ld accepted=%ld (%.1f%%)\n",
                stats.drafted, stats.accepted,
                stats.drafted ? 100.0 * stats.accepted / stats.drafted : 0.0);
    }

    cne::stream_prefetch_stop();
    double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    MemSnap snap1 = mem_snap();
    printf("\n[mem] minflt+%ld majflt+%ld rss=%.2f GiB hwm=%.2f GiB\n",
           snap1.minflt - snap0.minflt, snap1.majflt - snap0.majflt,
           snap1.rss_kib / 1048576.0, snap1.hwm_kib / 1048576.0);

    auto st = cache.stats();
    uint64_t gen_misses = st.misses - misses_before;

    cne::stream_check_windows();
    double hit = st.hits + st.misses ? 100.0 * st.hits / (st.hits + st.misses) : 0;
    auto tel = cne::stream_telemetry();
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
           cache.used_bytes() / 1048576.0, effective >> 10);
    printf("dedup requests     : %llu\n", (unsigned long long)st.dedup_requests);
    printf("audit checks run   : %ld | pending records: %zu\n",
           tel.audit_checks, tel.audit_pending);
    double fill_frac = secs > 0 ? 100.0 * tel.fill_s / secs : 0;
    printf("fill time          : %.2fs (%.1f%% of gen wall) over %llu batch fills\n",
           tel.fill_s, fill_frac, (unsigned long long)tel.fill_calls);
    printf("prefetched slices  : %llu\n", (unsigned long long)st.prefetched);
    if (tel.l2_mass > 0.0f)
        printf("[L2] dropped slices: %ld (mass=%.2f min_k=%d)\n",
               tel.l2_dropped, tel.l2_mass, tel.l2_min_k);

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
