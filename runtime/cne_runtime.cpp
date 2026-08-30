#include "cne_runtime.h"

#include "cne_stream_spec.h"

#include "cne/memory_budget.h"
#include "cne/kv_budget.h"
#include "cne/model_registry.h"
#include "cne/config.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace cne {

std::unique_ptr<Runtime> runtime_prepare(const RuntimeSettings& s) {
    auto rt = std::make_unique<Runtime>();

    ModelRegistry reg;
    if (!reg.build(std::string(s.model_path), rt->manifest)) {
        fprintf(stderr, "[cne] manifest FAILED: %s\n", reg.error().c_str());
        return nullptr;
    }

    // Dense policy: explicit env wins; otherwise AUTO from the memory regime.
    enum class DensePolicy { MMAP, WARM, ANON };
    DensePolicy dense = DensePolicy::MMAP;
    {
        const char* dp = env("DENSE");
        std::string d  = dp ? dp : "";
        MemoryBudget pre = MemoryBudget::detect();
        uint64_t dense_bytes = 0;
        for (const auto& ti : rt->manifest.tensors)
            if (ti.kind != TensorKind::ROUTED_EXPERT)
                dense_bytes += ti.bytes_total;
        Regime rg = classify((size_t)rt->manifest.file_size, pre.mem_available);
        rt->regime_str = regime_name(rg);
        printf("regime=%s (available %.1f GiB, dense %.2f GiB)\n",
               rt->regime_str.c_str(), pre.mem_available / 1073741824.0,
               dense_bytes / 1073741824.0);
        if (d.empty())
            dense = (rg != Regime::R0_RESIDENT &&
                     pre.mem_available > dense_bytes * 2)
                        ? DensePolicy::ANON
                        : DensePolicy::MMAP;
        else if (d == "warm") dense = DensePolicy::WARM;
        else if (d == "anon") dense = DensePolicy::ANON;
        rt->dense_policy_str =
            dense == DensePolicy::ANON ? "anon" :
            dense == DensePolicy::WARM ? "warm" : "mmap";
        printf("dense policy: %s\n", rt->dense_policy_str.c_str());
    }

    // Budget manager: clamp the cache cap to what this machine can hold.
    MemoryBudget budget = MemoryBudget::detect();
    budget.kv           = 64u << 20;
    budget.staging      = 64u << 20;
    budget.runtime_base = 512u << 20;
    if (dense == DensePolicy::ANON)
        for (const auto& ti : rt->manifest.tensors)
            if (ti.kind != TensorKind::ROUTED_EXPERT)
                budget.runtime_base += ti.bytes_total;
    size_t requested = s.cap_gib << 30;
    size_t effective = budget.clamp_cache_cap(requested);
    if (effective != requested)
        fprintf(stderr,
                "[budget] cache cap clamped: %zu GiB -> %zu GiB "
                "(available %.1f GiB)\n",
                requested >> 30, effective >> 30,
                budget.mem_available / 1073741824.0);
    printf("budget: available=%.1f GiB model=%.1f GiB regime=%s cap=%zu MiB\n",
           budget.mem_available / 1073741824.0,
           rt->manifest.file_size / 1073741824.0, rt->regime_str.c_str(),
           effective >> 20);
    rt->cache_cap = effective;

    // Expert-mass gating: DEFAULT UNSET = lossless full top-k. LOSSY when set.
    if (env("EXPERT_MASS")) {
        rt->l2_mass = (float)atof(env("EXPERT_MASS"));
        int l2_min_k = env("EXPERT_MIN_K") ? atoi(env("EXPERT_MIN_K")) : 2;
        if (rt->l2_mass <= 0.0f || rt->l2_mass > 1.0f) {
            fprintf(stderr,
                    "[expert-mass] invalid CNE_EXPERT_MASS - gating disabled\n");
            rt->l2_mass = 0.0f;
        } else {
            printf("*** [expert-mass] ACTIVE: mass=%.2f min_k=%d "
                   "(LOSSY profile - quality degradation expected) ***\n",
                   rt->l2_mass, l2_min_k);
        }
    }

    // Prefetch overlap: measured dead on current hardware/model, default OFF.
    PfMode pf_mode = PfMode::OFF;
    {
        const char* pf  = env("PREFETCH");
        std::string pfs = pf ? pf : "";
        if (pfs == "1" || pfs == "lookahead") pf_mode = PfMode::LOOKAHEAD;
        else if (pfs == "full")               pf_mode = PfMode::FULL;
    }

    rt->cache = std::make_unique<SliceCache>(CacheLimits{effective});

    StreamConfig cfg;
    cfg.rebind     = s.stream_on;
    cfg.full_fill  = env("FULL_FILL") != nullptr;
    cfg.step_fills = env("STEP_FILLS") != nullptr;
    cfg.l2_mass    = rt->l2_mass;
    cfg.l2_min_k   = env("EXPERT_MIN_K") ? atoi(env("EXPERT_MIN_K")) : 2;
    cfg.pf_mode    = pf_mode;
    stream_init(rt->manifest, *rt->cache, cfg);

    rt->odirect = stream_open_fill_backend(
        s.model_path, env("LANES") ? atoi(env("LANES")) : 4);
    rt->streaming = s.stream_on;
    stream_prefetch_start();

    if (dense == DensePolicy::WARM) {
        auto  t0 = std::chrono::steady_clock::now();
        FILE* f  = fopen(s.model_path, "rb");
        if (!f) { fprintf(stderr, "[dense] warm open failed\n"); return nullptr; }
        char   wbuf[1 << 20];
        size_t warmed = 0;
        for (const auto& ti : rt->manifest.tensors) {
            if (ti.kind == TensorKind::ROUTED_EXPERT) continue;
            if (fseeko(f, (off_t)ti.abs_offset, SEEK_SET) != 0) continue;
            uint64_t left = ti.bytes_total;
            while (left) {
                size_t c = left < sizeof(wbuf) ? (size_t)left : sizeof(wbuf);
                if (fread(wbuf, 1, c, f) != c) break;
                left -= c;
                warmed += c;
            }
        }
        fclose(f);
        double secs = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0).count();
        printf("dense=warm: %.1f MiB paged in (%.2fs)\n", warmed / 1048576.0,
               secs);
    }

    return rt;
}

bool runtime_load_llama(Runtime& rt, const RuntimeSettings& s) {
    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers    = 0;
    mparams.use_extra_bufts = false;
    rt.mtp_k = env("MTP") ? atoi(env("MTP")) : 0;
    if (rt.mtp_k < 0) rt.mtp_k = 0;
    if (rt.mtp_k > 0) mparams.load_mtp = true;

    rt.model = llama_model_load_from_file(s.model_path, mparams);
    if (!rt.model) {
        fprintf(stderr, "[cne] LOAD FAILED\n");
        return false;
    }
    rt.vocab = llama_model_get_vocab(rt.model);

    auto cparams = llama_context_default_params();
    int ctx_size = env("CTX") ? atoi(env("CTX")) : s.n_ctx;
    if (ctx_size < 64) ctx_size = 64;
    rt.n_ctx          = ctx_size;
    cparams.n_ctx         = ctx_size;
    cparams.n_batch       = ctx_size < 512 ? ctx_size : 512;
    cparams.n_ubatch      = 64;
    int n_threads = env("THREADS") ? atoi(env("THREADS")) : s.n_threads;
    if (n_threads < 1) n_threads = 1;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    if (env("KV_Q8")) {
        cparams.type_k = GGML_TYPE_Q8_0;
        cparams.type_v = GGML_TYPE_Q8_0;
    }
    cparams.flash_attn_type = env("FA") ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                        : LLAMA_FLASH_ATTN_TYPE_AUTO;
    cparams.cb_eval          = stream_cb_eval();
    cparams.cb_eval_user_data = nullptr;
    if (rt.mtp_k > 0) {
        spec_mtp_size_outputs(cparams, rt.mtp_k, (int)cparams.n_batch);
        fprintf(stderr, "[mtp] draft-mtp enabled: n_max=%d n_outputs_max=%u\n",
                rt.mtp_k, cparams.n_outputs_max);
    }

    uint32_t n_seq_max = 1;
    if (const char* sm = env("SESSION_MAX")) {
        const int n = atoi(sm);
        if (n > 1) n_seq_max = (uint32_t) n;
    }
    cparams.n_seq_max = n_seq_max;
    if (n_seq_max > 1) cparams.kv_unified = false;

    {
        ServingKvEstimate kv;
        kv.n_ctx         = (int) cparams.n_ctx;
        kv.n_seq_max     = (int) n_seq_max;
        kv.n_ctx_per_seq = n_seq_max > 0 ? (int) (cparams.n_ctx / n_seq_max)
                                         : (int) cparams.n_ctx;
        if (const char* kbpt = env("KV_BPT")) {
            const int v = atoi(kbpt);
            if (v > 0) kv.bytes_per_token = (size_t) v;
        }
        const Regime rg =
            classify((size_t) rt.manifest.file_size,
                     MemoryBudget::detect().mem_available);
        const size_t model_resident =
            rg == Regime::R0_RESIDENT ? (size_t) rt.manifest.file_size : 0;
        fprintf(stderr,
                "[cne] KV plan: ctx=%d (%d tok/seq x %u lanes) ~%.0f MiB est "
                "(CNE_KV_BPT=%zu)\n",
                kv.n_ctx, kv.n_ctx_per_seq, n_seq_max,
                kv.kv_bytes() / (1024.0 * 1024.0), kv.bytes_per_token);
        if (serving_kv_exceeds_headroom(kv, MemoryBudget::detect(),
                                        model_resident))
            fprintf(stderr,
                    "[cne] WARNING: projected KV may exceed anonymous "
                    "headroom — lower CNE_CTX or CNE_SESSION_MAX\n");
    }

    rt.ctx = llama_init_from_model(rt.model, cparams);
    if (!rt.ctx) {
        fprintf(stderr, "[cne] CONTEXT FAILED\n");
        return false;
    }
    // Warmup-off stays mandatory near/above RAM size; exception: MTP probing
    // keeps upstream's warmup-on behavior (same rule as the bench).
    if (rt.mtp_k == 0) llama_set_warmup(rt.ctx, false);

    if (rt.dense_policy_str == "anon") {
        stream_anon_scan_begin();
        llama_token b = llama_vocab_bos(rt.vocab);
        if (llama_decode(rt.ctx, llama_batch_get_one(&b, 1)))
            fprintf(stderr, "[dense] anon scan decode failed (continuing)\n");
        stream_anon_scan_end();
        llama_memory_clear(llama_get_memory(rt.ctx), true);
        printf("dense=anon: %zu tensors bound, %zu MiB anonymous\n",
               stream_dense_bound_count(), stream_dense_anon_bytes() >> 20);
    }
    return true;
}

void runtime_shutdown(Runtime& rt) {
    stream_prefetch_stop();
    if (rt.ctx) llama_free(rt.ctx);
    if (rt.model) llama_model_free(rt.model);
}

} // namespace cne
