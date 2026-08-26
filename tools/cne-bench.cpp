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

#include "cne_runtime.h"
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

    cne::RuntimeSettings rs;
    rs.model_path = argv[1];
    rs.cap_gib    = cap_gib;
    rs.n_ctx     = 256;      // bench default; CNE_CTX overrides
    rs.n_threads = 8;
    rs.stream_on = rebind;
    auto rt = cne::runtime_prepare(rs);
    if (!rt || !cne::runtime_load_llama(*rt, rs)) return 1;
    rt->cache->set_verify_next(verify_n);
    cne::Runtime& R = *rt;

    const char* prompt = cne::env("PROMPT") ? cne::env("PROMPT")
                                             : "The capital of France is";
    float mtp_p_min =
        cne::env("MTP_P_MIN") ? (float)atof(cne::env("MTP_P_MIN")) : 0.0f;

    // ---- Whole-model perplexity mode (policy-aware: expert-mass gating and
    // anon-dense residency apply) ----
    // Chunked cross-entropy over a plain-text corpus: windows of n_ctx tokens,
    // non-overlapping, memory cleared between windows. This is the canonical
    // whole-model quality gate because external llama-perplexity cannot apply
    // our callback policies (expert-mass gating, residency, streaming).
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
            n_seq = llama_tokenize(R.vocab, text.c_str(), (int)text.size(),
                                   seq.data(), (int)seq.size(), true, false);
            if (n_seq < 0 && (size_t)(-n_seq) == seq.size())
                seq.resize(seq.size() * 2);
        }
        seq.resize(n_seq);
        const int nv = llama_vocab_n_tokens(R.vocab);
        double nll = 0.0; long long ntok = 0;
        int wins = 0;
        for (size_t off = 0; off + 1 < (size_t)n_seq; off += (size_t)R.n_ctx) {
            const int n = (int)std::min((size_t)R.n_ctx, (size_t)n_seq - off);
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
            if (llama_decode(R.ctx, wb)) {
                llama_batch_free(wb);
                fprintf(stderr, "[ppl] decode failed at offset %zu\n", off);
                return 1;
            }
            llama_batch_free(wb);
            for (int i = 0; i + 1 < (int)n; i++) {
                const float* lg = llama_get_logits_ith(R.ctx, i);
                float mx = lg[0];
                for (int v2 = 1; v2 < nv; v2++) mx = std::max(mx, lg[v2]);
                double lse = mx;
                double s = 0.0;
                for (int v2 = 0; v2 < nv; v2++) s += std::exp((double)lg[v2] - mx);
                lse += std::log(s);
                nll += lse - (double)lg[seq[off + i + 1]];
                ntok++;
            }
            llama_memory_clear(llama_get_memory(R.ctx), true);
            wins++;
            fprintf(stderr, "[ppl] window %d done (%lld tokens evaluated)\n",
                    wins, ntok);
        }
        double ppl = ntok ? std::exp(nll / (double)ntok) : 0.0;
        printf("engine-ppl: %.4f (ntok %lld, windows %d, n_ctx %d)\n",
               ppl, ntok, wins, R.n_ctx);
        if (R.l2_mass > 0.0f)
            printf("[expert-mass] dropped slices total: %ld\n",
                   cne::stream_telemetry().l2_dropped);
        llama_free(R.ctx);
        llama_model_free(R.model);
        llama_backend_free();
        return 0;
    }

    std::vector<llama_token> toks(32);
    int n_tok = -1;
    // llama_tokenize contract: negative return = required token count.
    while (n_tok < 0) {
        if ((size_t)(-n_tok) > toks.size())
            toks.resize((size_t)(-n_tok));
        n_tok = llama_tokenize(R.vocab, prompt, (int)strlen(prompt),
                               toks.data(), (int)toks.size(), true, false);
        if (n_tok < 0 && (size_t)(-n_tok) == toks.size())
            toks.resize(toks.size() * 2);
    }
    toks.resize(n_tok);
    printf("prefill tokens: %d (n_ctx %d)\n", n_tok, R.n_ctx);
    if (cne::env("DEBUG_TOKIDS")) {
        printf("prompt ids:");
        for (int i = 0; i < n_tok; i++) printf(" %d", toks[i]);
        printf("\n");
    }
    if (R.mtp_k > 0) {
        // Speculative paths below do their own prefill.
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
        if (llama_decode(R.ctx, bp)) { fprintf(stderr, "[cne-bench] PREFILL FAILED\n"); return 1; }
        llama_batch_free(bp);
        llama_token il = toks.back();
        if (llama_decode(R.ctx, llama_batch_get_one(&il, 1))) {
            fprintf(stderr, "[cne-bench] PREFILL FAILED\n");
            return 1;
        }
    } else if (llama_decode(R.ctx, llama_batch_get_one(toks.data(), n_tok))) {
        fprintf(stderr, "[cne-bench] PREFILL FAILED\n");
        return 1;
    }

    printf("windows created: (see [geom] lines above) | cap: %zu MiB\n", R.cache_cap >> 20);

    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    uint64_t misses_before = R.cache->stats().misses;
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
        float* lg = llama_get_logits_ith(R.ctx, 0);
        int nv = llama_vocab_n_tokens(R.vocab);
        char lp[512];
        snprintf(lp, sizeof(lp), "%s/s%03d.f32", ldir, tag);
        FILE* lf = fopen(lp, "wb");
        if (lf) { fwrite(lg, sizeof(float), nv, lf); fclose(lf); }
    };
    for (int i = 0; i < n_gen; i++) {
        if (R.mtp_k > 0) break;   // spec paths own generation
        llama_token id = llama_sampler_sample(smpl, R.ctx, -1);
        if (llama_vocab_is_eog(R.vocab, id)) break;
        printf(" %d", id);
        fflush(stdout);
        produced++;
        fprintf(stderr, "[bench] step %d\n", i);
        cne::stream_set_step(i);
        if (llama_decode(R.ctx, llama_batch_get_one(&id, 1))) {
            fprintf(stderr, "\n[cne-bench] DECODE FAILED\n");
            return 1;
        }
        // stride limits tmpfs use on long canary runs (default: every token)
        if (!dump_every || ((long)i % dump_every) == 0)
            dump_logits(i);   // logits of THIS step's forward (produces token i+1)
        cne::stream_prefetch_kick_full();
        cne::stream_step_boundary();
    }

    // ---- MTP speculative decoding path (draft-mtp, greedy) ----
    if (R.mtp_k > 0 && produced < n_gen) {
        auto stats = cne::spec_mtp_generate(
                R.model, R.ctx, toks, R.mtp_k, mtp_p_min, n_gen - produced,
                cne::stream_cb_eval(),
                [](void* ud, llama_token id) {
                    fprintf((FILE*)ud, " %d", id);
                    fflush((FILE*)ud);
                },
                stdout);
        produced += stats.produced;
        fprintf(stderr,
                "[mtp] iterations=%ld drafted=%ld accepted=%ld (%.1f%%) "
                "partials=%ld produced=%d\n",
                stats.iterations, stats.drafted, stats.accepted,
                stats.partials,
                stats.drafted ? 100.0 * stats.accepted / stats.drafted : 0.0,
                stats.produced);
        double spec_wall = stats.draft_s + stats.process_s + stats.verify_s;
        long verify_rows = stats.iterations + stats.drafted;   // 1+k rows per iter
        fprintf(stderr,
                "[mtp] draft=%.2fs process=%.2fs verify=%.2fs (spec wall %.2fs)\n"
                "[mtp] per-iteration: draft %.0f ms | verify %.0f ms for %ld "
                "rows (%.0f ms/row)\n"
                "[mtp] R.cache_cap: %.0f ms/token over %d tokens "
                "(compare against your sequential arm)\n",
                stats.draft_s, stats.process_s, stats.verify_s, spec_wall,
                stats.iterations ? 1000.0 * stats.draft_s / stats.iterations : 0.0,
                stats.iterations ? 1000.0 * stats.verify_s / stats.iterations : 0.0,
                verify_rows,
                verify_rows ? 1000.0 * stats.verify_s / verify_rows : 0.0,
                stats.produced ? 1000.0 * spec_wall / stats.produced : 0.0,
                stats.produced);
    }

    cne::stream_prefetch_stop();
    double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    MemSnap snap1 = mem_snap();
    printf("\n[mem] minflt+%ld majflt+%ld rss=%.2f GiB hwm=%.2f GiB\n",
           snap1.minflt - snap0.minflt, snap1.majflt - snap0.majflt,
           snap1.rss_kib / 1048576.0, snap1.hwm_kib / 1048576.0);

    auto st = R.cache->stats();
    uint64_t gen_misses = st.misses - misses_before;

    cne::stream_check_windows();
    double hit = st.hits + st.misses ? 100.0 * st.hits / (st.hits + st.misses) : 0;
    auto tel = cne::stream_telemetry();
    printf("=== cne-bench summary ===\n");
    printf("tok/s              : %.2f\n", produced / secs);
    printf("produced           : %d in %.1fs\n", produced, secs);
    printf("cache hit rate     : %.2f%% (hits=%llu misses=%llu)\n", hit,
           (unsigned long long)st.hits, (unsigned long long)st.misses);
    printf("gen-phase misses   : %llu (%.1f MiB/step)\n",
           (unsigned long long)gen_misses,
           produced ? gen_misses * (double)R.manifest.tensors.front().bytes_per_expert /
                          produced / 1048576.0
                    : 0.0);
    printf("bytes loaded       : %.2f MiB | evictions: %llu | used: %.2f / %zu MiB\n",
           st.bytes_loaded / 1048576.0, (unsigned long long)st.evictions,
           R.cache->used_bytes() / 1048576.0, R.cache_cap >> 10);
    printf("dedup requests     : %llu\n", (unsigned long long)st.dedup_requests);
    printf("audit checks run   : %ld | pending records: %zu\n",
           tel.audit_checks, tel.audit_pending);
    double fill_frac = secs > 0 ? 100.0 * tel.fill_s / secs : 0;
    printf("fill time          : %.2fs (%.1f%% of gen wall) over %llu batch fills\n",
           tel.fill_s, fill_frac, (unsigned long long)tel.fill_calls);
    printf("prefetched slices  : %llu\n", (unsigned long long)st.prefetched);
    if (tel.l2_mass > 0.0f)
        printf("[expert-mass] dropped slices: %ld (mass=%.2f min_k=%d)\n",
               tel.l2_dropped, tel.l2_mass, tel.l2_min_k);

    llama_sampler_free(smpl);
    llama_free(R.ctx);
    llama_model_free(R.model);
    return 0;
}
