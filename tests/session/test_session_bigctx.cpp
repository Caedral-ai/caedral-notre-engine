// Big-context session E2E. Config: tests/e2e/session_bigctx.json
#include "cne_runtime.h"
#include "cne_session.h"
#include "e2e_config.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

int env_int(const char* name, int def) {
    const char* v = getenv(name);
    if (!v || !*v) return def;
    return atoi(v);
}

std::vector<llama_token> tokenize(const llama_vocab* vocab,
                                  const std::string& text) {
    std::vector<llama_token> out(32);
    while (true) {
        const int n = llama_tokenize(vocab, text.c_str(), (int) text.size(),
                                     out.data(), (int) out.size(), false, true);
        if (n >= 0) {
            out.resize((size_t) n);
            return out;
        }
        out.resize((size_t) (-n));
    }
}

std::vector<llama_token> build_long_prompt(const llama_vocab* vocab,
                                           size_t target_tokens) {
    const std::vector<llama_token> unit =
        tokenize(vocab, " Session context line for long prefill coverage.");
    if (unit.empty() || target_tokens == 0) return {};
    std::vector<llama_token> out;
    out.reserve(target_tokens);
    while (out.size() < target_tokens)
        out.insert(out.end(), unit.begin(), unit.end());
    out.resize(target_tokens);
    return out;
}

bool full_prefill_seq0(llama_context* ctx,
                       const std::vector<llama_token>& prompt) {
    cne::session_clear_seq(ctx, 0);
    const int n_batch = llama_n_batch(ctx);
    for (int off = 0; off < (int) prompt.size(); off += n_batch) {
        const int len = std::min(n_batch, (int) prompt.size() - off);
        if (llama_decode(ctx,
                         llama_batch_get_one(
                             const_cast<llama_token*>(prompt.data() + off),
                             len)))
            return false;
    }
    return true;
}

std::vector<llama_token> greedy_decode(cne::SessionSlot& slot,
                                       llama_context* ctx, llama_sampler* smpl,
                                       const llama_vocab* vocab, int n_gen) {
    std::vector<llama_token> out;
    out.reserve((size_t) n_gen);
    for (int i = 0; i < n_gen; i++) {
        const llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;
        out.push_back(id);
        cne::session_append_token(slot, id);
        if (!cne::session_decode_token(ctx, slot, id)) break;
    }
    return out;
}

llama_sampler* greedy_sampler() {
    llama_sampler* chain =
        llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    return chain;
}

bool tokens_equal(const std::vector<llama_token>& a,
                  const std::vector<llama_token>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++)
        if (a[i] != b[i]) return false;
    return true;
}

double elapsed_s(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - t0)
        .count();
}

int run_live(const char* model_path, const cne::e2e::Config& cfg) {
    const int n_ctx =
        cfg.runtime.n_ctx > 0 ? cfg.runtime.n_ctx : env_int("CNE_TEST_CTX", 8192);
    const int prompt_tok = cfg.prompt_tokens > 0
                               ? cfg.prompt_tokens
                               : env_int("CNE_TEST_PROMPT_TOK", 7000);
    const int n_gen =
        cfg.gen_tokens > 0 ? cfg.gen_tokens : env_int("CNE_TEST_GEN", 16);

    if (prompt_tok < 512 || prompt_tok + n_gen + 64 > n_ctx) {
        fprintf(stderr,
                "FAIL: bad sizes ctx=%d prompt_tok=%d gen=%d\n", n_ctx,
                prompt_tok, n_gen);
        return 1;
    }

    cne::RuntimeSettings rs;
    rs.model_path = model_path;
    rs.cap_gib    = cfg.runtime.cap_gib;
    rs.n_ctx      = n_ctx;
    rs.n_threads  = cfg.runtime.n_threads > 0 ? cfg.runtime.n_threads : 4;
    rs.stream_on  = cfg.runtime.stream_on;

    auto t0 = std::chrono::steady_clock::now();
    auto rt = cne::runtime_prepare(rs);
    if (!rt || !cne::runtime_load_llama(*rt, rs)) {
        fprintf(stderr, "FAIL: could not load model %s at ctx=%d\n", model_path,
                n_ctx);
        return 1;
    }
    fprintf(stderr, "[bigctx] loaded ctx=%u n_batch=%d (%.1fs)\n",
            llama_n_ctx(rt->ctx), llama_n_batch(rt->ctx), elapsed_s(t0));

    llama_context* ctx        = rt->ctx;
    const llama_vocab* vocab = rt->vocab;
    cne::SessionStore store(2);
    store.set_seq_capacity(llama_n_seq_max(ctx));

    const std::vector<llama_token> prompt1 =
        build_long_prompt(vocab, (size_t) prompt_tok);
    if (prompt1.size() != (size_t) prompt_tok) {
        fprintf(stderr, "FAIL: long prompt build (%zu != %d)\n",
                prompt1.size(), prompt_tok);
        return 1;
    }

    llama_sampler* smpl = greedy_sampler();
    cne::SessionSlot& slot = store.get_or_create("bigctx", ctx);

    t0 = std::chrono::steady_clock::now();
    cne::SessionPrefillStats st1;
    if (!cne::session_prefill(ctx, slot, prompt1, &st1)) {
        fprintf(stderr, "FAIL: turn-1 long session_prefill\n");
        return 1;
    }
    fprintf(stderr, "[bigctx] turn-1 prefill %zu tok (%.1fs)\n",
            st1.prefilled_tokens, elapsed_s(t0));

    if (st1.reused_tokens != 0 || st1.prefilled_tokens != prompt1.size()) {
        fprintf(stderr,
                "FAIL: turn-1 stats reused=%zu prefilled=%zu (want 0, %zu)\n",
                st1.reused_tokens, st1.prefilled_tokens, prompt1.size());
        return 1;
    }
    if (cne::kv_seq_len(ctx, slot.seq_id) != (int) prompt1.size()) {
        fprintf(stderr, "FAIL: KV len %d != prompt %zu after turn-1\n",
                cne::kv_seq_len(ctx, slot.seq_id), prompt1.size());
        return 1;
    }

    const std::vector<llama_token> gen1 =
        greedy_decode(slot, ctx, smpl, vocab, n_gen);
    if (gen1.empty()) {
        fprintf(stderr, "FAIL: turn-1 produced no tokens\n");
        return 1;
    }

    std::vector<llama_token> prompt2 = slot.kv_tokens;
    const std::vector<llama_token> tail =
        tokenize(vocab, " Follow-up after long context.");
    prompt2.insert(prompt2.end(), tail.begin(), tail.end());
    if (prompt2.size() > (size_t) n_ctx - (size_t) n_gen) {
        fprintf(stderr, "FAIL: prompt2 %zu exceeds ctx headroom\n",
                prompt2.size());
        return 1;
    }

    const size_t expect_reused = prompt1.size() + gen1.size();
    t0 = std::chrono::steady_clock::now();
    cne::SessionPrefillStats st2;
    if (!cne::session_prefill(ctx, slot, prompt2, &st2)) {
        fprintf(stderr, "FAIL: turn-2 session_prefill\n");
        return 1;
    }
    fprintf(stderr,
            "[bigctx] turn-2 reused=%zu prefilled=%zu (%.1fs)\n",
            st2.reused_tokens, st2.prefilled_tokens, elapsed_s(t0));

    if (st2.reused_tokens != expect_reused) {
        fprintf(stderr, "FAIL: turn-2 reused=%zu (want %zu)\n",
                st2.reused_tokens, expect_reused);
        return 1;
    }
    if (st2.prefilled_tokens != tail.size()) {
        fprintf(stderr, "FAIL: turn-2 prefilled=%zu (want %zu)\n",
                st2.prefilled_tokens, tail.size());
        return 1;
    }

    llama_sampler_reset(smpl);
    const std::vector<llama_token> gen2_session =
        greedy_decode(slot, ctx, smpl, vocab, n_gen);

    t0 = std::chrono::steady_clock::now();
    if (!full_prefill_seq0(ctx, prompt2)) {
        fprintf(stderr, "FAIL: stateless long prefill\n");
        return 1;
    }
    fprintf(stderr, "[bigctx] stateless prefill %zu tok (%.1fs)\n",
            prompt2.size(), elapsed_s(t0));

    llama_sampler_reset(smpl);
    cne::SessionSlot stateless;
    stateless.seq_id = 0;
    const std::vector<llama_token> gen2_stateless =
        greedy_decode(stateless, ctx, smpl, vocab, n_gen);

    if (!tokens_equal(gen2_session, gen2_stateless)) {
        fprintf(stderr, "FAIL: session vs stateless mismatch at big ctx\n");
        return 1;
    }

    llama_sampler_free(smpl);
    cne::runtime_shutdown(*rt);

    printf("session_bigctx live: OK (ctx=%d prompt=%d reused=%zu)\n", n_ctx,
           prompt_tok, st2.reused_tokens);
    return 0;
}

} // namespace

int main() {
    const std::string src = CNE_PROJECT_SOURCE_DIR;
    const std::string cfg_path =
        cne::e2e::discover_path("tests/e2e/session_bigctx_live.json", src);

    cne::e2e::Config cfg;
    std::string err;
    if (!cne::e2e::load(cfg_path, src, cfg, err)) {
        fprintf(stderr, "FAIL: e2e config %s (%s)\n", cfg_path.c_str(),
                err.c_str());
        return 1;
    }
    cne::e2e::apply_env(cfg);

    const std::string model = cne::e2e::resolve_model(cfg, src);
    if (model.empty()) {
        printf("skip: session_bigctx live (set model in %s or CNE_TEST_MODEL)\n",
               cfg_path.c_str());
        return 0;
    }
    if (access(model.c_str(), R_OK) != 0) {
        printf("skip: session_bigctx live (model not found: %s)\n",
                model.c_str());
        return 0;
    }
    return run_live(model.c_str(), cfg);
}
