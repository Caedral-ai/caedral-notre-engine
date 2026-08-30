// Session KV reuse integration test. Requires a real prepared GGUF via
// CNE_TEST_MODEL (skipped when unset). Verifies incremental prefill stats,
// greedy decode parity vs stateless, and warm KV across alternating users.
#include "cne_runtime.h"
#include "cne_session.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

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

void dump_tokens(const char* label, const std::vector<llama_token>& toks) {
    fprintf(stderr, "%s (%zu):", label, toks.size());
    for (llama_token t : toks) fprintf(stderr, " %d", t);
    fputc('\n', stderr);
}

int run_live(const char* model_path) {
    setenv("CNE_SESSION_MAX", "2", 1);

    cne::RuntimeSettings rs;
    rs.model_path = model_path;
    rs.cap_gib    = 4;
    rs.n_ctx      = 2048;
    rs.n_threads  = 4;
    rs.stream_on  = false;

    auto rt = cne::runtime_prepare(rs);
    if (!rt || !cne::runtime_load_llama(*rt, rs)) {
        fprintf(stderr, "FAIL: could not load model %s\n", model_path);
        return 1;
    }
    if (llama_n_seq_max(rt->ctx) < 2) {
        fprintf(stderr, "FAIL: need n_seq_max>=2 (got %u)\n",
                llama_n_seq_max(rt->ctx));
        return 1;
    }

    llama_context* ctx   = rt->ctx;
    const llama_vocab* vocab = rt->vocab;
    cne::SessionStore store(4);
    store.set_seq_capacity(llama_n_seq_max(ctx));

    const std::vector<llama_token> prompt1 =
        tokenize(vocab, "The capital of France is");
    if (prompt1.empty()) {
        fprintf(stderr, "FAIL: empty prompt1\n");
        return 1;
    }

    llama_sampler* smpl = greedy_sampler();
    cne::SessionSlot& slot = store.get_or_create("test", ctx);

    // ---- turn 1 (session path) ----
    cne::SessionPrefillStats st1;
    if (!cne::session_prefill(ctx, slot, prompt1, &st1)) {
        fprintf(stderr, "FAIL: turn-1 session_prefill\n");
        return 1;
    }
    if (st1.reused_tokens != 0 || st1.prefilled_tokens != prompt1.size()) {
        fprintf(stderr,
                "FAIL: turn-1 stats reused=%zu prefilled=%zu (want 0, %zu)\n",
                st1.reused_tokens, st1.prefilled_tokens, prompt1.size());
        return 1;
    }

    const int n_gen1 = 16;
    const std::vector<llama_token> gen1 =
        greedy_decode(slot, ctx, smpl, vocab, n_gen1);
    if (gen1.empty()) {
        fprintf(stderr, "FAIL: turn-1 produced no tokens\n");
        return 1;
    }
    if (slot.kv_tokens.size() != prompt1.size() + gen1.size()) {
        fprintf(stderr, "FAIL: slot size after turn-1 (%zu != %zu)\n",
                slot.kv_tokens.size(), prompt1.size() + gen1.size());
        return 1;
    }
    if (cne::kv_seq_len(ctx, slot.seq_id) != (int) slot.kv_tokens.size()) {
        fprintf(stderr, "FAIL: KV len %d != slot %zu after turn-1\n",
                cne::kv_seq_len(ctx, slot.seq_id), slot.kv_tokens.size());
        return 1;
    }

    // ---- turn 2 prompt: prior KV + new tail ----
    std::vector<llama_token> prompt2 = slot.kv_tokens;
    const std::vector<llama_token> tail =
        tokenize(vocab, " The river Seine flows through");
    prompt2.insert(prompt2.end(), tail.begin(), tail.end());

    const size_t expect_reused = prompt1.size() + gen1.size();

    cne::SessionPrefillStats st2;
    if (!cne::session_prefill(ctx, slot, prompt2, &st2)) {
        fprintf(stderr, "FAIL: turn-2 session_prefill\n");
        return 1;
    }
    if (st2.reused_tokens != expect_reused) {
        fprintf(stderr,
                "FAIL: turn-2 reused=%zu (want %zu) prefilled=%zu\n",
                st2.reused_tokens, expect_reused, st2.prefilled_tokens);
        return 1;
    }
    if (st2.prefilled_tokens != tail.size()) {
        fprintf(stderr,
                "FAIL: turn-2 prefilled=%zu (want %zu)\n",
                st2.prefilled_tokens, tail.size());
        return 1;
    }

    llama_sampler_reset(smpl);
    const int n_gen2 = 16;
    const std::vector<llama_token> gen2_session =
        greedy_decode(slot, ctx, smpl, vocab, n_gen2);

    if (!full_prefill_seq0(ctx, prompt2)) {
        fprintf(stderr, "FAIL: stateless prefill\n");
        return 1;
    }
    llama_sampler_reset(smpl);
    cne::SessionSlot stateless;
    stateless.seq_id = 0;
    const std::vector<llama_token> gen2_stateless =
        greedy_decode(stateless, ctx, smpl, vocab, n_gen2);

    if (!tokens_equal(gen2_session, gen2_stateless)) {
        fprintf(stderr, "FAIL: session vs stateless token mismatch\n");
        dump_tokens("session", gen2_session);
        dump_tokens("stateless", gen2_stateless);
        return 1;
    }

    // ---- alternating users: A -> B -> A (warm KV on A's seq) ----
    store.remove("test", ctx);
    cne::SessionSlot& slotA = store.get_or_create("user-a", ctx);
    const std::vector<llama_token> promptA1 =
        tokenize(vocab, "User alpha says hello");
    cne::SessionPrefillStats stA1;
    if (!cne::session_prefill(ctx, slotA, promptA1, &stA1)) return 1;
    const std::vector<llama_token> genA1 =
        greedy_decode(slotA, ctx, smpl, vocab, 8);
    const size_t a_warm = slotA.kv_tokens.size();

    cne::SessionSlot& slotB = store.get_or_create("user-b", ctx);
    assert(slotB.seq_id != slotA.seq_id);
    const std::vector<llama_token> promptB1 =
        tokenize(vocab, "User beta says goodbye");
    if (!cne::session_prefill(ctx, slotB, promptB1, nullptr)) return 1;
    (void) greedy_decode(slotB, ctx, smpl, vocab, 8);

    std::vector<llama_token> promptA2 = slotA.kv_tokens;
    const std::vector<llama_token> tailA =
        tokenize(vocab, " and asks again");
    promptA2.insert(promptA2.end(), tailA.begin(), tailA.end());

    cne::SessionPrefillStats stA2;
    if (!cne::session_prefill(ctx, slotA, promptA2, &stA2)) return 1;
    if (stA2.reused_tokens != a_warm) {
        fprintf(stderr,
                "FAIL: alternating users reused=%zu (want %zu)\n",
                stA2.reused_tokens, a_warm);
        return 1;
    }

    llama_sampler_free(smpl);
    cne::runtime_shutdown(*rt);

    printf("session_kv live: OK (turn2 reused=%zu alternating reused=%zu)\n",
           st2.reused_tokens, stA2.reused_tokens);
    return 0;
}

} // namespace

int main() {
    const char* model = getenv("CNE_TEST_MODEL");
    if (!model || !model[0]) {
        printf("skip: session_kv live (set CNE_TEST_MODEL)\n");
        return 0;
    }
    return run_live(model);
}
