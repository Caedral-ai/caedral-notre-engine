#include "cne_session.h"

#include <algorithm>
#include <cstdio>

namespace cne {

size_t token_common_prefix(const std::vector<llama_token>& a,
                           const std::vector<llama_token>& b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    for (; i < n && a[i] == b[i]; ++i) {}
    return i;
}

int kv_seq_len(llama_context* ctx) {
    llama_memory_t mem = llama_get_memory(ctx);
    if (!mem) return 0;
    const llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
    return pos_max < 0 ? 0 : (int) pos_max + 1;
}

static bool prefill_chunked(llama_context* ctx,
                            const llama_token* tokens, int n_tokens) {
    const int n_batch = llama_n_batch(ctx);
    for (int off = 0; off < n_tokens; off += n_batch) {
        const int len = std::min(n_batch, n_tokens - off);
        if (llama_decode(ctx, llama_batch_get_one(
                const_cast<llama_token*>(tokens + off), len))) {
            return false;
        }
    }
    return true;
}

bool session_prefill(llama_context* ctx, SessionSlot& slot,
                     const std::vector<llama_token>& prompt_tokens,
                     SessionPrefillStats* stats_out) {
    SessionPrefillStats stats;
    llama_memory_t mem = llama_get_memory(ctx);
    if (!mem || prompt_tokens.empty()) {
        if (stats_out) *stats_out = stats;
        return !prompt_tokens.empty() ? false : true;
    }

    const int n_kv = kv_seq_len(ctx);

    if (n_kv == 0) {
        if (!prefill_chunked(ctx, prompt_tokens.data(),
                             (int) prompt_tokens.size())) {
            return false;
        }
        slot.kv_tokens = prompt_tokens;
        stats.prefilled_tokens = prompt_tokens.size();
        if (stats_out) *stats_out = stats;
        return true;
    }

    if (slot.kv_tokens.size() != (size_t) n_kv) {
        fprintf(stderr,
                "[session] KV/token desync (kv=%d stored=%zu) — resetting slot\n",
                n_kv, slot.kv_tokens.size());
        session_reset_kv(ctx, slot);
        return session_prefill(ctx, slot, prompt_tokens, stats_out);
    }

    const size_t lcp =
        token_common_prefix(slot.kv_tokens, prompt_tokens);

    if (lcp < slot.kv_tokens.size()) {
        if (!llama_memory_seq_rm(mem, 0, (llama_pos) lcp, -1)) {
            fprintf(stderr, "[session] seq_rm failed at %zu — full reset\n",
                    lcp);
            session_reset_kv(ctx, slot);
            return session_prefill(ctx, slot, prompt_tokens, stats_out);
        }
        slot.kv_tokens.resize(lcp);
    }

    stats.reused_tokens = lcp;

    if (lcp < prompt_tokens.size()) {
        if (!prefill_chunked(ctx, prompt_tokens.data() + lcp,
                             (int) (prompt_tokens.size() - lcp))) {
            return false;
        }
        stats.prefilled_tokens = prompt_tokens.size() - lcp;
    }

    slot.kv_tokens = prompt_tokens;
    if (stats_out) *stats_out = stats;
    return true;
}

void session_append_token(SessionSlot& slot, llama_token id) {
    slot.kv_tokens.push_back(id);
}

void session_reset_kv(llama_context* ctx, SessionSlot& slot) {
    if (ctx) {
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) llama_memory_clear(mem, true);
    }
    slot.kv_tokens.clear();
}

SessionStore::SessionStore(size_t max_slots) : max_slots_(max_slots > 0 ? max_slots : 1) {}

SessionSlot& SessionStore::get_or_create(const std::string& id) {
    auto it = slots_.find(id);
    if (it == slots_.end()) {
        evict_lru_if_needed();
        touch(slots_[id]);
        slots_[id].id = id;
    } else {
        touch(it->second);
    }
    return slots_[id];
}

void SessionStore::remove(const std::string& id) { slots_.erase(id); }

void SessionStore::clear() { slots_.clear(); }

void SessionStore::touch(SessionSlot& slot) { slot.last_tick = ++tick_; }

void SessionStore::evict_lru_if_needed() {
    while (slots_.size() >= max_slots_) {
        auto it = slots_.begin();
        for (auto jt = slots_.begin(); jt != slots_.end(); ++jt) {
            if (jt->second.last_tick < it->second.last_tick) it = jt;
        }
        slots_.erase(it);
    }
}

} // namespace cne
