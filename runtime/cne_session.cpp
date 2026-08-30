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

int kv_seq_len(llama_context* ctx, llama_seq_id seq) {
    llama_memory_t mem = llama_get_memory(ctx);
    if (!mem) return 0;
    const llama_pos pos_max = llama_memory_seq_pos_max(mem, seq);
    return pos_max < 0 ? 0 : (int) pos_max + 1;
}

static void batch_add(llama_batch& batch, llama_token id, llama_pos pos,
                      llama_seq_id seq, bool logits) {
    const int i = batch.n_tokens;
    batch.token[i]     = id;
    batch.pos[i]       = pos;
    batch.n_seq_id[i]  = 1;
    batch.seq_id[i][0] = seq;
    batch.logits[i]    = logits;
    batch.n_tokens++;
}

static bool prefill_chunked(llama_context* ctx, llama_seq_id seq,
                            const llama_token* tokens, int n_tokens) {
    const int n_batch = llama_n_batch(ctx);
    llama_pos pos0    = (llama_pos) kv_seq_len(ctx, seq);
    for (int off = 0; off < n_tokens; off += n_batch) {
        const int len = std::min(n_batch, n_tokens - off);
        llama_batch batch = llama_batch_init(len, 0, 1);
        for (int j = 0; j < len; j++) {
            const bool logits = (off + j == n_tokens - 1);
            batch_add(batch, tokens[off + j], pos0 + off + j, seq, logits);
        }
        const int ret = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (ret) return false;
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
    if (slot.seq_id < 0) {
        fprintf(stderr, "[session] prefill without seq_id\n");
        return false;
    }
    const llama_seq_id seq = slot.seq_id;

    const int n_kv = kv_seq_len(ctx, seq);

    if (n_kv == 0) {
        if (!prefill_chunked(ctx, seq, prompt_tokens.data(),
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
                "[session] KV/token desync (seq=%d kv=%d stored=%zu) — reset\n",
                (int) seq, n_kv, slot.kv_tokens.size());
        session_reset_kv(ctx, slot);
        return session_prefill(ctx, slot, prompt_tokens, stats_out);
    }

    const size_t lcp =
        token_common_prefix(slot.kv_tokens, prompt_tokens);

    if (lcp < slot.kv_tokens.size()) {
        if (!llama_memory_seq_rm(mem, seq, (llama_pos) lcp, -1)) {
            fprintf(stderr, "[session] seq_rm failed seq=%d at %zu\n",
                    (int) seq, lcp);
            session_reset_kv(ctx, slot);
            return session_prefill(ctx, slot, prompt_tokens, stats_out);
        }
        slot.kv_tokens.resize(lcp);
    }

    stats.reused_tokens = lcp;

    if (lcp < prompt_tokens.size()) {
        if (!prefill_chunked(ctx, seq, prompt_tokens.data() + lcp,
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

bool session_decode_token(llama_context* ctx, SessionSlot& slot, llama_token id) {
    if (slot.seq_id < 0) return false;
    const llama_pos pos = (llama_pos) kv_seq_len(ctx, slot.seq_id);
    llama_batch batch   = llama_batch_init(1, 0, 1);
    batch_add(batch, id, pos, slot.seq_id, true);
    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return ret == 0;
}

void session_clear_seq(llama_context* ctx, llama_seq_id seq) {
    if (!ctx || seq < 0) return;
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) llama_memory_seq_rm(mem, seq, 0, -1);
}

void session_reset_kv(llama_context* ctx, SessionSlot& slot) {
    if (slot.seq_id >= 0) session_clear_seq(ctx, slot.seq_id);
    slot.kv_tokens.clear();
}

SessionStore::SessionStore(size_t max_slots) : max_slots_(max_slots > 0 ? max_slots : 1) {}

void SessionStore::set_seq_capacity(uint32_t n_seq_max) {
    if (n_seq_max < 1) n_seq_max = 1;
    n_seq_cap_ = n_seq_max;
    seq_free_.assign(n_seq_max, true);
    for (const auto& kv : slots_)
        if (kv.second.seq_id >= 0 &&
            (uint32_t) kv.second.seq_id < n_seq_max)
            seq_free_[(size_t) kv.second.seq_id] = false;
}

llama_seq_id SessionStore::alloc_seq() {
    for (uint32_t i = 0; i < n_seq_cap_; i++) {
        if (seq_free_[i]) {
            seq_free_[i] = false;
            return (llama_seq_id) i;
        }
    }
    return -1;
}

void SessionStore::free_seq(llama_seq_id seq) {
    if (seq < 0 || (uint32_t) seq >= n_seq_cap_) return;
    seq_free_[(size_t) seq] = true;
}

void SessionStore::evict_lru(llama_context* ctx) {
    if (slots_.empty()) return;
    auto it = slots_.begin();
    for (auto jt = slots_.begin(); jt != slots_.end(); ++jt) {
        if (jt->second.last_tick < it->second.last_tick) it = jt;
    }
    SessionSlot doomed = std::move(it->second);
    slots_.erase(it);
    session_reset_kv(ctx, doomed);
    if (doomed.seq_id >= 0) free_seq(doomed.seq_id);
    fprintf(stderr, "[session] evicted conv=%s seq=%d\n", doomed.id.c_str(),
            (int) doomed.seq_id);
}

SessionSlot& SessionStore::get_or_create(const std::string& id,
                                       llama_context* ctx) {
    auto it = slots_.find(id);
    if (it != slots_.end()) {
        touch(it->second);
        return it->second;
    }
    while (slots_.size() >= max_slots_) evict_lru(ctx);
    llama_seq_id seq = alloc_seq();
    if (seq < 0) {
        evict_lru(ctx);
        seq = alloc_seq();
    }
    touch(slots_[id]);
    slots_[id].id     = id;
    slots_[id].seq_id = seq;
    return slots_[id];
}

void SessionStore::remove(const std::string& id, llama_context* ctx) {
    auto it = slots_.find(id);
    if (it == slots_.end()) return;
    SessionSlot slot = std::move(it->second);
    slots_.erase(it);
    session_reset_kv(ctx, slot);
    if (slot.seq_id >= 0) free_seq(slot.seq_id);
}

void SessionStore::clear(llama_context* ctx) {
    for (auto& kv : slots_) session_reset_kv(ctx, kv.second);
    slots_.clear();
    std::fill(seq_free_.begin(), seq_free_.end(), true);
}

void SessionStore::touch(SessionSlot& slot) { slot.last_tick = ++tick_; }

} // namespace cne
