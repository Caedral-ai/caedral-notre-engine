#pragma once
// Server-side conversation slots: token-level prefix reuse across HTTP turns.
#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cne {

struct SessionPrefillStats {
    size_t reused_tokens  = 0;   // KV positions skipped (prefix hit)
    size_t prefilled_tokens = 0; // tokens decoded this request (prompt tail)
};

struct SessionSlot {
    std::string              id;
    std::vector<llama_token> kv_tokens;   // tokens represented in KV seq 0
    int64_t                  last_tick = 0;
};

// Longest common prefix length (token-level, not string-level).
size_t token_common_prefix(const std::vector<llama_token>& a,
                           const std::vector<llama_token>& b);

// Current filled length of sequence 0 in ctx (0 when empty).
int kv_seq_len(llama_context* ctx);

// Align KV with prompt: trim on edit, prefill only the new tail. On success,
// slot.kv_tokens == prompt_tokens. When KV is empty, always full-prefills
// prompt (session switch path).
bool session_prefill(llama_context* ctx, SessionSlot& slot,
                     const std::vector<llama_token>& prompt_tokens,
                     SessionPrefillStats* stats_out);

void session_append_token(SessionSlot& slot, llama_token id);

void session_reset_kv(llama_context* ctx, SessionSlot& slot);

class SessionStore {
public:
    explicit SessionStore(size_t max_slots = 8);

    SessionSlot& get_or_create(const std::string& id);
    void         remove(const std::string& id);
    void         clear();
    size_t       size() const { return slots_.size(); }
    size_t       max_slots() const { return max_slots_; }

private:
    void touch(SessionSlot& slot);
    void evict_lru_if_needed();

    size_t max_slots_;
    int64_t tick_ = 0;
    std::unordered_map<std::string, SessionSlot> slots_;
};

} // namespace cne
