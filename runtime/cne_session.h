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
    std::vector<llama_token> kv_tokens;   // tokens represented in KV for seq_id
    llama_seq_id             seq_id = -1; // llama sequence lane; assigned by store
    int64_t                  last_tick = 0;
};

// Longest common prefix length (token-level, not string-level).
size_t token_common_prefix(const std::vector<llama_token>& a,
                           const std::vector<llama_token>& b);

// Filled length of one sequence in ctx (0 when empty).
int kv_seq_len(llama_context* ctx, llama_seq_id seq = 0);

// Align KV with prompt on slot.seq_id: trim on edit, prefill only the new tail.
// On success, slot.kv_tokens == prompt_tokens.
bool session_prefill(llama_context* ctx, SessionSlot& slot,
                     const std::vector<llama_token>& prompt_tokens,
                     SessionPrefillStats* stats_out = nullptr);

void session_append_token(SessionSlot& slot, llama_token id);

// Decode one generated token into slot.seq_id (logits on).
bool session_decode_token(llama_context* ctx, SessionSlot& slot, llama_token id);

// Clear KV for slot.seq_id and token ledger.
void session_reset_kv(llama_context* ctx, SessionSlot& slot);

// Clear one sequence lane without touching slot metadata.
void session_clear_seq(llama_context* ctx, llama_seq_id seq);

class SessionStore {
public:
    explicit SessionStore(size_t max_slots = 8);

    void set_seq_capacity(uint32_t n_seq_max);

    // ctx is used to clear KV when evicting or removing a slot.
    SessionSlot& get_or_create(const std::string& id, llama_context* ctx);
    void         remove(const std::string& id, llama_context* ctx);
    void         clear(llama_context* ctx);
    size_t       size() const { return slots_.size(); }
    size_t       max_slots() const { return max_slots_; }

private:
    void touch(SessionSlot& slot);
    void evict_lru(llama_context* ctx);
    llama_seq_id alloc_seq();
    void         free_seq(llama_seq_id seq);

    size_t max_slots_;
    uint32_t n_seq_cap_ = 1;
    std::vector<bool> seq_free_{1, true};
    int64_t tick_ = 0;
    std::unordered_map<std::string, SessionSlot> slots_;
};

} // namespace cne
