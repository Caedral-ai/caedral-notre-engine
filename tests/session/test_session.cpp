#include "cne_session.h"

#include <cassert>
#include <cstdio>
#include <vector>

int main() {
    using cne::token_common_prefix;

    const std::vector<llama_token> a = {1, 2, 3, 4};
    const std::vector<llama_token> b = {1, 2, 9};
    assert(token_common_prefix(a, b) == 2);

    const std::vector<llama_token> c = {5, 6};
    assert(token_common_prefix(a, c) == 0);

    const std::vector<llama_token> d = {1, 2, 3, 4};
    assert(token_common_prefix(a, d) == 4);

    const std::vector<llama_token> e;
    assert(token_common_prefix(a, e) == 0);
    assert(token_common_prefix(e, a) == 0);

    // LRU slot cap
    cne::SessionStore store(2);
    cne::SessionSlot* s1 = store.get_or_create("a", nullptr);
    assert(s1);
    s1->kv_tokens = {10, 11};
    cne::SessionSlot* s2 = store.get_or_create("b", nullptr);
    assert(s2);
    s2->kv_tokens = {20};
    cne::SessionSlot* s3 = store.get_or_create("c", nullptr);
    assert(s3);
    (void) s3;
    assert(store.size() == 2);
    cne::SessionSlot* s1b = store.get_or_create("a", nullptr);
    assert(s1b);
    assert(s1b->kv_tokens.empty());

    // Distinct seq lanes per conversation
    cne::SessionStore lanes(8);
    lanes.set_seq_capacity(2);
    cne::SessionSlot* la = lanes.get_or_create("user-a", nullptr);
    cne::SessionSlot* lb = lanes.get_or_create("user-b", nullptr);
    assert(la && lb);
    assert(la->seq_id >= 0 && lb->seq_id >= 0);
    assert(la->seq_id != lb->seq_id);

    // Seq exhaustion evicts LRU conversation and reuses its lane
    cne::SessionStore cap(8);
    cap.set_seq_capacity(2);
    cne::SessionSlot* u1 = cap.get_or_create("u1", nullptr);
    assert(u1);
    u1->kv_tokens = {1};
    cne::SessionSlot* u2 = cap.get_or_create("u2", nullptr);
    assert(u2);
    u2->kv_tokens = {2};
    cne::SessionSlot* u3 = cap.get_or_create("u3", nullptr);
    assert(u3);
    assert(cap.size() == 2);
    assert(u3->seq_id >= 0);
    cne::SessionSlot* u1again = cap.get_or_create("u1", nullptr);
    assert(u1again);
    assert(u1again->kv_tokens.empty());

    const llama_seq_id lane_a = la->seq_id;
    lanes.remove("user-a", nullptr);
    cne::SessionSlot* lc = lanes.get_or_create("user-c", nullptr);
    assert(lc);
    assert(lc->seq_id == lane_a);

    printf("session_lcp: OK (lru + seq lanes)\n");
    return 0;
}
