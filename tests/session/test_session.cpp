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
    auto& s1 = store.get_or_create("a", nullptr);
    s1.kv_tokens = {10, 11};
    auto& s2 = store.get_or_create("b", nullptr);
    s2.kv_tokens = {20};
    auto& s3 = store.get_or_create("c", nullptr);
    (void) s3;
    assert(store.size() == 2);
    auto& s1b = store.get_or_create("a", nullptr);
    assert(s1b.kv_tokens.empty());

    // Distinct seq lanes per conversation
    cne::SessionStore lanes(8);
    lanes.set_seq_capacity(2);
    auto& la = lanes.get_or_create("user-a", nullptr);
    auto& lb = lanes.get_or_create("user-b", nullptr);
    assert(la.seq_id >= 0 && lb.seq_id >= 0);
    assert(la.seq_id != lb.seq_id);

    // Seq exhaustion evicts LRU conversation and reuses its lane
    cne::SessionStore cap(8);
    cap.set_seq_capacity(2);
    auto& u1 = cap.get_or_create("u1", nullptr);
    u1.kv_tokens = {1};
    auto& u2 = cap.get_or_create("u2", nullptr);
    u2.kv_tokens = {2};
    auto& u3 = cap.get_or_create("u3", nullptr);
    assert(cap.size() == 2);
    assert(u3.seq_id >= 0);
    auto& u1again = cap.get_or_create("u1", nullptr);
    assert(u1again.kv_tokens.empty());

    const llama_seq_id lane_a = la.seq_id;
    lanes.remove("user-a", nullptr);
    auto& lc = lanes.get_or_create("user-c", nullptr);
    assert(lc.seq_id == lane_a);

    printf("session_lcp: OK (lru + seq lanes)\n");
    return 0;
}
