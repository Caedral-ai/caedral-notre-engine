#include "cne_session.h"

#include <cassert>
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

    cne::SessionStore store(2);
    auto& s1 = store.get_or_create("a");
    s1.kv_tokens = {10, 11};
    auto& s2 = store.get_or_create("b");
    s2.kv_tokens = {20};
    auto& s3 = store.get_or_create("c");
    (void) s3;
    assert(store.size() == 2);

    return 0;
}
