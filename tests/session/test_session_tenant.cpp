#include "cne_session.h"

#include <cassert>
#include <cstdio>

int main() {
    cne::SessionStore store(4);
    store.set_max_slots_per_user(2);
    store.set_seq_capacity(4);

    cne::SessionSlot* a1 =
        store.get_or_create("alice:chat-1", nullptr, "alice");
    cne::SessionSlot* a2 =
        store.get_or_create("alice:chat-2", nullptr, "alice");
    assert(a1 && a2);
    a1->kv_tokens = {1, 2, 3};
    a2->kv_tokens = {4, 5};

    cne::SessionSlot* b1 =
        store.get_or_create("bob:chat-1", nullptr, "bob");
    assert(b1);
    b1->kv_tokens = {9};

    // Alice opens a third chat: evicts her LRU (chat-1), not Bob.
    cne::SessionSlot* a3 =
        store.get_or_create("alice:chat-3", nullptr, "alice");
    assert(a3);
    assert(store.size() == 3);

    cne::SessionSlot* a1again =
        store.get_or_create("alice:chat-1", nullptr, "alice");
    assert(a1again);
    assert(a1again->kv_tokens.empty());

    cne::SessionSlot* bob_again = store.get_or_create("bob:chat-1", nullptr, "bob");
    assert(bob_again && bob_again->kv_tokens.size() == 1);

    cne::SessionErr err = cne::SessionErr::None;
    cne::SessionSlot* hijack =
        store.get_or_create("bob:chat-1", nullptr, "alice", &err);
    assert(hijack == nullptr);
    assert(err == cne::SessionErr::OwnerMismatch);

    printf("session_tenant: OK\n");
    return 0;
}
