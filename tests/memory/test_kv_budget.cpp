#include "cne/kv_budget.h"
#include "cne/memory_budget.h"

#include <cassert>
#include <cstdio>

int main() {
    using namespace cne;

    MemoryBudget b;
    b.mem_total      = 16ull << 30;
    b.mem_available  = 14ull << 30;
    b.base_reserve   = 2ull << 30;
    b.kv             = 64u << 20;
    b.staging        = 64u << 20;
    b.runtime_base   = 512u << 20;

    const size_t lfm2 = (size_t) (13.4 * (1ull << 30));

    ServingKvEstimate chat =
        suggest_serving_kv(b, lfm2, Regime::R0_RESIDENT, false);
    assert(chat.n_seq_max >= 2);
    assert(chat.n_ctx_per_seq >= 512);
    assert(chat.n_ctx == chat.n_ctx_per_seq * chat.n_seq_max);
    assert(chat.kv_bytes() > 0);

    ServingKvEstimate mtp =
        suggest_serving_kv(b, lfm2, Regime::R0_RESIDENT, true);
    assert(mtp.n_seq_max == 1);
    assert(mtp.n_ctx == 1024);

    ServingKvEstimate tight =
        suggest_serving_kv(b, (size_t) (13.8 * (1ull << 30)),
                           Regime::R0_RESIDENT, false);
    assert(tight.n_seq_max == 1);

    assert(!serving_kv_exceeds_headroom(
        ServingKvEstimate{.n_ctx = 256, .n_seq_max = 1, .n_ctx_per_seq = 256},
        b, lfm2));

    printf("kv_budget: OK (chat ctx=%d seq=%d ~%zu MiB)\n", chat.n_ctx,
           chat.n_seq_max, chat.kv_bytes() >> 20);
    return 0;
}
