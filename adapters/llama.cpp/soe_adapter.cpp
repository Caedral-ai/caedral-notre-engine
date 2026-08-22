#include "soe_adapter.h"

#include "llama.h"

#include <atomic>

namespace soe::adapter {

const char *llama_version() {
    // Qualified on purpose: unqualified lookup would find soe::adapter::llama_version.
    return ::llama_version();
}

bool llama_backend_init_once() {
    static std::atomic<bool> initialized{false};
    bool expected = false;
    if (initialized.compare_exchange_strong(expected, true)) {
        ::llama_backend_init();
    }
    return true;
}

}
