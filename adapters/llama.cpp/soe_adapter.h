#pragma once
// Minimal seam between soe and the pinned llama.cpp.
// Purpose: prove the integration compiles and report upstream identity,
// so seam tests can fail fast on incompatible bumps. Plan ref: §13.
#include <cstdint>

namespace soe::adapter {

const char* llama_version();
bool llama_backend_init_once();

} // namespace soe::adapter
