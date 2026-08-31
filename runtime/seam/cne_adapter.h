#pragma once
// Minimal seam between cne and the pinned llama.cpp.
// Purpose: prove the integration compiles and report upstream identity,
#include <cstdint>

namespace cne::adapter {

const char *llama_version();
bool llama_backend_init_once();

}
