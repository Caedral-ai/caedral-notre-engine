#pragma once
// Tensor-kind classification. Generic suffix rules cover most architectures;
// architecture-specific overrides live here (model-specific code is allowed,
#include "soe/model.h"

#include <string>

namespace soe {

TensorKind classify_tensor(const std::string &architecture, const std::string &name);

// Extracts the layer index from "blk.N."-style names. Returns -1 if absent.
int parse_layer_index(const std::string &name);

}
