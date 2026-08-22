#pragma once
// Expert tensor discovery + pre-start validation.
// Fail closed: stream=1 with an incomplete registry must abort or fall back
// to mmap — never dereference nulls, never continue silently.
#include "soe/model.h"

#include <string>

namespace soe {

class ModelRegistry {
public:
    // Parses the GGUF, classifies tensors and validates the manifest.
    bool build(const std::string &gguf_path, ModelManifest &out);
    const std::string &error() const { return error_; }

private:
    std::string error_;
};

} // namespace soe
