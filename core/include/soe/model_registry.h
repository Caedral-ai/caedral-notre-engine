#pragma once
// Expert tensor discovery + pre-start validation.
// Fail closed: stream=1 with incomplete registry must abort or fall back to mmap.
// Plan ref: §5.
#include "soe/model.h"

namespace soe {

class ModelRegistry {
public:
    // Scan GGUF shard(s) and build the manifest. Returns false if any
    // expected expert tensor is missing or offsets are invalid.
    bool build(const std::string& gguf_path, ModelManifest& out);
};

} // namespace soe
