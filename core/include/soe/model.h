#pragma once
// Immutable per-model manifest generated at open time. Metadata-driven:
// no hardcoded layouts, axes, quant types or offsets. Plan ref: §5.1.
#include <cstdint>
#include <string>
#include <vector>

namespace soe {

struct TensorInfo {
    std::string name;
    uint64_t abs_offset; // data_offset + tensor_offset
    uint64_t bytes_total;
    int expert_count;
    int expert_axis;
    uint64_t bytes_per_expert;
    std::string quant_type;
    size_t io_alignment;
};

struct ModelManifest {
    std::string model_hash;
    uint32_t gguf_version = 3;
    std::string architecture;
    int n_layers = 0;
    int n_experts = 0;
    int n_experts_used = 0;
    std::vector<TensorInfo> tensors;
};

} // namespace soe
