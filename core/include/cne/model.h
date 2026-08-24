#pragma once
// Immutable per-model manifest generated at open time. Metadata-driven:
// no hardcoded layouts, axes, quant types or offsets.
#include <cstdint>
#include <string>
#include <vector>

namespace cne {

// Coarse functional class of a tensor. Drives residency policy:
// ROUTED_EXPERT -> expert LRU; everything else -> mandatory-resident set.
enum class TensorKind {
    ROUTED_EXPERT,
    SHARED_EXPERT,
    ATTENTION,
    RECURRENT,
    SCALE,
    EMBEDDING,
    OUTPUT,
    NORM,
    OTHER,
};

const char *tensor_kind_name(TensorKind kind);

struct TensorInfo {
    std::string name;
    TensorKind kind = TensorKind::OTHER;
    int layer = -1;          // -1 when not layer-scoped
    uint64_t abs_offset = 0; // data_offset + tensor_offset
    uint64_t bytes_total = 0;
    // Expert-axis fields; only meaningful for ROUTED_EXPERT.
    int expert_count = 0;
    int expert_axis = -1;
    uint64_t bytes_per_expert = 0;
    bool expert_contiguous = false; // expert axis is last (slowest varying)
    std::string quant_type;
};

struct ModelManifest {
    std::string model_hash; // sha256, filled by caller if computed
    std::string file_path;
    uint64_t file_size = 0;
    uint32_t gguf_version = 0;
    uint64_t data_offset = 0;
    size_t io_alignment = 32; // general.alignment
    std::string architecture;

    int n_layers = 0;
    int n_experts = 0;
    int n_experts_used = 0;

    std::vector<TensorInfo> tensors;

    // Convenience counters filled during validation.
    size_t routed_expert_tensors = 0;
    size_t shared_expert_tensors = 0;
    size_t misaligned_for_odirect = 0; // tensor start not 4096-aligned
    // Slice-level O_DIRECT alignment: slice k sits at start + k*bytes_per_expert.
    //   all_slices_aligned    : every slice 4096-aligned -> zero-copy possible
    //   uniform_misalignment  : all slices share the same misalignment (fixed
    //                           bounce offset works)
    //   scattered_alignment   : bytes_per_expert not 4096-multiple -> per-slice
    //                           bounce offsets required
    size_t all_slices_aligned = 0;
    size_t uniform_misalignment = 0;
    size_t scattered_alignment = 0;
};

}
