#include "soe/model_registry.h"

#include "soe/gguf.h"
#include "soe/tensor_classify.h"

#include <algorithm>
#include <filesystem>
#include <set>

namespace soe {

namespace {

struct Span {
    uint64_t begin;
    uint64_t end;
};

}

bool ModelRegistry::build(const std::string &gguf_path, ModelManifest &out) {
    namespace fs = std::filesystem;

    error_.clear();
    GgufReader r;
    if (!r.open(gguf_path)) {
        error_ = "gguf: " + r.error();
        return false;
    }

    if (!r.kv_str("general.architecture", out.architecture)) {
        error_ = "gguf: missing general.architecture";
        return false;
    }
    const std::string &arch = out.architecture;

    uint64_t v = 0;
    if (!r.kv_u64(arch + ".block_count", v)) {
        error_ = "gguf: missing " + arch + ".block_count";
        return false;
    }
    out.n_layers = (int)v;

    // MoE expert config. Fail closed when the arch declares experts but the
    // expected keys are missing.
    bool has_experts = r.has_kv(arch + ".expert_count");
    if (has_experts) {
        if (!r.kv_u64(arch + ".expert_count", v)) {
            error_ = "gguf: bad " + arch + ".expert_count";
            return false;
        }
        out.n_experts = (int)v;
        if (out.n_experts <= 0) {
            error_ = "gguf: non-positive expert_count";
            return false;
        }
        if (!r.kv_u64(arch + ".expert_used_count", v)) {
            error_ = "gguf: missing " + arch + ".expert_used_count";
            return false;
        }
        out.n_experts_used = (int)v;
    }

    out.file_path = gguf_path;
    out.file_size = (uint64_t)fs::file_size(fs::path(gguf_path));
    out.gguf_version = r.version();
    out.data_offset = r.data_offset();
    out.io_alignment = r.io_alignment();

    // Byte spans from file layout: sort by rel offset, span = delta to next.
    std::vector<const GgufReader::RawTensor *> sorted;
    sorted.reserve(r.raw_tensors().size());
    for (const auto &t : r.raw_tensors())
        sorted.push_back(&t);
    std::sort(sorted.begin(), sorted.end(), [](auto *a, auto *b) { return a->rel_offset < b->rel_offset; });

    out.tensors.reserve(sorted.size());
    for (size_t i = 0; i < sorted.size(); i++) {
        const auto *rt = sorted[i];
        TensorInfo ti;
        ti.name = rt->name;
        ti.kind = classify_tensor(arch, rt->name);
        ti.layer = parse_layer_index(rt->name);
        ti.abs_offset = r.data_offset() + rt->rel_offset;
        ti.bytes_total =
            (i + 1 < sorted.size()) ? sorted[i + 1]->rel_offset - rt->rel_offset : out.file_size - ti.abs_offset;

        if (ti.kind == TensorKind::ROUTED_EXPERT) {
            if (out.n_experts <= 0) {
                error_ = "gguf: routed expert tensor but no expert_count: " + rt->name;
                return false;
            }
            int axis = -1;
            for (int d = 0; d < (int)rt->dims.size(); d++)
                if ((int)rt->dims[d] == out.n_experts) {
                    axis = d;
                    break;
                }
            if (axis < 0 || !ti.bytes_total) {
                error_ = "gguf: expert axis not found in tensor: " + rt->name;
                return false;
            }
            ti.expert_count = out.n_experts;
            ti.expert_axis = axis;
            ti.expert_contiguous = (axis == (int)rt->dims.size() - 1);
            ti.bytes_per_expert = ti.bytes_total / (uint64_t)out.n_experts;
            out.routed_expert_tensors++;
        } else if (ti.kind == TensorKind::SHARED_EXPERT) {
            out.shared_expert_tensors++;
        }
        if (ti.abs_offset % 4096 != 0 && ti.kind == TensorKind::ROUTED_EXPERT)
            out.misaligned_for_odirect++;
        if (ti.kind == TensorKind::ROUTED_EXPERT) {
            if (ti.bytes_per_expert % 4096 == 0) {
                if (ti.abs_offset % 4096 == 0)
                    out.all_slices_aligned++;
                else
                    out.uniform_misalignment++;
            } else {
                out.scattered_alignment++;
            }
        }

        ti.quant_type = GgufReader::ggml_type_name(rt->ggml_type);
        out.tensors.push_back(std::move(ti));
    }

    // Fail-closed structural checks.
    std::set<uint64_t> seen_offsets;
    for (const auto &t : out.tensors) {
        if (t.abs_offset < out.data_offset || t.abs_offset + t.bytes_total > out.file_size) {
            error_ = "gguf: tensor span out of file bounds: " + t.name;
            return false;
        }
        if (!seen_offsets.insert(t.abs_offset).second) {
            error_ = "gguf: duplicate tensor offset: " + t.name;
            return false;
        }
    }
    if (has_experts && out.routed_expert_tensors == 0) {
        error_ = "gguf: arch declares experts but none found";
        return false;
    }
    return true;
}

}
