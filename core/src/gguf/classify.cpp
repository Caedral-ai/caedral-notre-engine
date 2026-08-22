#include "soe/tensor_classify.h"

#include <cstdlib>
#include <cstring>

namespace soe {

namespace {

bool has_suffix(const std::string &s, const char *suf) {
    size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

} // namespace

int parse_layer_index(const std::string &name) {
    const std::string marker = "blk.";
    size_t p = name.find(marker);
    if (p == std::string::npos)
        return -1;
    p += marker.size();
    size_t end = name.find('.', p);
    if (end == std::string::npos || end == p)
        return -1;
    for (size_t i = p; i < end; i++)
        if (name[i] < '0' || name[i] > '9')
            return -1;
    return std::atoi(name.c_str() + p);
}

TensorKind classify_tensor(const std::string &arch, const std::string &name) {
    // Architecture-specific overrides go here first. qwen35moe needs none:
    // the generic rules below already separate its full taxonomy.
    (void)arch;

    // Scales first: e.g. "ffn_gate_exps.input_scale" is a companion of an
    // expert tensor, not an expert itself.
    if (has_suffix(name, ".scale") || has_suffix(name, ".input_scale"))
        return TensorKind::SCALE;
    // Routed experts appear fused as "<base>_exps[.<quant-suffix>]"; the
    // marker sits mid-name ("ffn_gate_exps.weight"), never as pure suffix.
    if (name.find("_exps") != std::string::npos)
        return TensorKind::ROUTED_EXPERT;
    if (name.find("shexp") != std::string::npos)
        return TensorKind::SHARED_EXPERT;
    if (name.compare(0, 4, "ssm.") == 0 || name.find("ssm_") != std::string::npos || name.compare(0, 5, "conv.") == 0)
        return TensorKind::RECURRENT;
    if (name.compare(0, 5, "attn.") == 0 || name.find("attn_") != std::string::npos)
        return TensorKind::ATTENTION;
    if (name.compare(0, 11, "token_embd.") == 0)
        return TensorKind::EMBEDDING;
    if (name.compare(0, 7, "output.") == 0)
        return TensorKind::OUTPUT;
    if (name.find("output_norm") != std::string::npos || name.find(".attn_norm") != std::string::npos)
        return TensorKind::NORM;
    return TensorKind::OTHER;
}

} // namespace soe
