#pragma once
// Expert slice addressing within fused GGUF tensors.
#include <cstdint>

namespace soe {

struct ExpertKey {
    int layer;
    int expert_id;
    // tensor_kind: gate/up/down/etc.
    const char *tensor_kind;
    int shard;

    bool operator==(const ExpertKey &o) const {
        return layer == o.layer && expert_id == o.expert_id && shard == o.shard && tensor_kind == o.tensor_kind;
    }
};

}
