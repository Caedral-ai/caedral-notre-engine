// Synthetic-GGUF fixture tests for the parser/registry, plus an optional
// live check against a real model via SOE_TEST_MODEL (skipped when unset).
#include "soe/gguf.h"
#include "soe/model_registry.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace soe;

namespace {

void w(FILE *f, const void *p, size_t n) {
    assert(fwrite(p, 1, n, f) == n);
}
void w_u32(FILE *f, uint32_t v) {
    w(f, &v, 4);
}
void w_u64(FILE *f, uint64_t v) {
    w(f, &v, 8);
}
void w_str(FILE *f, const std::string &s) {
    w_u64(f, s.size());
    w(f, s.data(), s.size());
}
void w_tensor_info(FILE *f, const std::string &name, std::vector<uint64_t> dims, uint32_t type, uint64_t off) {
    w_str(f, name);
    w_u32(f, (uint32_t)dims.size());
    for (uint64_t d : dims)
        w_u64(f, d);
    w_u32(f, type);
    w_u64(f, off);
}

// Minimal v3 GGUF: 2 layers worth of tensors, arch "testmoe", 4 experts.
std::string write_fixture(const std::string &path, bool expert_axis_last = true, bool corrupt_magic = false) {
    FILE *f = fopen(path.c_str(), "wb");
    assert(f);
    uint32_t magic = corrupt_magic ? 0xdeadbeef : 0x46554747;
    uint32_t version = 3;
    uint64_t n_tensors = 5;
    uint64_t n_kv = 5;
    w(f, &magic, 4);
    w_u32(f, version);
    w_u64(f, n_tensors);
    w_u64(f, n_kv);

    w_str(f, "general.architecture");
    w_u32(f, 8 /*STRING*/);
    w_str(f, "testmoe");
    w_str(f, "general.alignment");
    w_u32(f, 4 /*UINT32*/);
    w_u32(f, 32);
    w_str(f, "testmoe.block_count");
    w_u32(f, 4 /*UINT32*/);
    w_u32(f, 2);
    w_str(f, "testmoe.expert_count");
    w_u32(f, 4 /*UINT32*/);
    w_u32(f, 4);
    w_str(f, "testmoe.expert_used_count");
    w_u32(f, 4 /*UINT32*/);
    w_u32(f, 2);

    // Tensor dir (offsets relative to data start; data begins aligned to 32).
    // Layout plan: header ends ~unaligned -> pad handled by parser.
    // We just need consistent relative offsets with 32B-aligned starts.
    std::vector<std::pair<std::vector<uint64_t>, uint64_t>> spec;
    // attn q: [8,8] f32 = 256B ; gate_exps [8,2,4] f32 = 256B (axis2 last)
    // down_exps non-contiguous variant [8,4,2] when !expert_axis_last
    // shexp [8,2] f32 = 64B ; token_embd [8] f32 = 32B
    w_tensor_info(f, "blk.0.attn_q.weight", {8, 8}, 0, 0);
    w_tensor_info(f, "blk.0.ffn_gate_exps.weight",
                  expert_axis_last ? std::vector<uint64_t>{8, 2, 4} : std::vector<uint64_t>{8, 4, 2}, 0, 256);
    w_tensor_info(f, "blk.0.ffn_gate_shexp.weight", {8, 2}, 0, 512);
    w_tensor_info(f, "blk.1.attn_q.weight", {8, 8}, 0, 576);
    w_tensor_info(f, "token_embd.weight", {8}, 0, 832);
    // Data section: 608 bytes total from rel 0..864.
    for (int i = 0; i < 864; i++)
        fputc(0xAB, f);
    fclose(f);
    return path;
}

void test_good_fixture() {
    std::string p = write_fixture("/tmp/opencode/soe_test_good.gguf");
    ModelRegistry reg;
    ModelManifest m;
    assert(reg.build(p, m));
    assert(m.architecture == "testmoe");
    assert(m.n_layers == 2);
    assert(m.n_experts == 4);
    assert(m.tensors.size() == 5);

    const TensorInfo *exps = nullptr;
    const TensorInfo *shexp = nullptr;
    for (const auto &t : m.tensors) {
        if (t.name == "blk.0.ffn_gate_exps.weight")
            exps = &t;
        if (t.name == "blk.0.ffn_gate_shexp.weight")
            shexp = &t;
    }
    assert(exps && exps->kind == TensorKind::ROUTED_EXPERT);
    assert(exps->layer == 0);
    assert(exps->bytes_total == 256);
    assert(exps->bytes_per_expert == 64);
    assert(exps->expert_axis == 2 && exps->expert_contiguous);
    assert(shexp && shexp->kind == TensorKind::SHARED_EXPERT);
    assert(shexp->expert_count == 0);
    assert(m.routed_expert_tensors == 1);
    printf("ok: good fixture\n");
}

void test_non_contiguous_flagged_not_fatal() {
    std::string p = write_fixture("/tmp/opencode/soe_test_nc.gguf", false);
    ModelRegistry reg;
    ModelManifest m;
    assert(reg.build(p, m));
    bool found = false;
    for (const auto &t : m.tensors)
        if (t.name == "blk.0.ffn_gate_exps.weight") {
            found = true;
            assert(t.expert_axis == 1 && !t.expert_contiguous);
        }
    assert(found);
    printf("ok: non-contiguous expert axis recorded\n");
}

void test_corrupt_magic_fails_closed() {
    std::string p = write_fixture("/tmp/opencode/soe_test_bad.gguf", true, true);
    ModelRegistry reg;
    ModelManifest m;
    assert(!reg.build(p, m));
    assert(reg.error().find("magic") != std::string::npos);
    printf("ok: corrupt magic fails closed\n");
}

void test_live_model_if_available() {
    const char *env = getenv("SOE_TEST_MODEL");
    if (!env || !*env) {
        printf("skip: live model (set SOE_TEST_MODEL)\n");
        return;
    }
    ModelRegistry reg;
    ModelManifest m;
    if (!reg.build(env, m)) {
        fprintf(stderr, "LIVE BUILD FAILED: %s\n", reg.error().c_str());
        exit(1);
    }
    printf("live model: arch=%s layers=%d experts=%d used=%d tensors=%zu "
           "(routed=%zu shared=%zu odirect_misaligned=%zu) data_offset=%llu\n",
           m.architecture.c_str(), m.n_layers, m.n_experts, m.n_experts_used, m.tensors.size(), m.routed_expert_tensors,
           m.shared_expert_tensors, m.misaligned_for_odirect, (unsigned long long)m.data_offset);
    printf("slice alignment: all=%zu uniform_misaligned=%zu scattered=%zu\n", m.all_slices_aligned,
           m.uniform_misalignment, m.scattered_alignment);
    assert(m.architecture == "qwen35moe");
    assert(m.n_layers == 40 && m.n_experts_used > 0);
    assert(m.routed_expert_tensors >= 3 * (size_t)m.n_layers);
    size_t contiguous_routed = 0;
    for (const auto &t : m.tensors)
        if (t.kind == TensorKind::ROUTED_EXPERT && t.expert_contiguous)
            contiguous_routed++;
    assert(contiguous_routed > 0);
}

}

int main() {
    test_good_fixture();
    test_non_contiguous_flagged_not_fatal();
    test_corrupt_magic_fails_closed();
    test_live_model_if_available();
    printf("ALL GGUF TESTS PASSED\n");
    return 0;
}
