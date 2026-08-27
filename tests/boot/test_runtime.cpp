// Runtime boot-sequence tests (pre-llama stages) against the synthetic GGUF
// fixture, plus an optional live check via CNE_TEST_MODEL (skipped when
// unset). The llama load stage cannot run on synthetic artifacts.
#include "cne_runtime.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace cne;

namespace {

bool g_werr = false;
void w(FILE * f, const void * p, size_t n) {
    if (fwrite(p, 1, n, f) != n) g_werr = true;
}
void w_u32(FILE * f, uint32_t v) { w(f, &v, 4); }
void w_u64(FILE * f, uint64_t v) { w(f, &v, 8); }
void w_str(FILE * f, const std::string & s) {
    w_u64(f, s.size());
    w(f, s.data(), s.size());
}
void w_tensor_info(FILE * f, const std::string & name,
                   std::vector<uint64_t> dims, uint32_t type, uint64_t off) {
    w_str(f, name);
    w_u32(f, (uint32_t)dims.size());
    for (uint64_t d : dims)
        w_u64(f, d);
    w_u32(f, type);
    w_u64(f, off);
}

// Same fixture layout as tests/gguf/test_gguf_parse.cpp (registry-validated).
std::string write_fixture(const std::string & path) {
    FILE * f = fopen(path.c_str(), "wb");
    assert(f);
    uint32_t magic     = 0x46554747;
    uint32_t version   = 3;
    uint64_t n_tensors = 5;
    uint64_t n_kv      = 5;
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

    w_tensor_info(f, "blk.0.attn_q.weight", {8, 8}, 0, 0);
    w_tensor_info(f, "blk.0.ffn_gate_exps.weight", {8, 2, 4}, 0, 256);
    w_tensor_info(f, "blk.0.ffn_gate_shexp.weight", {8, 2}, 0, 512);
    w_tensor_info(f, "blk.1.attn_q.weight", {8, 8}, 0, 576);
    w_tensor_info(f, "token_embd.weight", {8}, 0, 832);
    for (int i = 0; i < 864; i++)
        fputc(0xAB, f);
    fclose(f);
    if (g_werr) fprintf(stderr, "fixture write FAILED\n");
    return path;
}

} // namespace

int main() {
    const std::string path = "/tmp/cne_test_runtime.gguf";
    write_fixture(path);

    RuntimeSettings rs;
    rs.model_path = path.c_str();
    rs.cap_gib    = 1;
    rs.n_ctx      = 256;
    rs.stream_on  = true;

    auto rt = runtime_prepare(rs);
    if (!rt) {
        fprintf(stderr, "FAIL: runtime_prepare returned null\n");
        return 1;
    }
    if (rt->manifest.tensors.size() != 5) {
        fprintf(stderr, "FAIL: expected 5 tensors, got %zu\n",
                rt->manifest.tensors.size());
        return 1;
    }
    if (!rt->cache || rt->cache_cap == 0 || rt->cache_cap > (1ull << 30)) {
        fprintf(stderr, "FAIL: cache missing or cap out of range (%zu)\n",
                rt->cache_cap);
        return 1;
    }
    if (rt->regime_str.empty() || rt->dense_policy_str.empty()) {
        fprintf(stderr, "FAIL: regime/dense policy not populated\n");
        return 1;
    }
    if (!rt->streaming) {
        fprintf(stderr, "FAIL: streaming flag not propagated\n");
        return 1;
    }
    printf("runtime_prepare on synthetic fixture: OK "
           "(tensors=%zu cap=%zu MiB regime=%s dense=%s)\n",
           rt->manifest.tensors.size(), rt->cache_cap >> 20,
           rt->regime_str.c_str(), rt->dense_policy_str.c_str());

    if (const char * live = getenv("CNE_TEST_MODEL")) {
        rs.model_path = live;
        auto lr       = runtime_prepare(rs);
        if (!lr || !runtime_load_llama(*lr, rs)) {
            fprintf(stderr, "FAIL: live model load\n");
            return 1;
        }
        if (!lr->model || !lr->ctx || !lr->vocab) {
            fprintf(stderr, "FAIL: live llama plane incomplete\n");
            return 1;
        }
        printf("live runtime_load_llama: OK\n");
        runtime_shutdown(*lr);
    }

    remove(path.c_str());
    printf("PASS\n");
    return 0;
}
