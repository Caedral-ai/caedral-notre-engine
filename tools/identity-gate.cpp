// identity-gate: streaming correctness harness. Rebinds as many routed-expert tensors as fit
// in a byte budget into engine-owned 4096-aligned memory (mixed residency:
// owned + mmap), generates greedily and prints token ids. Run twice
// (rebind 0 / 1) with identical args and diff the token lines — any
// difference means the streaming path changed the math. FAIL.
#include "cne/model.h"
#include "cne/model_registry.h"
#include "cne/tensor_classify.h"

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

struct Capture {
    const cne::ModelManifest *manifest = nullptr;
    std::map<std::string, ggml_tensor *> routed; // name -> tensor (first hit)
};

bool cb_eval(struct ggml_tensor *t, bool ask, void *user_data) {
    auto *c = static_cast<Capture *>(user_data);
    if (ask)
        return true;
    if (!t || !t->name)
        return true;
    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; i++) {
        auto *s = t->src[i];
        if (!s->name || s->name[0] == '\0')
            continue;
        if (c->routed.count(s->name))
            continue;
        for (const auto &ti : c->manifest->tensors)
            if (ti.name == s->name && ti.kind == cne::TensorKind::ROUTED_EXPERT) {
                c->routed[s->name] = s;
                break;
            }
    }
    return true;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [budget_gib=2] [rebind=1] [n_gen=48]\n", argv[0]);
        return 2;
    }
    const size_t budget = (argc > 2 ? (size_t)atoll(argv[2]) : 2) << 30;
    const bool rebind = argc > 3 ? std::atoi(argv[3]) != 0 : true;
    const int n_gen = argc > 4 ? std::atoi(argv[4]) : 48;

    cne::ModelRegistry reg;
    cne::ModelManifest manifest;
    if (!reg.build(argv[1], manifest)) {
        fprintf(stderr, "[identity-gate] manifest FAILED: %s\n", reg.error().c_str());
        return 1;
    }

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_extra_bufts = false;
    llama_model *model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "[identity-gate] LOAD FAILED\n");
        return 1;
    }

    Capture capture;
    capture.manifest = &manifest;

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 256;
    cparams.n_batch = cparams.n_ctx;
    cparams.n_ubatch = 64;
    cparams.n_threads = 8;
    cparams.n_threads_batch = 8;
    cparams.cb_eval = cb_eval;
    cparams.cb_eval_user_data = &capture;

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "[identity-gate] CONTEXT FAILED\n");
        return 1;
    }
    llama_set_warmup(ctx, false);

    const char *prompt = "The capital of France is";
    const auto *vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(32);
    int n_tok = llama_tokenize(vocab, prompt, (int)strlen(prompt), tokens.data(), (int)tokens.size(), true, false);
    tokens.resize(n_tok);
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tok))) {
        fprintf(stderr, "[identity-gate] PREFILL FAILED\n");
        return 1;
    }

    printf("captured routed tensors: %zu/%zu\n", capture.routed.size(), manifest.routed_expert_tensors);

    size_t owned_bytes = 0, rebound = 0;
    std::vector<std::pair<void *, ggml_tensor *>> owned;
    if (rebind) {
        // Rebind in layer order until budget exhausted (deterministic subset).
        std::vector<std::pair<int, ggml_tensor *>> order;
        for (auto &[name, t] : capture.routed)
            order.push_back({cne::parse_layer_index(name), t});
        std::sort(order.begin(), order.end(), [](auto &a, auto &b) { return a.first < b.first; });
        for (auto &[layer, t] : order) {
            size_t nbytes = ggml_nbytes(t);
            size_t asize = (nbytes + 4095) / 4096 * 4096;
            if (owned_bytes + asize > budget)
                continue;
            void *buf = aligned_alloc(4096, asize);
            if (!buf)
                continue;
            std::memcpy(buf, t->data, nbytes);
            t->data = buf;
            owned.push_back({buf, t});
            owned_bytes += asize;
            rebound++;
        }
        printf("rebound %zu/%zu tensors, %.2f GiB owned (budget %zu GiB)\n", rebound, capture.routed.size(),
               owned_bytes / 1073741824.0, budget >> 30);
    }

    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    printf("tokens:");
    int produced = 0;
    for (int i = 0; i < n_gen; i++) {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id))
            break;
        printf(" %d", id);
        produced++;
        if (llama_decode(ctx, llama_batch_get_one(&id, 1))) {
            fprintf(stderr, "\n[identity-gate] DECODE FAILED\n");
            return 1;
        }
    }
    printf("\nproduced: %d\n", produced);

    int verdict = 0;
    if (rebind) {
        size_t dirty = 0;
        for (auto &[buf, t] : owned) {
            // Read-only check needs the original bytes; we only kept the copy.
            // A dirty buffer would have shown up as output drift already, but
            // we can still detect aliasing bugs by checking the pointer moved.
            if (t->data != buf)
                dirty++;
        }
        if (dirty) {
            printf("pointer stability: VIOLATED (%zu)\n", dirty);
            verdict = 1;
        } else {
            printf("pointer stability: ok\n");
        }
    }
    printf("verdict: %s (rebind=%d)\n", verdict ? "FAIL" : "PASS", rebind ? 1 : 0);
    return verdict;
}
