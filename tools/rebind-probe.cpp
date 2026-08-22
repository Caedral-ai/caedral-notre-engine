// rebind-probe: prove the engine can own an expert tensor's bytes.
// Copies one routed-expert tensor into engine-owned aligned memory,
// repoints the ggml tensor, generates greedily and checks two invariants:
//   1) output tokens identical to the non-rebound run (run twice, diff)
//   2) weights are read-only: our buffer must be untouched after compute
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Capture {
    const char *target = nullptr;
    ggml_tensor *found = nullptr;
};

bool cb_eval(struct ggml_tensor *t, bool ask, void *user_data) {
    auto *c = static_cast<Capture *>(user_data);
    if (ask)
        return true;
    if (!t || c->found || !t->name)
        return true;
    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; i++) {
        auto *s = t->src[i];
        if (s->name && std::strcmp(s->name, c->target) == 0) {
            c->found = s;
            return true;
        }
    }
    return true;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [tensor] [rebind 0|1]\n", argv[0]);
        return 2;
    }
    const char *tensor_name = argc > 2 ? argv[2] : "blk.0.ffn_gate_exps.weight";
    bool rebind = argc > 3 ? std::atoi(argv[3]) != 0 : false;

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_extra_bufts = false;
    llama_model *model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "[rebind-probe] LOAD FAILED\n");
        return 1;
    }

    Capture capture;
    capture.target = tensor_name;

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
        fprintf(stderr, "[rebind-probe] CONTEXT FAILED\n");
        return 1;
    }
    llama_set_warmup(ctx, false);

    const char *prompt = "hello";
    const auto *vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(32);
    int n_tok = llama_tokenize(vocab, prompt, (int)strlen(prompt), tokens.data(), (int)tokens.size(), true, false);
    tokens.resize(n_tok);
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tok))) {
        fprintf(stderr, "[rebind-probe] PREFILL FAILED\n");
        return 1;
    }
    if (!capture.found) {
        fprintf(stderr, "[rebind-probe] TARGET TENSOR NOT SEEN: %s\n", tensor_name);
        return 1;
    }

    ggml_tensor *t = capture.found;
    size_t nbytes = ggml_nbytes(t);
    printf("target: %s | type=%s | bytes=%zu\n", tensor_name, ggml_type_name(t->type), nbytes);

    void *owned = nullptr;
    void *original = t->data;
    if (rebind) {
        size_t asize = (nbytes + 4095) / 4096 * 4096;
        owned = aligned_alloc(4096, asize);
        if (!owned) {
            fprintf(stderr, "[rebind-probe] ALLOC FAILED\n");
            return 1;
        }
        std::memcpy(owned, original, nbytes);
        if (std::memcmp(owned, original, nbytes) != 0) {
            fprintf(stderr, "[rebind-probe] COPY MISMATCH BEFORE REBIND\n");
            return 1;
        }
        t->data = owned;
        printf("rebound data -> engine-owned aligned buffer\n");
    }

    const int n_gen = 8;
    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    printf("tokens:");
    for (int i = 0; i < n_gen; i++) {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id))
            break;
        printf(" %d", id);
        if (llama_decode(ctx, llama_batch_get_one(&id, 1))) {
            fprintf(stderr, "\n[rebind-probe] DECODE FAILED\n");
            return 1;
        }
    }
    printf("\n");

    int verdict = 0;
    if (rebind) {
        if (std::memcmp(owned, original, nbytes) != 0) {
            printf("read-only invariant: VIOLATED\n");
            verdict = 1;
        } else {
            printf("read-only invariant: ok\n");
        }
    }
    printf("verdict: %s (rebind=%d)\n", verdict ? "FAIL" : "PASS", rebind ? 1 : 0);
    return verdict;
}
