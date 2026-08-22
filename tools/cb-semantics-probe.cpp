// cb-semantics-probe: determines whether the eval callback's ask phase can
// serve expert demand WITHOUT patching llama.cpp.
//
// Semantics A (eager): at ask=true for MUL_MAT_ID, src tensors are already
//   computed -> the i32 router-ids tensor holds valid values -> we could
//   fetch missing slices synchronously inside the callback.
// Semantics B (build-only): ids->data is null/garbage at ask time -> no safe
//   demand point -> direct kernel patch required.
#include "soe/model_registry.h"

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct Stats {
    long ask_matmul = 0;
    long ask_with_data = 0;
    long ask_valid_values = 0;
    std::vector<int> first_ids;
};

bool cb_eval(ggml_tensor *t, bool ask, void *ud) {
    auto *st = static_cast<Stats *>(ud);
    if (!ask || !t || t->op != GGML_OP_MUL_MAT_ID)
        return true;
    st->ask_matmul++;
    ggml_tensor *ids = nullptr;
    ggml_tensor *w = nullptr;
    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; i++) {
        auto *s = t->src[i];
        if (!ids && s->type == GGML_TYPE_I32)
            ids = s;
        else if (!w && s->ne[2] > 1)
            w = s; // fused weights have expert axis
    }
    if (!ids)
        return true;
    if (!ids->data) {
        printf("  ask#%ld: ids present but data NULL\n", st->ask_matmul);
        return true;
    }
    st->ask_with_data++;
    int n = (int)ggml_nelements(ids);
    if (n <= 0 || n > 4096)
        return true;
    std::vector<int32_t> v(n);
    memcpy(v.data(), ids->data, sizeof(int32_t) * n);
    bool plausible = true;
    for (int i = 0; i < n; i++)
        if (v[i] < 0 || v[i] >= (int)(w ? w->ne[2] : 256)) {
            plausible = false;
            break;
        }
    if (plausible) {
        st->ask_valid_values++;
        if (st->first_ids.size() < 3)
            for (int i = 0; i < n && (int)st->first_ids.size() < 3; i++)
                st->first_ids.push_back(v[i]);
    }
    return true;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    soe::ModelRegistry reg;
    soe::ModelManifest manifest;
    if (!reg.build(argv[1], manifest)) {
        fprintf(stderr, "[cb-probe] manifest FAILED: %s\n", reg.error().c_str());
        return 1;
    }

    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_extra_bufts = false;
    llama_model *model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "[cb-probe] LOAD FAILED\n");
        return 1;
    }

    static Stats stats;
    auto cparams = llama_context_default_params();
    cparams.n_ctx = 128;
    cparams.n_batch = cparams.n_ctx;
    cparams.n_ubatch = 64;
    cparams.n_threads = 8;
    cparams.n_threads_batch = 8;
    cparams.cb_eval = cb_eval;
    cparams.cb_eval_user_data = &stats;

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "[cb-probe] CONTEXT FAILED\n");
        return 1;
    }
    llama_set_warmup(ctx, false);

    const char *prompt = "hello";
    const auto *vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(16);
    int n_tok = llama_tokenize(vocab, prompt, (int)strlen(prompt), toks.data(), (int)toks.size(), true, false);
    if (llama_decode(ctx, llama_batch_get_one(toks.data(), n_tok))) {
        fprintf(stderr, "[cb-probe] DECODE FAILED\n");
        return 1;
    }

    printf("MUL_MAT_ID ask-phase observations:\n");
    printf("  nodes seen        : %ld\n", stats.ask_matmul);
    printf("  ids data readable : %ld\n", stats.ask_with_data);
    printf("  values plausible  : %ld\n", stats.ask_valid_values);
    printf("  sample ids        :");
    for (int v : stats.first_ids)
        printf(" %d", v);
    printf("\n");

    bool semantic_a =
        stats.ask_matmul > 0 && stats.ask_with_data == stats.ask_matmul && stats.ask_valid_values == stats.ask_matmul;
    printf("verdict: %s\n", semantic_a ? "SEMANTICS A - demand-serving possible in callback (no fork)"
                                       : "SEMANTICS B - build-time only; direct kernel patch required (D7 allows)");
    return semantic_a ? 0 : 1;
}
