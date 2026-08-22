// graph-probe: capture the tensors llama.cpp actually touches during one
// decode and reconcile them against the ModelManifest built from the file.
// Fail-closed seam between "what the file says" and "what inference does".
#include "soe/model.h"
#include "soe/model_registry.h"

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

struct Capture {
    std::set<std::string> leaf_names; // named leaves = weights & states
    bool saw_ask_phase = false;
    size_t ask_calls = 0;
    size_t data_calls = 0;
    std::vector<std::string> first_nodes;
};

bool cb_eval(struct ggml_tensor *t, bool ask, void *user_data) {
    auto *c = static_cast<Capture *>(user_data);
    if (ask) {
        c->saw_ask_phase = true;
        c->ask_calls++;
        return true;
    }
    c->data_calls++;
    if (!t || !t->name)
        return true;
    if (c->first_nodes.size() < 5)
        c->first_nodes.push_back(t->name);
    bool is_leaf = true;
    for (int i = 0; i < GGML_MAX_SRC; i++)
        if (t->src[i]) {
            is_leaf = false;
            break;
        }
    if (is_leaf && t->name[0])
        c->leaf_names.insert(t->name);
    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; i++) {
        auto *s = t->src[i];
        if (s->name && s->name[0]) {
            bool s_leaf = true;
            for (int j = 0; j < GGML_MAX_SRC; j++)
                if (s->src[j]) {
                    s_leaf = false;
                    break;
                }
            if (s_leaf)
                c->leaf_names.insert(s->name);
        }
    }
    return true;
}

bool weight_like(const std::string &n) {
    return n.compare(0, 4, "blk.") == 0 || n.compare(0, 11, "token_embd.") == 0 || n.compare(0, 7, "output.") == 0 ||
           n.compare(0, 9, "position_") == 0;
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
        fprintf(stderr, "[graph-probe] manifest build FAILED: %s\n", reg.error().c_str());
        return 1;
    }

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_extra_bufts = false; // CPU no-repack: weights consumed as stored
    llama_model *model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "[graph-probe] LOAD FAILED\n");
        return 1;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 256;
    cparams.n_batch = cparams.n_ctx;
    cparams.n_ubatch = 64;
    cparams.n_threads = 8;
    cparams.n_threads_batch = 8;

    Capture capture;
    cparams.cb_eval = cb_eval;
    cparams.cb_eval_user_data = &capture;

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "[graph-probe] CONTEXT FAILED\n");
        return 1;
    }
    llama_set_warmup(ctx, false);

    const char *prompt = "hello";
    const auto *vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(32);
    int n_tok = llama_tokenize(vocab, prompt, (int)strlen(prompt), tokens.data(), (int)tokens.size(), true, false);
    tokens.resize(n_tok);
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tok))) {
        fprintf(stderr, "[graph-probe] DECODE FAILED\n");
        return 1;
    }

    // Reconcile: every weight-like leaf must exist in the manifest.
    std::set<std::string> manifest_names;
    size_t routed_total = 0, routed_seen = 0;
    for (const auto &t : manifest.tensors) {
        manifest_names.insert(t.name);
        if (t.kind == soe::TensorKind::ROUTED_EXPERT) {
            routed_total++;
            if (capture.leaf_names.count(t.name))
                routed_seen++;
        }
    }
    std::vector<std::string> unknown;
    for (const auto &n : capture.leaf_names)
        if (!manifest_names.count(n))
            unknown.push_back(n);

    printf("graph-probe report\n");
    printf("  ask phase observed : %s (ask=%zu, data=%zu)\n", capture.saw_ask_phase ? "yes" : "NO", capture.ask_calls,
           capture.data_calls);
    for (const auto &n : capture.first_nodes)
        printf("  node sample: %s\n", n.c_str());
    printf("  named graph leaves : %zu\n", capture.leaf_names.size());
    printf("  matched manifest   : %zu\n", capture.leaf_names.size() - unknown.size());
    printf("  routed experts seen: %zu/%zu\n", routed_seen, routed_total);
    printf("  unmatched leaves   : %zu\n", unknown.size());
    for (size_t i = 0; i < unknown.size() && i < 20; i++) {
        printf("    - %s%s\n", unknown[i].c_str(), weight_like(unknown[i]) ? "  [WEIGHT-LIKE!]" : "");
    }

    bool fail = !capture.saw_ask_phase || routed_seen != routed_total;
    for (const auto &n : unknown)
        if (weight_like(n))
            fail = true;
    printf("  verdict            : %s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
