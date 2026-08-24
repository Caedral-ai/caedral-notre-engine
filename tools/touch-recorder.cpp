// touch-recorder: records the slice-level expert touch stream of real
// decoding. For every MUL_MAT_ID node we log (step, fused-tensor name,
// routed expert ids) by reading the i32 ids tensor at execution time.
// Shared experts are verified to fire every step and measured, but stay
// out of the stream: mandatory-resident, never cached.
//
// Trace format (text):
//   #T <fused-name> <slice_bytes>        header, one per routed tensor
//   #M <shared_bytes> <dense_other_bytes>
//   S <step> <fused-name> <id,id,...>    one line per touch
#include "cne/model.h"
#include "cne/model_registry.h"

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

FILE *g_trace = nullptr;
long g_step = 0;

struct Capture {
    const cne::ModelManifest *manifest = nullptr;
    std::map<std::string, const cne::TensorInfo *> routed;
    std::set<std::string> shared_names;
    std::set<std::string> shared_seen_step;
    long steps_all_shared_seen = 0;
};

Capture g_cap;

bool cb_eval(struct ggml_tensor *t, bool ask, void *user_data) {
    (void)user_data;
    if (!g_trace || ask || !t || !t->name)
        return true;

    // Shared-expert residency verification (any op referencing them).
    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; i++) {
        auto *s = t->src[i];
        if (s->name && g_cap.shared_names.count(s->name))
            g_cap.shared_seen_step.insert(s->name);
    }

    if (t->op != GGML_OP_MUL_MAT_ID)
        return true;

    // src layout: fused weights + activations + i32 ids.
    ggml_tensor *w = nullptr;
    ggml_tensor *ids = nullptr;
    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; i++) {
        auto *s = t->src[i];
        if (!ids && s->type == GGML_TYPE_I32)
            ids = s;
        else if (!w && s->name && g_cap.routed.count(s->name))
            w = s;
    }
    if (!w || !ids || !ids->data)
        return true;

    int n = (int)ggml_nelements(ids);
    if (n <= 0 || n > 4096)
        return true;
    std::vector<int32_t> v(n);
    memcpy(v.data(), ids->data, sizeof(int32_t) * n);

    fprintf(g_trace, "S %ld %s", g_step, w->name);
    for (int i = 0; i < n; i++)
        fprintf(g_trace, "%c%d", i ? ',' : ' ', v[i]);
    fprintf(g_trace, "\n");
    return true;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <trace.out> [n_gen=96]\n", argv[0]);
        return 2;
    }
    int n_gen = argc > 3 ? atoi(argv[3]) : 96;

    cne::ModelRegistry reg;
    cne::ModelManifest manifest;
    if (!reg.build(argv[1], manifest)) {
        fprintf(stderr, "[touch-recorder] manifest FAILED: %s\n", reg.error().c_str());
        return 1;
    }

    FILE *out = fopen(argv[2], "w");
    if (!out) {
        fprintf(stderr, "[touch-recorder] cannot write %s\n", argv[2]);
        return 1;
    }
    g_trace = out;
    g_cap.manifest = &manifest;
    size_t shared_bytes = 0, dense_other_bytes = 0;
    for (const auto &ti : manifest.tensors) {
        if (ti.kind == cne::TensorKind::ROUTED_EXPERT) {
            g_cap.routed[ti.name] = &ti;
            fprintf(out, "#T %s %llu\n", ti.name.c_str(), (unsigned long long)ti.bytes_per_expert);
        } else if (ti.kind == cne::TensorKind::SHARED_EXPERT) {
            g_cap.shared_names.insert(ti.name);
            shared_bytes += ti.bytes_total;
        } else if (ti.kind != cne::TensorKind::SCALE) {
            dense_other_bytes += ti.bytes_total;
        }
    }
    fprintf(out, "#M %zu %zu\n", shared_bytes, dense_other_bytes);
    printf("mandatory-resident: shared=%.2f MiB | dense(other)=%.2f MiB | "
           "routed slice bytes/step (cold ceiling): %.2f MiB\n",
           shared_bytes / 1048576.0, dense_other_bytes / 1048576.0, [&] {
               double s = 0;
               for (auto &[_, ti] : g_cap.routed)
                   s += (double)ti->bytes_per_expert * manifest.n_experts_used / manifest.n_experts_used;
               // per step: n_used slices per routed tensor
               s = 0;
               for (auto &[_, ti] : g_cap.routed)
                   s += (double)ti->bytes_per_expert * manifest.n_experts_used;
               return s / 1048576.0;
           }());

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_extra_bufts = false;
    llama_model *model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "[touch-recorder] LOAD FAILED\n");
        return 1;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 256;
    cparams.n_batch = cparams.n_ctx;
    cparams.n_ubatch = 64;
    cparams.n_threads = 8;
    cparams.n_threads_batch = 8;
    cparams.cb_eval = cb_eval;
    cparams.cb_eval_user_data = nullptr;

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "[touch-recorder] CONTEXT FAILED\n");
        return 1;
    }
    llama_set_warmup(ctx, false);

    const char *prompt = "The capital of France is";
    const auto *vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(32);
    int n_tok = llama_tokenize(vocab, prompt, (int)strlen(prompt), toks.data(), (int)toks.size(), true, false);
    toks.resize(n_tok);
    if (llama_decode(ctx, llama_batch_get_one(toks.data(), n_tok))) {
        fprintf(stderr, "[touch-recorder] PREFILL FAILED\n");
        return 1;
    }

    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    long produced = 0;
    for (int i = 0; i < n_gen; i++) {
        g_cap.shared_seen_step.clear();
        g_step = produced;
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id))
            break;
        if (llama_decode(ctx, llama_batch_get_one(&id, 1))) {
            fprintf(stderr, "[touch-recorder] DECODE FAILED\n");
            return 1;
        }
        produced++;
        if ((int)g_cap.shared_seen_step.size() == (int)g_cap.shared_names.size())
            g_cap.steps_all_shared_seen++;
    }
    fclose(out);

    printf("trace: %s | steps=%ld | shared-expert coverage: %ld/%ld steps full\n", argv[2], produced,
           g_cap.steps_all_shared_seen, produced);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
