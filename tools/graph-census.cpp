// graph-census: classify + time ggml ops during LFM2 decode (n_tokens=1 per sample).
// Phase A1/A2 — kernel analysis. No CNE runtime required.
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Ns = std::chrono::nanoseconds;

enum class MatBucket {
    ROUTER,
    EXPERT_GATE,
    EXPERT_UP,
    EXPERT_DOWN,
    ATTN,
    SHORTCONV,
    DENSE_FFN,
    EMBED_OUTPUT,
    OTHER,
    UNKNOWN,
};

const char * bucket_name(MatBucket b) {
    switch (b) {
    case MatBucket::ROUTER: return "moe_router";
    case MatBucket::EXPERT_GATE: return "moe_expert_gate";
    case MatBucket::EXPERT_UP: return "moe_expert_up";
    case MatBucket::EXPERT_DOWN: return "moe_expert_down";
    case MatBucket::ATTN: return "attention";
    case MatBucket::SHORTCONV: return "shortconv";
    case MatBucket::DENSE_FFN: return "dense_ffn";
    case MatBucket::EMBED_OUTPUT: return "embed_output";
    case MatBucket::OTHER: return "other_mat";
    default: return "unknown";
    }
}

// Coarse buckets for prioritization.
const char * rollup_name(MatBucket b, ggml_op op) {
    if (op == GGML_OP_MUL_MAT_ID) {
        if (b == MatBucket::EXPERT_GATE || b == MatBucket::EXPERT_UP)
            return "moe_expert_q4k"; // B3 eligible
        if (b == MatBucket::EXPERT_DOWN)
            return "moe_expert_q6k_down"; // B3 NOT eligible
    }
    if (b == MatBucket::ROUTER)
        return "moe_router";
    if (b == MatBucket::ATTN)
        return "attention";
    if (b == MatBucket::SHORTCONV)
        return "shortconv";
    if (b == MatBucket::DENSE_FFN)
        return "dense_ffn";
    if (b == MatBucket::EMBED_OUTPUT)
        return "embed_output";
    return nullptr;
}

MatBucket classify_weight(const char * name) {
    if (!name || !name[0])
        return MatBucket::UNKNOWN;
    if (strstr(name, "ffn_gate_inp"))
        return MatBucket::ROUTER;
    if (strstr(name, "ffn_gate_exps"))
        return MatBucket::EXPERT_GATE;
    if (strstr(name, "ffn_up_exps"))
        return MatBucket::EXPERT_UP;
    if (strstr(name, "ffn_down_exps"))
        return MatBucket::EXPERT_DOWN;
    if (strstr(name, "attn_") || strstr(name, ".attn."))
        return MatBucket::ATTN;
    if (strstr(name, "shortconv"))
        return MatBucket::SHORTCONV;
    if (strstr(name, "ffn_gate") || strstr(name, "ffn_up") || strstr(name, "ffn_down"))
        return MatBucket::DENSE_FFN;
    if (strstr(name, "token_embd") || strstr(name, "output"))
        return MatBucket::EMBED_OUTPUT;
    return MatBucket::OTHER;
}

enum class OpGroup {
    MUL_MAT,
    MUL_MAT_ID,
    NORM,
    ACTIVATION,
    ROPE,
    SOFTMAX_ARGSORT,
    CONV_SSM,
    ELEM_BINARY,
    COPY_VIEW,
    OTHER,
};

OpGroup op_group(ggml_op op) {
    switch (op) {
    case GGML_OP_MUL_MAT: return OpGroup::MUL_MAT;
    case GGML_OP_MUL_MAT_ID: return OpGroup::MUL_MAT_ID;
    case GGML_OP_RMS_NORM: return OpGroup::NORM;
    case GGML_OP_ROPE:
    case GGML_OP_ROPE_BACK: return OpGroup::ROPE;
    case GGML_OP_SOFT_MAX:
    case GGML_OP_ARGSORT: return OpGroup::SOFTMAX_ARGSORT;
    case GGML_OP_SSM_CONV: return OpGroup::CONV_SSM;
    case GGML_OP_UNARY: return OpGroup::ACTIVATION;
    case GGML_OP_ADD:
    case GGML_OP_MUL:
    case GGML_OP_DIV: return OpGroup::ELEM_BINARY;
    case GGML_OP_CPY:
    case GGML_OP_VIEW:
    case GGML_OP_RESHAPE:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
    case GGML_OP_CONT: return OpGroup::COPY_VIEW;
    default: return OpGroup::OTHER;
    }
}

const char * group_name(OpGroup g) {
    switch (g) {
    case OpGroup::MUL_MAT: return "mul_mat";
    case OpGroup::MUL_MAT_ID: return "mul_mat_id";
    case OpGroup::NORM: return "norm";
    case OpGroup::ACTIVATION: return "activation";
    case OpGroup::ROPE: return "rope";
    case OpGroup::SOFTMAX_ARGSORT: return "softmax_argsort";
    case OpGroup::CONV_SSM: return "conv_ssm";
    case OpGroup::ELEM_BINARY: return "elem_binary";
    case OpGroup::COPY_VIEW: return "copy_view";
    default: return "other";
    }
}

struct Census {
    bool active = false;
    bool time_nodes = false;
    size_t total_nodes = 0;
    int64_t graph_ns = 0; // sum of per-node wall time (counted once)
    Clock::time_point node_t0{};

    std::map<std::string, size_t> op_counts;
    std::map<std::string, int64_t> op_ns;
    std::map<std::string, size_t> group_counts;
    std::map<std::string, int64_t> group_ns;
    std::map<std::string, size_t> mat_bucket_counts;
    std::map<std::string, int64_t> mat_bucket_ns;
    std::map<std::string, int64_t> rollup_ns;
    std::map<std::string, std::string> mat_bucket_sample;
};

void add_time(std::map<std::string, int64_t> & m, const std::string & key, int64_t dt) {
    m[key] += dt;
}

bool cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    auto * c = static_cast<Census *>(user_data);
    if (!c->active)
        return false;

    if (ask) {
        if (c->time_nodes)
            c->node_t0 = Clock::now();
        return true; // force per-node compute + sync for accurate timing
    }

    if (!t)
        return true;

    int64_t dt = 0;
    if (c->time_nodes)
        dt = std::chrono::duration_cast<Ns>(Clock::now() - c->node_t0).count();

    c->total_nodes++;
    const char * opn = ggml_op_name(t->op);
    c->op_counts[opn]++;
    if (c->time_nodes) {
        c->graph_ns += dt;
        add_time(c->op_ns, opn, dt);
    }

    OpGroup g = op_group(t->op);
    const char * gn = group_name(g);
    c->group_counts[gn]++;
    if (c->time_nodes)
        add_time(c->group_ns, gn, dt);

    if (t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID) {
        const ggml_tensor * w = t->src[0];
        const char * wname = (w && w->name[0]) ? w->name : "?";
        MatBucket b = classify_weight(wname);
        const char * bn = bucket_name(b);
        c->mat_bucket_counts[bn]++;
        if (c->time_nodes)
            add_time(c->mat_bucket_ns, bn, dt);
        if (const char * rn = rollup_name(b, t->op))
            c->rollup_ns[rn] += dt;
        if (!c->mat_bucket_sample.count(bn)) {
            char buf[256];
            snprintf(buf, sizeof(buf), "w=%s type=%s ne=[%lld,%lld,%lld,%lld] act_ne=[%lld,%lld,%lld,%lld]",
                     wname, w ? ggml_type_name(w->type) : "?",
                     w ? (long long) w->ne[0] : 0, w ? (long long) w->ne[1] : 0,
                     w ? (long long) w->ne[2] : 0, w ? (long long) w->ne[3] : 0,
                     (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
            c->mat_bucket_sample[bn] = buf;
        }
    }
    return true;
}

void print_count_pct(const std::map<std::string, size_t> & m, size_t total, const char * title) {
    printf("\n%s\n", title);
    printf("%-24s %8s %8s\n", "category", "count", "pct");
    std::vector<std::pair<std::string, size_t>> v(m.begin(), m.end());
    std::sort(v.begin(), v.end(), [](auto & a, auto & b) { return a.second > b.second; });
    for (const auto & kv : v) {
        double pct = total ? (100.0 * kv.second / total) : 0.0;
        printf("%-24s %8zu %7.1f%%\n", kv.first.c_str(), kv.second, pct);
    }
}

void print_time_pct(const std::map<std::string, int64_t> & m, int64_t total_ns, const char * title) {
    printf("\n%s\n", title);
    printf("%-24s %10s %8s %10s\n", "category", "ms", "pct", "us/op");
    std::vector<std::pair<std::string, int64_t>> v(m.begin(), m.end());
    std::sort(v.begin(), v.end(), [](auto & a, auto & b) { return a.second > b.second; });
    for (const auto & kv : v) {
        double pct = total_ns ? (100.0 * kv.second / total_ns) : 0.0;
        double ms = kv.second / 1e6;
        printf("%-24s %10.3f %7.1f%%\n", kv.first.c_str(), ms, pct);
    }
}

void print_time_pct_with_counts(const std::map<std::string, int64_t> & ns,
                                const std::map<std::string, size_t> & counts,
                                int64_t total_ns,
                                const char * title) {
    printf("\n%s\n", title);
    printf("%-24s %10s %8s %10s\n", "category", "ms", "pct", "us/op");
    std::vector<std::pair<std::string, int64_t>> v(ns.begin(), ns.end());
    std::sort(v.begin(), v.end(), [](auto & a, auto & b) { return a.second > b.second; });
    for (const auto & kv : v) {
        double pct = total_ns ? (100.0 * kv.second / total_ns) : 0.0;
        double ms = kv.second / 1e6;
        size_t n = counts.count(kv.first) ? counts.at(kv.first) : 0;
        double us_op = n ? (kv.second / 1e3 / n) : 0.0;
        printf("%-24s %10.3f %7.1f%% %10.1f\n", kv.first.c_str(), ms, pct, us_op);
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <model.gguf> [n_ctx=4096] [threads=4] [decode_steps=3]\n",
                argv[0]);
        return 2;
    }
    const int n_ctx = argc > 2 ? atoi(argv[2]) : 4096;
    const int n_threads = argc > 3 ? atoi(argv[3]) : 4;
    const int decode_steps = argc > 4 ? atoi(argv[4]) : 3;

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_extra_bufts = false;

    Census census;
    census.time_nodes = true;
    auto cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = 512;
    cparams.n_ubatch = 512;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.cb_eval = cb_eval;
    cparams.cb_eval_user_data = &census;

    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "[graph-census] load failed\n");
        return 1;
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "[graph-census] context failed\n");
        return 1;
    }
    llama_set_warmup(ctx, false);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const char * prompt = "The capital of France is";
    std::vector<llama_token> toks(64);
    int n_tok = llama_tokenize(vocab, prompt, (int) strlen(prompt), toks.data(), (int) toks.size(), true, false);
    if (n_tok < 0) {
        toks.resize((size_t) -n_tok);
        n_tok = llama_tokenize(vocab, prompt, (int) strlen(prompt), toks.data(), (int) toks.size(), true, false);
    }
    toks.resize((size_t) n_tok);

    if (llama_decode(ctx, llama_batch_get_one(toks.data(), n_tok))) {
        fprintf(stderr, "[graph-census] prefill failed\n");
        return 1;
    }

    // warm decode (not counted)
    llama_token warm = 0;
    census.active = false;
    if (llama_decode(ctx, llama_batch_get_one(&warm, 1))) {
        fprintf(stderr, "[graph-census] warm decode failed\n");
        return 1;
    }

    auto wall0 = Clock::now();
    census.active = true;
    for (int s = 0; s < decode_steps; s++) {
        llama_token id = 0;
        if (llama_decode(ctx, llama_batch_get_one(&id, 1))) {
            fprintf(stderr, "[graph-census] decode step %d failed\n", s);
            return 1;
        }
    }
    census.active = false;
    auto wall1 = Clock::now();
    double wall_ms = std::chrono::duration_cast<std::chrono::microseconds>(wall1 - wall0).count() / 1000.0;

    const double scale = 1.0 / decode_steps;
    printf("graph-census: decode profile (avg of %d steps, after prefill+warm)\n", decode_steps);
    printf("model: %s\n", argv[1]);
    printf("n_ctx=%d threads=%d nodes/step=%.0f wall_ms/step=%.2f tok_per_s=%.2f\n",
           n_ctx, n_threads, census.total_nodes * scale, wall_ms / decode_steps,
           1000.0 * decode_steps / wall_ms);

    print_count_pct(census.group_counts, census.total_nodes, "=== op groups by node count (total over run) ===");

    size_t mat_total = 0;
    for (const auto & kv : census.mat_bucket_counts)
        mat_total += kv.second;
    print_count_pct(census.mat_bucket_counts, mat_total, "=== matmul buckets by node count ===");

    const int64_t avg_ns = (int64_t) (census.graph_ns * scale);
    print_time_pct(census.group_ns, census.graph_ns, "=== op groups by wall time (total over run) ===");
    print_time_pct_with_counts(census.mat_bucket_ns, census.mat_bucket_counts, census.graph_ns,
                               "=== matmul buckets by wall time ===");
    print_time_pct(census.rollup_ns, census.graph_ns, "=== kernel rollup (matmul wall time) ===");

    int64_t rollup_sum = 0;
    for (const auto & kv : census.rollup_ns)
        rollup_sum += kv.second;
    int64_t graph_other = census.graph_ns - rollup_sum;
    printf("\n=== graph overhead (non-rollup ops, instrumented) ===\n");
    printf("graph_other              %10.3f ms  %7.1f%%\n",
           graph_other / 1e6, census.graph_ns ? (100.0 * graph_other / census.graph_ns) : 0.0);

    printf("\n=== per-step averages ===\n");
    printf("instrumented_ms/step : %.3f\n", avg_ns / 1e6);
    printf("wall_ms/step       : %.3f\n", wall_ms / decode_steps);
    printf("tok/s (wall)       : %.2f\n", 1000.0 * decode_steps / wall_ms);

    printf("\n=== matmul shape samples ===\n");
    for (const auto & kv : census.mat_bucket_sample)
        printf("%s: %s\n", kv.first.c_str(), kv.second.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
