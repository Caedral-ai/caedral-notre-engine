// cne-server: OpenAI-compatible HTTP/SSE endpoint on the shared cne runtime.
// Single session; greedy default (lossless), temperature opt-in per request.
#include "cne/cache.h"
#include "cne/memory_budget.h"
#include "cne/model.h"
#include "cne/model_registry.h"
#include "cne/config.h"

#include "cne_runtime.h"
#include "cne_stream_cb.h"
#include "cne_stream_spec.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "llama.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

namespace {

// ---- engine state -----------------------------------------------------------

struct Engine {
    std::string        model_path;
    std::string        model_name;
    std::string        regime_str;
    std::string        dense_policy_str;
    bool               odirect  = false;
    bool               streaming = true;   // CNE_STREAM=0 = naive mmap decode
    llama_model*       model     = nullptr;
    llama_context*     ctx       = nullptr;
    const llama_vocab* vocab     = nullptr;
    int                n_ctx     = 0;
    size_t             cache_cap = 0;
    int                mtp_k     = 0;
    float              mtp_p_min = 0.0f;
    bool               think_default = true;   // CNE_THINK=0 flips the default
    bool               arch_qwen3    = false;  // enables the empty-think prefix
    std::string        chat_template;   // empty = model default template
    std::mutex         gen_mutex;       // single-decode: serialize generation
};

Engine g_engine;
std::atomic<long long> g_req_counter{0};

json error_body(const std::string& message, const std::string& type) {
    return json{
        {"error",
         { {"message", message}, {"type", type}, {"param", nullptr},
           {"code", nullptr} }}};
}

void set_json(httplib::Response& res, const json& j) {
    res.set_content(j.dump(), "application/json");
}

// ---- token <-> text helpers --------------------------------------------------

std::vector<llama_token> tokenize(const std::string& text, bool add_special) {
    std::vector<llama_token> out(32);
    // negative return = required token count
    while (true) {
        int n = llama_tokenize(g_engine.vocab, text.c_str(), (int)text.size(),
                               out.data(), (int)out.size(), add_special, true);
        if (n >= 0) { out.resize(n); return out; }
        if ((size_t)(-n) > out.size()) out.resize((size_t)(-n));
    }
}

std::string detokenize(const std::vector<llama_token>& toks) {
    std::string buf(64, '\0');
    while (true) {
        int n = llama_detokenize(g_engine.vocab, toks.data(), (int)toks.size(),
                                 buf.data(), (int)buf.size(), false, false);
        if (n >= 0) { buf.resize(n); return buf; }
        buf.resize((size_t)(-n));
    }
}

// Incremental-safe emission: detokenize the accumulated suffix so multi-token
// UTF-8 glyphs are never split across SSE chunks.
struct Emitter {
    std::vector<llama_token> toks;
    size_t                   sent = 0;
    bool                     strip_think = false;   // hide <think> blocks
    bool                     stripped    = false;

    void push(llama_token id) { toks.push_back(id); }

    // Visible text: drop <think>...</think> spans. An unclosed <think>
    // holds everything after it back until the close arrives (stream-safe:
    // the filter re-runs over the full accumulated text each call).
    std::string visible(const std::string& full) {
        if (!strip_think) return full;
        std::string out;
        size_t pos = 0;
        while (true) {
            size_t b = full.find("<think>", pos);
            if (b == std::string::npos) {
                out += full.substr(pos);
                break;
            }
            size_t e = full.find("</think>", b);
            if (e == std::string::npos) {
                // think block still open - emit nothing from here on yet
                out += full.substr(0, 0);
                break;
            }
            out += full.substr(pos, b - pos);
            pos   = e + 8;   // strlen("</think>")
            stripped = true;
        }
        if (stripped) {
            // swallow the newline the template puts right after </think>
            size_t lead = out.find_first_not_of("\n \t");
            out.erase(0, lead == std::string::npos ? out.size() : lead);
        }
        return out;
    }

    std::string drain() {
        std::string vis = visible(detokenize(toks));
        if (vis.size() <= sent) return {};
        std::string piece = vis.substr(sent);
        sent = vis.size();
        return piece;
    }

    // Failure path: if think suppression held back EVERYTHING, hand back the
    // raw output so the user gets text instead of an empty response.
    std::string rescue_all() {
        if (!strip_think || sent > 0 || toks.empty()) return {};
        return detokenize(toks);
    }
};

// Streaming handoff between the generation thread and the HTTP sink.
struct TokenStream {
    std::mutex              m;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool                    done   = false;
    bool                    failed = false;
    std::atomic<bool>       aborted{false};   // client disconnected

    void push(std::string s) {
        { std::lock_guard<std::mutex> l(m); q.push_back(std::move(s)); }
        cv.notify_one();
    }
    void finish(bool ok) {
        { std::lock_guard<std::mutex> l(m); done = true; failed = !ok; }
        cv.notify_all();
    }
    // blocks until a piece is available or the stream ends
    bool next(std::string& s) {
        std::unique_lock<std::mutex> l(m);
        cv.wait(l, [&] { return done || aborted.load() || !q.empty(); });
        if (!q.empty()) { s = std::move(q.front()); q.pop_front(); return true; }
        return false;
    }
};

// ---- generation ----------------------------------------------------------------

struct EmitCtx {
    Emitter*     em     = nullptr;
    TokenStream* stream = nullptr;   // null = non-streaming
    std::string  text;                // non-streaming accumulation
    long long    emitted = 0;
};

// Shared token emission for both decode arms.
void emit_token(EmitCtx& e, llama_token id) {
    if (e.stream && e.stream->aborted.load()) return;
    e.em->push(id);
    std::string piece = e.em->drain();
    if (!piece.empty()) {
        if (e.stream) e.stream->push(std::move(piece));
        else          e.text += piece;
    }
    e.emitted++;
}

void emit_token_cb(void* ud, llama_token id) {
    emit_token(*static_cast<EmitCtx*>(ud), id);
}

// ---- engine thread -----------------------------------------------------------
// llama contexts are created and warmed on one thread; every decode must run
// on that same thread. HTTP handlers only prepare state and submit jobs.
std::deque<std::function<void()>> g_jobs;
std::mutex                        g_jobs_m;
std::condition_variable           g_jobs_cv;
bool                              g_jobs_stop = false;

void engine_thread_loop() {
    while (true) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> l(g_jobs_m);
            g_jobs_cv.wait(l, [] { return g_jobs_stop || !g_jobs.empty(); });
            if (g_jobs_stop && g_jobs.empty()) return;
            job = std::move(g_jobs.front());
            g_jobs.pop_front();
        }
        job();
    }
}

void submit_job(std::function<void()> f) {
    {
        std::lock_guard<std::mutex> l(g_jobs_m);
        g_jobs.push_back(std::move(f));
    }
    g_jobs_cv.notify_one();
}

// Sequential decode arm. Mirrors cne-bench's loop exactly: sample after each
// forward, emit, then feed the token back in. Returns tokens emitted,
// -1 on decode failure.
long long generate_sequential(Engine& eng, const std::vector<llama_token>& prompt,
                              llama_sampler* smpl, int n_gen, EmitCtx& emit,
                              TokenStream* stream, std::string& finish_reason) {
    if (llama_decode(eng.ctx, llama_batch_get_one(
                const_cast<llama_token*>(prompt.data()), (int)prompt.size()))) {
        fprintf(stderr, "[server] PREFILL FAILED\n");
        finish_reason = "error";
        return -1;
    }

    const int ctx_n_gen = n_gen;
    for (int i = 0; i < ctx_n_gen; i++) {
        if (stream && stream->aborted.load()) { finish_reason = "abort"; break; }
        llama_token id = llama_sampler_sample(smpl, eng.ctx, -1);
        if (llama_vocab_is_eog(eng.vocab, id)) { finish_reason = "stop"; break; }
        emit_token(emit, id);
        cne::stream_set_step(i);
        if (llama_decode(eng.ctx, llama_batch_get_one(&id, 1))) {
            fprintf(stderr, "[server] DECODE FAILED at step %d\n", i);
            finish_reason = "error";
            return i;
        }
        cne::stream_prefetch_kick_full();
        cne::stream_step_boundary();
    }
    if (finish_reason.empty()) finish_reason = "length";
    return emit.emitted;
}

// Draft-MTP speculative arm (greedy only - lossless contract).
// spec_mtp_generate owns prefill and step hooks internally.
long long generate_mtp(Engine& eng, const std::vector<llama_token>& prompt,
                       int n_gen, EmitCtx& emit, TokenStream* stream,
                       std::string& finish_reason) {
    cne::SpecStats st = cne::spec_mtp_generate(
            eng.model, eng.ctx, prompt, eng.mtp_k, eng.mtp_p_min, n_gen,
            cne::stream_cb_eval(), emit_token_cb, &emit);

    fprintf(stderr,
            "[mtp] iterations=%ld drafted=%ld accepted=%ld (%.1f%%) "
            "partials=%ld produced=%d\n",
            st.iterations, st.drafted, st.accepted,
            st.drafted ? 100.0 * st.accepted / st.drafted : 0.0,
            st.partials, st.produced);

    if (st.produced <= 0 && finish_reason.empty()) {
        fprintf(stderr, "[server] MTP GENERATION FAILED\n");
        finish_reason = "error";
        return -1;
    }
    if (finish_reason.empty())
        finish_reason = st.produced < n_gen ? "stop" : "length";
    return emit.emitted;
}

// ---- request handling ------------------------------------------------------------

llama_sampler* build_sampler(float temp, float top_p, uint32_t seed) {
    auto sp = llama_sampler_chain_default_params();
    llama_sampler* chain = llama_sampler_chain_init(sp);
    if (temp <= 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
        if (top_p > 0.0f && top_p < 1.0f)
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    }
    return chain;
}

bool apply_chat_template(const json& messages, std::string& out,
                         std::string& err) {
    std::vector<std::string> roles, contents;
    for (const auto& m : messages) {
        if (!m.contains("role") || !m.contains("content") ||
            !m["role"].is_string() || !m["content"].is_string()) {
            err = "each message needs string 'role' and 'content'";
            return false;
        }
        roles.push_back(m["role"].get<std::string>());
        contents.push_back(m["content"].get<std::string>());
    }
    std::vector<llama_chat_message> msgs;
    msgs.reserve(roles.size());
    for (size_t i = 0; i < roles.size(); i++)
        msgs.push_back({ roles[i].c_str(), contents[i].c_str() });

    const char* tmpl = g_engine.chat_template.empty()
                           ? nullptr
                           : g_engine.chat_template.c_str();
    size_t cap = 4096;
    std::string buf;
    int32_t n = 0;
    while (true) {
        buf.resize(cap);
        n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true,
                                      buf.data(), (int32_t)buf.size());
        if (n >= 0 && (size_t)n < cap) break;   // fits (n excludes NUL)
        if (n <= 0 && cap > (1u << 24)) {
            err = "chat template failed";
            return false;
        }
        cap = n > 0 ? (size_t)n + 1 : cap * 2;
    }
    out.assign(buf, 0, (size_t)n);
    return true;
}

json completion_response(const std::string& id, const std::string& model,
                         const std::string& text, const std::string& finish,
                         long long prompt_toks, long long completion_toks) {
    return json{
        {"id", id},
        {"object", "chat.completion"},
        {"created", std::time(nullptr)},
        {"model", model},
        {"choices",
         json::array({ {
             {"index", 0},
             {"message", { {"role", "assistant"}, {"content", text} }},
             {"finish_reason", finish},
         } })},
        {"usage",
         { {"prompt_tokens", prompt_toks},
           {"completion_tokens", completion_toks},
           {"total_tokens", prompt_toks + completion_toks} }}};
}

json chunk_base(const std::string& id, const std::string& model) {
    return json{
        {"id", id},
        {"object", "chat.completion.chunk"},
        {"created", std::time(nullptr)},
        {"model", model},
        {"choices", json::array({
             { {"index", 0}, {"delta", json::object()}, {"finish_reason", nullptr} }
         })}};
}

void handle_chat(httplib::Response& res, const json& body) {
    Engine& eng = g_engine;

    bool  want_stream = body.value("stream", false);
    float temperature = body.value("temperature", 0.0f);
    float top_p       = body.value("top_p", 1.0f);
    uint32_t seed     = body.value("seed", 0u);
    int max_tokens =
        body.value("max_tokens", body.value("max_completion_tokens", 256));
    if (max_tokens < 1) max_tokens = 1;

    // Thinking control. Default = model behavior (think on). Override:
    //   request: "chat_template_kwargs": {"enable_thinking": false}
    //   server:  CNE_THINK=0 (request may re-enable with enable_thinking:true)
    bool think_enabled = eng.think_default;
    if (body.contains("chat_template_kwargs") &&
        body["chat_template_kwargs"].contains("enable_thinking") &&
        body["chat_template_kwargs"]["enable_thinking"].is_boolean())
        think_enabled =
            body["chat_template_kwargs"]["enable_thinking"].get<bool>();

    json messages = body["messages"];

    std::string rendered, err;
    if (!apply_chat_template(messages, rendered, err)) {
        res.status = 400;
        set_json(res, error_body(err, "invalid_request_error"));
        return;
    }

    // Thinking off, Qwen3-family style: newer qwen3 artifacts ignore the
    // /no_think soft switch and expect the ASSISTANT PREFIX to carry an
    // already-closed empty think block - exactly what the official Jinja
    // template emits for enable_thinking=false. Gated on the artifact's
    // declared architecture (metadata-driven, no name guessing).
    if (!think_enabled && eng.arch_qwen3) {
        while (!rendered.empty() &&
               (rendered.back() == '\n' || rendered.back() == ' '))
            rendered.pop_back();
        rendered += "\n<think>\n\n</think>\n\n";
    }
    // Bisect knob: bypass the chat template entirely and mimic cne-bench's
    // raw-text tokenization (add_special=true, parse_special=false).
    std::vector<llama_token> prompt;
    if (const char* raw = cne::env("RAW_PROMPT")) {
        // exact cne-bench tokenization
        std::vector<llama_token> out(32);
        while (true) {
            int n = llama_tokenize(g_engine.vocab, raw, (int)strlen(raw),
                                   out.data(), (int)out.size(), true, false);
            if (n >= 0) { out.resize(n); break; }
            out.resize((size_t)(-n));
        }
        prompt = out;
    } else {
        prompt = tokenize(rendered, /*add_special=*/false);
    }
    if (cne::env("DEBUG_TOKIDS")) {
        fprintf(stderr, "prompt ids:");
        for (auto t : prompt) fprintf(stderr, " %d", t);
        fprintf(stderr, "\n");
    }
    if (cne::env("DEBUG_PROMPT"))
        fprintf(stderr, "[debug] rendered prompt (%zu chars):\n%s\n<<<END>>>\n",
                rendered.size(), rendered.c_str());

    // Context guard: fail loudly instead of the mid-run abort the default
    // small context produces under MTP ("failed to find a memory slot").
    int headroom = eng.n_ctx - (int)prompt.size() - 4;
    if (headroom < 1) {
        res.status = 400;
        set_json(res, error_body(
            "prompt of " + std::to_string(prompt.size()) +
                " tokens does not fit context of " + std::to_string(eng.n_ctx) +
                "; raise CNE_CTX",
            "context_length_exceeded"));
        return;
    }
    const int n_gen = std::min(max_tokens, headroom);

    // Single-decode runtime: serialize generation across connections.
    std::lock_guard<std::mutex> gen_lock(eng.gen_mutex);

    const auto t0 = std::chrono::steady_clock::now();
    const std::string id =
        "chatcmpl-cne-" + std::to_string(++g_req_counter);
    llama_sampler* smpl = build_sampler(temperature, top_p, seed);
    std::string finish_reason;
    auto log_stats = [&](long long out, bool ok) {
        const double secs = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr,
                "[server] prompt %zu tok | %lld tok out | %.2fs | %.2f tok/s | "
                "%s\n",
                prompt.size(), out, secs, secs > 0 ? out / secs : 0.0,
                ok ? finish_reason.c_str() : "FAILED");
    };

    if (want_stream) {
        // Heap-held state: the content provider runs after this handler
        // returns, so everything it touches must outlive the handler's stack.
        struct StreamState {
            TokenStream              stream;
            Emitter                  emitter;
            EmitCtx                  emit;
            std::string              id, finish_reason;
            std::vector<llama_token> prompt_tokens;
            llama_sampler*           smpl = nullptr;
            bool                     header_sent = false;
            bool                     final_sent  = false;
            std::atomic<bool>        job_done{false};
            ~StreamState() { if (smpl) llama_sampler_free(smpl); }
        };
        auto st      = std::make_shared<StreamState>();
        st->id       = id;
        st->smpl     = smpl;
        st->prompt_tokens = prompt;
        st->emit     = { &st->emitter, &st->stream, {}, 0 };
        st->emitter.strip_think = !think_enabled;

        // generation runs on the engine thread; provider drains as chunks
        submit_job([st, &eng, n_gen, temperature]() {
            long long n =
                eng.mtp_k > 0 && temperature <= 0.0f
                    ? generate_mtp(eng, st->prompt_tokens, n_gen, st->emit,
                                   &st->stream, st->finish_reason)
                    : generate_sequential(eng, st->prompt_tokens,
                                          st->smpl, n_gen, st->emit,
                                          &st->stream, st->finish_reason);
            // think suppression swallowed everything -> flush raw output
            std::string rescue = st->emitter.rescue_all();
            if (!rescue.empty()) {
                fprintf(stderr, "[server] WARNING: think suppression produced "
                                "no visible text; flushing raw output\n");
                if (!st->stream.aborted.load()) st->stream.push(std::move(rescue));
            }
            llama_memory_clear(llama_get_memory(eng.ctx), true);  // flush retained spec state
            st->stream.finish(n >= 0 && st->finish_reason != "error");
        });

        res.status = 200;
        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider(
            "text/event-stream",
            [st, &eng](size_t, httplib::DataSink& sink) -> bool {
                // httplib pulls repeatedly until we return false: each call
                // blocks for one piece (or completion), writes it, and keeps
                // the connection open until the final chunk is out.
                TokenStream* stream = &st->stream;
                auto send_chunk = [&](const json& j) {
                    std::string s = "data: " + j.dump() + "\n\n";
                    return sink.write(s.data(), s.size());
                };
                if (st->final_sent) return false;

                if (!st->header_sent) {
                    st->header_sent = true;
                    json first = chunk_base(st->id, eng.model_name);
                    first["choices"][0]["delta"] = {
                        {"role", "assistant"}, {"content", ""}};
                    if (!send_chunk(first)) stream->aborted.store(true);
                    return !stream->aborted.load();
                }

                std::string piece;
                if (stream->next(piece)) {
                    json ch = chunk_base(st->id, eng.model_name);
                    ch["choices"][0]["delta"] = { {"content", piece} };
                    if (!send_chunk(ch)) stream->aborted.store(true);
                    return true;
                }

                // generation ended
                if (st->job_done.exchange(true) == false)
                    fprintf(stderr,
                            "[server] streamed %lld tok | finish=%s\n",
                            st->emit.emitted, st->finish_reason.c_str());
                if (!stream->failed && !stream->aborted.load()) {
                    json last = chunk_base(st->id, eng.model_name);
                    last["choices"][0]["finish_reason"] = st->finish_reason;
                    send_chunk(last);
                    std::string done = "data: [DONE]\n\n";
                    sink.write(done.data(), done.size());
                }
                st->final_sent = true;
                return false;
            });
        return;
    }

    // Non-streaming: submit and block until the engine thread is done.
    Emitter emitter;
    emitter.strip_think = !think_enabled;
    EmitCtx emit{ &emitter, nullptr, {}, 0 };
    long long n = 0;
    bool gen_done = false;
    std::mutex gen_m;
    std::condition_variable gen_cv;

    submit_job([&]() {
        n = eng.mtp_k > 0 && temperature <= 0.0f
                ? generate_mtp(eng, prompt, n_gen, emit, nullptr, finish_reason)
                : generate_sequential(eng, prompt, smpl, n_gen, emit, nullptr,
                                      finish_reason);
        std::lock_guard<std::mutex> l(gen_m);
        std::string rescue = emitter.rescue_all();
        if (!rescue.empty()) {
            fprintf(stderr, "[server] WARNING: think suppression produced "
                            "no visible text; flushing raw output\n");
            emit.text += rescue;
        }
        gen_done = true;
        gen_cv.notify_one();
    });
    std::unique_lock<std::mutex> l(gen_m);
    gen_cv.wait(l, [&] { return gen_done; });
    llama_sampler_free(smpl);
    llama_memory_clear(llama_get_memory(eng.ctx), true);

    if (finish_reason == "error" || n < 0) {
        res.status = 500;
        set_json(res, 
            error_body("generation failed (see server logs)", "internal_error"));
        return;
    }
    log_stats(emit.emitted, true);
    set_json(res, completion_response(id, eng.model_name, emit.text,
                                     finish_reason, (long long)prompt.size(),
                                     emit.emitted));
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [host=127.0.0.1] [port=8080]\n",
                argv[0]);
        return 2;
    }
    const char* host = argc > 2 ? argv[2] : "127.0.0.1";
    int         port = argc > 3 ? atoi(argv[3]) : 8080;

    // ---- boot via the shared runtime module (same path as cne-bench) ----

    g_engine.model_path = argv[1];

    cne::RuntimeSettings rs;
    rs.model_path = argv[1];
    rs.cap_gib =
        cne::env("CACHE_GIB") ? (size_t)atoll(cne::env("CACHE_GIB")) : 8;
    rs.n_ctx     = 1024;   // serving default; 256 aborts near token ~250 under MTP
    rs.n_threads = 8;
    rs.stream_on =
        cne::env("STREAM") ? atoi(cne::env("STREAM")) != 0 : true;

    auto rt = cne::runtime_prepare(rs);
    if (!rt || !cne::runtime_load_llama(*rt, rs)) return 1;
    g_engine.mtp_k     = rt->mtp_k;
    g_engine.mtp_p_min =
        cne::env("MTP_P_MIN") ? (float)atof(cne::env("MTP_P_MIN")) : 0.0f;
    if (rt->mtp_k > 0 && rt->n_ctx < 1024)
        fprintf(stderr, "[mtp] WARNING: CNE_CTX=%d aborts near token ~250 "
                        "under speculation; recommend CNE_CTX=1024+\n",
                rt->n_ctx);
    if (rt->mtp_k > 0)
        fprintf(stderr,
                "*** [mtp] CNE_MTP=%d requested. UPSTREAM BUG #26425 affects "
                "multi-request serving: output may degrade across requests "
                "(measured on this artifact). cne-bench remains the MTP "
                "measurement vehicle. ***\n",
                rt->mtp_k);

    {
        char arch[64] = {0};
        if (llama_model_meta_val_str(rt->model, "general.architecture",
                                     arch, sizeof(arch)) == 0)
            g_engine.arch_qwen3 = strncmp(arch, "qwen3", 5) == 0;
    }
    {
        auto pos  = g_engine.model_path.find_last_of('/');
        std::string base = pos == std::string::npos
                               ? g_engine.model_path
                               : g_engine.model_path.substr(pos + 1);
        if (base.size() > 5 && base.compare(base.size() - 5, 5, ".gguf") == 0)
            base.resize(base.size() - 5);
        g_engine.model_name = base;
    }

    if (const char* ct = cne::env("CHAT_TEMPLATE")) g_engine.chat_template = ct;
    if (cne::env("THINK") && strcmp(cne::env("THINK"), "0") == 0)
        g_engine.think_default = false;

    // handlers read engine state through the long-lived g_engine view
    g_engine.regime_str       = rt->regime_str;
    g_engine.dense_policy_str = rt->dense_policy_str;
    g_engine.odirect          = rt->odirect;
    g_engine.streaming        = rs.stream_on;
    g_engine.cache_cap        = rt->cache_cap;
    g_engine.model            = rt->model;
    g_engine.ctx              = rt->ctx;
    g_engine.vocab            = rt->vocab;
    g_engine.n_ctx            = rt->n_ctx;



    if (const char* ct = cne::env("CHAT_TEMPLATE")) g_engine.chat_template = ct;
    if (cne::env("THINK") && strcmp(cne::env("THINK"), "0") == 0)
        g_engine.think_default = false;

    // handlers read engine state through the long-lived g_engine view
    g_engine.regime_str       = rt->regime_str;
    g_engine.dense_policy_str = rt->dense_policy_str;
    g_engine.odirect          = rt->odirect;
    g_engine.streaming        = rs.stream_on;
    g_engine.cache_cap        = rt->cache_cap;
    g_engine.model            = rt->model;
    g_engine.ctx              = rt->ctx;
    g_engine.vocab            = rt->vocab;
    g_engine.n_ctx            = rt->n_ctx;

    const int n_threads =
        cne::env("THREADS") ? atoi(cne::env("THREADS")) : 8;

    // All decodes happen on one dedicated thread: the context is created and
    // warmed here, and llama contexts must be driven by one thread.
    // CNE_MAIN_ENGINE=1 instead promotes the MAIN thread to engine duty
    // (bench-identical topology; bisect knob).
    const bool main_engine = cne::env("MAIN_ENGINE") != nullptr;
    std::thread engine_th;
    if (!main_engine) engine_th = std::thread(engine_thread_loop);

    // ---- HTTP surface ----

    httplib::Server svr;

    svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        set_json(res, json{
            {"status", "ok"},
            {"model", g_engine.model_name},
            {"regime", g_engine.regime_str},
            {"dense_policy", g_engine.dense_policy_str},
            {"odirect", g_engine.odirect},
            {"streaming", g_engine.streaming},
            {"mtp", g_engine.mtp_k > 0 ? json(g_engine.mtp_k) : json(nullptr)},
            {"thinking_default", g_engine.think_default},
            {"cache_cap_mib", g_engine.cache_cap >> 20},
            {"n_ctx", g_engine.n_ctx}});
    });

    svr.Get("/v1/models", [&](const httplib::Request&,
                              httplib::Response& res) {
        set_json(res, json{
            {"object", "list"},
            {"data", json::array({
                 { {"id", g_engine.model_name},
                   {"object", "model"},
                   {"created", 0},
                   {"owned_by", "cne"} },
             })}});
    });

    svr.Post("/v1/chat/completions",
             [&](const httplib::Request& req, httplib::Response& res) {
                 json body;
                 try {
                     body = json::parse(req.body);
                 } catch (...) {
                     res.status = 400;
                     set_json(res, error_body("invalid JSON body",
                                             "invalid_request_error"));
                     return;
                 }
                 if (!body.contains("messages") || !body["messages"].is_array() ||
                     body["messages"].empty()) {
                     res.status = 400;
                     set_json(res, error_body(
                         "'messages' must be a non-empty array",
                         "invalid_request_error"));
                     return;
                 }
                 handle_chat(res, body);
             });

    printf("cne-server: model=%s regime=%s dense=%s stream=%d mtp_k=%d "
           "think=%d ctx=%d threads=%d\n",
           g_engine.model_name.c_str(), g_engine.regime_str.c_str(),
           g_engine.dense_policy_str.c_str(), (int)g_engine.streaming,
           g_engine.mtp_k, (int)g_engine.think_default, g_engine.n_ctx,
           n_threads);

    // MTP WARNING - upstream #26425 (open): draft-mtp retains state across
    // generations inside one process; output degrades after the first
    // invocation(s) on Qwen3.6-class artifacts. Single-shot harnesses
    // (cne_bench) are unaffected because they make ONE call per process.
    // Default here is OFF; enable explicitly with CNE_MTP=k knowing the
    // trade. Post-generation memory clears below blunt the retention but do
    // not eliminate it.
    if (g_engine.mtp_k > 0)
        fprintf(stderr,
                "*** [mtp] CNE_MTP=%d requested. UPSTREAM BUG #26425 affects "
                "multi-request serving: output may degrade across requests "
                "(measured on this artifact). cne-bench remains the MTP "
                "measurement vehicle. ***\n",
                g_engine.mtp_k);

    printf("listening on http://%s:%d (single session; requests serialize)\n",
           host, port);
    fflush(stdout);

    // Bisect mode CNE_MAIN_ENGINE=1: HTTP on a side thread, engine loop on
    // the MAIN thread - mirrors cne-bench's thread topology exactly.
    if (cne::env("MAIN_ENGINE")) {
        if (!svr.bind_to_port(host, port)) {
            fprintf(stderr, "[cne-server] bind FAILED on %s:%d\n", host, port);
            return 1;
        }
        std::thread http_th([&] { svr.listen_after_bind(); });
        engine_thread_loop();   // main thread = engine thread
        {
            std::lock_guard<std::mutex> l(g_jobs_m);
            g_jobs_stop = true;
        }
        g_jobs_cv.notify_all();
        svr.stop();
        http_th.join();
    } else {
        if (!svr.listen(host, port)) {
            fprintf(stderr, "[cne-server] listen FAILED on %s:%d\n", host, port);
            return 1;
        }
    }

    cne::stream_prefetch_stop();
    {
        std::lock_guard<std::mutex> l(g_jobs_m);
        g_jobs_stop = true;
    }
    g_jobs_cv.notify_all();
    if (engine_th.joinable()) engine_th.join();
    cne::runtime_shutdown(*rt);
    llama_backend_free();
    return 0;
}
