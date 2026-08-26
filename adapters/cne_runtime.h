#pragma once
// Shared boot sequence for cne binaries (bench, server): manifest, regime,
// budget clamp, cache/stream wiring, llama model/context. Drivers stay thin.
#include "cne/cache.h"
#include "cne/model.h"
#include "cne_stream_cb.h"

#include "llama.h"

#include <memory>
#include <string>

namespace cne {

struct RuntimeSettings {
    const char* model_path = nullptr;
    const char* draft_model_path = nullptr;   // self-spec subset-expert copy
    size_t cap_gib   = 8;     // requested expert-cache GiB (budget-clamped)
    int    n_ctx     = 1024;  // default; CNE_CTX overrides
    int    n_threads = 8;     // default; CNE_THREADS overrides
    bool   stream_on = true;  // demand-serving windows/fills on?
};

struct Runtime {
    // metadata / memory plane
    ModelManifest               manifest;
    std::unique_ptr<SliceCache> cache;
    size_t      cache_cap    = 0;   // post-clamp bytes
    std::string regime_str, dense_policy_str;
    bool        odirect   = false;
    bool        streaming = true;
    float       l2_mass   = 0.0f;  // expert-mass gating (0 = lossless)
    int         mtp_k     = 0;     // draft depth (0 = off)

    // llama plane (valid after runtime_load_llama)
    llama_model*       model = nullptr;
    llama_context*     ctx   = nullptr;
    const llama_vocab* vocab = nullptr;
    int                n_ctx = 0;

    // self-speculative drafter (subset-expert artifact copy), optional
    llama_model*       draft_model = nullptr;
    llama_context*     draft_ctx   = nullptr;
};

// Stages 1-2: manifest -> dense policy -> budget clamp -> cache/stream/O_DIRECT
// -> warm residency. Returns nullptr on failure (reason logged). Does not
// touch llama; testable against synthetic fixtures.
std::unique_ptr<Runtime> runtime_prepare(const RuntimeSettings& s);

// Stage 3: llama backend init, model load (load_mtp when prepared for it),
// context creation (cb_eval wired), warmup policy, anon-dense scan.
// Returns false on failure (reason logged).
bool runtime_load_llama(Runtime& rt, const RuntimeSettings& s);

// Prefetch worker stop + llama teardown.
void runtime_shutdown(Runtime& rt);

} // namespace cne
