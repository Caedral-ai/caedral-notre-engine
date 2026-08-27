#pragma once
// Demand-serving runtime: router-harvest callback, expert windows, slice
// fills, expert-mass gating, prefetch overlap, anon-dense binding.
//
// This is the engine runtime, extracted from the measurement driver.
// All state lives inside stream_cb.cpp under a single-decode assumption;
// the driver only configures, drives step boundaries, and reads telemetry.
#include "cne/cache.h"
#include "cne/model.h"

#include "llama.h"

#include <cstddef>
#include <cstdint>

namespace cne {

enum class PfMode { OFF, FULL, LOOKAHEAD };

struct StreamConfig {
    bool  rebind     = true;   // windows + fills + repoint (stream ON)
    bool  full_fill  = false;  // debug: fill whole windows at creation
    bool  step_fills = false;  // telemetry: per-step fill dump
    float l2_mass    = 0.0f;   // 0 = disabled (lossless full-k)
    int   l2_min_k   = 2;
    PfMode pf_mode   = PfMode::OFF;
};

// Binds manifest + cache; builds the dense-name set for anon binding.
void stream_init(const ModelManifest& manifest, SliceCache& cache,
                 const StreamConfig& cfg);

// Fill backend: opens the model file O_DIRECT when its layout allows and
// wires source (+ optional lane scheduler) into the cache. Returns true
// when O_DIRECT mode is active (memcpy-from-mmap otherwise).
bool stream_open_fill_backend(const char* model_path, int lanes);
bool stream_use_odirect();

// Callback to install in llama_context_params.cb_eval.
ggml_backend_sched_eval_callback stream_cb_eval();

// ANON dense policy: one observed decode binds every dense weight touched by
// src edges into an aligned anonymous copy. Begin before the scan decode,
// end after it (memory must be cleared by the driver).
void   stream_anon_scan_begin();
void   stream_anon_scan_end();
size_t stream_dense_bound_count();
size_t stream_dense_anon_bytes();

// Prefetch worker lifecycle (no-op unless pf_mode != OFF).
void stream_prefetch_start();
void stream_prefetch_stop();
void stream_prefetch_kick_full();   // FULL mode: speculate next step routing

// Generation-loop hooks.
void stream_set_step(long step);    // driver reports the current decode step
void stream_step_boundary();        // id-slot rotation + lookahead dedup reset

struct StreamTelemetry {
    double   fill_s;         // total time inside batch fills
    uint64_t fill_calls;
    long     audit_checks;
    size_t   audit_pending;
    long     l2_dropped;
    float    l2_mass;
    int      l2_min_k;
};
StreamTelemetry stream_telemetry();

// Full-window integrity walk (memcpy mode only): every expert slice must
// match the original mapping byte-for-byte. Prints findings to stderr.
// Compiled in only with -DCNE_AUDIT (CMake: -DCNE_AUDIT=ON); no-op otherwise.
void stream_check_windows();

} // namespace cne
