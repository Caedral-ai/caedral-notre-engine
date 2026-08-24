#pragma once
// Draft-MTP speculative decoding: drafts k tokens per step with the model's
// native Multi-Token Prediction head (nextn tensors) and verifies them in one
// batched target forward. Accepted tokens are identical to non-speculative
// greedy decoding - the target always verifies over the full vocabulary.
//
// Requires a GGUF that preserves the nextn/MTP tensors (load_mtp = true at
// model load) and a context created with LLAMA_CONTEXT_TYPE_DEFAULT.
#include "llama.h"

#include <vector>

namespace cne {

struct SpecStats {
    long drafted  = 0;   // draft tokens proposed
    long accepted = 0;   // draft tokens verified and kept
    int  produced = 0;   // tokens emitted (excludes EOS)
    // Timing telemetry (seconds). draft = drafter forwards; process =
    // speculative state catch-up (also drafter-side work); verify = the
    // batched target forward that validates 1+k tokens per iteration.
    double draft_s   = 0;
    double process_s = 0;
    double verify_s  = 0;
    long   iterations = 0;
    long   partials    = 0;   // rounds with a rejected draft (checkpoint replay)
};

// Size the target context's output buffers for verify batches of up to
// n_max+1 logits rows. Call BEFORE creating the target context.
void spec_mtp_size_outputs(struct llama_context_params& cparams,
                           int n_max, int n_batch);

// Greedy draft-mtp generation loop.
//
//  - prompt: full tokenized prompt (special tokens included).
//  - n_max: maximum draft depth per iteration.
//  - p_min: minimum draft-token probability; shorter drafts on uncertain
//           steps (0 = always draft the full depth).
//  - n_gen: generation budget (produced stops at this count).
//  - cb: demand-serving eval callback, installed on BOTH the draft context
//        created here and expected on the caller's target context.
//  - on_token: invoked once per emitted token (never for EOS).
SpecStats spec_mtp_generate(llama_model* model,
                            llama_context* ctx,
                            const std::vector<llama_token>& prompt,
                            int n_max,
                            float p_min,
                            int n_gen,
                            ggml_backend_sched_eval_callback cb,
                            void (*on_token)(void* ud, llama_token id),
                            void* ud);

} // namespace cne
