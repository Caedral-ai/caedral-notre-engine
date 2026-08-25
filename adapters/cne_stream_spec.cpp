// Draft-MTP speculative generation loop. Mirrors upstream's
// speculative-simple example; every accepted token is verified by the full
// model, so output is lossless with respect to the target's own greedy path.
#include "cne_stream_spec.h"

#include "cne_stream_cb.h"

#include "common.h"
#include "sampling.h"
#include "speculative.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <vector>

namespace cne {

void spec_mtp_size_outputs(llama_context_params& cparams, int n_max, int n_batch) {
    auto lim = common_speculative_get_output_limits(n_batch, 1, n_max);
    cparams.n_outputs_max         = (uint32_t)lim.total;
    cparams.n_outputs_max_per_seq = (uint32_t)lim.per_seq;
}

SpecStats spec_mtp_generate(llama_model* model,
                            llama_context* ctx,
                            const std::vector<llama_token>& prompt,
                            int n_max,
                            float p_min,
                            int n_gen,
                            ggml_backend_sched_eval_callback cb,
                            void (*on_token)(void*, llama_token),
                            void* ud,
                            int (*poll_stop)(void*)) {
    SpecStats stats;
    if (prompt.empty() || n_gen <= 0)
        return stats;

    common_init();

    common_params params;
    params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
    params.speculative.draft.n_max = n_max;
    params.speculative.draft.p_min = p_min;
    params.sampling.temp = 0.0f;   // greedy acceptance
    params.cb_eval = cb;           // demand-serving in the draft graph too
    // Both contexts must see the SAME weight buffers: a draft context with
    // extra buffer types would repack the shared quantized weights and
    // silently switch kernels for the target as well.
    params.no_extra_bufts = true;

    auto spec_init = common_speculative_init_from_params(params, model, ctx);
    if (!spec_init || !spec_init->context()) {
        fprintf(stderr, "\n[cne] MTP DRAFT CONTEXT FAILED\n");
        return stats;
    }
    params.speculative.draft.ctx_tgt = ctx;
    params.speculative.draft.ctx_dft = spec_init->context();
    llama_context* ctx_dft = params.speculative.draft.ctx_dft;

    // common_context_can_seq_rm is destructive (clears memory, decodes dummy
    // tokens). Run it before prefill, once per process - calling it after the
    // prompt is primed erases conditioning.
    static bool  ckpt_probed  = false;
    static bool  ckpt_use_tgt = false;
    static bool  ckpt_use_dft = false;
    if (!ckpt_probed) {
        ckpt_use_tgt =
            common_context_can_seq_rm(ctx) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        ckpt_use_dft = common_context_can_seq_rm(ctx_dft) ==
                       COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        ckpt_probed = true;
    }
    const bool use_ckpt_tgt = ckpt_use_tgt;
    const bool use_ckpt_dft = ckpt_use_dft;
    common_prompt_checkpoint ckpt;

    std::unique_ptr<common_speculative, decltype(&common_speculative_free)>
        spec(common_speculative_init(params.speculative, 1),
             &common_speculative_free);
    if (!spec) {
        fprintf(stderr, "\n[cne] SPECULATOR INIT FAILED\n");
        return stats;
    }

    const auto* vocab = llama_model_get_vocab(model);

    // Prompt through the target (all but last token), then hand the state to
    // the speculator. The last token joins the first verify batch instead -
    // its logits are what produce the first sampled continuation.
    // Chunked prefill: batches are capped at n_batch (a longer single
    // batch aborts inside llama_decode); the drafter processes each chunk
    // so its state stays in lockstep with the target.
    {
        const int n_batch = llama_n_batch(ctx);
        for (size_t off = 0; off + 1 < prompt.size(); off += n_batch) {
            const size_t len =
                std::min<size_t>(n_batch, prompt.size() - 1 - off);
            llama_batch chunk = llama_batch_init((int)len, 0, 1);
            for (size_t j = 0; j < len; j++)
                common_batch_add(chunk, prompt[off + j], (llama_pos)(off + j),
                                 { 0 }, false);
            if (llama_decode(ctx, chunk)) {
                fprintf(stderr, "\n[cne] PREFILL FAILED (mtp) at offset %zu\n",
                        off);
                llama_batch_free(chunk);
                return stats;
            }
            if (!common_speculative_process(spec.get(), chunk)) {
                fprintf(stderr, "\n[cne] SPEC PROMPT PROCESS FAILED\n");
                llama_batch_free(chunk);
                return stats;
            }
            llama_batch_free(chunk);
        }
    }

    llama_token id_last = prompt.back();
    std::vector<llama_token> prompt_tgt(prompt.begin(), prompt.end() - 1);
    int n_past = (int) prompt.size() - 1;

    common_speculative_begin(spec.get(), 0, prompt_tgt);

    common_sampler_ptr smpl(common_sampler_init(model, params.sampling));
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx), 0, 1);
    std::vector<llama_token> draft;
    long step = 0;

    while (stats.produced < n_gen) {
        if (poll_stop && poll_stop(ud)) break;
        if (draft.empty()) {
            ckpt.update_pos(
                    prompt_tgt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx), 0),
                    llama_memory_seq_pos_max(llama_get_memory(ctx), 0));
            if (use_ckpt_dft)
                ckpt.update_dft(ctx_dft, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

            int n_draft_max = (int) llama_n_ctx(ctx) - n_past - 2;
            n_draft_max = std::min(n_draft_max, n_gen - stats.produced - 1);
            n_draft_max = std::max(n_draft_max, 0);

            common_speculative_get_draft_params(spec.get(), 0) = {
                /* .drafting = */ true,
                /* .n_max    = */ n_draft_max,
                /* .n_past   = */ n_past,
                /* .id_last  = */ id_last,
                /* .prompt   = */ &prompt_tgt,
                /* .result   = */ &draft,
            };
            auto t_draft0 = std::chrono::steady_clock::now();
            common_speculative_draft(spec.get());
            stats.draft_s += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_draft0).count();
            if (getenv("CNE_MTP_DEBUG"))
                fprintf(stderr, "[mtp-dbg] drafted %zu tokens this call\n",
                        draft.size());

            if (!draft.empty() && use_ckpt_tgt)
                ckpt.update_tgt(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

            if (ctx_dft) {
                if (use_ckpt_dft)
                    ckpt.load_dft(ctx_dft, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                llama_memory_seq_rm(llama_get_memory(ctx_dft), 0,
                                    ckpt.pos_max + 1, -1);
            }
        }

        common_batch_clear(batch_tgt);
        common_batch_add(batch_tgt, id_last, n_past++, { 0 }, true);
        for (size_t i = 0; i < draft.size(); ++i)
            common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
        if (getenv("CNE_MTP_DEBUG"))
            fprintf(stderr, "[mtp-dbg] verify %zu draft rows\n", draft.size());

        auto t_verify0 = std::chrono::steady_clock::now();
        if (llama_decode(ctx, batch_tgt)) {
            fprintf(stderr, "\n[cne] DECODE FAILED (mtp verify)\n");
            break;
        }
        stats.verify_s += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_verify0).count();
        stats.iterations++;

        auto t_proc0 = std::chrono::steady_clock::now();
        if (!common_speculative_process(spec.get(), batch_tgt)) {
            fprintf(stderr, "\n[cne] SPEC PROCESS FAILED\n");
            break;
        }
        stats.process_s += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_proc0).count();

        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt)
            smpl_save.reset(common_sampler_clone(smpl.get()));

        const size_t n_draft = draft.size();
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx, draft);
        GGML_ASSERT(!ids.empty());

        // Partial acceptance: restore checkpoints so unverified tail tokens
        // leave both memories, then replay them as next iteration's draft.
        if (use_ckpt_tgt && ids.size() - 1 < n_draft) {
            draft = std::move(ids);
            ckpt.load_tgt(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(ctx), 0, ckpt.pos_max + 1, -1);
            if (ctx_dft) {
                ckpt.load_dft(ctx_dft, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                llama_memory_seq_rm(llama_get_memory(ctx_dft), 0,
                                    ckpt.pos_max + 1, -1);
            }
            prompt_tgt.resize(ckpt.n_tokens);
            smpl = std::move(smpl_save);
            n_past = (int) prompt_tgt.size();
            continue;
        }

        common_speculative_accept(spec.get(), 0, ids.size() - 1);
        n_past         += (int) ids.size() - 1;
        stats.drafted  += (long) n_draft;
        stats.accepted += (long) ids.size() - 1;

        bool stop = false;
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];
            step++;
            stream_set_step(step);
            if (llama_vocab_is_eog(vocab, id_last)) { stop = true; break; }
            if (getenv("CNE_DEBUG_TOKIDS")) fprintf(stderr, "%d ", id_last);
            if (on_token)
                on_token(ud, id_last);
            stats.produced++;
        }
        if (stop) break;

        // Trim rejected draft tokens from BOTH memories: stale positions
        // poison every subsequent decode (recurrent layers included).
        draft.clear();
        llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past, -1);
        if (ctx_dft)
            llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, n_past, -1);
    }
    llama_batch_free(batch_tgt);
    return stats;
}

} // namespace cne
