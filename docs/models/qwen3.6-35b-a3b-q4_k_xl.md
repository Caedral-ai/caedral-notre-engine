# Qwen3.6-35B-A3B — UD-Q4_K_XL (lossy velocity profile)

> Companion to `qwen3.6-35b-a3b.md` (lossless q8 reference). This artifact is
> the canonical RUNTIME model since the 2026-08-23 pivot: whole-model q4-family
> dynamic quantization, accepted bounded quality loss for ~half the streaming
> bytes per cache miss.
>
> STATUS: ⚠️ OPEN ISSUE — stock llama.cpp (v0.2.0 pinned, CPU) produces NaN
> logits on this quant (see section 4). Do not use for quality gates until
> resolved.

---

## 1. Identity

| Field | Value |
|---|---|
| Source | [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) |
| File | `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` |
| Quantization | Unsloth Dynamic Q4_K_XL (imatrix-calibrated; mixed q4_k/q5_k/q6_k/q8_0/f32 per-tensor) |
| Downloaded size | 22,974,495,712 bytes (~21.4 GiB) |
| GGUF architecture key | `qwen35moe` (same as q8 reference) |
| Regeneration script | `tools/download-q4.sh` (fetch + soe-prepare align) |

## 2. Tensor-type census (observed via registry, both raw and prepared)

| type | count | notes |
|---|---|---|
| f32 | 361 | norms, biases, precision-critical weights |
| q4_k | 78 | attention + routed-expert tensors |
| q5_k | 38 | attention + selected weights |
| q6_k | 4 | output.head family |
| q8_0 | 252 | embeddings + precision-retained weights |

- Routed experts are **Q4_K**: expert slice = 720,896 bytes
  (0.647× the q8 slice — the L1/I-O lever).
- Shared experts (`ffn_*_shexp`) retained at higher precision than routed.
- Prepared variant: identical type distribution, io_alignment=4096,
  misaligned4096=0 — soe-prepare preserved types byte-faithfully.

## 3. Streaming geometry

- Expert slices uniform per tensor (span/n_experts), 4096-multiple after
  prepare → O_DIRECT-clean, no bounce buffers.
- Miss bytes per token ≈ half of the q8 profile at identical routing.

## 4. OPEN ISSUE — NaN in stock inference

Stock `llama-perplexity` (v0.2.0 pinned build, CPU, default settings)
produces **NaN perplexity from chunk 1** on both the raw download and the
aligned prepared file. Our own engine reproduces the failure identically
(all-zero greedy tokens, pure-mmap arm included) — i.e., the issue is
UPSTREAM of the streaming path.

Ruled out:
- soe-prepare corruption (raw download also NaNs; type census identical).
- Our callback/policy flow (rebind=0 pure-mmap arm also NaNs).

Remaining hypotheses:
1. v0.2.0 CPU kernel bug with one of the Q4_K_XL tensor-type/layout
   combinations (unsloth dynamic quants target newer llama.cpp releases).
2. A tensor-type in this quant requires runtime support absent from the
   pinned build.

Next steps: bisect by tensor family (build a hybrid artifact keeping
attention at q8, experts at q4_k) or test the same quant on a newer
llama.cpp release. If neither resolves: switch profile to MXFP4_MOE or
UD-Q4_K_M and repeat the census.

---

## 5. Engine decisions bound to this artifact

- Same as the q8 reference doc (D13 no-repack, original ids, full-size
  windows, slice-level streaming) — the engine is quant-agnostic; all
  geometry comes from each loaded artifact's manifest.
- Quality gates for this profile run against the LOCKED lossless numbers
  (PTB-16: 13.3874 ±0.62 standalone / 16.9666 engine-ppl protocol) and the
  saved drift tokens (models/eval/lossless_ref_64.toks).
