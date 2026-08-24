# Qwen3.6-35B-A3B — UD-Q4_K_XL + MTP (canonical runtime model)

> This is THE model the engine runs. Whole-model q4-family dynamic
> quantization with native Multi-Token Prediction layers preserved for
> lossless speculative decoding.
>
> STATUS: ✅ WORKING — verified on latest llama.cpp build, coherent output,
> streaming identity PASS within q4 profile.

---

## 1. Identity

| Field | Value |
|---|---|
| Source | [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) (MTP layers preserved) |
| Base model | [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| File | `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` |
| Quantization | Unsloth Dynamic Q4_K_XL (imatrix-calibrated; mixed q4_k/q5_k/q6_k/q8_0/f32 per-tensor) |
| Downloaded size | 22,853,663,008 bytes (~21.3 GiB) |
| GGUF architecture key | `qwen35moe` |
| Tensors | 753 (includes MTP prediction head) |
| Regeneration script | `tools/download-qwen3.6-35b-a3b-q4_k_xl.sh` |

## 2. Tensor-type census (prepared variant)

| type | count | notes |
|---|---|---|
| bf16 | 2 | precision-critical |
| f32 | 368 | norms, biases, precision-retained weights |
| q4_k | 80 | attention + routed-expert tensors |
| q5_k | 40 | attention + selected weights |
| q6_k | 3 | output.head family |
| q8_0 | 260 | embeddings + precision-retained weights |

Prepared variant: io_alignment=4096, all routed slices 4096-aligned,
types preserved byte-faithfully through cne-prepare.

## 3. Streaming geometry

- Expert slices uniform per tensor (span/n_experts), 4096-multiple after
  prepare → O_DIRECT-clean, no bounce buffers.
- Miss bytes per token ≈ half of the q8 profile at identical routing.

## 4. MTP speculative decoding

This artifact preserves the native Multi-Token Prediction head. When
enabled via llama.cpp's `load_mtp = true` + `ctx_type = LLAMA_CONTEXT_TYPE_MTP`,
the engine generates multiple tokens per forward pass:

- Community-reported speedup: 1.4–2.2× on Qwen3.6 family
- **Lossless by mathematical proof**: every draft token is verified against
  the full model; accepted tokens are identical to non-speculative inference
- Enable via env: `CNE_MTP=1` (legacy `SOE_MTP` accepted) in our bench

No quality trade-off needed — MTP gives speed without changing any weight or computation result.

---

## 5. Engine decisions bound to this artifact

- Same as the q8 reference doc (D13 no-repack, original ids, full-size
  windows, slice-level streaming) — the engine is quant-agnostic; all
  geometry comes from each loaded artifact's manifest.
- Quality gates for this profile run against the LOCKED lossless numbers
  and the saved drift tokens (models/eval/lossless_ref_64.toks).
