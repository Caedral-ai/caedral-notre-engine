# Qwen3.6-35B-A3B — first supported model (reference)

Facts below were observed on the actual artifacts, not copied from marketing
material.

Canonical runtime artifact (since 2026-08-23):
`models/qwen3.6-35b-a3b-q4_k_xl/Qwen3.6-35B-A3B-UD-Q4_K_XL-prepared.gguf`
(unsloth dynamic q4 family, expert tensors 4096-aligned by soe-prepare).

Original observation baseline was the upstream Q8_0 build
(`Qwen3.6-35B-A3B-Q8_0.gguf`, sha256 `d1a39580…2794a59`,
36,903,140,320 bytes); architecture facts carry over — quantization is
per-tensor metadata, and all streaming geometry derives from each loaded
artifact's own manifest.

## Identity

| Field | Value |
|---|---|
| HF base model | [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| GGUF source | [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) |
| File (runtime) | `Qwen3.6-35B-A3B-UD-Q4_K_XL-prepared.gguf` |
| Quantization | unsloth UD-Q4_K_XL dynamic (whole-model q4 family; canonical since the q4 profile pivot) |
| GGUF architecture key | `qwen35moe` |
| Params | 34.66 B (llama_model_n_params) |
| Layers | 40 |
| Vocab | 248,320 |
| Train context | 262,144 |

## Skeleton: HYBRID (not pure-attention)

Load-time observation (pinned llama.cpp v0.2.0):

- **10 of 40 layers use attention KV cache** (layers 3, 7, …, 39);
  KV @ ctx 1024 = 20 MiB (f16).
- **30 layers are Gated Delta Net recurrent** (`ssm_*` tensors,
  `llama_memory_recurrent`): R state 2.81 MiB + S state 60.00 MiB f32,
  fixed-size (does not grow with context).
- Extra graph node families: Lightning Indexer, chunked/autoregressive
  Gated Delta Net, DeepSeek-V4-HC fused ops.
- Consequences for the engine: a mandatory-resident set beyond dense
  (shared experts + ssm/attn + scales), shape-class warmup instead of
  tensor-data warmup, per-layer-type adaptive overlap window, an ASK-phase
  node allowlist, and TTFT as a first-class metric.

## Tensor taxonomy seen in the file

Per MoE layer (names as stored):

- Routed experts (fused, streamed by the engine): `blk.N.ffn_gate_exps`,
  `blk.N.ffn_down_exps`, `blk.N.ffn_up_exps`
- Shared expert (dense-like, never cached as expert slice):
  `blk.N.ffn_{gate,down,up}_shexp`
- Recurrent skeleton: `blk.N.ssm_{alpha,beta,out}` (+ `.scale`,
  `.input_scale` companions)
- Attention skeleton (10 layers): `attn_qkv` (fused), `attn_q/k/v/output`
  variants, `attn_gate`
- Companion scale tensors exist throughout (`.scale`, `.input_scale`) —
  classify as SCALE, resident like dense.

## Engine decisions bound to this model

- **CPU no-repack**: loaded with `use_extra_bufts = false` (D13). Verified on
  q8_0 (loader logged a CPU_REPACK fallback to plain CPU); applies to all
  quants in v1 regardless of type.
- Expert slices are consumed exactly as stored on disk (original tensor ids,
  full-size buffers).
- Vision `mmproj` files are out of scope (text-only v1).
