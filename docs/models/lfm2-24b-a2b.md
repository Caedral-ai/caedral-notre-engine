# LFM2-24B-A2B

Second supported artifact. Liquid AI's hybrid MoE - first non-Qwen
architecture validated on the engine.

## Identity

| property | value |
|---|---|
| Source | `LiquidAI/LFM2-24B-A2B-GGUF`, file `LFM2-24B-A2B-Q4_K_M.gguf` |
| sha256 | `eb4d2d4d4e61b795726c2f526c4434ca6bc725ad7a783691b58681f025cf58f2` |
| Total / active params | 24B / 2.3B |
| Architecture | `lfm2_moe`: 40 layers = 30 conv + 10 attention (NOT GDN like Qwen3.6) |
| Context | 32,768 native |
| MTP | none - sequential arm only (`mtp_k=0`) |
| Template | ChatML-like via the model's own Jinja; no think blocks |
| Download | `./tools/download-lfm2-24b-a2b.sh` (fetch + align, resumable) |

## Validation (2026-08-25, i5-1135G7 / 16 GB)

- Alignment census: `routed=114 all_aligned=114 misaligned=0 -> OK`
- Demand-serving callback works unmodified on the conv-hybrid graph
  (generic `ffn_moe_argsort`/MUL_MAT_ID node matching)
- Serving smoke: prompt-conditioned answer, finish=stop, clean aborts

## Measured

| config | tok/s | notes |
|---|---|---|
| naive t4 | **2.71** | best config; t6 2.61, callback-off identical (2.63) |
| naive t8 | 1.31 cold / ~2 warm | first-touch page faults dominate when cold |
| stream t4 | 1.51 | hit-rate 52% -> below the 0.90 rule, streaming OFF |

majflt ~20k/run in naive mode: hot set (~9.6 GiB touched of the 13.4 GiB
file) exceeds MemAvailable (~10.4 GiB), so the page cache re-reads across
runs. Expected at this ratio; streaming does not pay until ~1.6x RAM.

## Subset-expert self-spec (spike closed, removed)

A 2026-08 V1 spike tried training-free subset-expert drafting for LFM2 (no
MTP head): a drafter with fewer active experts, full-model greedy verify.
Identity gate passed; velocity gate failed — **~5 tok/s vs ~11 naive**
(warm, `CNE_STREAM=0`, 4 threads). Native top-K is already short; shrinking
K' cannot amortize draft+verify overhead on CPU.

The experiment code and probe GGUF copies were removed from the tree
(2026-08-26). LFM2 stays on naive sequential decode. Chronicle:
`internal-docs/VELOCITY_II.md`.

## Operating profile (owner ruling: NO-STREAM)

```sh
CNE_STREAM=0 CNE_THREADS=4 ./build/server/cne_server \
    models/lfm2-24b-a2b/LFM2-24B-A2B-Q4_K_M-prepared.gguf
```

Auto-policy note: the classifier picked `R1_NEAR_LIMIT dense=anon` and it
boots and serves correctly. No MTP warning applies. Think control gate is
qwen3-specific and correctly inert here.
