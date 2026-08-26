# LFM2-24B-A2B

Second supported artifact. Liquid AI's hybrid MoE — first non-Qwen
architecture validated on the engine.

## Identity

| property | value |
|---|---|
| Source | `LiquidAI/LFM2-24B-A2B-GGUF`, file `LFM2-24B-A2B-Q4_K_M.gguf` |
| sha256 | `eb4d2d4d4e61b795726c2f526c4434ca6bc725ad7a783691b58681f025cf58f2` |
| Total / active params | 24B / 2.3B |
| Architecture | `lfm2_moe`: 40 layers = 30 shortconv + 10 GQA attention |
| Context | 32,768 native |
| MTP | none — sequential arm only (`mtp_k=0`) |
| Template | ChatML-like via the model's own Jinja; no think blocks |
| Download | `./tools/download-lfm2-24b-a2b.sh` (fetch + align, resumable) |

## Validation (2026-08-25, i5-1135G7 / 16 GB)

- Alignment census: `routed=114 all_aligned=114 misaligned=0 -> OK`
- Demand-serving callback works unmodified on the conv-hybrid graph
  (generic `ffn_moe_argsort`/MUL_MAT_ID node matching)
- Serving smoke: prompt-conditioned answer, finish=stop, clean aborts

## Measured (2026-08-26, reference hardware)

Warm steady-state decode on the prepared artifact. LFM2 uses **4 threads**
on this chip (unlike Qwen, where 6 threads wins). Run-to-run variance
±~0.2 tok/s when warm; ±15–20% across cold sessions.

| config | tok/s | notes |
|---|---|---|
| **warm dense, t4, ctx 4096** | **~10.9–11.0** | recommended serving profile |
| warm dense, t4, 300 tok fixed | **10.84** | `CNE_IGNORE_EOS=1` bench-only knob |
| mmap cold, t4, first run | ~6.0 | majflt ~6k until page cache hot |
| stream t4 | ~1.5 | hit-rate 52% — below the 0.90 rule; streaming OFF |
| subset-expert self-spec | ~5 | closed negative; code removed |

`llama-bench` tg128 on the same machine reports **~9.3 tok/s** (mmap,
t4) — the engine's warm `CNE_DENSE=warm` path is ~15% faster by
eliminating decode-time page faults on this ~1.4× RAM ratio.

Prefill (llama-bench, t4): **~41–44 tok/s** at pp512–4096 — roughly
**4× decode**, not the older ~2× anecdote.

### Where decode time goes (profiling, 2026-08-26)

One decode forward is dominated by Q4 matrix ops, not shortconv:

| op bucket | graph nodes | est. wall time |
|---|---|---|
| `MUL_MAT` + `MUL_MAT_ID` (dense + MoE GEMV) | ~89% of heavy ops | **dominant** |
| `SSM_CONV` (30 shortconv layers) | ~10% of nodes | **~1%** (microbench) |
| attention / norms / routing | remainder | small |

Build audit: CNE already ships `-march=native`, OpenMP, and
`GGML_USE_CPU_REPACK` (~5% vs repack off). No further cmake flag win
measured on Tiger Lake.

## Subset-expert self-spec (spike closed, removed)

A 2026-08 V1 spike tried training-free subset-expert drafting for LFM2 (no
MTP head): a drafter with fewer active experts, full-model greedy verify.
Identity gate passed; velocity gate failed — **~5 tok/s vs ~11 warm
sequential** (`CNE_STREAM=0`, 4 threads). Native top-K is already short;
shrinking K' cannot amortize draft+verify overhead on CPU.

The experiment code and probe GGUF copies were removed from the tree
(2026-08-26). LFM2 stays on naive sequential decode. Chronicle:
`internal-docs/VELOCITY_II.md`.

## Operating profile (owner ruling: NO-STREAM)

Recommended lossless serving config:

```sh
CNE_STREAM=0 CNE_DENSE=warm CNE_THREADS=4 CNE_CTX=4096 \
  ./build/server/cne_server \
  models/lfm2-24b-a2b/LFM2-24B-A2B-Q4_K_M-prepared.gguf
```

| knob | value | why |
|---|---|---|
| `CNE_STREAM=0` | no-stream | expert-cache hit-rate 52% → net slower than naive |
| `CNE_DENSE=warm` | dense preload | 0 majflt during decode; ~543 MiB paged in at boot |
| `CNE_THREADS=4` | compute | best warm tok/s on this chip for LFM2 |
| `CNE_CTX=4096` | context | 1024+ for long sessions; bench default 256 is too small |
| `CNE_MTP` | unset | artifact has no MTP / nextn tensors |
| `CNE_FA` | off | no measurable gain at ctx 256–1024 on LFM2 |

`CNE_DENSE=mmap` ties ~11 tok/s when the page cache is already hot and
uses ~0.5 GiB less RSS; use mmap if RSS is tight and cold-start fault
storms are acceptable.

### Reproduce (bench)

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# steady-state throughput (fixed 300 tokens; bench-only EOS override)
CNE_STREAM=0 CNE_DENSE=warm CNE_THREADS=4 CNE_CTX=4096 CNE_IGNORE_EOS=1 \
  ./build/tools/cne_bench \
  models/lfm2-24b-a2b/LFM2-24B-A2B-Q4_K_M-prepared.gguf \
  0 300 32 0
```

Auto-policy note: the classifier picked `R1_NEAR_LIMIT dense=anon` and it
boots and serves correctly. No MTP warning applies. Think control gate is
qwen3-specific and correctly inert here.

See also [BENCHMARKS.md](../BENCHMARKS.md) § LFM2.
