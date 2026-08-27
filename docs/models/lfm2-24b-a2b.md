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
| **warm dense, t4, ctx 4096** | **~10.5–11.5** | `cne_server` wall clock; B3 on |
| warm dense, t4, llama-bench tg128 | **~8.6–11.5** | high session variance; see B3 § |
| warm dense, t4, 300 tok fixed | **10.84** | pre-B3 `cne_bench`; re-bench after fork pin |
| llama-bench tg128 (fork B3 p3) | **11.48 ± 0.46** | `cne/lfm2-b3` @ `8d2440243`, t4, pp512 warmup |
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

### B3 MoE kernel (2026-08-26, fork `cne/lfm2-b3`)

LFM2 routes **top-4** experts (`expert_used_count=4`). Early B3 work gated on
`n_ids==2` and did not apply. Phase 3 adds a `mul_mat_id` decode fast path for
top-2/top-4, single-token batches (`ne11==1`, `ne12==1`), Q4_K repacked weights.

B3 is **three mechanisms**, not one fused SIMD kernel:

| piece | what it does |
|---|---|
| Fast-path dispatch | Skips generic `mul_mat_id` row loop when top-K experts each have one row |
| q8 activation cache | Gate and up share `src1` on decode — second float→q8 quantize is skipped |
| `ggml_gemv_q4_K_8x8_q8_K_4vx` | Intended fused 4-expert GEMV |

On **x86 (Tiger Lake)**, `4vx`/`2vx` have **no native implementation** —
`arch-fallback.h` aliases them to `_generic`, which is four sequential calls to
the existing single-expert `ggml_gemv_q4_K_8x8_q8_K` (AVX-VNNI in
`arch/x86/repack.cpp`). Most measured B3 gain is therefore **q8 cache +
dispatch**, not true multi-expert SIMD fusion.

Toggle: `CNE_MOE_B3=1` (default) / `CNE_MOE_B3=0` (slow path, same tokens).

#### Measured velocity

| session | test | B3 on | B3 off | delta |
|---|---|---|---|---|
| 2026-08-26 (fork pin) | `llama-bench` tg128, single sweep | **11.48 ± 0.46** | 10.35 ± 0.20 | +10.9% |
| 2026-08-27 | `llama-bench` tg128, **5 runs/arm** | **8.61 ± 0.31** | 8.21 ± 0.24 | **+4.8%** |
| 2026-08-27 | `cne_server`, 1000 tok × 3 | **10.44 ± 0.11** | 10.50 ± 0.11 | ~0% (noise) |

The original +11% figure was a single tight session. Re-measurement shows
**~5% decode-only** on microbench and **no visible wall-clock gain** on live
server (HTTP + prefill dilute the kernel win). High run-to-run variance
(±15–20% cold; ±0.1–0.3 tok/s warm) — always sweep multiple runs.

Lossless: same greedy tokens (`cne_identity_gate`). Code:
`third_party/llama.cpp/ggml/src/ggml-cpu/repack.cpp` (~4483 q8 cache, ~4565
fast path). Chronicle: `internal-docs/LFM2_VELOCITY_RESEARCH.md`.

#### A/B reproduce

```sh
# microbench (decode-only; stop server first to avoid OOM contention)
for i in {1..5}; do ./bench/scripts/lfm2/tg128-microbench.sh; done
for i in {1..5}; do CNE_MOE_B3=0 ./bench/scripts/lfm2/tg128-microbench.sh; done
column -t bench/results/lfm2-tg128.history.tsv

# live server (restart with matching CNE_MOE_B3)
CNE_MOE_B3=1 CNE_STREAM=0 CNE_DENSE=warm CNE_THREADS=4 CNE_CTX=4096 \
  ./build/server/cne_server --config models/server.json
CNE_BENCH_MAX_TOKENS=1000 ./bench/scripts/lfm2/server-velocity.sh
```

### B4 gate‖up fusion (closed negative, 2026-08)

Prepare-time stacked `ffn_gate_up_exps` (38 fewer `MUL_MAT_ID`/token) passed
identity but measured **slower than B3** on tg128 (~13.9 vs ~14.3 tok/s warm).
Runtime fusion added ~8 GiB RSS → OOM on 16 GB. Reverted; LFM2 artifact stays
unfused (428 tensors: separate `ffn_gate_exps` + `ffn_up_exps`).

### Kernel roadmap (lossless, next levers)

Decode is ~89% Q4 `MUL_MAT` + `MUL_MAT_ID`. Closed or sub-5% elsewhere
(streaming, self-spec, `CNE_FA`, shortconv ~1%, MTP N/A). Ranked next work:

| priority | kernel | est. decode gain | status |
|---|---|---|---|
| 1 | **True x86 AVX-VNNI `4vx`** (one activation, 4 weight blocks) | 10–20% | not built — B3 calls generic 4× sequential today |
| 2 | **Down-proj `4vx`** (`ffn_down_exps`, 3rd MoE matmul/layer) | 5–10% | not built |
| 3 | **Router decode GEMV** (38 small `MUL_MAT`/token) | 3–8% | not built |

All require identity-gate PASS on `cne/lfm2-b3`. No lossy shortcut (`CNE_EXPERT_MASS`,
self-spec, IQ4_XS) is recommended for this profile unless quality trade is explicit.

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
| `CNE_MOE_B3` | unset (on) | B3 `mul_mat_id` fast path; set `0` for A/B |
| `CNE_FA` | off | no measurable gain at ctx 256–4096 on LFM2 |

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
