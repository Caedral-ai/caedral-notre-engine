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
| Template | ChatML-like via the model's own Jinja; **no think blocks** |
| Thinking | **None** — no think blocks; `"think"` in `server.json` has no effect on generation |
| Download | `./tools/scripts/download-lfm2-24b-a2b.sh` (fetch + align, resumable) |

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
| **warm dense, t4, ctx 4096** | **~10.5–11.5** | `cne_server` wall clock; `CNE_KERNELS=1` |
| warm dense, t4, llama-bench tg250 | **9.50 ± 0.17** | kernels on, 5-run mean (2026-08-27) |
| warm dense, t4, llama-bench tg128 | **~8.6–11.5** | high session variance; see kernels § |
| warm dense, t4, 300 tok fixed | **10.84** | pre-B3 `cne_bench`; re-bench after fork pin |
| llama-bench tg128 (custom kernels) | **11.48 ± 0.46** | `cne/cpu-kernels` @ `8d2440243`, t4, pp512 warmup |
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

### Custom kernels (`CNE_KERNELS`, fork `cne/cpu-kernels`)

LFM2 routes **top-4** experts (`expert_used_count=4`). The fork adds a
`mul_mat_id` decode fast path for top-2/top-4, single-token batches
(`ne11==1`, `ne12==1`), repacked Q4_K gate/up and Q6_K down weights.

All fork hooks share one runtime toggle:

| `CNE_KERNELS` | behavior |
|---|---|
| `1` (default) | q8 activation cache + MoE dispatch + fused q4/q6 `2vx`/`4vx` GEMV |
| `0` | stock llama.cpp `mul_mat_id` path (same greedy tokens) |

| piece | what it does |
|---|---|
| q8 activation cache | Gate and up share `src1` on decode — second float→q8 quantize skipped |
| Fast-path dispatch | Skips generic `mul_mat_id` row loop when top-K experts each have one row |
| `ggml_gemv_q4_K_8x8_q8_K_4vx` | Fused AVX2 gate/up GEMV (`arch/x86/repack_mmid.inl`) |
| `ggml_gemv_q6_K_8x8_q8_K_4vx` | Fused AVX2 expert-down GEMV (`arch/x86/repack_q6k.inl`) |

Toggle: `CNE_KERNELS=1` (default) / `CNE_KERNELS=0` (stock llama.cpp), or
`"kernels": true/false` in `models/server.json` (via `cne_setup`).

#### Measured velocity

| session | test | kernels on | kernels off | delta |
|---|---|---|---|---|
| 2026-08-27 | `llama-bench` tg250, **5 runs/arm** | **9.50 ± 0.17** | 8.59 ± 0.15 | **+10.6%** |
| 2026-08-27 | `llama-bench` tg128, **5 runs/arm** | **8.61 ± 0.31** | 8.21 ± 0.24 | +4.8% |
| 2026-08-26 | `llama-bench` tg128, single sweep | 11.48 ± 0.46 | 10.35 ± 0.20 | +10.9% |
| 2026-08-27 | `cne_server`, 1000 tok × 3 | 10.44 ± 0.11 | 10.50 ± 0.11 | ~0% (noise) |

Use **tg250 5-run sweeps** as the primary kernel A/B probe; tg128 has higher
variance. Live server wall clock dilutes the kernel win (HTTP + prefill).

Lossless: same greedy tokens (`cne_identity_gate`). Code:
`third_party/llama.cpp/ggml/src/ggml-cpu/repack.cpp` (~4483 q8 cache, ~4565
fast path). Chronicle: `internal-docs/chronicles/LFM2_VELOCITY_RESEARCH.md` (history);
active kernel playbook: `internal-docs/kernels/CPU_KERNELS.md`.

#### A/B reproduce

```sh
# tg128 microbench (decode-only; stop server first to avoid OOM contention)
for i in {1..5}; do ./bench/scripts/lfm2/tg128-microbench.sh; done
for i in {1..5}; do CNE_KERNELS=0 ./bench/scripts/lfm2/tg128-microbench.sh; done
column -t bench/results/lfm2-tg128.history.tsv

# tg250 sweep (5 runs/arm; rebuild llama-bench from fork first)
cmake --build /tmp/llama-bench-build --target llama-bench -j
for k in 1 0; do for i in {1..5}; do
  CNE_KERNELS=$k /tmp/llama-bench-build/bin/llama-bench \
    -m models/lfm2-24b-a2b/LFM2-24B-A2B-Q4_K_M-prepared.gguf \
    -t 4 -p 0 -n 250 -r 1 -b 512 -ub 512
done; done

# live server (restart with matching CNE_KERNELS)
CNE_KERNELS=1 CNE_STREAM=0 CNE_DENSE=warm CNE_THREADS=4 CNE_CTX=4096 \
  ./build/server/cne_server --config models/server.json
CNE_BENCH_MAX_TOKENS=1000 ./bench/scripts/lfm2/server-velocity.sh
```

#### Compatibility

Two levels: whether the **fork path runs at all**, and whether you get the
**full fused AVX2 SIMD** (what the +10.6% tg250 number measures).

**Path does not run** (stock llama `mul_mat_id` — same tokens, no speedup):

| condition | why |
|---|---|
| `CNE_KERNELS=0` | explicit off |
| upstream llama.cpp (not fork `cne/cpu-kernels`) | code not present |
| GPU backend (CUDA / Metal / Vulkan) | hooks are `ggml-cpu` only |
| prefill or batched decode (`ne11>1` or `ne12>1`) | fast path is single-token decode only |
| MoE top-K not 2 or 4 | LFM2 uses top-4; other K values miss dispatch |
| weights not repacked q4_K / q6_K (`GGML_CPU_REPACK` off) | needs `block_q4_K` / `block_q6_K`, 8×8 repack layout |
| other models / graphs | MoE top-K not 2/4, fused gate‖up, or non-repacked quants |

**Other likely-compatible models** (same `mul_mat_id` + top-4 MoE pattern; **not
bench'd in CNE yet** — run `cne_identity_gate` before trusting velocity):

| model | arch | top-K | notes |
|---|---|---|---|
| **LFM2-8B-A1B** | `lfm2moe` | 4 | Same family as 24B-A2B; [LiquidAI/LFM2-8B-A1B-GGUF](https://huggingface.co/LiquidAI/LFM2-8B-A1B-GGUF) |
| **LFM2.5-8B-A1B** | `lfm2moe` | 4 | Successor checkpoint; **4 GiB `dense=anon`** or 16 GiB mmap profile — **docs/models/lfm2.5-8b-a1b.md** |
| **SmallThinker** (1B / 4B / 20B MoE) | `smallthinker` | 4 | Separate gate/up/down experts; no shortconv — MoE hooks should match |

Models with default top-K **8+** (e.g. Qwen3.6 MoE) do **not** hit the fast path
unless metadata is overridden to 2 or 4 (changes routing — not recommended for
lossless claims).

**Path runs, partial speedup** (dispatch + q8 cache; fused `4vx` falls back to
sequential single-expert GEMV):

| machine / build | notes |
|---|---|
| x86 without AVX2 (pre-Haswell, or portable `-march` build) | compiles, no fused SIMD |
| ARM / Apple Silicon (M-series, etc.) | no `arch/x86/repack_mmid.inl` / `repack_q6k.inl` |
| RISC-V, POWER, `GGML_CPU_GENERIC` | generic fallbacks only |

**Full speed** (reference: **+10.6%** tg250, i5-1135G7):

| requirement |
|---|
| x86_64 + **AVX2** (Intel Haswell 2013+, AMD Zen+, Tiger Lake class) |
| CNE fork `cne/cpu-kernels`, `GGML_CPU_REPACK=ON`, `CNE_KERNELS=1` |
| LFM2 prepared Q4_K_M, **CPU decode**, single token, top-4 MoE |

```
CNE_KERNELS=0              → always stock llama
CNE_KERNELS=1 + GPU        → stock (GPU matmul path)
CNE_KERNELS=1 + prefill    → stock mul_mat_id loop
CNE_KERNELS=1 + LFM2 decode + x86 AVX2 → full custom kernels
CNE_KERNELS=1 + LFM2 decode + ARM/old x86 → partial (dispatch/cache only)
```

### B4 gate‖up fusion (closed negative, 2026-08)

Prepare-time stacked `ffn_gate_up_exps` (38 fewer `MUL_MAT_ID`/token) passed
identity but measured **slower than B3** on tg128 (~13.9 vs ~14.3 tok/s warm).
Runtime fusion added ~8 GiB RSS → OOM on 16 GB. Reverted; LFM2 artifact stays
unfused (428 tensors: separate `ffn_gate_exps` + `ffn_up_exps`).

### Kernel roadmap (lossless, next levers)

Per-op census (`cne_graph_census`, 2026-08-27): MoE gate/up **39%**, MoE down
**27%**, shortconv **14%**, router **1%**. Custom kernels (q4+q6 fused GEMV)
are **shipped** under `CNE_KERNELS`. Ranked next work:

| priority | kernel | est. decode bucket | status |
|---|---|---|---|
| 1 | **Shortconv decode `MUL_MAT`** | ~4% wall (post-MoE) | **partial** — `mul_mat_decode.inl` skips chunk pool |
| 2 | **AVX-VNNI inner loop** for q4 `4vx` | incremental on shipped path | backlog |
| 3 | **Router decode GEMV** | 1.1% wall | deprioritized |

All require identity-gate PASS on `cne/cpu-kernels`. No lossy shortcut (`CNE_EXPERT_MASS`,
self-spec, IQ4_XS) is recommended for this profile unless quality trade is explicit.

## Subset-expert self-spec (spike closed, removed)

A 2026-08 V1 spike tried training-free subset-expert drafting for LFM2 (no
MTP head): a drafter with fewer active experts, full-model greedy verify.
Identity gate passed; velocity gate failed — **~5 tok/s vs ~11 warm
sequential** (`CNE_STREAM=0`, 4 threads). Native top-K is already short;
shrinking K' cannot amortize draft+verify overhead on CPU.

The experiment code and probe GGUF copies were removed from the tree
(2026-08-26). LFM2 stays on naive sequential decode. Chronicle:
`internal-docs/kernels/VELOCITY_II.md`.

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
| `kernels` / `CNE_KERNELS` | on (`true` in `server.json`) | fork MoE + decode matmul kernels (+12.3% tg250); `0`/`false` for stock A/B |
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

### Live integration tests

Default GGUF for `tests/e2e/*.json`. After download:

```sh
ctest --test-dir build -R 'server_e2e_live|session_kv_live' --output-on-failure
```

`session_bigctx_live` (~8 min) exercises 7k-token prefill at ctx=8192.
Details: [TESTING.md](../TESTING.md).

See also [BENCHMARKS.md](../BENCHMARKS.md) § LFM2.
