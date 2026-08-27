# Benchmarks: reference hardware, measured gains, reproducible configs

All numbers below are real measurements taken with `cne_bench` on the
reference machine described here. Reproduction commands are included for
every row. Run-to-run variance is ±15–20% between sessions (desktop noise);
the robust signals are major-fault counts and cache hit-rate.

## Reference hardware

| item | value |
|---|---|
| CPU | Intel Core i5-1135G7 (Tiger Lake, 4C/8T) |
| ISA | AVX2, AVX-512, AVX-512 VNNI, FMA, F16C (`-march=native` build) |
| RAM | 16 GB DDR4 |
| Storage | NVMe SSD |
| OS | Linux, CPU-only inference (no GPU anywhere in the pipeline) |
| Isolation | all runs inside `systemd-run --user --scope -p MemoryMax=14G` |

Model-to-RAM ratio ≈ 1.4× — regime R2 boundary. At this ratio the engine's
streaming/cache machinery is at its *friendliest* operating point; the same
artifact runs on machines with far less RAM via R3/R4 streaming.

## Reference model

| item | value |
|---|---|
| Model | Qwen3.6-35B-A3B (MoE, ~3B active/token, hybrid GDN/attention) |
| Artifact | unsloth UD-Q4_K_XL dynamic 4-bit GGUF, **MTP layers preserved** |
| Size | 22.9 GB download → 21.3 GiB after `cne_prepare` alignment |

```sh
./tools/download-qwen3.6-35b-a3b-q4_k_xl.sh   # fetch + sha256 verify + prepare
```

## Measured velocity gains

500-token generations, greedy, context 1024:

| config | tok/s | gain vs previous step |
|---|---|---|
| q4_K_XL artifact, naive mmap, 8 threads (stock config) | 2.19 | — |
| same, **6 threads** (physical cores beat SMT) | **4.00** | **+82%** |
| same + draft-MTP k=8 p_min=0.5 | **4.86** | **+20%** |
| same, streaming ON (expert cache 93.9% hit-rate) | 4.52 | −2.4% = within noise |

Cumulative: **~2.2× over the stock configuration — all with bit-exact
lossless output.**

### What each gain is made of

- **Thread tuning (+82%)**: the default "use every logical core" strategy
  costs ~2× on this chip — two SMT siblings fight over one FP unit. The
  engine's sweep found 6 threads optimal; `CNE_THREADS=6`.
- **Draft-MTP speculation (+20%)**: the model's native Multi-Token
  Prediction head drafts up to 8 tokens per step; the full model verifies
  every one over the whole vocabulary. **Accepted output is exactly what
  plain greedy decoding produces — 0% quality loss by construction.**
  The confidence floor (`CNE_MTP_P_MIN=0.5`) eliminates replay rounds.
- **Streaming at this ratio**: cost-free stability — bounded RSS, no fault
  storms — at a noise-level velocity delta. Its velocity case opens as the
  model/RAM ratio grows.

## Quality gates behind these numbers

Every configuration above passes:

- **Identity gate**: generation with streaming ON vs OFF is token-exact
  identical (greedy); re-verified after every engine change.
- **Perplexity harness**: locked PTB protocol; lossless reference
  PPL 13.39 ±0.62.
- **Memory cap**: steady-state RSS inside the enforced budget, verified
  under `MemoryMax` isolation.

Speculation telemetry for the headline config: 100% acceptance, zero
partial rounds across all published runs.

## Reproduce

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./tools/download-qwen3.6-35b-a3b-q4_k_xl.sh

# baseline: naive mmap
CNE_THREADS=6 ./build/tools/cne_bench \
    models/qwen3.6-35b-a3b-q4_k_xl-mtp/Qwen3.6-35B-A3B-UD-Q4_K_XL-prepared.gguf \
    8 500 64 0

# recommended fast config (lossless): speculation + threads
CNE_CTX=1024 CNE_MTP=8 CNE_MTP_P_MIN=0.5 CNE_THREADS=6 \
    ./build/tools/cne_bench \
    models/qwen3.6-35b-a3b-q4_k_xl-mtp/Qwen3.6-35B-A3B-UD-Q4_K_XL-prepared.gguf \
    8 500 64 0

# same, streaming mode
CNE_CTX=1024 CNE_MTP=8 CNE_MTP_P_MIN=0.5 CNE_THREADS=6 \
    ./build/tools/cne_bench <gguf> 8 500 64 1
```

Bench CLI: `<gguf> [cache_cap_gib=8] [n_gen=64] [verify_n=64] [stream=1]`.

Notes:

- long generations need `CNE_CTX=1024+`; the 256 default aborts near token
  ~250 when speculating
- do not pin threads to physical cores manually — the I/O lane workers need
  CPUs too (measured regression)
- KV-cache q8_0 is a measured CPU regression on this class; expert-mass
  gating is inert on this model (flat routers)

Full feature guidance: [FEATURES.md](FEATURES.md).

## LFM2-24B-A2B (second artifact)

No MTP head — sequential decode only. Hybrid shortconv + MoE; **4 threads**
beats 6/8 on the reference chip (opposite of Qwen). Full profile:
[models/lfm2-24b-a2b.md](models/lfm2-24b-a2b.md).

### Measured (i5-1135G7, prepared Q4_K_M)

| test | tok/s | tool / config |
|---|---|---|
| decode warm steady-state | **~10.5–11.5** | `cne_server` / `cne_bench`, `CNE_STREAM=0 CNE_DENSE=warm CNE_THREADS=4` |
| decode tg128 (B3 on, 5-run mean) | **8.61 ± 0.31** | `llama-bench`, t4, 2026-08-27 re-measure |
| decode tg128 (B3 off, 5-run mean) | 8.21 ± 0.24 | same; **+4.8%** B3 delta (not +11%) |
| decode tg128 (B3 p3 fork, single session) | **11.48 ± 0.46** | `cne/lfm2-b3` @ `8d2440243` — treat as upper bound, high variance |
| server 1000 tok wall-clock | **~10.5** | `cne_server`; B3 on/off within noise at this length |
| decode fixed 300 tok | **10.84** | pre-B3 `cne_bench`; re-measure after fork pin |
| decode tg128 (pre-B3, mmap) | **9.26 ± 1.12** | kernel floor without warm dense |
| prefill pp512–4096 | **~41–44** | `llama-bench`, t4 |

**B3 A/B scripts:** `bench/scripts/lfm2/tg128-microbench.sh` (tg128 microbench),
`bench/scripts/lfm2/server-velocity.sh` (live HTTP). History TSVs under
`bench/results/` (gitignored). See [bench/README.md](../bench/README.md).

Decode is **~4× slower than prefill** on this stack. Q4 `MUL_MAT` +
`MUL_MAT_ID` dominate the decode graph (~89% of heavy ops); shortconv
(`SSM_CONV`) is ~1% of wall time in microbench.

Build: Release with `-march=native`, `GGML_CPU_REPACK=ON` (CNE default).
Repack OFF costs ~5% decode; an SSE4.2-only build costs **~2.7×** — do not
ship generic x86 binaries for this model class.

### Reproduce

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./tools/download-lfm2-24b-a2b.sh

# serving (recommended)
CNE_STREAM=0 CNE_DENSE=warm CNE_THREADS=4 CNE_CTX=4096 \
  ./build/server/cne_server \
  models/lfm2-24b-a2b/LFM2-24B-A2B-Q4_K_M-prepared.gguf

# bench throughput (300-token fixed run)
CNE_STREAM=0 CNE_DENSE=warm CNE_THREADS=4 CNE_CTX=4096 CNE_IGNORE_EOS=1 \
  ./build/tools/cne_bench \
  models/lfm2-24b-a2b/LFM2-24B-A2B-Q4_K_M-prepared.gguf \
  0 300 32 0
```

**Measurement caveat:** `cne_bench` that restarts the process each run can
show 5–6 tok/s with heavy majflt even when `CNE_DENSE=warm` is set — use a
long-lived `cne_server` or `llama-bench tg` for clean decode probes unless
the warm path is already resident.

A 2026-08 subset-expert self-spec spike measured **~5 tok/s** (~2.2× slower
than warm sequential) despite passing the identity gate; code and probe
artifacts were removed from the tree.

B4 prepare-time gate‖up fusion passed identity but was **slower than B3** and
was reverted. Next lossless kernel work: true x86 AVX-VNNI `4vx`, down-proj
`4vx`, router decode GEMV — see [models/lfm2-24b-a2b.md](models/lfm2-24b-a2b.md)
§ Kernel roadmap.
