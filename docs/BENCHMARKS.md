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
| q8 profile, naive mmap (retired reference) | 0.57 | — |
| q4_K_XL artifact, naive mmap, 8 threads | 2.19 | 3.8× (quantization of artifact) |
| same, **6 threads** (physical cores beat SMT) | **4.00** | **+82%** |
| same + draft-MTP k=8 p_min=0.5 | **4.86** | **+20%** |
| same, streaming ON (expert cache 93.9% hit-rate) | 4.52 | −2.4% = within noise |

Cumulative: **8.5× from the retired q8 profile, ~2.2× from stock-config
q4, +20% from speculation — all with bit-exact lossless output.**

### What each gain is made of

- **Artifact quantization (q8 → q4_K_XL)**: smaller weights = fewer bytes
  per forward. This is an artifact choice, enabled by the engine loading
  any quant without code changes.
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
