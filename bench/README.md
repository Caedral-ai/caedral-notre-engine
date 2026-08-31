# Benchmark harness

Reproducible velocity scripts for CNE. **Scripts are tracked; outputs are not.**

```
bench/
  scripts/          runnable harnesses (by model / scenario)
  results/          local logs and TSV history (gitignored)
```

Build `llama-bench` separately when needed (not part of the main CNE build):

```sh
cmake -S third_party/llama.cpp -B /tmp/llama-bench-build \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build /tmp/llama-bench-build --target llama-bench -j
export LLAMA_BENCH=/tmp/llama-bench-build/bin/llama-bench
```

**Important:** stop `cne_server` before microbench runs on 16 GB machines — concurrent
load causes OOM (exit 137).

## LFM2

| Script | What it measures |
|---|---|
| `scripts/lfm2/tg250-kernel-ab.sh` | **Canonical** kernel A/B: tg250, pp512, 5 reps/arm |
| `scripts/lfm2/tg128-microbench.sh` | `llama-bench` tg128 decode (quick smoke) |
| `scripts/lfm2/server-velocity.sh` | Live `cne_server` wall-clock tok/s |

```sh
# Canonical kernel A/B (tg250, pp512 warmup, 5 reps per arm)
./bench/scripts/lfm2/tg250-kernel-ab.sh
# Latest (2026-08-28, i5-1135G7, 21767de7d): on 11.92±0.42 vs off 10.61±0.13 (+12.3%)

# Quick tg128 smoke (append history)
for i in {1..5}; do ./bench/scripts/lfm2/tg128-microbench.sh; done
for i in {1..5}; do CNE_KERNELS=0 ./bench/scripts/lfm2/tg128-microbench.sh; done
column -t bench/results/lfm2-tg128.history.tsv

# Manual tg250 (same as script)
cmake -S third_party/llama.cpp -B /tmp/llama-bench-build \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build /tmp/llama-bench-build --target llama-bench -j
MODEL=models/lfm2-24b-a2b/LFM2-24B-A2B-Q4_K_M-prepared.gguf
for k in 1 0; do
  CNE_KERNELS=$k /tmp/llama-bench-build/bin/llama-bench \
    -m "$MODEL" -t 4 -p 512 -n 250 -r 5 -b 512 -ub 512
done
# Prior (2026-08-27): on 9.50±0.17 vs off 8.59±0.15 (+10.6%)

# Server (restart with matching CNE_KERNELS)
CNE_KERNELS=1 CNE_STREAM=0 CNE_DENSE=warm CNE_THREADS=4 CNE_CTX=4096 \
  ./build/server/cne_server --config models/server.json
CNE_BENCH_MAX_TOKENS=1000 ./bench/scripts/lfm2/server-velocity.sh
```

See [docs/BENCHMARKS.md](../docs/BENCHMARKS.md) and
[docs/models/lfm2-24b-a2b.md](../docs/models/lfm2-24b-a2b.md) for measured numbers.

## LFM2.5

| Script | What it measures |
|---|---|
| `scripts/lfm2.5/memory-profile.sh` | RSS (VmHWM), tok/s, hit-rate across mmap / anon / stream profiles |

```sh
# Memory + velocity probes (writes bench/results/lfm25-memory.tsv)
./bench/scripts/lfm2.5/memory-profile.sh

# 250-token sustained decode (canonical length for profile comparison)
# 4 GiB — anon, no stream, tuned (t4, ctx 1024): ~11.1 tok/s, ~2.9 GiB peak
CNE_STREAM=0 CNE_DENSE=anon CNE_KERNELS=1 CNE_THREADS=4 CNE_CTX=1024 CNE_IGNORE_EOS=1 \
  ./build/tools/cne_bench \
  models/lfm2.5-8b-a1b/lfm25-8b-a1b-UD-Q4_K_XL-prepared.gguf 0 250 250 0

# 16 GiB+ — mmap, no stream, tuned (t4, ctx 2048): ~12.0 tok/s
CNE_STREAM=0 CNE_DENSE=mmap CNE_KERNELS=1 CNE_THREADS=4 CNE_CTX=2048 CNE_IGNORE_EOS=1 \
  ./build/tools/cne_bench \
  models/lfm2.5-8b-a1b/lfm25-8b-a1b-UD-Q4_K_XL-prepared.gguf 0 250 250 0

# 4 GiB — anon + 3 GiB stream cache, tuned (t4, lanes 2): ~9.1 tok/s, 97% hit
CNE_STREAM=1 CNE_DENSE=anon CNE_KERNELS=1 CNE_THREADS=4 CNE_LANES=2 CNE_CTX=1024 CNE_IGNORE_EOS=1 \
  ./build/tools/cne_bench \
  models/lfm2.5-8b-a1b/lfm25-8b-a1b-UD-Q4_K_XL-prepared.gguf 3 250 250 1
```

See [docs/models/lfm2.5-8b-a1b.md](../docs/models/lfm2.5-8b-a1b.md) for serving knobs.
