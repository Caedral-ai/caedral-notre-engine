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
