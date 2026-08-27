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
| `scripts/lfm2/tg128-microbench.sh` | `llama-bench` tg128 decode (kernel A/B via `CNE_MOE_B3`) |
| `scripts/lfm2/server-velocity.sh` | Live `cne_server` wall-clock tok/s |

```sh
# Microbench (5-run sweep)
for i in {1..5}; do ./bench/scripts/lfm2/tg128-microbench.sh; done
for i in {1..5}; do CNE_MOE_B3=0 ./bench/scripts/lfm2/tg128-microbench.sh; done
column -t bench/results/lfm2-tg128.history.tsv

# Server (restart with matching CNE_MOE_B3)
CNE_MOE_B3=1 CNE_STREAM=0 CNE_DENSE=warm CNE_THREADS=4 CNE_CTX=4096 \
  ./build/server/cne_server --config models/server.json
CNE_BENCH_MAX_TOKENS=1000 ./bench/scripts/lfm2/server-velocity.sh
```

See [docs/BENCHMARKS.md](../docs/BENCHMARKS.md) and
[docs/models/lfm2-24b-a2b.md](../docs/models/lfm2-24b-a2b.md) for measured numbers.
