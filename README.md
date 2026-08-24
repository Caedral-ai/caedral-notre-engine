# Caedral Notre Engine (soe)

**Streaming of Experts engine**: run Mixture-of-Experts models whose GGUF file is
larger than available RAM, by streaming expert weights from NVMe/flash on demand —
losslessly, with explicit memory budgets and predictable behavior.

Built on top of `llama.cpp` as the inference runtime; residency policy and expert
I/O live in a dedicated layer outside of it.

> Status: **pre-alpha / project bootstrap.**

## Why

**Useful AI should run on the hardware people already have.** State-of-the-art
models are locked behind datacenter GPUs and cloud APIs — most people cannot
afford racks of H100s just to run a good model locally. MoE architectures change
the equation: they are sparse, so the active compute per token is tiny compared
to the total parameter count. A CPU plus a fast SSD can plausibly run models
that would otherwise demand tens of GB of VRAM. Making that practical is a step
toward **democratizing local AI**: private, censorship-resistant,
subscription-free inference on ordinary laptops, mini-PCs and edge devices —
your model, your data, your machine.


## Architecture (target)

```
Client (OpenAI SDK / Open WebUI / n8n)
        │  HTTP/SSE
        ▼
    soe-server ── auth · quotas · sessions · scheduler
        ▼
    soe-runtime ── llama.cpp context + router observer
        ▼                     ▼
  Memory manager         Expert I/O pipeline
  dense/KV/budget        O_DIRECT · N lanes · overlap
        ▼                     ▼
       Expert LRU cache (shared per model)
                    ▼
            NVMe / Flash (GGUF shards)
```

Core principles:

- **Lossless by default** — streaming never changes the math; identity gates
  (`stream ON == OFF`, token-exact) block every release.
- **Explicit memory budget** — dense anon + expert LRU + KV + staging ≤ RAM budget;
  page cache is never the line of defense.
- **Metadata-driven** — no hardcoded layouts, axes, quant types or offsets.
- **O_DIRECT first-class** for expert misses, bounce buffers for misaligned slices.
- **Overlap** I/O and compute across layers (N+1 window) to hide flash latency.
- **Fork discipline** — the llama.cpp fork carries only the minimal expert-ready
  hook; all product logic lives outside it.

## Repository layout

```
core/include/soe/   public headers (config, model, cache, io, metrics…)
core/src/           implementation: gguf/, moe/, cache/, io/, memory/,
                    runtime/, metrics/
adapters/           llama.cpp adapter + platform-specific code
server/             soe-server: HTTP, OpenAI-compatible API, scheduler
cli/                command-line interface
tools/              inspection & benchmarking utilities
tests/              correctness/ gguf/ cache/ io/ memory/ server/
third_party/        llama.cpp (upstream submodule)
```


## When streaming helps

The stream delivers meaningful speedup only when the model is **at least 1.6×
the available RAM** (regime R3). Below that threshold, plain mmap inference
performs equally well or better because the OS page cache already does the
job. The regime classifier reports which situation you are in at load time.

## When the stream helps

The stream delivers meaningful speedup only when the model is **at least 1.6×
the available RAM** (regime R3). Below that threshold, plain mmap inference
performs equally well or better. The regime classifier reports which situation
you are in at load time.

## Supported models

First target — streaming in regime R3 (model ≫ RAM):

| Model | Quant | Source file | Size |
|---|---|---|---|
| [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (MoE, 3B active) | UD-Q4_K_XL + MTP | [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) | ~22.4 GiB |

```sh
./tools/download-qwen3.6-35b-a3b-q4_k_xl.sh
# → models/qwen3.6-35b-a3b-q4_k_xl/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf          (~22.9 GB)
# → models/qwen3.6-35b-a3b-q4_k_xl/Qwen3.6-35B-A3B-UD-Q4_K_XL-prepared.gguf (aligned, runtime artifact)
```

MTP speculative decoding is **lossless**: every draft token is verified
against the full model, so output is identical to non-speculative inference —
just faster.

The prepared file is the runtime artifact: all expert tensors 4096-aligned
for O_DIRECT streaming. The unaligned download can be deleted after the
gate passes.

Chosen because its Q8 GGUF far exceeds typical desktop RAM while the 3B active
parameters keep CPU compute tractable.

### Quality

Unsloth's own benchmarks (150+ KL-divergence runs, 9 TB of artifacts) rank
UD-Q4_K_XL as **SOTA** for this model family — sitting on the Pareto frontier
for quality-per-byte: attention and shared-expert weights stay at higher
precision; only routed experts are q4_k. In practice, chat / coding /
reasoning quality is indistinguishable from full precision.

## Building

Not yet functional — bootstrap phase. Planned:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

`third_party/llama.cpp` will be pinned as an upstream-patched submodule.

## References

- llama.cpp server API: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

## License

MIT — see [LICENSE](LICENSE).
