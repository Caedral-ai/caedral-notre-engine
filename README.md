# Caedral Notre Engine (cne)

**A MoE inference engine for low-RAM, CPU-only machines.** Run
Mixture-of-Experts models that don't fit in memory by letting the engine pick
and combine the right optimizations for your hardware and model: expert
streaming from NVMe/flash, dense-residency policies, speculative decoding,
memory budgets — losslessly by default, with predictable behavior.

Built on top of `llama.cpp` as the inference runtime; the residency policy,
feature selection and expert I/O live in a dedicated layer outside of it.

> Status: **pre-alpha.**

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

## How it works

At load time the engine classifies the situation into a memory regime
(model size vs available RAM) and activates features accordingly:

| Regime | Model vs RAM | What the engine does |
|---|---|---|
| R0 | model ≪ RAM | reports that plain llama.cpp is the better tool |
| R1 | model ≈ RAM | anon-dense weights + budget enforcement |
| R2 | model 1–4× RAM | expert cache + streaming + speculation |
| R3 | model 4–8× RAM | full streaming pipeline |
| R4 | > 8× RAM | everything on, aggressive (opt-in) compression |

Core principles:

- **Lossless by default** — nothing changes the model's math silently;
  quality-affecting modes are explicit opt-in flags.
- **Explicit memory budget** — dense anon + expert cache + KV + staging ≤
  RAM budget; the page cache is never the line of defense.
- **Metadata-driven** — no hardcoded layouts, axes, quant types or offsets;
  all geometry comes from each loaded artifact's manifest.
- **Fail closed** — incomplete discovery aborts; errors are never swallowed.

## Architecture

```
Client (OpenAI SDK / Open WebUI / n8n)
        │  HTTP/SSE
        ▼
    cne-server ── auth · quotas · sessions · scheduler
        ▼
    cne-runtime ── regime classifier → feature activation
        ▼                     ▼
  Memory manager         Feature modules
  budgets/regimes        streaming · speculation · precision
        ▼                     ▼
     Expert LRU cache  ←  O_DIRECT I/O lanes
                     ▼
             NVMe / Flash (GGUF shards)
```

## Repository layout

```
core/include/cne/          public headers (model, memory, config…)
core/src/gguf/             GGUF reader, tensor classification, registry
core/src/memory/           memory budgets + regime classification
core/src/features/         feature implementations (streaming/ today,
                           speculation & precision landing next)
adapters/                  llama.cpp seam: demand-serving runtime +
                           draft-MTP speculative decoding
server/                    cne-server: HTTP, OpenAI-compatible API (planned)
cli/                       command-line interface (planned)
tools/                     measurement drivers & probes (cne-bench,
                           cne-prepare, identity gate, …)
tests/                     mirrors core areas
third_party/               llama.cpp (pinned upstream submodule)
```

## When streaming helps

Expert streaming delivers meaningful speedup only when the model is
meaningfully larger than available RAM. Below that threshold plain mmap
inference performs equally well or better because the OS page cache already
does the job — and even then, streaming's first-order value is a stable RSS
envelope and fault-free operation rather than raw velocity. The regime
classifier reports which situation you are in at load time and deactivates
streaming when it doesn't pay.

## Supported models

First target:

| Model | Quant | Source file | Size |
|---|---|---|---|
| [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (MoE, 3B active) | UD-Q4_K_XL, MTP tensors preserved | [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) | ~22.9 GB |

```sh
./tools/download-qwen3.6-35b-a3b-q4_k_xl.sh
# → models/qwen3.6-35b-a3b-q4_k_xl-mtp/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf          (~22.9 GB)
# → models/qwen3.6-35b-a3b-q4_k_xl-mtp/...-prepared.gguf (4096-aligned runtime artifact)
```

The prepared file is the runtime artifact: all expert tensors are
4096-aligned for O_DIRECT reads. The unaligned download can be deleted once
the correctness gates pass on the prepared file.

### Quality

Unsloth's own benchmarks (150+ KL-divergence runs, 9 TB of artifacts) rank
UD-Q4_K_XL as **SOTA** for this model family — sitting on the Pareto frontier
for quality-per-byte: attention and shared-expert weights stay at higher
precision; only routed experts are q4_k. In practice, chat / coding /
reasoning quality is indistinguishable from full precision.

MTP speculative decoding is **lossless**: every draft token is verified
against the full model over the full vocabulary, so accepted output is what
the target model would have produced anyway — just faster.

## Building

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Main artifacts: `build/tools/cne_streaming_bench` (measurement driver),
`build/tools/cne_prepare` (GGUF alignment tool). `third_party/llama.cpp` is
pinned as an upstream submodule; `common` is linked by tools for speculative
decoding, never by the product core.

## References

- llama.cpp server API: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

## License

MIT — see [LICENSE](LICENSE).
