<div align="center">

<img src="https://caedral.com/icon.svg" alt="Caedral" width="96"/>

# Caedral Notre Engine

**`cne`** · A MoE inference engine for low-RAM, CPU-only machines.

Run Mixture-of-Experts models that don't fit in memory by letting the engine
choose and combine the right optimizations for your hardware and model —
with explicit memory budgets, predictable behavior, and lossless output
by default.

![License](https://img.shields.io/badge/license-MIT-green)
![Language](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20CPU--only-FCC624?logo=linux&logoColor=black)
![Status](https://img.shields.io/badge/status-pre--alpha-orange)
![Kernel](https://img.shields.io/badge/kernel-llama.cpp-c17?logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiI+PGNpcmNsZSBjeD0iOCIgY3k9IjgiIHI9IjciIGZpbGw9IiM4ODAiLz48L3N2Zz4)

*Built on top of [llama.cpp](https://github.com/ggml-org/llama.cpp) as the
inference kernel. Memory regimes, feature selection, expert caching and
speculative decoding live in a dedicated layer outside it.*

</div>

---

> **Status: pre-alpha.** The streaming pipeline and measurement tooling are
> working end-to-end; the serving layer and user-facing profiles are next.
> See [Roadmap](#roadmap).

## Why

**Useful AI should run on the hardware people already have.**

State-of-the-art models are locked behind datacenter GPUs and cloud APIs.
MoE architectures change the equation: they are sparse — each token activates
only a few experts — so active compute per token is tiny compared to the total
parameter count. A CPU plus a fast SSD can run models that would otherwise
demand tens of GB of VRAM. Making that practical is a step toward
democratizing local AI: private, subscription-free inference on ordinary
laptops, mini-PCs and edge devices. **Your model, your data, your machine.**

The catch: the *file* is still huge, and RAM is not. The engine exists to
close that gap without changing what the model computes.

## The concept: one engine, many features, auto-selected

There is no single trick that makes big models fit small machines. What works
depends on the situation. At load time a **regime classifier** measures the
machine and inspects the model artifact, then decides which features activate:

| Regime | Model vs available RAM | Streaming | Speculation | Residency & budget |
|---|---|---|---|---|
| **R0** | much smaller than RAM | off — the OS page cache already delivers every byte; streaming can only add overhead | available (same math as upstream) | plain mmap; the engine stays out of the way |
| **R1** | about equal to RAM | off — page cache is competitive at this ratio | on when measured to pay | anon-dense weights: fault-free decode where vanilla mmap thrashes |
| **R2** | 1–4× RAM | on — expert cache absorbs routing locality; misses stream from NVMe | on when measured to pay | budgets enforced; anon dense above the thrash line |
| **R3** | 4–8× RAM | on — the core regime; most experts cannot stay resident | on when measured to pay | full budget enforcement |
| **R4** | over 8× RAM | on — streaming is the only way to run at all | on; aggressive (opt-in) compression becomes the deciding lever | tightest budgets |

**When speculation makes sense.** Drafting only pays if the accepted tokens
per verify pass outweigh the extra draft plus batched-verify compute. The
engine enables it per regime and measures acceptance live; where it does not
pay (small models, low-acceptance domains, or CPUs where batched matmuls
cost more than sequential ones), it deactivates automatically instead of
guessing.

**When streaming makes sense.** Below roughly 1.6× model-to-RAM ratio the
page cache does the same job for free, so streaming stays off. From R2
upward the cache-plus-stream pipeline wins on stability (orders of magnitude
fewer page faults) and, as the ratio grows, on velocity.

### The guarantee: never worse than llama.cpp

The engine's floor is parity with plain `llama.cpp` on the same artifact:

- output is **identical by default** (lossless contract, verified by
  token-exact identity gates), and
- any feature that does not measurably help on the detected hardware/model
  pair is **deactivated**, so overhead never accumulates where there is no
  value.

In practice: equal quality everywhere, equal-or-better velocity, and
strictly better behavior — bounded memory, no fault storms — exactly where
the model pushes against the machine's limits.

Users state intent; the engine resolves settings:

```json
{ "quality": "lossless" }   // or "balanced" | "fast"  (planned)
```

## Feature set

| Feature | Status | What it does |
|---|---|---|
| Expert cache + NVMe streaming | ![](https://img.shields.io/badge/status-working-brightgreen) | LRU-cached expert slices, filled on demand via O_DIRECT through parallel I/O lanes; misses read only what the router asked for |
| GGUF alignment (`cne_prepare`) | ![](https://img.shields.io/badge/status-working-brightgreen) | one-time pass that 4096-aligns every expert tensor so O_DIRECT needs no bounce buffers |
| Dense residency policies | ![](https://img.shields.io/badge/status-working-brightgreen) | mmap / pre-warmed / anonymous copies — chosen per regime; anon eliminates page-fault storms near the RAM boundary |
| Memory budget manager | ![](https://img.shields.io/badge/status-working-brightgreen) | clamps any requested cache size to what the machine can actually hold; never relies on page-cache luck |
| Regime classification | ![](https://img.shields.io/badge/status-working-brightgreen) | R0–R4 detection at load time drives feature selection |
| Draft-MTP speculation | ![](https://img.shields.io/badge/status-working-brightgreen) | native Multi-Token Prediction head drafts k tokens per step; full model verifies. Lossless by construction (0% quality loss); tuned config measured at 4.86 tok/s vs 4.00 naive on the reference machine (+20%) |
| Mixed-precision serving | ![](https://img.shields.io/badge/status-designed-blue) | cache-missed experts served from lower-precision sidecars (bounded quality trade, opt-in) |
| OpenAI-compatible server | ![](https://img.shields.io/badge/status-planned-lightgrey) | SSE endpoint on top of the extracted runtime |

Full per-feature guidance — including when *not* to use each one — lives in
**[docs/FEATURES.md](docs/FEATURES.md)**.

## Principles

- **Never worse than llama.cpp.** Parity is the floor: identical output by
  default, and any feature that does not pay on the detected hardware/model
  pair is switched off.
- **Lossless by default.** Nothing changes the model's math silently.
  Quality-affecting modes exist only behind explicit opt-in flags.
- **Explicit memory budget.** Dense weights + expert cache + KV + staging ≤
  RAM budget, enforced by clamping at load. The OS page cache is never the
  line of defense.
- **Metadata-driven.** No hardcoded tensor names, axes, quant types or
  offsets. All geometry comes from the loaded artifact's manifest; the same
  binary serves any quant.
- **Fail closed.** Incomplete discovery or misaligned reads abort loudly;
  errors are never swallowed.
- **Measure before building.** Features ship only with paired A/B numbers;
  mechanisms that measure as no-ops stay default-off (and documented).

## Architecture

```
Client (OpenAI SDK / Open WebUI / n8n)
        │  HTTP/SSE                      (planned)
        ▼
    cne-server ── auth · quotas · sessions · scheduler
        ▼
    cne-runtime ── regime classifier → feature activation
        ▼                              ▼
  Memory manager                  Feature modules
  budgets · regimes               streaming · speculation · precision
        ▼                              ▼
     Expert LRU cache  ←────  O_DIRECT I/O lanes
                     ▼
             NVMe / Flash (GGUF shards)
```

Today the runtime speaks through `cne_bench` (measurement driver); the
server wraps the same API surface later.

## Repository layout

```
core/include/cne/          public headers (model, memory, config…)
core/src/gguf/             GGUF reader, tensor classification, manifest registry
core/src/memory/           memory budgets + regime classification
core/src/features/
  streaming/               slice cache · O_DIRECT file · I/O lane scheduler
adapters/                  llama.cpp seam:
                             cne_stream_cb.cpp   demand-serving runtime
                             cne_stream_spec.cpp draft-MTP generation loop
tools/                     drivers & probes:
                             cne_bench       end-to-end bench
                             cne_prepare     GGUF alignment tool
                             cne_identity_gate  correctness harness
tests/features/            unit tests mirroring core areas
docs/                      FEATURES.md · per-model notes
third_party/               llama.cpp (pinned upstream submodule)
```

## Quick start

Requirements: Linux, CMake ≥ 3.16, C++17 compiler, ~50 GB free disk for the
reference model.

```sh
# build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# fetch + align the reference model (~22.9 GB download, sha256-verified)
./tools/download-qwen3.6-35b-a3b-q4_k_xl.sh

# run: naive mmap baseline
./build/tools/cne_bench \
    models/qwen3.6-35b-a3b-q4_k_xl-mtp/Qwen3.6-35B-A3B-UD-Q4_K_XL-prepared.gguf \
    8 64 0 0

# run: streaming mode (last arg 1 = stream ON)
CNE_LANES=4 ./build/tools/cne_bench \
    models/qwen3.6-35b-a3b-q4_k_xl-mtp/Qwen3.6-35B-A3B-UD-Q4_K_XL-prepared.gguf \
    8 64 0 1
```

Bench CLI: `<gguf> [cache_cap_gib=8] [n_gen=64] [verify_n=64] [stream=1]`.
The cache cap is automatically clamped to the machine's real budget.

<details>
<summary><strong>Environment knobs (development)</strong></summary>

| Variable | Values | Effect |
|---|---|---|
| `CNE_DENSE` | `mmap` \| `warm` \| `anon` | dense-weight residency policy (default: auto by regime) |
| `CNE_LANES` | N | parallel slice-read workers (default 4) |
| `CNE_MTP` | `1` \| k | enable draft-MTP speculative decoding (depth k) |
| `CNE_EXPERT_MASS` | 0 < x ≤ 1 | **lossy**: drop tail experts below cumulative routing mass |
| `CNE_EXPERT_MIN_K` | N | minimum experts kept when mass gating is active |
| `CNE_CTX` | N | context size |
| `CNE_PROMPT` | text | custom prompt |
| `CNE_PPL_FILE` | path | whole-model perplexity mode over a text corpus |

Legacy `SOE_*` spellings are accepted everywhere during migration.

</details>

## Correctness

Every optimization must pass its gate before it ships:

- **Stream identity** — generation with streaming ON vs OFF must be
  token-exact (greedy). This gate blocked every release so far and passes
  on the reference artifact today.
- **Memory cap** — steady-state RSS stays within the enforced budget,
  verified under `systemd-run --scope MemoryMax` isolation.
- **Artifact integrity** — every filled expert slice byte-compared against
  its source; window audits run continuously in memcpy mode.
- **Perplexity harness** — locked PTB-16 protocol for whole-model quality;
  required for anything that trades quality for speed.

Reference model note: MTP speculative decoding verifies every drafted token
against the full model over the full vocabulary, so accepted output equals
plain greedy decoding — speedup without changing results. Quality loss of
the tuned config (`CNE_MTP=8 CNE_MTP_P_MIN=0.5 CNE_THREADS=6`) is 0%;
the lossy expert-mass knob never activates on this model (measured:
zero dropped slices), and KV q8_0 was rejected as a CPU regression.

## Roadmap

1. Streaming pipeline, budget manager, regime classification, tooling — **done**
2. Draft-MTP speculation: performance evaluation on CPU (telemetry +
   depth/probability sweeps); stays flag-gated until it beats sequential
3. Speculation telemetry: separate draft/verify timing to decide viability
   per hardware class
4. Mixed-precision miss serving (opt-in lossy profile)
5. Runtime hardening → `cne-server` with OpenAI-compatible SSE
6. User-facing quality profiles (`lossless` / `balanced` / `fast`)
7. Multi-user serving, autotune, packaging

Non-goals: GPU offloading, training/fine-tuning, dense-model optimization
(that's llama.cpp's job), Windows/macOS support initially.

---

<div align="center">

Part of the [Caedral](https://caedral.com) ecosystem —
prepaid AI infrastructure for automation agencies.

MIT — see [LICENSE](LICENSE).

</div>
