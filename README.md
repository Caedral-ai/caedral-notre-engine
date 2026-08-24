# Caedral Notre Engine (cne)

**A MoE inference engine for low-RAM, CPU-only machines.** Run
Mixture-of-Experts models that don't fit in RAM by letting the engine choose
and combine the right optimizations for your hardware and model — with
explicit memory budgets, predictable behavior, and lossless output by
default.

Built on top of `llama.cpp` as the inference kernel. Everything else — memory
regimes, feature selection, expert caching, speculative decoding — lives in a
dedicated layer outside it.

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
| R0 | model much smaller than RAM | **off** — the OS page cache already delivers every byte; streaming can only add overhead | available (same math as upstream) | plain mmap; the engine stays out of the way |
| R1 | model about equal to RAM | **off** — page cache is competitive at this ratio | on when measured to pay | anon-dense weights: fault-free decode where vanilla mmap thrashes |
| R2 | model 1-4x RAM | **on** — expert cache absorbs routing locality; misses stream from NVMe | on when measured to pay | budgets enforced; anon dense above the thrash line |
| R3 | model 4-8x RAM | **on** — the core regime; most experts cannot stay resident | on when measured to pay | full budget enforcement |
| R4 | model over 8x RAM | **on** — streaming is the only way to run at all | on; aggressive (opt-in) compression becomes the deciding lever | tightest budgets |

**When speculation makes sense.** Drafting only pays if the accepted tokens
per verify pass outweigh the extra draft plus batched-verify compute. The
engine enables it per regime and measures acceptance live; where it does not
pay (small models, low-acceptance domains, or CPUs where batched matmuls
cost more than sequential ones), it deactivates automatically instead of
guessing.

**When streaming makes sense.** Below roughly 1.6x model-to-RAM ratio the
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

Manual environment variables remain available for development and
measurement (`CNE_*`, see below) — but they are knobs, not the interface.

### Feature set

| Feature | Status | What it does |
|---|---|---|
| Expert cache + NVMe streaming | working | LRU-cached expert slices, filled on demand via O_DIRECT through parallel I/O lanes; misses read only what the router asked for |
| GGUF alignment (`cne-prepare`) | working | one-time pass that 4096-aligns every expert tensor so O_DIRECT needs no bounce buffers |
| Dense residency policies | working | mmap / pre-warmed / anonymous copies — chosen per regime; anon eliminates page-fault storms near the RAM boundary |
| Memory budget manager | working | clamps any requested cache size to what the machine can actually hold; never relies on page-cache luck |
| Regime classification | working | R0–R4 detection at load time drives feature selection |
| Draft-MTP speculation | experimental | native Multi-Token Prediction head drafts k tokens per step; full model verifies. Lossless by construction; CPU economics under evaluation |
| Mixed-precision serving | designed | cache-missed experts served from lower-precision sidecars (bounded quality trade, opt-in) |
| OpenAI-compatible server | planned | SSE endpoint on top of the extracted runtime |

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

Today the runtime speaks through `cne-bench` (measurement driver); the
server wraps the same API surface later.

## Repository layout

```
core/include/cne/          public headers (model, memory, config…)
core/src/gguf/             GGUF reader, tensor classification, manifest registry
core/src/memory/           memory budgets + regime classification
core/src/features/
  streaming/               slice cache · O_DIRECT file · I/O lane scheduler
adapters/                  llama.cpp seam:
                             stream_cb.cpp   demand-serving runtime
                             stream_spec.cpp draft-MTP generation loop
tools/                     drivers & probes:
                             cne_streaming_bench   end-to-end bench
                             cne_prepare           GGUF alignment tool
                             cne_identity_gate     correctness harness
                             cne_probe_*           graph/rebind/callback probes
tests/                     unit tests mirroring core areas
third_party/               llama.cpp (pinned upstream submodule)
docs/models/               per-model notes bound to specific artifacts
docs/FEATURES.md           feature guide: what exists, when to use it
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
./build/tools/cne_streaming_bench \
    models/qwen3.6-35b-a3b-q4_k_xl-mtp/Qwen3.6-35B-A3B-UD-Q4_K_XL-prepared.gguf \
    8 64 0 0

# run: streaming mode (last arg 1 = rebind/stream ON)
CNE_LANES=4 ./build/tools/cne_streaming_bench \
    models/qwen3.6-35b-a3b-q4_k_xl-mtp/Qwen3.6-35B-A3B-UD-Q4_K_XL-prepared.gguf \
    8 64 0 1
```

Bench CLI: `<gguf> [cache_cap_gib=8] [n_gen=64] [verify_n=64] [stream=1]`.
The cache cap is automatically clamped to the machine's real budget.

### Environment knobs (development)

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
plain greedy decoding — speedup without changing results.

## Roadmap

1. streaming pipeline, budget manager, regime classification, tooling
2. draft-MTP speculation: performance evaluation on CPU (telemetry +
   depth/probability sweeps); keep flag-gated until it beats sequential
3. Speculation telemetry: separate draft/verify timing to decide viability
   per hardware class
4. Mixed-precision miss serving (opt-in lossy profile)
5. Runtime extraction hardening → `cne-server` with OpenAI-compatible SSE
6. User-facing quality profiles (`lossless` / `balanced` / `fast`)
7. Multi-user serving, autotune, packaging

Non-goals: GPU offloading, training/fine-tuning, dense-model optimization
(that's llama.cpp's job), Windows/macOS support initially.

## License

MIT — see [LICENSE](LICENSE).
