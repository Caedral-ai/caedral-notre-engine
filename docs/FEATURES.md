# cne feature guide

This document describes every feature the engine provides today, what it
does, and — most importantly — **when you should and should not use it**.

If you only remember one rule: **the engine's job is to make these decisions
for you.** Every knob below exists so you can override or inspect behavior,
not because you are expected to tune it by hand.

Quick decision table:

| Your situation | What the engine should do | Feature doing the work |
|---|---|---|
| Model fits comfortably in RAM | nothing special; stay out of the way | regime classification (R0) |
| Model about the size of RAM | fault-free decode, bounded memory | anon-dense residency + budget manager |
| Model 1-4x RAM | cache hot experts, stream cold ones | expert streaming + O_DIRECT |
| Model 4-8x+ RAM | full streaming pipeline | expert streaming (core regime) |
| Generation feels slow but memory is fine | draft multiple tokens per step | draft-MTP speculation |
| You accept a quality trade for speed | skip low-weight experts | expert-mass gating (lossy, opt-in) |
| You want to verify quality yourself | measure perplexity directly | engine-ppl mode |

---

## 1. Regime classification

**Status:** working · **Effort required from you:** none

At load time the engine compares the model artifact size against the
machine's *actually available* memory (not installed RAM — reclaimable page
cache is accounted honestly) and classifies the situation into a regime:

| Regime | Model vs available RAM | Consequence |
|---|---|---|
| R0 | model much smaller than RAM | plain inference; the engine stays out of the way |
| R1 | model about equal to RAM | residency + budget features activate |
| R2 | model 1-4x RAM | streaming features activate |
| R3 | model 4-8x RAM | full streaming pipeline |
| R4 | over 8x RAM | everything on; compression becomes necessary |

The detected regime is printed at startup (`regime=...`).

**When you use it:** always — it runs automatically.
**When it matters:** it is the reason you do not need to know any of the
rest of this document.

## 2. Memory budget manager

**Status:** working · **Effort required from you:** none

Every byte the engine controls — dense weight copies, the expert cache,
context state, staging buffers — is counted into a budget that is clamped to
what the machine can actually hold at load time, with a safety reserve for
machine-state drift while your generation runs.

Practical consequence: an oversized cache request degrades gracefully to a
smaller working cache instead of getting the process killed mid-generation.

**When you use it:** always — automatic.
**When to override:** pass a smaller `cap_gib` argument to the bench if you
need headroom for other applications; never a larger one (it gets clamped
anyway).

## 3. Dense residency policies (`CNE_DENSE`)

**Status:** working

Controls how non-expert weights live in memory during generation:

| Policy | What happens | Use when |
|---|---|---|
| `mmap` | weights stay file-backed; kernel pages them in/out | default for small models (R0/R1); zero setup cost |
| `warm` | engine pre-reads dense spans once before generating | rare: dedicated batch machines where first-token latency doesn't matter |
| `anon` | engine binds every dense weight to a private anonymous copy | models near/above RAM size: eliminates major-page-fault storms (~100x fewer faults measured); costs one extra copy of dense bytes |

Default: chosen automatically from the regime (anon above the thrash line,
mmap below). Explicit values win over the auto choice.

**Use when:** leave it automatic.
**Avoid when:** never set `anon` on a machine where the model barely fits —
the copy needs real memory and the budget manager will shrink your expert
cache to compensate.

## 4. Expert streaming (slice cache + O_DIRECT + I/O lanes)

**Status:** working · **Enabled:** bench last argument `1` (default); `0`
disables it for A/B comparison

The core feature for models bigger than RAM. The router decides which
experts each token needs; the engine keeps recently used expert slices in an
LRU cache in RAM and reads misses straight from the NVMe device with
`O_DIRECT` (no page-cache pollution), parallelized across I/O lanes.

Telemetry printed per run: hit-rate, bytes loaded, evictions, fill-time
share of the generation wall.

**Use when:** the model is at least ~1.6x available RAM (regime R2+). Below
that the OS page cache does the same job for free and streaming stays
pointless — which is why the classifier keeps it off there.

**Avoid when:** R0/R1 — run with streaming disabled and compare if unsure;
the identity guarantee means output is identical either way, so this is
purely a velocity/stability question.

Requirements: run the artifact through `cne-prepare` first (see section 6);
unaligned artifacts fall back to buffered copies automatically.

## 5. Draft-MTP speculation (`CNE_MTP`)

**Status:** working on CPU — net-positive velocity with the tuned config

Uses the model's native Multi-Token Prediction head to draft up to k tokens
per step and verifies them all in one batched forward through the full
model. Accepted tokens are exactly what greedy decoding would produce — the
target verifies every draft over the full vocabulary, so this is lossless by
construction: **0% quality loss**, byte-for-byte the same distribution as
plain decoding. The only observable difference is cosmetic near-tie stream
divergence from the different sampling path.

Measured on the reference machine (i5-1135G7, 16 GB RAM, 500-token runs,
Qwen3.6-35B-A3B Q4_K_XL):

| config | tok/s | note |
|---|---|---|
| naive, 8 threads | 2.19 | thread-starved baseline |
| naive, 6 threads | 4.00 | SMT contention was costing ~2x |
| **MTP k=8 p_min=0.5, 6 threads** | **4.86 / 4.75** | **+20% over fair naive; 100% acceptance, 0 partials** |

Depth plateaus at k=8 (drafter yields ~2.5 confident tokens/step regardless
of max depth). Measured dead ends, do not re-run: KV cache q8_0 (quant
overhead beats bandwidth savings on CPU), expert-mass gating (this model's
routing mass is too evenly spread to ever drop an expert).

```sh
# recommended fast config (lossless)
CNE_MTP=8 CNE_MTP_P_MIN=0.5 CNE_THREADS=6 CNE_CTX=1024 \
    ./build/tools/cne_bench model.gguf
```

Requires an artifact with MTP tensors preserved (the reference model download
script fetches exactly that) and loads them via `load_mtp`.

**Use when:** single-user interactive decode — this is now the fastest
lossless configuration measured on this hardware class.

**Avoid when:** serving multiple concurrent requests — draft overhead does
not parallelize as well as plain decode.

## 6. Artifact alignment (`cne-prepare`)

**Status:** working · **One-time cost per artifact**

Rewrites a GGUF so every expert tensor starts at a 4096-aligned offset
(bumps `general.alignment` and repacks offsets cumulatively; output remains
loadable by stock llama.cpp). Aligned artifacts allow true O_DIRECT reads;
unaligned ones force buffered fallback paths.

```sh
./build/tools/cne_prepare model.gguf
# -> model-prepared.gguf (canonical runtime artifact)
```

Idempotent: running it on an already-prepared file is safe.

**Use when:** always, once per downloaded artifact — the download script
runs it for you.
**Avoid when:** never; there is no downside. The unaligned original can be
deleted after gates pass on the prepared file.

## 7. Expert-mass gating (`CNE_EXPERT_MASS`, `CNE_EXPERT_MIN_K`) — LOSSY

**Status:** working, default OFF

A quality-for-speed trade you must opt into explicitly. At each token, the
engine walks the ranked routing weights and stops filling experts once
cumulative mass passes the threshold — tail experts contribute exactly
nothing because their slices are never read.

```sh
CNE_EXPERT_MASS=0.75 CNE_EXPERT_MIN_K=2 \
    ./build/tools/cne_bench model.gguf
```

Loud telemetry marks every active run; dropped-slice counts are reported.

**Use when:** you accept measurable quality degradation for fewer bytes read
— typically extreme-RAM-pressure situations (R4) or batch jobs where slight
quality loss is acceptable. Also valuable on future models whose routers are
trained to be peaked (the reference model's router is flat by design, so
mild thresholds drop almost nothing here).

**Avoid when:** you care about output quality. This is the engine's only
feature that changes results without verifying against the full model.
Never combine casually with the perplexity gates' expectations — measure
before trusting it.

## 8. Whole-model perplexity mode (`CNE_PPL_FILE`)

**Status:** working

Runs chunked cross-entropy evaluation over a plain-text corpus inside the
engine, applying whichever runtime policies you have enabled. External
perplexity tools cannot reproduce callback-side policies, so this is the
canonical way to measure quality effects of gating/residency/streaming.

```sh
CNE_PPL_FILE=corpus.txt ./build/tools/cne_bench model.gguf
```

**Use when:** before and after enabling any lossy feature; report both
numbers.
**Avoid when:** comparing against numbers produced by different toolchains —
absolute values are only comparable within the same harness and protocol.

## 9. Prefetch overlap (`CNE_PREFETCH`)

**Status:** shipped, default OFF

Speculatively fills expert slices for the next step using the previous
step's routing. Measured as a no-op or regression on current hardware
(adequate caching already captures routing locality, and the prefetcher
contends with demand fills on the cache lock).

**Use when:** essentially never today; revisit on machines where fills are
cheap relative to compute (very fast storage) or models far beyond cache
reach (R4).
**Avoid when:** everywhere else — the flag exists so the mechanism can be
re-measured as hardware changes, not because it currently helps.

## 10. Diagnostics (development)

These exist for engine work, not for end users:

| Knob | Purpose |
|---|---|
| `CNE_CTX`, `CNE_PROMPT` | context size / custom prompt |
| `CNE_DUMP_DST`, `CNE_DUMP_LOGITS(_EVERY)` | dump tensors/logits for divergence hunts |
| `CNE_STEP_FILLS`, `CNE_FULL_FILL` | fill telemetry / whole-window fill probe |
| `CNE_LAYER_LIMIT` | restrict demand-serving to N layers (bisecting) |
| `CNE_SPLIT_PREFILL`, `CNE_MTP_NODRAFT` | prefill-shape and draft-isolation bisects |

## The standing guarantees

Whichever features you enable or disable:

1. **Output parity** — with all lossy features off (the default), generation
   is token-exact identical to plain llama.cpp greedy decoding on the same
   artifact. Verified continuously by the identity gate.
2. **Bounded memory** — the process respects its enforced budget regardless
   of knob combinations; requests beyond the machine's capacity are clamped,
   not crashed.
3. **No silent trade-offs** — anything that changes results prints loud
   telemetry and requires an explicit opt-in variable.
