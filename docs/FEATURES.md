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
| Generation feels slow but memory is fine | draft multiple tokens per step | draft-MTP speculation (Qwen3.6 only) |
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
CNE_MTP=8 CNE_MTP_P_MIN=0.5 CNE_THREADS=6 CNE_FA=1 CNE_CTX=1024 \
    ./build/tools/cne_bench model.gguf
```

Flash attention (`CNE_FA=1`) adds ~3% at ctx 1024 and scales with context
length.

Requires an artifact with MTP tensors preserved (the reference model download
script fetches exactly that) and loads them via `load_mtp`.

**Use when:** single-user interactive decode — this is now the fastest
lossless configuration measured on this hardware class.

**Avoid when:** serving multiple concurrent requests — draft overhead does
not parallelize as well as plain decode.

**Incompatible with conversation sessions (`conversation_id`).** On
`cne_server`, MTP and multi-turn KV reuse are **mutually exclusive** — pick
one per deployment (`CNE_MTP=0` for chat sessions; unset `conversation_id`
when MTP is on). When `CNE_MTP>0`, the server ignores `conversation_id` and
runs stateless (full prefill + KV clear every request). Reasons:

1. **Two KV caches** — MTP keeps target and draft contexts in sync; session
   reuse only tracks the target. Resuming a chat would leave the drafter on
   stale state.
2. **KV rollbacks** — MTP trims rejected draft tails with `seq_rm` and
   checkpoints mid-generation; sessions assume a stable prefix that only
   grows (or trims on history edit).
3. **Separate code paths** — `spec_mtp_generate()` always full-prefills;
   it does not call incremental `session_prefill()`.
4. **Serving hazard** — upstream draft-MTP can degrade across back-to-back
   requests (#26425); sessions intentionally retain KV, which would compound
   that risk.

A combined MTP+sessions mode is possible later (draft sync + parity tests);
not supported in v1. See also §11 (server sessions).

**Streaming interaction (measured 2026-08-24 at ~1.4x RAM):** speculation's
advantage decays as cache hit-rate drops, for three reasons:

1. A verify block needs the union of several draft positions' expert slices
   resident *simultaneously* — the per-step working set grows with depth,
   which increases thrash exactly when the cache is smallest relative to
   demand.
2. Rejected draft tails waste fills that were already paid for in NVMe time;
   in compute-bound mode the same waste is only discarded ALU work.
3. Block-shaped fetches scramble LRU recency ordering, degrading eviction
   choices under pressure.

Measurement (500 tokens, ctx 1024, k=8 p_min=0.5 t=6): stream 4.52 / 4.44
tok/s vs naive 4.63 / 4.40 — a −2.4% / −0.9% delta, within run-to-run noise.
Hit-rate held at 93.9% under block-shaped fetches; fills took 35% of wall
but overlapped with compute through the I/O lanes; outputs were
byte-identical across arms; 100% acceptance in all runs.

**Verdict:** MTP shows **little-to-no gain in streaming mode** even at this
friendly operating point (~1.4× RAM): stream 4.52 / 4.44 tok/s vs naive
4.63 / 4.40 — at best a wash once noise is accounted for. The mechanism:
verify blocks need several draft positions' expert slices resident
simultaneously (bigger per-step working set), rejected draft tails waste
NVMe fills already paid for, and block-shaped fetches scramble LRU recency.

**Operational rule: below ~90% expert-cache hit-rate, MTP is net-NEGATIVE —
turn it off** (`CNE_MTP` unset). Fill latency dominates wall time in that
regime, so every extra slice touched per verify block and every rejected
draft tail costs real I/O wait that sequential decode would amortize or
avoid. Above 90%, the cost is bounded (~2%) and acceptable for the +20%
naive-mode win. A future release will enforce this automatically via the
regime classifier; until then, watch the `cache hit rate` telemetry line
and disable MTP manually when it dips under 90%.

Note: long generations need `CNE_CTX=1024` or higher — with MTP active the
default 256-token context aborts around token ~250 ("failed to find a
memory slot").

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
| `CNE_THREADS` | compute threads (default 8; 6 measured fastest on 4c/8t hardware) |
| `CNE_KV_Q8` | q8_0 KV cache (measured CPU regression, kept for re-measurement) |
| `CNE_VERBOSE` | restore the diagnostic stderr prints (`[geom]` window geometry, `[cb]` graph-node dump, `[cne] rebound` notices) that are silent by default in normal runs |
| `CNE_DUMP_DST`, `CNE_DUMP_LOGITS(_EVERY)` | dump tensors/logits for divergence hunts |
| `CNE_STEP_FILLS`, `CNE_FULL_FILL` | fill telemetry / whole-window fill probe |
| `CNE_LAYER_LIMIT` | restrict demand-serving to N layers (bisecting) |
| `CNE_SPLIT_PREFILL`, `CNE_MTP_NODRAFT` | prefill-shape and draft-isolation bisects |
| `CNE_IGNORE_EOS` | bench-only: keep generating after EOS for fixed-length throughput runs (does not change model math; not for serving) |
| `CNE_KERNELS` | CNE custom ggml-cpu kernels in `repack.cpp` (default **on**). `1` = fork fast path (q8 activation cache, MoE `mul_mat_id` dispatch, fused q4/q6 `2vx`/`4vx` GEMV). `0` = stock llama.cpp path; same tokens, for A/B. Measured **+10.6%** on `llama-bench` tg250 (5 runs/arm, i5-1135G7, t4). Scripts: `bench/scripts/lfm2/tg128-microbench.sh`, `server-velocity.sh`. See **§ Custom kernels — compatibility** below. |

### Custom kernels (`CNE_KERNELS`) — compatibility

One toggle controls all fork hooks in `ggml-cpu/repack.cpp`. Tokens stay
identical whether on or off; only the CPU kernel path changes.

**Validated:** LFM2-24B-A2B prepared Q4_K_M (+10.6% tg250 vs `CNE_KERNELS=0`).

**Likely compatible** (same top-4 MoE `mul_mat_id` pattern; not bench'd here yet):

- **LFM2 MoE variants** — `lfm2moe` arch, e.g. LFM2-8B-A1B, LFM2.5-8B-A1B
- **SmallThinker MoE** — `smallthinker` arch, `expert_used_count=4`, separate gate/up/down tensors

Run `cne_identity_gate` on any new artifact before velocity claims.

**Does not run** (falls back to stock llama `mul_mat_id`):

- `CNE_KERNELS=0`, or build without the `cne/cpu-kernels` fork
- GPU inference (CUDA / Metal / Vulkan) — CPU repack hooks are not used
- Prefill / multi-token batches — fast path requires single-token decode (`ne11==1`, `ne12==1`)
- MoE top-K other than 2 or 4
- `GGML_CPU_REPACK` off, or non-repacked weight layouts

**Runs with partial gain** (dispatch + q8 activation cache; no fused AVX2 `4vx`):

- x86 CPUs without AVX2 at compile time
- ARM / Apple Silicon, RISC-V, POWER, generic CPU builds

**Full fused SIMD** (measured +10.6% tg250 vs `CNE_KERNELS=0`):

- x86_64 + AVX2, fork build with `GGML_CPU_REPACK=ON`, `CNE_KERNELS=1`
- LFM2-24B-A2B prepared Q4_K_M (or other likely-compatible MoE above), CPU-only decode

Model-specific detail and decision tree: [models/lfm2-24b-a2b.md](models/lfm2-24b-a2b.md) § Compatibility.

Build-time debug machinery: configure with **`-DCNE_AUDIT=ON`** to compile
in the slice-corruption audit (per-fill records verified at the consuming
matmul's post-compute edge, reported as `[audit] ... SLICE CORRUPT`) and the
full-window integrity walk `stream_check_windows()`. Default builds compile
these out entirely — zero overhead, and the integrity check is a no-op.

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

## 11. OpenAI-compatible server (`cne_server`)

**Status:** working (single context; optional multi-turn KV reuse) · **Built with:** `-DCNE_BUILD_SERVER=ON`

Serving endpoint on the same runtime the bench measures - identical regime
classification, budget clamp, and demand-serving path.

| endpoint | what |
|---|---|
| `POST /v1/chat/completions` | OpenAI format; `"stream": true` for SSE; `temperature`/`top_p`/`seed` passthrough; greedy default = lossless |
| `GET /v1/models` | single-model list |
| `GET /health` | regime, dense policy, streaming, MTP depth, ctx, cache cap, **sessions**, **queue** |

Server-specific knobs:

| Variable | Effect |
|---|---|
| `CNE_THINK=0` | thinking off by default (requests may re-enable via `chat_template_kwargs.enable_thinking`) |
| `CNE_STREAM=0` | naive mmap decode - measured faster at ~1.4x RAM; streaming pays above ~1.6x |
| `CNE_KERNELS=0` | stock llama.cpp ggml-cpu path (A/B); default on when unset |
| `CNE_CACHE_GIB=N` | expert cache cap before budget clamping |
| `CNE_MAX_REQ_S=N` | wall budget per request in seconds; loud abort on exceed (default off) |
| `CNE_SESSION=0` | disable conversation KV reuse (default on when `conversation_id` is sent) |
| `CNE_SESSION_MAX=N` | LRU cap on tracked conversations; also sets `n_seq_max` KV lanes at boot (default 8 if unset; `cne_setup` usually suggests 2) |
| `CNE_KV_BPT=N` | KV bytes/token estimate for boot warnings (default ~20480) |

| `CNE_API_MODE` | `1` | Enable API key auth + `chat_id` tenancy |
| `CNE_API_KEY` | secret | Single development key |
| `CNE_API_KEYS` | `k1,k2` | Comma-separated keys |
| `CNE_API_KEYS_FILE` | path | One key per line (`key` or `key user_id`) |
| `CNE_API_RPM` | N | Per-user requests/minute (0 = off) |
| `CNE_SESSION_MAX_PER_USER` | N | Max parked chats per user (default 2 in API mode) |

Config file keys: `api_mode`, `api_keys` (array), `api_keys_file`, `api_rpm`,
`session_max_per_user`. See `tools/api_keys.example.txt` and **docs/GATEWAY.md**
for the two-tier key model (internal vs client keys).

**Public API:** run **`cne_gateway`** in front of API-mode `cne_server` on
`127.0.0.1`. Clients never call `cne_server` directly. Gateway docs:
**docs/GATEWAY.md** · serving architecture: **docs/SERVING.md**.

Multi-turn KV reuse: send the same `conversation_id` on each turn (JSON
field or `X-Conversation-Id` header). Turn 2+ prefills only the new prompt
tail; logs `[session] reused=N prefilled=M`. Different users can share one
server (requests still serialize); each `conversation_id` keeps its own KV
lane until global LRU eviction. Omit `conversation_id` for stateless behavior
(seq 0 only, cleared after each request). `clear_conversation: true` drops
cached KV for that id.

**Context lanes:** total `ctx` is split **evenly** at boot
(`per_lane = ctx / session_max`). Unused tokens in one chat are not given to
another. There is no server-side auto-truncate or summarize when a lane fills
— clients (or your API proxy) must trim the `messages` they send. If a user
opens more distinct `conversation_id`s than `session_max`, the least recently
used conversation is evicted (any user). There is no per-tenant isolation in
v1; see **docs/SERVING.md** for API-layer mitigations and the engine roadmap.

**Decode:** one request at a time (`gen_mutex`); multiple users are not batched
into one forward pass. `session_max` parks KV only.

**Requires `CNE_MTP=0` (or unset).** Sessions and draft-MTP cannot be
combined on the server — see §5 for why. Use sequential decode for
multi-turn or multi-user chat; use MTP only for stateless single-shot
requests.

The server also consumes a confirmed config file written by `cne-setup`
(`models/server.json`): precedence is environment variable > config file >
built-in default, and every resolved knob is logged with its source at
boot. Supported keys include `stream`, `kernels`, `dense`, `mtp`, `threads`,
`ctx`, `think`, `cache_gib`, `max_req_s`, `session_max`. See **docs/SETUP.md**.

Behavior notes:

- Requests serialize (single-decode runtime); `/health` reports
  `queue.waiting` and `queue.active`. Waits over 50ms log `[queue]`.
- A client disconnecting
  mid-stream stops generation cleanly - the socket is polled for liveness
  instead of trusting write() alone, so abandoned requests cannot hog the
  engine slot.
- Thinking-off uses the Qwen3-family empty-think assistant prefix and is
  deterministic once active. If a request budget expires while the model
  is inside a suppressed think block, the response carries an explicit
  notice instead of reasoning content.
- Context: prompts are prefilled in batch-sized chunks, so long agent
  system prompts (several thousand tokens) are safe. Size `ctx` to your
  workload; context costs ~20 KiB/token on hybrid-attention artifacts,
  plus KV headroom for generation.

**Live tests:** `server_e2e_live` forks this binary and checks
`/v1/chat/completions` plus session KV reuse. Config and `ctest` commands:
**docs/TESTING.md**.

**Multi-user API:** auth, quotas, fair session eviction, and client context
policy are documented in **docs/SERVING.md** (operate with a proxy now; tenant
sessions planned in-engine).
