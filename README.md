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

> **Status: pre-alpha.** Streaming pipeline, serving layer and the first-run
> setup CLI are working end-to-end; hardening and user-facing profiles are
> next. See [Roadmap](#roadmap).

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
guessing. Streaming adds one more condition: verify blocks touch several
draft positions' experts at once, so under cache pressure speculation turns
into wasted NVMe fills — **below ~90% expert-cache hit-rate, MTP is a net
loss and stays off**.

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
| Draft-MTP speculation | ![](https://img.shields.io/badge/status-working-brightgreen) | native Multi-Token Prediction head drafts k tokens per step; full model verifies. Lossless by construction (0% quality loss); tuned config measured at 4.86 tok/s vs 4.00 naive on the reference machine (+20%). **Not combinable with server `conversation_id` sessions** — see below |
| Custom CPU kernels (`CNE_KERNELS`) | ![](https://img.shields.io/badge/status-working-brightgreen) | fused AVX2 MoE GEMV in the pinned llama.cpp fork — q8 activation cache, `mul_mat_id` dispatch, q4 gate/up and q6 expert-down `2vx`/`4vx`. One toggle (`CNE_KERNELS=1` default); token-identical to stock. **+10.6%** on LFM2 tg250 vs `CNE_KERNELS=0` (i5-1135G7, t4) |
| OpenAI-compatible server | ![](https://img.shields.io/badge/status-working-brightgreen) | `/v1/chat/completions` (SSE), `/v1/models`, `/health` (sessions + queue); multi-turn KV via `conversation_id`; API mode (`chat_id`, per-user caps) |
| API gateway (`cne_gateway`) | ![](https://img.shields.io/badge/status-working-brightgreen) | Public API: client API keys, rate limits, proxies to `cne_server` on localhost — **docs/GATEWAY.md** |

Full per-feature guidance — including when *not* to use each one — lives in
**[docs/FEATURES.md](docs/FEATURES.md)**. Reference hardware, measured
velocity gains and reproduction commands: **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.

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

**Local / single-user** — client talks to `cne_server` directly:

```
Client (OpenAI SDK / Open WebUI / n8n)
        │  HTTP/SSE
        ▼
    cne_server ── OpenAI-compatible API
        ▼
    cne_runtime ── regime classifier → feature activation
        ▼                              ▼
  Memory manager                  Feature modules
        ▼                              ▼
     Expert LRU cache  ←────  O_DIRECT I/O lanes → NVMe / Flash (GGUF)
```

**Public multi-user API** — do not expose `cne_server` on the internet; use the
gateway (**docs/GATEWAY.md**, **docs/SERVING.md**):

```
Clients (Bearer API key)
        │  HTTPS (nginx/Caddy optional)
        ▼
    cne_gateway :8090     client keys, RPM, maps key → user_id
        │  internal API key + X-User-Id + chat_id
        ▼
    cne_server :8080      API mode, 127.0.0.1 only
        ▼
    (same runtime stack as above)
```

`cne_bench` (measurement) and `cne_setup` (confirmed `server.json` + optional
API mode fields) sit beside the server; the gateway reads `gateway.json`.

## Repository layout

Full guide: **[docs/STRUCTURE.md](docs/STRUCTURE.md)**.

```
core/include/cne/          public headers (model, memory, config…)
core/src/                  engine library (cne_core)
  gguf/                    GGUF reader, tensor classification, manifest registry
  memory/                  memory budgets + regime classification
  features/streaming/      slice cache · O_DIRECT file · I/O lane scheduler
runtime/                   llama.cpp seam + demand-serving runtime (cne_runtime)
  seam/                    thin llama backend probe (cne_adapter.*)
  cne_runtime.cpp          shared boot sequence (bench + server)
  cne_stream_cb.cpp        demand-serving runtime (fills, cache, anon dense)
  cne_stream_spec.cpp      draft-MTP generation loop
server/                    cne_server — OpenAI-compatible HTTP/SSE endpoint
gateway/                   cne_gateway — API-key public proxy (Path B)
cli/                       cne_setup — first-run config (detect, suggest, confirm)
tools/                     shipped binaries + operator scripts
  scripts/                 model download, canary, perplexity helpers
  cne_prepare, cne_bench, cne_identity_gate, dev probes (see tools/README.md)
bench/                     velocity harness scripts + local results (bench/README.md)
tests/                     unit and integration tests
  e2e/                     JSON configs for live tests (env + runtime)
  gguf/ memory/ features/streaming/   core library
  boot/ session/ server/   runtime + session + HTTP live checks
  perplexity/              PL drift-gate unit tests
docs/                      FEATURES · SETUP · SERVING · GATEWAY · TESTING · …
third_party/llama.cpp      pinned kernel fork (branch cne/cpu-kernels; kernels in `ggml-cpu/cne/`)
models/                    local GGUF artifacts (gitignored)
internal-docs/             private kernel/ops research (separate git repo, gitignored)
```

## Quick start

Requirements: Linux, CMake ≥ 3.16, C++17 compiler, ~50 GB free disk for the
reference model.

```sh
# build (include tests + server + setup for live E2E)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCNE_BUILD_SERVER=ON -DCNE_BUILD_CLI=ON -DCNE_BUILD_TESTS=ON
cmake --build build -j

# fetch + align a model (both scripts verify sha256 and run cne_prepare)
./tools/scripts/download-qwen3.6-35b-a3b-q4_k_xl.sh   # Qwen3.6-35B-A3B UD-Q4_K_XL + MTP (~22.9 GB)
./tools/scripts/download-lfm2-24b-a2b.sh              # LFM2-24B-A2B Q4_K_M, no-stream profile (~14.4 GB)
./tools/scripts/download-lfm2.5-8b-a1b.sh             # LFM2.5-8B-A1B AtomicChat UD-Q4_K_XL (~5.2 GB)
```

Both models then appear in `cne_setup`'s artifact picker automatically.
Recommended profiles:

| model | total / active | profile | notes |
|---|---|---|---|
| Qwen3.6-35B-A3B | 35B / 3B | mmap-dense + MTP k=8 | lossless speculation; needs `ctx ≥ 1024` |
| LFM2-24B-A2B | 24B / 2.3B | **no-stream** + warm dense, t4, ctx 4096 | hybrid conv+MoE; no MTP; ~10.5 tok/s server |
| LFM2.5-8B-A1B | 8.5B / ~1.5B | **4 GiB:** `dense=anon`, stream off, ctx 1024 · **16 GiB:** mmap-dense, ~18 tok/s | AtomicChat UD-Q4_K_XL; **docs/models/lfm2.5-8b-a1b.md** |

Bench CLI: `<gguf> [cache_cap_gib=8] [n_gen=64] [verify_n=64] [stream=1]`.
The cache cap is automatically clamped to the machine's real budget.

### Serve (OpenAI-compatible)

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCNE_BUILD_SERVER=ON -DCNE_BUILD_CLI=ON
cmake --build build --target cne_server cne_setup -j

# first run: detect hardware, preview the regime, confirm every setting
# interactively (or -y to accept the suggestions). Writes models/server.json;
# aborting writes nothing.
./build/cli/cne_setup

# boot with zero env vars - the confirmed config drives everything
./build/server/cne_server
```

Prefer explicit control? Skip `cne_setup` and pass knobs directly:

```sh
CNE_DENSE=mmap CNE_MTP=8 CNE_MTP_P_MIN=0.5 CNE_THREADS=6 \
    ./build/server/cne_server \
    models/qwen3.6-35b-a3b-q4_k_xl-mtp/Qwen3.6-35B-A3B-UD-Q4_K_XL-prepared.gguf \
    127.0.0.1 8080
```

Precedence: environment variables > `server.json` > built-in defaults; every
resolved knob is logged with its source at boot.

```sh
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":64}'
```

Single decode slot: requests serialize. Greedy by default (lossless);
`temperature`/`top_p`/`seed` honored per request. Disable thinking with
`CNE_THINK=0` or per-request `"chat_template_kwargs":{"enable_thinking":false}`.
Abandoned streaming requests abort within one poll interval instead of
hogging the engine slot; `CNE_MAX_REQ_S=<seconds>` adds an optional wall
cap per request.

**Chat sessions vs MTP** — choose one per `server.json` / env, not both:

| Goal | Config |
|---|---|
| Multi-turn or multi-user chat (`conversation_id`) | `CNE_MTP=0` (default) |
| Speculative decode speed (stateless) | `CNE_MTP=k`, no `conversation_id` |

With MTP on, the server ignores `conversation_id` and clears KV after every
request. Sessions need sequential decode so KV can be reused safely; MTP uses
dual contexts and mid-step KV rollbacks that session bookkeeping does not
track yet. Full rationale: **[docs/FEATURES.md](docs/FEATURES.md)** §5 and §11.

Full endpoint/knob reference in `docs/FEATURES.md` §11;
step-by-step setup in **[docs/SETUP.md](docs/SETUP.md)**.
**Multi-user API** (auth, `session_max`, proxy layer, roadmap):
**[docs/SERVING.md](docs/SERVING.md)**. **API-key gateway (Path B):**
**[docs/GATEWAY.md](docs/GATEWAY.md)**.
Live integration tests (JSON configs, `ctest`): **[docs/TESTING.md](docs/TESTING.md)**.

### Multi-user API (gateway)

For a **public** API, bind `cne_server` to `127.0.0.1` only and put
**`cne_gateway`** in front. Users get a **client API key** (no login/JWT).

```sh
# 1. Setup with API mode (writes models/server.json + models/api_keys.txt)
./build/cli/cne_setup -y

# 2. Engine (localhost, API mode from server.json)
./build/server/cne_server

# 3. Gateway (separate terminal)
cp gateway/gateway.json.example gateway/gateway.json   # set internal_api_key
cp gateway/api_keys.example.txt gateway/api_keys.local.txt  # client keys
./tools/scripts/run-gateway.sh

# 4. Client calls the gateway (not cne_server)
curl http://127.0.0.1:8090/v1/chat/completions \
  -H "Authorization: Bearer <client-api-key>" \
  -H "X-Chat-Id: my-chat" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":32}'
```

Two key tiers: **client API key** (gateway, per user) and **internal key**
(`models/api_keys.txt` / `CNE_INTERNAL_API_KEY`, gateway → engine only).
Full guide: **[docs/GATEWAY.md](docs/GATEWAY.md)** · architecture:
**[docs/SERVING.md](docs/SERVING.md)**.

Gateway unit tests: `cd gateway && pip install -r requirements.txt && PYTHONPATH=. pytest tests -q`.
Live stack: `ctest --test-dir build -R server_gateway_live --output-on-failure`
(needs GGUF + `gateway/.venv`).

<details>
<summary><strong>Environment knobs (development)</strong></summary>

| Variable | Values | Effect |
|---|---|---|
| `CNE_DENSE` | `mmap` \| `warm` \| `anon` | dense-weight residency policy (default: auto by regime). **`anon`** on 4 GiB MoE hosts: private dense copy + expert mmap trim (see **docs/models/lfm2.5-8b-a1b.md**) |
| `CNE_DENSE_DROP_MMAP` | `0` \| unset | unset = drop file-backed RSS after `anon` bind (default on); `0` = debug / A/B only |
| `CNE_KERNELS` | `1` \| `0` | custom ggml-cpu kernels in the pinned fork (default on); `0` = stock llama A/B |
| `CNE_LANES` | N | parallel slice-read workers (default 4) |
| `CNE_MTP` | `1` \| k | enable draft-MTP speculative decoding (depth k); **incompatible with server `conversation_id` sessions** |
| `CNE_MTP_P_MIN` | 0 < x ≤ 1 | draft-token confidence floor; keeps proposals honest, eliminates replay rounds |
| `CNE_SESSION` | `0` | disable conversation KV reuse on the server (default on when `conversation_id` is sent) |
| `CNE_SESSION_MAX` | N | LRU cap on tracked conversations + KV lanes; splits `ctx` evenly (see **docs/SERVING.md**) |
| `CNE_THREADS` | N | compute threads (default 8); physical-core count beats SMT on many laptops |
| `CNE_FA` | set = on | flash attention; ~+3% at ctx 1024, scales with context |
| `CNE_KV_Q8` | set = on | q8_0 KV cache; measured CPU regression on the reference machine, kept for re-measurement |
| `CNE_EXPERT_MASS` | 0 < x ≤ 1 | **lossy**: drop tail experts below cumulative routing mass |
| `CNE_EXPERT_MIN_K` | N | minimum experts kept when mass gating is active |
| `CNE_CTX` | N | context size |
| `CNE_MAX_REQ_S` | seconds | wall budget per request; loud abort on exceed (default off) |
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

On `cne_server`, do not enable MTP when using `conversation_id` for
multi-turn chat — the two modes are mutually exclusive (see **Chat sessions
vs MTP** above and `docs/FEATURES.md` §5).

**Live tests** — with a GGUF on disk:

```sh
ctest --test-dir build -R server_e2e_live --output-on-failure          # LFM2 HTTP
ctest --test-dir build -R server_e2e_qwen_live --output-on-failure     # Qwen HTTP
ctest --test-dir build -R server_e2e_lfm25_live --output-on-failure    # LFM2.5 HTTP
ctest --test-dir build -R 'server_api_live|server_api_per_user_live' --output-on-failure
ctest --test-dir build -R server_gateway_live --output-on-failure       # + gateway venv
```

See **[docs/TESTING.md](docs/TESTING.md)**.

## Roadmap

1. Streaming pipeline, budget manager, regime classification, tooling — **done**
2. Draft-MTP speculation: performance evaluation on CPU (telemetry +
   depth/probability sweeps); stays flag-gated until it beats sequential
3. Speculation telemetry: separate draft/verify timing to decide viability
   per hardware class
4. Mixed-precision miss serving (opt-in lossy profile)
5. `cne-server` + **API gateway** for multi-tenant serving — **done** (API mode,
   `cne_gateway`, live E2E); fairness/autoscale in **docs/SERVING.md** §5
6. User-facing quality profiles (`lossless` / `balanced` / `fast`) —
   config-file plumbing shipped via `cne-setup`
7. Tenant-aware sessions, fairness, autotune, packaging — see **docs/SERVING.md** §5

Non-goals: GPU offloading, training/fine-tuning, dense-model optimization
(that's llama.cpp's job), Windows/macOS support initially.

---

<div align="center">

Part of the [Caedral](https://caedral.com) ecosystem —
prepaid AI infrastructure for automation agencies.

MIT — see [LICENSE](LICENSE).

</div>
