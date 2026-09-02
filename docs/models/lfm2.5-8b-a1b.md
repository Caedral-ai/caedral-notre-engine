# LFM2.5-8B-A1B — AtomicChat UD-Q4_K_XL (canonical runtime model)

> Liquid AI's LFM2.5 MoE checkpoint at 8.5B total / ~1.5B active parameters.
> Same `lfm2moe` family as LFM2-24B-A2B (top-4 experts, hybrid shortconv + GQA),
> but small enough for **mmap-dense** on 16 GiB-class laptops, or **`dense=anon`**
> on **4 GiB** hosts (UD-Q4_K_XL, no smaller quant required).
>
> STATUS: **WORKING** — download + prepare OK, `cne_identity_gate` PASS,
> HTTP E2E (`server_e2e_lfm25_live`) PASS (2026-08-31).

---

## 1. Identity

| Field | Value |
|---|---|
| Source | [`AtomicChat/LFM2.5-8B-A1B-GGUF`](https://huggingface.co/AtomicChat/LFM2.5-8B-A1B-GGUF) |
| Base model | [`LiquidAI/LFM2.5-8B-A1B`](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) |
| File | `lfm25-8b-a1b-UD-Q4_K_XL.gguf` |
| Quantization | imatrix **UD-Q4_K_XL** (dynamic 4-bit; embeddings + output at Q8_0) |
| Downloaded size | 5,219,053,088 bytes (~4.86 GiB) |
| sha256 | `283b12943743c5bb54b7f9fc8f7076c8ea32d163a06fb0e6f525178e7232c588` |
| GGUF architecture key | `lfm2moe` |
| Experts | 32 routed, top-4 active |
| Context (native) | 128,000 tokens |
| MTP | none — sequential decode only (`mtp_k=0`) |
| Chat template | `LFM2.5-8B-A1B.jinja` |
| Thinking | **Always on** — LFM2.5 always reasons; think blocks (or plain narration) appear in every reply. **`"think": false` does not disable thinking** — the server always returns reasoning content. Budget **`max_tokens` 128+** for short answers. |
| Regeneration script | `tools/scripts/download-lfm2.5-8b-a1b.sh` |
| Prepared artifact | `models/lfm2.5-8b-a1b/lfm25-8b-a1b-UD-Q4_K_XL-prepared.gguf` |

## 2. Why this quant?

CNE pins **AtomicChat UD-Q4_K_XL** (not the official LiquidAI `Q4_K_M`) because:

- **imatrix calibration** — better 4-bit quality than plain post-training quants
- **UD-Q4_K_XL** — same ~5.2 GiB footprint as `Q4_K_M`, with higher-precision
  embedding and output layers (same pattern as the Qwen3.6 artifact in this repo)

| AtomicChat quant | size | notes |
|---|---|---|
| **UD-Q4_K_XL** (default) | ~5.2 GiB | CNE canonical; **fits 4 GiB RAM with `dense=anon`** |
| Q4_K_M | ~5.2 GiB | standard imatrix K-quant |
| Q6_K | ~7.0 GiB | near-lossless |
| IQ4_XS | ~4.6 GiB | smaller 4-bit imatrix (optional; not required for 4 GiB) |

## 3. Prepare / alignment

After download, `cne_prepare` writes the prepared variant:

```
validation: io_alignment=4096 routed=66 all_aligned=66 misaligned=0 -> OK
```

Prepared size: ~4.86 GiB. Expert tensors are 4096-aligned for O_DIRECT streaming
when `stream` is on. Profile by host:

| RAM | `stream` | `dense` | `ctx` | notes |
|---|---|---|---|---|
| **4 GiB** | off | **anon** | 1024 | mmap drop + per-step expert trim (~2.9 GiB peak) |
| **4 GiB** (faster) | on | anon | 1024 | `cache_gib: 3` (~3.8 GiB peak, ~7.6 tok/s) |
| **16 GiB+** | off | mmap | 2048–4096 | default velocity (~18 tok/s) |

## 4. Validation (2026-08-31)

| check | result |
|---|---|
| sha256 after download | OK (pinned in download script) |
| `cne_prepare` alignment | 66/66 routed tensors, 4096-aligned |
| `cne_identity_gate` | **PASS** (greedy tokens, rebind=0/1) |
| `CNE_KERNELS` tg250 A/B (`llama-bench`, t4, pp512, 5 reps) | **PASS** — **14.79 ± 0.23** vs **12.66 ± 0.11** off (**+16.8%**) |
| Regime on 16 GiB host | **R0_RESIDENT** (~11.5 GiB RAM available) |
| `server_e2e_lfm25_live` | **PASS** — chat + session KV reuse |

Custom CPU kernels (`CNE_KERNELS=1`): **validated** on this artifact — **+16.8%**
tg250 vs stock (`llama-bench`, t4, pp512, 5 reps, 2026-08-31). Same top-4 MoE
fast path as LFM2-24B-A2B. Re-run `cne_identity_gate` after any quant swap.
Internal plan: `internal-docs/plans/LFM2.5-8B-A1B_STREAM_KERNELS.md`.

## 5. Measured (2026-08-31, i5-1135G7-class host)

### Kernel A/B (`llama-bench` tg250, pp512 warmup, t4, 5 reps/arm, `CNE_STREAM=0`)

| `CNE_KERNELS` | tg250 tok/s | notes |
|---|---|---|
| **1** (on) | **14.79 ± 0.23** | fused MoE q4/q6 GEMV |
| 0 (off) | 12.66 ± 0.11 | stock llama path |
| **delta** | **+16.8%** | gate: ≥ +5% (same bar as LFM2-24B) |

Graph census (`cne_graph_census`): MoE matmul **~48%** of instrumented wall
(q4 gate/up **31.6%**, q6 down **16.7%**); shortconv **17.6%**; embed/output q8_0
**18.1%** (stock path).

```sh
./bench/scripts/lfm2.5/tg250-kernel-ab.sh
# → bench/results/lfm25-tg250-kernels.tsv
```

### Smoke (64 generated tokens, `cne_bench`)

| profile | RSS (HWM) | tok/s | notes |
|---|---|---|---|
| mmap-dense, stream off, ctx 2048, t4 | **~4.96 GiB** | **~12–15** | `llama-bench` tg250 **14.79**; `cne_bench` ~12 @ 250 tok |
| **dense anon, stream off, ctx 1024** | **~2.9 GiB peak** | **~12** | **4 GiB target** — mmap drop + per-step expert trim |
| dense anon, stream on, 3 GiB cache | ~3.8 GiB peak | ~7.6 | 90% hit-rate @ 64 tok |
| stream on, 1 GiB expert cache (mmap dense) | ~6.2 GiB | ~1.9 | legacy — do not use for 4 GiB |

### Sustained decode (250 generated tokens, `cne_bench`, greedy, `CNE_IGNORE_EOS=1`)

| profile | RSS (HWM) | tok/s | wall | notes |
|---|---|---|---|---|
| **anon, stream off, ctx 1024, t4** | **2.89 GiB** | **11.1** | 22.5 s | **4 GiB recommended** (tuned) |
| mmap-dense, stream off, ctx 2048, t4 | 4.96 GiB | **12.0** | 20.8 s | **≥ 6 GiB velocity** (tuned) |
| anon, stream on, **3 GiB** cache, t4, **lanes 2** | **4.31 GiB** | **11.14** | ~22 s | **96.9% hit**; kernels on, prefetch off |
| anon, stream on, **2 GiB** cache, t4, lanes 4 | 3.65 GiB | 4.65 | ~54 s | **83% hit** — below 90% rule; avoid |

At 250 tokens, **anon + no stream** and **stream + 3 GiB cache** are now neck-and-neck
on velocity (~**11 tok/s**); **2 GiB cache** collapses to ~**4.7 tok/s** (I/O-bound).
Use streaming only when LRU residency is required (multi-tenant / colocation).

#### Expert cache sweep (250 tok, `CNE_STREAM=1`, `CNE_KERNELS=1`, `CNE_PREFETCH=0`)

`cne_bench` greedy, `CNE_IGNORE_EOS=1`, `dense=anon`, `ctx=1024`, `threads=4`,
i5-1135G7-class host, 2026-08-31.

| `cache_gib` | `CNE_LANES` | tok/s | hit % | HWM | notes |
|---|---|---|---|---|---|
| **3** | **2** | **11.14** | **96.88** | **4.31 GiB** | **tuned streaming default** |
| 3 | 4 | 10.84 | 96.88 | 4.35 GiB | |
| 3 | 2 (mmap) | 10.20 | 96.88 | 8.73 GiB | higher RSS, slower |
| **2** | **4** | **4.65** | **82.94** | 3.65 GiB | best @ 2 GiB; still I/O-bound |
| 2 | 2 | 4.39 | 82.94 | 3.65 GiB | |

**Rule of thumb:** stay **≥ 3 GiB** expert cache on this artifact for **≥ 90%**
hit-rate. Dropping to 2 GiB costs ~**2.4×** throughput despite lower peak RSS.

```sh
# 3 GiB — tuned streaming (t4, lanes 2, prefetch off)
CNE_STREAM=1 CNE_CACHE_GIB=3 CNE_KERNELS=1 CNE_PREFETCH=0 \
CNE_THREADS=4 CNE_LANES=2 CNE_DENSE=anon CNE_CTX=1024 CNE_IGNORE_EOS=1 \
  ./build/tools/cne_bench \
  models/lfm2.5-8b-a1b/lfm25-8b-a1b-UD-Q4_K_XL-prepared.gguf 3 250 250 1

# 2 GiB — not recommended (83% hit, ~4.7 tok/s)
CNE_STREAM=1 CNE_CACHE_GIB=2 CNE_KERNELS=1 CNE_PREFETCH=0 \
CNE_THREADS=4 CNE_LANES=4 CNE_DENSE=anon CNE_CTX=1024 CNE_IGNORE_EOS=1 \
  ./build/tools/cne_bench \
  models/lfm2.5-8b-a1b/lfm25-8b-a1b-UD-Q4_K_XL-prepared.gguf 2 250 250 1
```

#### No-stream tuning (250 tok, `CNE_STREAM=0`, `CNE_KERNELS=1`)

**4 GiB — `dense=anon`** (peak RSS **~2.89 GiB** across all sweeps):

**Threads** (`ctx=1024`):

| threads | tok/s (best run) | notes |
|---|---|---|
| 2 | 8.36 | |
| 3 | 7.41 | |
| **4** | **11.11** | **tuned default** |
| 6 | 10.39 | flat vs t4 |
| 8 | 6.00 | **avoid** — SMT regression |

**Context** (`threads=4`):

| ctx | tok/s (best run) | peak RSS |
|---|---|---|
| 512 | 7.39 | 2.88 GiB |
| **1024** | **11.11** | 2.89 GiB | **recommended** |
| 2048 | 9.85 | 2.90 GiB | same RSS band; use for longer sessions |
| 4096 | 8.43 | 2.92 GiB | diminishing returns |

`CNE_KERNELS=0` and `CNE_FA=1` were flat (±3%) on anon at t4/ctx 1024.

**16 GiB+ — `dense=mmap`** (RSS **~4.96 GiB**, independent of ctx):

| threads | tok/s (best run, ctx 2048) | notes |
|---|---|---|
| 2 | 9.26 | |
| **4** | **12.01** | **tuned default** |
| 6 | 10.43 | |
| 8 | 4.46 | **avoid** |

| ctx (t4) | tok/s (best run) |
|---|---|
| 1024 | 7.73 |
| **2048** | **12.01** |
| 4096 | 8.35 |

Repeat runs under host load showed **~7 tok/s** (anon) and **~8.3 tok/s**
(mmap) — treat **t4 + avoid t8** as the stable finding; absolute tok/s varies
with machine state. Peak numbers above are best-of sweep on a quiet host.

**Tuned no-stream profiles:**

| host | config | tok/s (best) | peak RSS |
|---|---|---|---|
| **4 GiB** | `dense=anon`, t4, ctx 1024 | **11.1** | **2.89 GiB** |
| **16 GiB+** | `dense=mmap`, t4, ctx 2048 | **12.0** | 4.96 GiB |

#### Streaming tuning (`dense=anon`, 3 GiB cache, 250 tok, `CNE_PREFETCH=0`)

Hit-rate was **96.88%** across cache/lane sweeps below. RSS **~4.3 GiB** HWM
with 3 GiB cache (vs ~3.65 GiB @ 2 GiB cache).

**Cache size** (`CNE_LANES=2`, `CNE_KERNELS=1`, t4) — see table in §5 above.
Prefer **3 GiB**; 2 GiB falls to **~83%** hit and **~4.7 tok/s**.

**Threads** (`CNE_LANES=4` default, earlier sweep @ 3 GiB):

| threads | tok/s | wall | notes |
|---|---|---|---|
| 2 | 7.50 | 33.3 s | |
| **4** | **7.84** | 31.9 s | best compute/I/O balance (pre-kernel re-measure) |
| 6 | 7.75 | 32.3 s | flat vs t4 |
| 8 | 3.00 | 83.4 s | **avoid** — SMT starves I/O fill workers |

**I/O lanes** (`CNE_THREADS=4`, `CNE_KERNELS=1`, 3 GiB, 2026-08-31):

| lanes | tok/s | HWM | notes |
|---|---|---|---|
| **2** | **11.14** | 4.31 GiB | **tuned default** |
| 4 | 10.84 | 4.35 GiB | |

**Tuned streaming profile:** `cache_gib: 3`, `threads=4`, `CNE_LANES=2`,
`prefetch: false`, `kernels: true` → **11.14 tok/s** @ 250 tok (matches tuned
no-stream on a quiet host).

Reproduce smoke: `./bench/scripts/lfm2.5/memory-profile.sh`

### 4 GiB memory target

UD-Q4_K_XL fits **≤ 4 GiB RSS** with **`CNE_DENSE=anon`** (no smaller quant needed):

1. **No MAP_POPULATE** on load when `dense=anon` (llama fork).
2. **Dense weights** copied to anonymous memory (~522 MiB); file-backed dense pages dropped immediately.
3. **Expert weights** mmap-trimmed after boot and **each decode step** when stream is off (demand-paged from disk; peak ~2.9 GiB HWM on 250-token run).
4. Optional: **`stream: on` + 3 GiB cache** — peak **~4.3 GiB** @ 250 tok.
   Tuned **`threads=4`, `CNE_LANES=2`, `prefetch: false`** → **11.14 tok/s**
   (96.9% hit). **Do not use 2 GiB** cache (~83% hit, ~4.7 tok/s).

```sh
# 4 GiB profile — no stream, tuned (t4, ctx 1024)
CNE_STREAM=0 CNE_DENSE=anon CNE_KERNELS=1 CNE_THREADS=4 CNE_CTX=1024 CNE_IGNORE_EOS=1 \
  ./build/tools/cne_bench \
  models/lfm2.5-8b-a1b/lfm25-8b-a1b-UD-Q4_K_XL-prepared.gguf 0 250 250 0

# 4 GiB profile — stream + 3 GiB cache (tuned: t4, lanes 2, prefetch off)
CNE_STREAM=1 CNE_CACHE_GIB=3 CNE_KERNELS=1 CNE_PREFETCH=0 CNE_THREADS=4 CNE_LANES=2 CNE_CTX=1024 CNE_IGNORE_EOS=1 \
  ./build/tools/cne_bench \
  models/lfm2.5-8b-a1b/lfm25-8b-a1b-UD-Q4_K_XL-prepared.gguf 3 250 250 1

# 16 GiB velocity — no stream, tuned (t4, ctx 2048)
CNE_STREAM=0 CNE_DENSE=mmap CNE_KERNELS=1 CNE_THREADS=4 CNE_CTX=2048 CNE_IGNORE_EOS=1 \
  ./build/tools/cne_bench \
  models/lfm2.5-8b-a1b/lfm25-8b-a1b-UD-Q4_K_XL-prepared.gguf 0 250 250 0
```

Disable mmap drop (debug only): `CNE_DENSE_DROP_MMAP=0`.

## 6. Serving profile

**16 GiB+ (velocity):**

| knob | value | notes |
|---|---|---|
| `stream` | off | lowest overhead on R0 |
| `dense` | mmap | ~4.96 GiB RSS |
| `ctx` | **2048** | tuned @ 250 tok; 4096 regresses |
| `threads` | **4** | avoid 8 (SMT) |
| `kernels` | on | |

**4 GiB RAM:**

| knob | value | notes |
|---|---|---|
| `stream` | off | per-step expert trim keeps RSS under cap |
| `dense` | **anon** | required — binds ~522 MiB dense, drops expert mmap |
| `ctx` | **1024** | tuned; 2048 OK for longer sessions (same ~2.9 GiB peak) |
| `threads` | **4** | avoid 8 (SMT) |
| `kernels` | on | |
| `cache_gib` | 0 | stream off |

Optional: `stream: on`, `cache_gib: 3`, `prefetch: false`, `dense: anon`,
`threads: 4`, env `CNE_LANES=2` for LRU expert residency (~**11.1 tok/s @ 250
tok**, ~**4.3 GiB** peak, **97% hit**). Use **`cache_gib: 2` only if RAM is
hard-capped — expect ~**4.7 tok/s** and **~83%** hit.

Example `models/server.json` for **4 GiB + streaming** (multi-tenant / LRU):

```json
{
  "model": "lfm2.5-8b-a1b/lfm25-8b-a1b-UD-Q4_K_XL-prepared.gguf",
  "stream": true,
  "prefetch": false,
  "cache_gib": 3,
  "dense": "anon",
  "ctx": 1024,
  "threads": 4,
  "kernels": true,
  "mtp": 0,
  "session_max": 1,
  "think": true,
  "host": "127.0.0.1",
  "port": 8080
}
```

Set **`CNE_LANES=2`** in the server environment (not yet in `server.json`).

Example `models/server.json` for **4 GiB** (single-session, fastest):

```json
{
  "model": "lfm2.5-8b-a1b/lfm25-8b-a1b-UD-Q4_K_XL-prepared.gguf",
  "stream": false,
  "prefetch": false,
  "dense": "anon",
  "ctx": 1024,
  "threads": 4,
  "kernels": true,
  "mtp": 0,
  "session_max": 1,
  "think": true,
  "host": "127.0.0.1",
  "port": 8080
}
```

Common:

| knob | value | notes |
|---|---|---|
| `session_max` | 2 | splits `ctx` evenly per parked chat |
| `mtp` | 0 | no MTP head in this GGUF |
| `think` | **ignored** | LFM2.5 **always thinks**. `"think": false` and `enable_thinking: false` do not turn reasoning off — output is never stripped. Use **Qwen3.6** or **LFM2-24B** if you need no-think APIs. Set **`max_tokens` 128+**. Gateway `allow_thinking` does not change LFM2.5. |

## 7. Quick start

```sh
# fetch + verify + prepare (~5 min on a good connection)
./tools/scripts/download-lfm2.5-8b-a1b.sh

# write models/server.json (on 4 GiB hosts, set dense=anon — see §6)
./build/cli/cne_setup -y --model lfm2.5

# serve
./build/server/cne_server
```

Smoke test:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}],
       "max_tokens":128}'
```

For a **public multi-user API**, put `cne_gateway` in front — see
**docs/GATEWAY.md** (chat policy: `allow_thinking`, `max_tokens_per_request`)
and **docs/SERVING.md**.

## 8. Live integration tests

Config: `tests/e2e/server_e2e_lfm25_live.json`

```sh
ctest --test-dir build -R server_e2e_lfm25_live --output-on-failure
```

Sets `CNE_API_MODE=0` so the test is not blocked by a local `models/server.json`
with API mode enabled. (`think` is ignored on LFM2.5 — model always reasons.)

See [TESTING.md](../TESTING.md).

## 9. Related

- LFM2-24B-A2B (kernels, velocity A/B): [lfm2-24b-a2b.md](lfm2-24b-a2b.md)
- Custom kernels: [FEATURES.md](../FEATURES.md) § Custom kernels
- Setup guide: [SETUP.md](../SETUP.md)
