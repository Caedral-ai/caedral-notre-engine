# LFM2.5-8B-A1B — AtomicChat UD-Q4_K_XL (canonical runtime model)

> Liquid AI's LFM2.5 MoE checkpoint at 8.5B total / ~1.5B active parameters.
> Same `lfm2moe` family as LFM2-24B-A2B (top-4 experts, hybrid shortconv + GQA),
> but small enough to run **mmap-dense** on 16 GiB-class laptops.
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
| Chat template | `LFM2.5-8B-A1B.jinja` (`enable_thinking` supported) |
| Regeneration script | `tools/scripts/download-lfm2.5-8b-a1b.sh` |
| Prepared artifact | `models/lfm2.5-8b-a1b/lfm25-8b-a1b-UD-Q4_K_XL-prepared.gguf` |

## 2. Why this quant?

CNE pins **AtomicChat UD-Q4_K_XL** (not the official LiquidAI `Q4_K_M`) because:

- **imatrix calibration** — better 4-bit quality than plain post-training quants
- **UD-Q4_K_XL** — same ~5.2 GiB footprint as `Q4_K_M`, with higher-precision
  embedding and output layers (same pattern as the Qwen3.6 artifact in this repo)

| AtomicChat quant | size | notes |
|---|---|---|
| **UD-Q4_K_XL** (default) | ~5.2 GiB | CNE canonical |
| Q4_K_M | ~5.2 GiB | standard imatrix K-quant |
| Q6_K | ~7.0 GiB | near-lossless |
| IQ4_XS | ~4.6 GiB | smaller 4-bit imatrix |

## 3. Prepare / alignment

After download, `cne_prepare` writes the prepared variant:

```
validation: io_alignment=4096 routed=66 all_aligned=66 misaligned=0 -> OK
```

Prepared size: ~4.86 GiB. Expert tensors are 4096-aligned for O_DIRECT streaming
if you ever enable `stream` on larger hosts; on 16 GiB machines **mmap-dense**
(`stream: false`) is the typical profile.

## 4. Validation (2026-08-31)

| check | result |
|---|---|
| sha256 after download | OK (pinned in download script) |
| `cne_prepare` alignment | 66/66 routed tensors, 4096-aligned |
| `cne_identity_gate` | **PASS** (greedy tokens, rebind=1) |
| Regime on 16 GiB host | **R0_RESIDENT** (~11.5 GiB RAM available) |
| `server_e2e_lfm25_live` | **PASS** — chat + session KV reuse |

Custom CPU kernels (`CNE_KERNELS=1`): same top-4 MoE fast path as LFM2-24B-A2B.
Run `cne_identity_gate` after any quant swap before velocity claims.

## 5. Serving profile

Suggested starting point (`cne_setup` on a 16 GiB machine):

| knob | value | notes |
|---|---|---|
| `stream` | off | weights fit RAM; streaming not needed |
| `dense` | mmap or warm | R0_RESIDENT — either works |
| `ctx` | 4096–8192 | native 128K is impractical on CPU without huge RAM |
| `session_max` | 2 | splits `ctx` evenly per parked chat |
| `mtp` | 0 | no MTP head in this GGUF |
| `threads` | 4 | LFM family on Tiger Lake–class chips |
| `kernels` | on | MoE decode fast path (see **docs/models/lfm2-24b-a2b.md**) |
| `think` | off for APIs | `CNE_THINK=0` or `enable_thinking: false` per request |

Velocity: expect **well above** LFM2-24B-A2B on the same CPU (smaller active
MoE). Bench numbers not yet pinned — re-run after hardware changes.

## 6. Quick start

```sh
# fetch + verify + prepare (~5 min on a good connection)
./tools/scripts/download-lfm2.5-8b-a1b.sh

# write models/server.json (pick the prepared artifact)
./build/cli/cne_setup -y --model lfm2.5

# serve
./build/server/cne_server
```

Smoke test:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}],
       "chat_template_kwargs":{"enable_thinking":false},
       "max_tokens":32}'
```

For a **public multi-user API**, put `cne_gateway` in front — see
**docs/GATEWAY.md** and **docs/SERVING.md**.

## 7. Live integration tests

Config: `tests/e2e/server_e2e_lfm25_live.json`

```sh
ctest --test-dir build -R server_e2e_lfm25_live --output-on-failure
```

Sets `CNE_API_MODE=0` so the test is not blocked by a local `models/server.json`
with API mode enabled. Thinking is off (`CNE_THINK=0` + `enable_thinking: false`).

See [TESTING.md](../TESTING.md).

## 8. Related

- LFM2-24B-A2B (kernels, velocity A/B): [lfm2-24b-a2b.md](lfm2-24b-a2b.md)
- Custom kernels: [FEATURES.md](../FEATURES.md) § Custom kernels
- Setup guide: [SETUP.md](../SETUP.md)
