# Live integration tests

Opt-in tests that load a real GGUF (LFM2 by default). Fast unit tests run
without a model; live tests **skip** when the configured artifact is missing.

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCNE_BUILD_TESTS=ON -DCNE_BUILD_SERVER=ON
cmake --build build -j
```

## JSON configs (`tests/e2e/`)

Live tests read env and runtime settings from JSON instead of long `ctest`
`ENVIRONMENT` strings. Each ctest name maps to a config file:

| ctest name | config file | what it checks |
|---|---|---|
| `server_e2e_live` | `server_e2e_live.json` | LFM2: fork `cne_server` → chat + session KV reuse |
| `server_api_live` | `server_api_live.json` | API mode: auth, `chat_id`, per-user sessions |
| `server_api_per_user_live` | `server_api_per_user_live.json` | API mode: `session_max_per_user` eviction (fails fast on boot errors) |
| `server_gateway_live` | `server_gateway_live.json` | API-key gateway → CNE: chat, KV reuse |
| `server_e2e_qwen_live` | `server_e2e_qwen_live.json` | Qwen3.6: same HTTP path; `CNE_MTP=0`, thinking off |
| `session_kv_live` | `session_kv_live.json` | LFM2 runtime session KV reuse, parity, alternating `conversation_id`s (no auth) |
| `session_kv_qwen_live` | `session_kv_qwen_live.json` | Qwen3.6 runtime session KV (slow) |
| `session_bigctx_live` | `session_bigctx_live.json` | LFM2 ~7k-token prefill at ctx=8192 (slow) |

Example (`server_e2e_live.json`):

```json
{
  "description": "LFM2 HTTP E2E — cne_server fork, chat + session KV reuse",
  "model": "models/lfm2-24b-a2b/LFM2-24B-A2B-Q4_K_M-prepared.gguf",
  "env": {
    "CNE_STREAM": "0",
    "CNE_MTP": "0",
    "CNE_CTX": "2048",
    "CNE_CACHE_GIB": "0",
    "CNE_THREADS": "4",
    "CNE_SESSION_MAX": "2"
  },
  "options": {
    "port": 0,
    "boot_timeout_s": 180
  }
}
```

| JSON field | purpose |
|---|---|
| `model` | GGUF path (relative to repo root) |
| `env` | applied via `setenv` before boot (`CNE_*` keys) |
| `runtime` | `cap_gib`, `n_ctx`, `n_threads`, `stream_on` for direct runtime tests |
| `options` | test-specific: `port`, `boot_timeout_s`, `prompt_tokens`, `gen_tokens` |
| `chat` | HTTP E2E: `conversation_id`, `max_tokens`, `think_off` (Qwen3) |

**Precedence:** `CNE_TEST_MODEL` overrides `model`; `CNE_E2E_CONFIG` overrides
the default config path.

## Run

From the **build** directory (not the repo root):

```sh
# HTTP + API + gateway (slow — needs GGUF; gateway test needs Python venv)
ctest --test-dir build -R 'server_e2e_live|server_api_live|server_api_per_user_live|server_gateway_live' --output-on-failure

# Qwen models (if downloaded)
ctest --test-dir build -R 'server_e2e_qwen_live|session_kv_qwen_live' --output-on-failure

# fast subset only (unit tests; no GGUF)
ctest --test-dir build -LE slow --output-on-failure

# one test
ctest --test-dir build -R server_e2e_live --output-on-failure
```

Custom config or model:

```sh
CNE_E2E_CONFIG=tests/e2e/server_e2e_live.json \
  ctest --test-dir build -R server_e2e_live --output-on-failure

CNE_TEST_MODEL=/path/to/model.gguf \
  ctest --test-dir build -R session_kv_live --output-on-failure
```

Fetch the default LFM2 artifact:

```sh
./tools/scripts/download-lfm2-24b-a2b.sh
```

Qwen3.6 reference model (~21 GiB):

```sh
./tools/scripts/download-qwen3.6-35b-a3b-q4_k_xl.sh
ctest --test-dir build -R server_e2e_qwen_live --output-on-failure
```

Multi-user API and gateway: **docs/SERVING.md**, **docs/GATEWAY.md**.

**Gateway unit tests** (no model):

```sh
cd gateway && python -m venv .venv && .venv/bin/pip install -r requirements.txt
ctest --test-dir build -R gateway_unit --output-on-failure
```

**Qwen + sessions:** MTP and `conversation_id` are mutually exclusive — Qwen
E2E configs set `CNE_MTP=0`. Use separate MTP benchmarks for speculative
decode (`docs/FEATURES.md` §5).

## Labels and timeouts

| Label | tests | typical time |
|---|---|---|
| *(none)* | `session_lcp`, `session_tenant`, `api_unit`, `gateway_unit`, `kv_budget`, … | < 1 s |
| `slow` | `server_e2e_live`, `server_e2e_qwen_live`, `server_api_live`, `server_api_per_user_live`, `server_gateway_live`, `session_kv_qwen_live`, `session_bigctx_live` | ~20 s – few min |

`session_kv_live` loads the full model but finishes in ~20–40 s on LFM2.
`server_api_per_user_live` ~30 s (LFM2). Qwen HTTP E2E ~60–80 s each.

## Gateway live test

`server_gateway_live` forks `cne_server` + `cne_gateway`, checks 401 without a
client key, runs two-turn chat with KV reuse. Requires:

```sh
cd gateway && python -m venv .venv && .venv/bin/pip install -r requirements.txt
ctest --test-dir build -R server_gateway_live --output-on-failure
```

## Server E2E details

`server_e2e_live` forks `cne_server`, waits for `/health`, posts two
non-streaming `/v1/chat/completions` requests with the same
`conversation_id`, and asserts turn 2 logs session KV reuse. Server stderr
is written to `/tmp/cne_server_e2e_<pid>.log` during the run.

Manual smoke (same knobs as the JSON config):

```sh
CNE_STREAM=0 CNE_MTP=0 CNE_CTX=2048 CNE_CACHE_GIB=0 CNE_SESSION_MAX=2 \
  ./build/server/cne_server models/lfm2-24b-a2b/LFM2-24B-A2B-Q4_K_M-prepared.gguf \
  127.0.0.1 8080
```

## Unit tests (no model)

```sh
ctest --test-dir build -LE slow -E 'session_kv_live|session_bigctx_live|server_e2e_live'
```

Or run everything except slow:

```sh
ctest --test-dir build -LE slow
```

Note: `session_kv_live` is not labeled `slow` but still needs the GGUF; use
`-E` above if the model is not on disk.
