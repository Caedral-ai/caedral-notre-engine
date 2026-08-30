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
| `server_e2e_live` | `server_e2e_live.json` | fork `cne_server` → `/health`, `/v1/models`, 2-turn chat + `[session] reused=` |
| `session_kv_live` | `session_kv_live.json` | runtime session KV reuse, parity, alternating users |
| `session_bigctx_live` | `session_bigctx_live.json` | ~7k-token chunked prefill at ctx=8192, turn-2 reuse (slow) |

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

**Precedence:** `CNE_TEST_MODEL` overrides `model`; `CNE_E2E_CONFIG` overrides
the default config path.

## Run

From the **build** directory (not the repo root):

```sh
# all live tests (includes slow — several minutes on LFM2)
ctest --test-dir build -R 'server_e2e_live|session_kv_live|session_bigctx_live' --output-on-failure

# fast subset only (unit tests + session_lcp; no GGUF)
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

## Labels and timeouts

| Label | tests | typical LFM2 time |
|---|---|---|
| *(none)* | `session_lcp`, `kv_budget`, `memory_budget`, … | < 1 s |
| `slow` | `server_e2e_live`, `session_bigctx_live` | ~20 s – ~9 min |

`session_kv_live` loads the full model but finishes in ~20–40 s.

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
