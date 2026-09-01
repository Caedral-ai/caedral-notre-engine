# cne-setup: server setup guide

`cne-setup` configures `cne-server` for your machine. It detects your
hardware, shows what the engine WOULD pick for the detected model, and lets
you confirm or override every setting before anything is written.

The philosophy in one line: **detection informs, you decide.** Nothing is
applied automatically; aborting at any point writes nothing.

---

## 1. Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCNE_BUILD_SERVER=ON -DCNE_BUILD_CLI=ON -DCNE_BUILD_TESTS=ON
cmake --build build --target cne_prepare cne_setup cne_server -j
```

`cne-setup` links only the engine core - it never loads a model, so it runs
in seconds even before any artifact exists on disk.

Debug builds: add `-DCNE_AUDIT=ON` to compile in the slice-audit and
window-integrity verification machinery (see docs/FEATURES.md §10). Not
needed for normal use.

## 2. Get a model

If you have a GGUF already, place it under `models/`. For the reference
artifact:

```sh
./tools/scripts/download-qwen3.6-35b-a3b-q4_k_xl.sh   # reference model (~22.9 GB)
./tools/scripts/download-lfm2-24b-a2b.sh              # LFM2-24B-A2B, no-stream profile (~14.4 GB)
./tools/scripts/download-lfm2.5-8b-a1b.sh             # LFM2.5-8B-A1B AtomicChat UD-Q4_K_XL (~5.2 GB)
```

Both scripts verify sha256 and run the one-time alignment pass
(`cne_prepare`). Prepared artifacts are listed first by `cne-setup` and
marked `-prepared.gguf`. Per-model profiles: `docs/models/lfm2-24b-a2b.md`,
`docs/models/lfm2.5-8b-a1b.md`, `docs/models/qwen3.6-35b-a3b-q4_k_xl.md`.

The download script ends by running the one-time alignment pass
(`cne_prepare`) that expert streaming requires. Prepared artifacts are
listed first by the CLI and marked `-prepared.gguf`.

## 3. Run the setup

```sh
./build/cli/cne_setup            # models/ is the default directory
```

What happens:

1. **Artifact pick** - lists every `.gguf` found (recursively), prepared
   ones first.
2. **Hardware + regime preview** - RAM available, logical threads, free
   disk, and the regime class the engine would assign, e.g.:

   ```
   artifact : Qwen3.6-35B-A3B-UD-Q4_K_XL-prepared.gguf (21.3 GiB)
   hardware : 12.5 GiB RAM available / 19.2 GiB total, 8 logical threads
   regime   : R2_ABOVE_RAM (preview only - nothing is auto-applied)
   ```

3. **Confirm loop** - one prompt per setting. Each prompt shows the
   suggested value, its reason, and the accepted values:

   ```
   stream decode         [off] (on | off) :
   ```

   - `Enter` keeps the suggestion
   - type a value to override (invalid input re-prompts with the accepted
     grammar)
   - `!` leaves the setting at the engine default (omitted from the file)

4. **Summary + write gate** - the resolved configuration is printed; only
   an explicit confirmation writes the file. Answering `n`, Ctrl-C or EOF
   anywhere leaves no file behind.

### Non-interactive use

```sh
./build/cli/cne_setup -y                          # accept all suggestions
CNE_SETUP_THREADS=8 CNE_SETUP_PORT=9090 ./build/cli/cne_setup   # preset some items
```

With piped/absent stdin, any item lacking a preset aborts loudly instead of
guessing. Invalid presets fail with exit code 2.

Accepted value grammars:

| Setting | Values |
|---|---|
| stream decode | `on` \| `off` |
| prefetch overlap (`prefetch`) | `on` \| `off`; streaming only — maps to `CNE_PREFETCH` (default off) |
| dense residency | `mmap` \| `warm` \| `anon` \| empty = engine auto |
| MTP draft depth | whole number >= 0 (0 = off) |
| threads | whole number >= 1 |
| context size | whole number >= 1 |
| conversation lanes (`session_max`) | whole number 1–64; KV lanes + LRU cap on `conversation_id` |
| API mode (`api_mode`) | on \| off; multi-user auth + `chat_id` tenancy |
| sessions per user (`session_max_per_user`) | whole number 1–64; per-user LRU when API mode on |
| API rate limit (`api_rpm`) | requests/minute per user; 0 = off |
| internal keys file (`api_keys_file`) | path relative to models dir (gateway secret) |
| expert cache cap | GiB number; empty or 0 = budget manager clamps |
| thinking default | `on` \| `off` |
| wall cap per request | seconds, 0 = off |
| host | IP or hostname |
| port | 1-65535 |

Exit codes: `0` written · `1` usage/environment problem · `2` invalid
preset or flag · `130` aborted.

## 4. What was suggested, and why

Suggestions come from measurements on the reference machine/hardware class
(see docs/BENCHMARKS.md); each one prints its reason inline:

- **stream off** below R3 - the expert cache already holds the hot set;
  naive mmap decode measured equal-or-better there
- **prefetch off** — speculative expert I/O overlap during streaming;
  default off (measured neutral/regressive); only applies when `stream` is on
- **dense mmap** when MTP is active - measured win under speculation-era
  memory footprints
- **MTP 8** (p_min 0.5) when the artifact carries MTP tensors AND the
  regime is not deep-streaming - lossless ~+20% over sequential on the
  reference machine; net-negative below ~90% expert-cache hit-rate, so it
  stays off in R3+
- **threads** - physical-core count beats SMT for both decode arms
- **ctx 1024 floor** - MTP aborts near token ~250 below this
- **LFM2-24B-A2B** (no MTP tensors) - stream off, dense `warm`, kernels on,
  4 threads, ctx 4096; streaming measured slower at this RAM ratio (~1.4×). See
  `docs/models/lfm2-24b-a2b.md`
- **LFM2.5-8B-A1B** — on **≤ 4 GiB** hosts: stream off, dense **`anon`**, t4,
  ctx 1024 (~11 tok/s @ 250 tok, ~2.9 GiB peak). Streaming + 3 GiB cache:
  t4, **`CNE_LANES=2`** (~9 tok/s). On **16 GiB+**: stream off, dense `mmap`,
  t4, ctx 2048 (~12 tok/s). **Avoid 8 threads** on both paths. See
  `docs/models/lfm2.5-8b-a1b.md`
- **conversation lanes** (`session_max`) - when MTP is off, typically 2 (3 on
  comfortable RAM); 1 when MTP is on or memory is tight. Sets `CNE_SESSION_MAX`
  and splits total `ctx` evenly across lanes (`ctx / session_max` tokens per
  chat). Requires `mtp: 0` for multi-turn API use.

You can override any of them; nothing changes model math silently.

## 5. The config file

Written where you pointed `--config` (default `models/server.json`):

```json
{
  "ctx": 4096,
  "dense": "mmap",
  "host": "127.0.0.1",
  "kernels": true,
  "max_req_s": 0.0,
  "model": "qwen3.6-35b-a3b-q4_k_xl-mtp/Qwen3.6-35B-A3B-UD-Q4_K_XL-prepared.gguf",
  "mtp": 0,
  "port": 8080,
  "session_max": 2,
  "session_max_per_user": 2,
  "api_mode": true,
  "api_keys_file": "api_keys.txt",
  "api_rpm": 120,
  "stream": false,
  "think": true,
  "threads": 4
}
```

With API mode, `cne_setup` can also write `models/api_keys.txt` (internal key for
the gateway). Client-facing keys stay in `gateway/api_keys.local.txt` — see
**docs/GATEWAY.md**.

Example above is tuned for **multi-turn / multi-user chat** (`mtp: 0`,
`session_max: 2` → 2048 tokens per parked conversation). For stateless MTP
speed, use `"mtp": 8`, `"mtp_p_min": 0.5`, omit `session_max` or set it to
`1`, and do not send `conversation_id` — see **docs/FEATURES.md** §5.

Notes:

- `model` is relative to the config file's directory
- `prefetch` is written as `true`/`false` (default off); requires `"stream": true`
  to have any effect — maps to `CNE_PREFETCH=1` / `lookahead`
- keys left at engine default are omitted entirely - every key present in
  the file traces back to a choice you confirmed

## 6. Boot the server

```sh
./build/server/cne_server
```

No arguments needed if `models/server.json` exists. Discovery order:
`--config PATH` flag > `server.json` next to the model argument >
`models/server.json`.

Every applied key is logged with its source at boot:

```
[config] loaded models/server.json
[config] KERNELS    = 1 (config)
[config] MTP        = 8 (config)
[config] THREADS    kept env override
listening on http://127.0.0.1:8080 ...
```

Precedence: **environment variable > config file > built-in default.**
Exported `CNE_*` knobs always beat the file, so temporary overrides never
require editing config.

Then test it:

```sh
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":64}'
```

Any OpenAI-compatible client works locally: point Open WebUI at
`http://127.0.0.1:8080/v1`.

For **multiple users on a public API**, do **not** expose `cne_server` on the
internet. Use **`cne_gateway`** (client API keys) in front of API-mode
`cne_server` on `127.0.0.1` — **docs/GATEWAY.md** and **docs/SERVING.md**.

## 6b. API gateway (Path B)

After `cne_setup` with API mode enabled (`api_mode`, `session_max_per_user`,
`api_rpm`, `api_keys_file` in `models/server.json`):

```sh
# engine (reads models/server.json; bind 127.0.0.1 in production)
./build/server/cne_server

# gateway — copy examples, edit keys, then:
cp gateway/gateway.json.example gateway/gateway.json
cp gateway/api_keys.example.txt gateway/api_keys.local.txt
./tools/scripts/run-gateway.sh
```

| File | Purpose |
|---|---|
| `models/server.json` | Engine knobs + `api_mode` |
| `models/api_keys.txt` | **Internal** key (gateway → engine only) |
| `gateway/gateway.json` | Gateway listen port, upstream, internal key, **chat policy** (`allow_thinking`, `max_tokens_per_request`) |
| `gateway/api_keys.local.txt` | **Client** keys (`<key> <user_id>` per line) |

Clients call `http://<gateway>:8090/v1/...` with
`Authorization: Bearer <client-api-key>` and `X-Chat-Id` (or JSON `chat_id`).

Chat policy (thinking block, per-answer token cap) is configured in
`gateway/gateway.json` — see **docs/GATEWAY.md** §2 (Chat policy).

Install gateway deps once: `cd gateway && python -m venv .venv && pip install -r requirements.txt`.

## 7. Automated live tests

With the LFM2 artifact on disk:

```sh
ctest --test-dir build -R server_e2e_live --output-on-failure
ctest --test-dir build -R 'server_api_live|server_api_per_user_live' --output-on-failure
```

Gateway live test (needs `gateway/.venv` + `pip install -r requirements.txt`):

```sh
ctest --test-dir build -R server_gateway_live --output-on-failure
```

Edit env knobs in `tests/e2e/*.json` instead of exporting long `CNE_*` lists.
Full matrix: **docs/TESTING.md**.

## 8. Troubleshooting

| Symptom | Meaning | Fix |
|---|---|---|
| `no .gguf artifacts` | wrong directory | pass the dir: `cne_setup path/to/models` |
| `only N free on disk` warning | artifact larger than free space | free space or ignore (download/prepare will fail later otherwise) |
| `finish_reason: "length"` in clients | max_tokens budget reached | client-side: raise `max_tokens` |
| answer cut off mid-"thinking" notice | request ended inside suppressed reasoning | raise `max_tokens` or disable thinking |
| boot: `no 'model' key in ...` | config missing model + none on argv | re-run `cne_setup` or pass model path |
| port already in use | another instance running | change port via setup or `CNE_SETUP_PORT` |

For engine-level knobs beyond this guide (lanes, FA, KV quant, debug
switches), see **docs/FEATURES.md**.
