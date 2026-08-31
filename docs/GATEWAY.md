# CNE API Gateway (Path B)

API-key-authenticated public API in front of `cne_server`. Each user receives a
client API key out-of-band. There is no login endpoint. End users never see the
internal CNE API key or set `X-User-Id` themselves.

```
Client (API key)  →  gateway :8090  →  cne_server :8080 (API mode, localhost)
```

## 1. Prerequisites

1. Build and configure `cne_server` with **API mode** on `127.0.0.1:8080`.

Easiest path: run `./build/cli/cne_setup` (enable API mode) and
`./build/server/cne_server` — see **docs/SETUP.md** §6b. Manual env example:

```sh
export CNE_API_MODE=1
export CNE_API_KEY=internal-only-key    # gateway uses this; clients never do
export CNE_MTP=0
export CNE_SESSION_MAX=3
export CNE_SESSION_MAX_PER_USER=2
./build/server/cne_server
```

2. Install gateway deps (once):

```sh
cd gateway
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or: `tools/scripts/run-gateway.sh` (creates venv automatically).

## 2. Configure gateway

Copy `gateway/gateway.json.example` → `gateway/gateway.json` (gitignored) or set
`CNE_GATEWAY_CONFIG`. **Precedence:** environment variable > JSON file > default.

```json
{
  "host": "127.0.0.1",
  "port": 8090,
  "upstream": "http://127.0.0.1:8080",
  "internal_api_key": "same-as-cne_server-CNE_API_KEY",
  "api_keys_file": "api_keys.local.txt",
  "rpm_per_user": 120,
  "allow_thinking": false,
  "max_tokens_per_request": 512
}
```

| JSON key | Env override | Default | Meaning |
|---|---|---|---|
| `internal_api_key` | `CNE_INTERNAL_API_KEY` | — | Same value as `CNE_API_KEY` on `cne_server` |
| `api_keys_file` | `CNE_GATEWAY_API_KEYS_FILE` | `api_keys.example.txt` | Client keys: `<api_key> <user_id>` per line |
| `upstream` | `CNE_UPSTREAM` | `http://127.0.0.1:8080` | `cne_server` base URL |
| `host` | `CNE_GATEWAY_HOST` | `127.0.0.1` | Bind address |
| `port` | `CNE_GATEWAY_PORT` | `8090` | Listen port |
| `rpm_per_user` | `CNE_GATEWAY_RPM` | `120` | Per-user requests/minute at gateway |
| `allow_thinking` | `CNE_GATEWAY_ALLOW_THINKING` | `false` | When `false`, clients cannot enable thinking (`chat_template_kwargs.enable_thinking`); gateway returns 403 if they try |
| `max_tokens_per_request` | `CNE_GATEWAY_MAX_TOKENS` | `0` (unlimited) | Cap on `max_tokens` per chat completion; omitted client `max_tokens` is set to this cap |
| — | `CNE_GATEWAY_CONFIG` | `gateway/gateway.json` | Config file path (if present) |

Extra env-only client key sources: `CNE_GATEWAY_API_KEY`, `CNE_GATEWAY_API_KEYS`.

At least one client key must be loaded (file and/or env).

Client keys file format (`gateway/api_keys.example.txt`):

```
# <api_key> <user_id>
cne_sk_alice_dev_replace_me_in_production alice
cne_sk_bob_dev_replace_me_in_production bob
```

```sh
cp gateway/gateway.json.example gateway/gateway.json   # edit internal_api_key
./tools/scripts/run-gateway.sh
```

Put TLS in nginx/Caddy in front of the gateway (`tools/nginx/cne_api.conf.example`).

### Chat policy

The gateway enforces **per-request limits** on `POST /v1/chat/completions`
before forwarding to `cne_server`. Policy applies to both streaming and
non-streaming requests.

This is separate from engine knobs in `models/server.json` (e.g. `"think": false`).
Even if the server default allows thinking, clients **cannot** turn it on through
the gateway unless you set `"allow_thinking": true` in `gateway.json`.

| Policy | Config key | What it does |
|---|---|---|
| **Thinking** | `allow_thinking` | When `false` (default): gateway sets `chat_template_kwargs.enable_thinking: false` on every chat request. If a client sends `enable_thinking: true`, the gateway returns **403** and does not call the engine. When `true`: client value is passed through unchanged. |
| **Answer length** | `max_tokens_per_request` | When `> 0`: caps `max_tokens` per completion. Client values above the cap are **clamped**; omitted `max_tokens` is set to the cap. When `0` (default): no gateway limit (engine/client default applies). |

**Thinking** (models such as LFM2.5 that support `` blocks):

```sh
# Blocked when allow_thinking is false (default) — 403, no upstream call
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H "Authorization: Bearer <client-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"hello"}],
    "chat_template_kwargs":{"enable_thinking":true}
  }'
# → {"detail":"thinking is disabled by gateway policy"}

# Allowed only when gateway.json has "allow_thinking": true
```

**Max tokens** (`max_tokens_per_request: 512`):

| Client sends | Forwarded to engine |
|---|---|
| *(omitted)* | `max_tokens: 512` |
| `max_tokens: 128` | `max_tokens: 128` |
| `max_tokens: 4096` | `max_tokens: 512` (clamped) |
| `max_tokens: "lots"` | **400** — must be a positive integer |

Policy fields appear on `GET /health` under `gateway`:

```json
{
  "status": "ok",
  "gateway": {
    "auth": "api_key",
    "upstream": "http://127.0.0.1:8080",
    "client_keys": 2,
    "allow_thinking": false,
    "max_tokens_per_request": 512
  },
  "cne": { "...": "..." }
}
```

Implementation: `gateway/cne_gateway/policy.py` (`apply_chat_policy`).

## 3. Client flow

Issue each user an API key (admin). The client sends it on every request:

```sh
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H "Authorization: Bearer cne_sk_alice_dev_replace_me_in_production" \
  -H "X-Chat-Id: support-42" \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"hello"}],
    "max_tokens": 64,
    "stream": false
  }'
```

The gateway:

1. Validates the client API key → maps to `user_id` → `X-User-Id` for CNE
2. Enforces **policy** from `gateway.json`: caps `max_tokens`, blocks or forces off thinking
3. Forwards with `Authorization: Bearer <CNE_INTERNAL_API_KEY>`
4. Passes `chat_id` (header `X-Chat-Id` or JSON `chat_id`)

### Health

```sh
curl -s http://127.0.0.1:8090/health | jq .
```

Returns gateway status plus upstream `/health` from `cne_server`.

## 4. Endpoints

| Path | Auth | Proxies to |
|---|---|---|
| `POST /v1/chat/completions` | client API key | CNE chat |
| `GET /v1/models` | client API key | CNE models |
| `GET /health` | none | gateway + CNE health |

## 5. Tests

```sh
cd gateway && .venv/bin/pip install -r requirements.txt
PYTHONPATH=. pytest tests -q
```

| File | Coverage |
|---|---|
| `tests/test_gateway.py` | Auth, proxy headers, config load |
| `tests/test_policy.py` | `allow_thinking`, `max_tokens_per_request`, streaming, `/health` |

Or from build tree (if CMake finds Python):

```sh
ctest --test-dir build -R gateway_unit --output-on-failure
```

**Live E2E (gateway + engine):** requires GGUF, `gateway/.venv` with
`pip install -r requirements.txt`, then:

```sh
ctest --test-dir build -R server_gateway_live --output-on-failure
```

## 6. Production notes

- Issue and rotate client keys out-of-band; revoke by removing a line from the keys file.
- Never expose `cne_server` on a public interface; only the gateway.
- Rotate `CNE_INTERNAL_API_KEY` on compromise (re-issue all client keys if needed).
- Set **`max_tokens_per_request`** to bound completion cost and latency per user.
- Keep **`allow_thinking: false`** on public APIs unless you explicitly want clients
  to opt into reasoning tokens (LFM2.5, Qwen3, etc.).

See also **docs/SERVING.md** for engine session limits and sizing.
