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
  "rpm_per_user": 120
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
2. Forwards with `Authorization: Bearer <CNE_INTERNAL_API_KEY>`
3. Passes `chat_id` (header `X-Chat-Id` or JSON `chat_id`)

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

See also **docs/SERVING.md** for engine session limits and sizing.
