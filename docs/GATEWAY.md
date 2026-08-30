# CNE API Gateway (Path B)

JWT-authenticated public API in front of `cne_server`. End users never see the
internal CNE API key or set `X-User-Id` themselves.

```
Client (JWT)  →  gateway :8090  →  cne_server :8080 (API mode, localhost)
```

## 1. Prerequisites

1. Build and configure `cne_server` with **API mode** on `127.0.0.1:8080`:

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

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `CNE_GATEWAY_JWT_SECRET` | yes | — | HS256 signing secret (long random string) |
| `CNE_INTERNAL_API_KEY` | yes | — | Same value as `CNE_API_KEY` on `cne_server` |
| `CNE_GATEWAY_USERS_FILE` | no | `gateway/users.example.txt` | `user:password` lines for login |
| `CNE_UPSTREAM` | no | `http://127.0.0.1:8080` | `cne_server` base URL |
| `CNE_GATEWAY_HOST` | no | `127.0.0.1` | Bind address |
| `CNE_GATEWAY_PORT` | no | `8090` | Listen port |
| `CNE_GATEWAY_JWT_TTL_S` | no | `86400` | Token lifetime (seconds) |
| `CNE_GATEWAY_RPM` | no | `120` | Per-user requests/minute at gateway |

```sh
export CNE_GATEWAY_JWT_SECRET="$(openssl rand -hex 32)"
export CNE_INTERNAL_API_KEY=internal-only-key
export CNE_GATEWAY_USERS_FILE=gateway/users.example.txt
./tools/scripts/run-gateway.sh
```

Put TLS in nginx/Caddy in front of the gateway (`tools/nginx/cne_api.conf.example`).

## 3. Client flow

### Login

```sh
curl -s http://127.0.0.1:8090/v1/auth/token \
  -H 'Content-Type: application/json' \
  -d '{"username":"alice","password":"changeme"}'
```

Response:

```json
{"access_token":"<jwt>","token_type":"bearer","expires_in":86400}
```

### Chat (OpenAI-compatible)

```sh
TOKEN=...
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Chat-Id: support-42" \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"hello"}],
    "max_tokens": 64,
    "stream": false
  }'
```

The gateway:

1. Validates JWT → `sub` becomes `X-User-Id` for CNE
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
| `POST /v1/auth/token` | none | issues JWT |
| `POST /v1/chat/completions` | JWT | CNE chat |
| `GET /v1/models` | JWT | CNE models |
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

- Replace `users.example.txt` with your user store or swap `/v1/auth/token` for
  OAuth/OIDC in a fork — the proxy layer stays the same.
- Never expose `cne_server` on a public interface; only the gateway.
- Rotate `CNE_GATEWAY_JWT_SECRET` and `CNE_INTERNAL_API_KEY` on compromise.

See also **docs/SERVING.md** for engine session limits and sizing.
