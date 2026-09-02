# Docker deployment (engine only)

Ship **`cne_server`** in a container. Model weights and `server.json` stay on
the host under `models/` (mounted read-only). **Your production gateway** —
nginx, Kong, a custom API service, or `cne_gateway` on the host — sits in
front and is **not** part of this image.

```
Client  →  your gateway / API layer  →  cne_server :8080 (container)
                ↑
         TLS, client auth, quotas, WAF, billing, …
```

`cne_gateway` in this repo is a **reference proxy** for dev and small
deployments (**docs/GATEWAY.md**). Production usually keeps the engine in
Docker and runs a separate edge layer — that works fine.

For bare-metal engine setup see **docs/SETUP.md** §6.

**Recommended private production layout** (Caddy + `cne_gateway` on host, engine
in Docker): see `internal-docs/production-caddy-gateway.md` (local runbook,
gitignored).

---

## 1. Prerequisites

1. **Submodules** (llama.cpp is required to build):

   ```sh
   git submodule update --init --recursive
   ```

2. **Model + config** on the host:

   ```sh
   cmake -B build -DCMAKE_BUILD_TYPE=Release \
         -DCNE_BUILD_SERVER=ON -DCNE_BUILD_CLI=ON
   cmake --build build --target cne_setup -j
   ./tools/scripts/download-lfm2.5-8b-a1b.sh   # or your artifact
   ./build/cli/cne_setup                       # writes models/server.json
   ```

   For multi-user serving through **any** gateway, enable **API mode** in setup
   (`api_mode: true`). That writes `models/api_keys.txt` — the **internal**
   key your gateway uses when calling the engine.

---

## 2. Build and run

**Docker Compose** (binds engine to `127.0.0.1:8080` on the host):

```sh
docker compose up --build -d
```

**Plain `docker run`:**

```sh
docker build -t cne-engine .
docker run -d --name cne_server \
  -v "$PWD/models:/models:ro" \
  -p 127.0.0.1:8080:8080 \
  --restart unless-stopped \
  cne-engine
```

Health:

```sh
curl -s http://127.0.0.1:8080/health | jq .
```

The image also ships **`cne_prepare`** and **`cne_setup`** for one-off use:

```sh
docker run --rm -v "$PWD/models:/models" --entrypoint cne_setup cne-engine -y
```

**Config paths:** `"model": "lfm2.5-8b-a1b/foo-prepared.gguf"` in `server.json`
is resolved relative to `/models`.

### Smoke test without downloading

If you already have a prepared artifact under `./models` (e.g. from a prior
`download-lfm2.5-8b-a1b.sh` on the host), hardlink-copy it into
`.docker-test/models` and exercise the image:

```sh
./tools/scripts/docker-smoke.sh           # build, up, /health
./tools/scripts/docker-smoke.sh --chat    # + one /v1/chat/completions
./tools/scripts/docker-smoke.sh --keep    # leave container up after success
```

Uses `CNE_MODELS_DIR` for compose; production still mounts `./models` after
a real download + `cne_setup`.

---

## 3. Plugging in your production gateway

The engine exposes an **OpenAI-compatible HTTP API** when API mode is on. Your
gateway must:

| Requirement | Detail |
|---|---|
| **Upstream URL** | `http://<engine-host>:8080` (e.g. `http://127.0.0.1:8080` if gateway is on the same machine) |
| **Internal auth** | `Authorization: Bearer <key>` where the key matches `models/api_keys.txt` |
| **User tenancy** | Set `X-User-Id: <user>` on proxied requests (engine enforces per-user session quotas) |
| **Chat threads** | Forward `X-Chat-Id` or JSON `chat_id`; engine maps to `conversation_id` |
| **Do not expose engine publicly** | Only your gateway should be on the internet; keep `cne_server` on a private network or `127.0.0.1` |

Reference behaviour (headers, policy knobs, rate limits) is implemented in
`gateway/cne_gateway/` and documented in **docs/GATEWAY.md** — copy the parts
you need into your own service.

**Same host (gateway not in Docker):**

```sh
# engine in Docker on localhost:8080
docker compose up -d

# reference gateway on the host
./tools/scripts/run-gateway.sh   # upstream http://127.0.0.1:8080
```

**Kubernetes / separate VM:** publish `cne_server` as a ClusterIP / internal
load balancer; point your gateway deployment at that DNS name.

**nginx only (no Python gateway):** see `tools/nginx/cne_api.conf.example` —
terminate TLS at nginx and proxy `/v1/` to the engine (inject the internal
Bearer key server-side).

---

## 4. Production notes

| Topic | Guidance |
|---|---|
| **Engine-only Docker** | Right default when you bring your own gateway — simpler image, clear boundary. |
| **Models in the image** | Do **not** bake GGUF into the image — mount a volume or attach block storage. |
| **Memory** | Container limit ≥ model RSS + `cache_gib` + KV overhead from `server.json`. |
| **CPU** | Pin `threads` in `server.json`; set compose/k8s CPU limits to match. |
| **Scaling** | One container ≈ one model + one decode queue. Scale with more engine replicas behind your gateway/LB. |
| **TLS** | Terminate at your gateway or nginx/Caddy — not inside the engine container. |

---

## 5. Files

| Path | Purpose |
|---|---|
| `Dockerfile` | Multi-stage build for `cne_server`, `cne_prepare`, `cne_setup` |
| `docker-compose.yml` | Single-service compose (engine on `127.0.0.1:8080`) |
| `.dockerignore` | Excludes `models/`, `build/`, venvs from build context |

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `no 'model' key` / missing GGUF | run `cne_setup` on the host; confirm `./models` mount |
| `401` from engine | gateway Bearer key ≠ `models/api_keys.txt` |
| gateway cannot connect | engine still booting (large models take minutes) or wrong host/port |
| build fails on empty `third_party/llama.cpp` | `git submodule update --init --recursive` |
| OOM kill | raise memory limit or lower `cache_gib` / `session_max` |
