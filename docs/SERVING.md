# Multi-user API serving

How to expose `cne_server` to multiple users over an HTTP API: what works
today, what belongs in your API layer, and what we plan to add in the engine.

For build, config, and knobs see **docs/SETUP.md** and **docs/FEATURES.md**
§11. For live tests see **docs/TESTING.md**.

---

## 1. What v1 gives you

| Capability | Status |
|---|---|
| OpenAI-compatible `/v1/chat/completions` (incl. SSE) | **Done** |
| Greedy default (lossless); `temperature` / `top_p` / `seed` per request | **Done** |
| Multi-turn KV reuse via `conversation_id` | **Done** (`CNE_MTP=0`) |
| Several parked conversations (`session_max` / `CNE_SESSION_MAX`) | **Done** |
| Request queue (one generation at a time) | **Done** (`/health` → `queue.*`) |
| Client disconnect stops generation | **Done** |
| Per-request wall cap (`max_req_s` / `CNE_MAX_REQ_S`) | **Done** |
| API keys, per-user quotas, fair eviction | **Engine (API mode)** + **gateway** — see below |
| Dynamic context per chat (use what you need) | **Not supported** — fixed lanes |
| Auto-truncate / summarize old turns | **Not in server** — client or API layer |
| Parallel decode across users | **Not supported** — serial queue |

**One engine process = one model, one decode slot, shared KV budget.** Multiple
users are supported by **queuing** requests and optionally **parking** a small
number of conversation KV caches.

---

## 2. Core concepts

### `conversation_id` (chat thread)

Send the same id on every turn of one chat (JSON field or `X-Conversation-Id`
header). Turn 2+ reuses KV and only prefills the new tail:

```
[session] conv=alice-chat-7 reused=1200 prefilled=42 prompt=1242
```

Omit `conversation_id` for stateless requests (seq 0, cleared after each
request). Use `"clear_conversation": true` to drop KV for one id.

**Requires `CNE_MTP=0`.** MTP and sessions are mutually exclusive — see
**docs/FEATURES.md** §5.

### `session_max` (“conversation lanes” in `cne_setup`)

Config key / env: `session_max` → `CNE_SESSION_MAX`.

Controls two linked limits:

1. **KV lanes in the model** — `n_seq_max` at boot; total `ctx` is split
   **equally** across lanes (see below).
2. **LRU cap** on tracked `conversation_id`s — when full, the **least
   recently used** conversation is evicted before creating a new one.

Setup suggests a value from RAM + model size (`suggest_serving_kv()`). Typical
chat serving: `session_max: 2`, `mtp: 0`.

### Context is split equally (not by usage)

At boot:

```
per_lane_tokens = ctx / session_max   (rounded down to llama padding)
```

Example: `ctx: 4096`, `session_max: 2` → **2048 tokens per lane**, always.

- A short chat does **not** donate spare context to a long chat.
- Only one active lane still gets only its slice, not the full `ctx`.
- When a lane fills, decode fails or returns context errors — the server does
  **not** auto-drop old messages.

**Who trims history?** The **client** (Open WebUI, your app, or API gateway)
must send a `messages` array that fits the lane. The server tokenizes whatever
it receives.

### Isolation with API mode + gateway

With **`api_mode: true`** on `cne_server` and the **gateway** in front:

- Clients use **client API keys** at the gateway; the **internal key** never
  leaves localhost.
- The gateway maps each key → `user_id` and sends `X-User-Id` + `chat_id`.
- CNE builds `conversation_id` as `{user}:{chat}` and rejects cross-user access.
- **`session_max_per_user`** evicts LRU chats **within** a user before touching
  another user's slots.

Without API mode (direct `cne_server`), `conversation_id` is still an
unauthenticated global string — use the gateway for public multi-user hosting.

### Serial decode (not batched multi-user)

Two users never share one `llama_decode` batch. User B waits on the queue
while user A generates. `session_max` only parks KV between turns; it does not
parallelize compute.

---

## 3. Recommended stack for a public API

Do **not** expose `cne_server` directly on the internet. Put an API layer in
front.

```
Clients (API key)  →  gateway :8090  →  cne_server :8080 (API mode, 127.0.0.1)
```

**Path B (recommended):** use the API-key gateway — **docs/GATEWAY.md**. Each
user gets a client API key; the gateway forwards to CNE with the internal API
key and `X-User-Id` from the key mapping.

```
┌───────────────────────────────────────┐
│  cne_gateway (API key, :8090)          │  ← Path B — docs/GATEWAY.md
│  - validates Bearer key on /v1/chat/*  │
│  - maps key → X-User-Id                │
└───────────────────────────────────────┘
        │ internal API key + X-User-Id
        ▼
┌───────────────────────────────────────┐
│  cne_server (API mode, 127.0.0.1)      │  ← this repo
│  - OpenAI /v1/chat/completions         │
│  - KV reuse, queue, /health            │
└───────────────────────────────────────┘
```

### API layer responsibilities (Path B gateway + CNE)

| Item | Where |
|---|---|
| **Authentication** | `cne_gateway` — client API keys (`Authorization: Bearer`) |
| **Stable id namespace** | Gateway → CNE `chat_id` as `{user}:{chat}` |
| **Per-user session limit** | `cne_server` — `session_max_per_user` |
| **Rate limits** | Gateway `rpm_per_user` / `CNE_GATEWAY_RPM` + nginx |
| **Thinking + answer token cap** | Gateway `allow_thinking`, `max_tokens_per_request` — **docs/GATEWAY.md** |
| **Context trim** | `cne_server` — per-lane auto-trim |
| **TLS + bind** | CNE on `127.0.0.1`; TLS in front of gateway |
| **Observability** | `GET /health` on gateway and CNE |

### Open WebUI and similar clients

Open WebUI stores chat history locally and sends the full `messages` list each
turn. It usually does **not** send `conversation_id` unless you customize the
integration.

| Approach | KV reuse on CNE | Context control |
|---|---|---|
| Stock Open WebUI → `cne_server` | Often **none** (no `conversation_id`) | Client UI / model ctx settings |
| Thin proxy adds `conversation_id` | **Yes** | Proxy trims + maps WebUI chat id |
| Your own client | **Yes** | Full control |

For multi-user hosting, point clients at **`cne_gateway`** (`docs/GATEWAY.md`).
It validates client API keys, maps `X-Chat-Id` → CNE sessions, and never
exposes the internal API key.

---

## 4. Sizing `ctx` and `session_max`

| Goal | Suggestion |
|---|---|
| Multi-user chat API | `mtp: 0`, `session_max: 2` (or 3 if RAM allows) |
| Long agent threads | Raise `ctx`; remember **per-lane** cap is `ctx / session_max` |
| Maximize one heavy chat | `session_max: 1` (full `ctx` to one parked session) |
| Stateless speed | `session_max: 1`, no `conversation_id`, optional MTP (no sessions) |

KV RAM scales roughly with `ctx × session_max × ~20 KiB/token` (hybrid
models; tune with `CNE_KV_BPT`). Boot logs:

```
[cne] KV plan: ctx=4096 (2048 tok/seq x 2 lanes) ~160 MiB est ...
```

Run `cne_setup -y` and check `models/server.json` for suggested `ctx` and
`session_max`.

---

## 5. Roadmap (engine + server)

Ordered for multi-tenant API serving. Items marked **API** can be done outside
this repo without waiting on engine work.

### Phase 0 — Operate safely (**API** + gateway, now)

- [x] API key auth on `cne_server` (`CNE_API_MODE`, `api_keys_file`, …)
- [x] **`cne_gateway`** — minimal reference proxy: client API keys, RPM, chat
  policy, OpenAI proxy (**docs/GATEWAY.md**; replace/wrap for production scale)
- [x] `X-User-Id` + server-owned `chat_id` → `conversation_id`
- [x] Per-user session cap (`session_max_per_user` / `CNE_SESSION_MAX_PER_USER`)
- [x] Message trim to per-lane context before generate
- [x] `/health` exposes `api.*`, `n_ctx_per_seq`, queue + sessions
- [x] Example nginx TLS/rate-limit config: `tools/nginx/cne_api.conf.example`
- [x] `cne_setup` writes API mode fields + internal keys file
- [ ] TLS termination in production (deploy nginx/Caddy — see example config)
- [ ] Monitor `/health` queue depth in your orchestrator (poll `queue.waiting`)

### Phase 1 — Tenant-aware sessions (server)

- [x] `X-User-Id` on authenticated requests
- [x] Evict LRU **within user** before global eviction (`SessionStore`)
- [x] Reject `conversation_id` not owned by authenticated user
- [x] Config: `cne_setup` documents `session_max` vs `session_max_per_user`
- [x] Tests: `session_tenant`, `server_api_live`, `server_api_per_user_live`,
  `server_gateway_live` (**docs/TESTING.md**)

### Phase 2 — Fairness and limits (server + API)

- [ ] Max queue wait / 429 when overloaded
- [ ] Optional per-user concurrent request cap (still one decode; block duplicate tabs)
- [ ] Structured metrics export (Prometheus) from queue + sessions
- [ ] Documented max prompt tokens using `llama_n_ctx_seq()` (fix headroom check)

### Phase 3 — Context lifecycle (server)

- [ ] Optional **ctx shift** per lane (drop oldest tokens, keep system + tail)
- [ ] Research **unified KV pool** (`kv_unified`) for dynamic sharing across lanes
- [ ] Optional server-side summarization hook (callout only; model/policy external)

### Phase 4 — Throughput (research)

- [ ] Decode scheduler: fair round-robin across active lanes (still 1 tok/step each)
- [ ] Batched decode when multiple lanes ready same tick (`CNE_MTP=0` only)
- [ ] Multiple engine workers (separate processes) behind load balancer — ops heavy

Phases 1–2 are the minimum to call multi-user API serving **fair**. Phase 0
can go live today with **`cne_gateway`** + API-mode `cne_server`.

---

## 6. Request contract (reference)

### Multi-turn chat via gateway (recommended for public API)

```http
POST http://gateway:8090/v1/chat/completions
Authorization: Bearer <client-api-key>
X-Chat-Id: support-42
Content-Type: application/json

{
  "messages": [{"role": "user", "content": "hello"}],
  "max_tokens": 64
}
```

The gateway forwards to `cne_server` with the internal key and `X-User-Id` from
the key mapping. Use `chat_id` in JSON instead of `X-Chat-Id` if you prefer.

### Multi-turn chat (direct `cne_server`, API mode or legacy)

```http
POST /v1/chat/completions
Content-Type: application/json
Authorization: Bearer <api-key>
X-User-Id: alice
X-Chat-Id: chat-9

{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "follow-up"}
  ],
  "max_tokens": 512,
  "stream": true
}
```

Legacy (no API mode): use `conversation_id` or `X-Conversation-Id` instead of
`chat_id` / `X-User-Id`.

### Reset a thread

```json
{
  "conversation_id": "user-42:chat-9",
  "clear_conversation": true,
  "messages": [{"role": "user", "content": "start over"}],
  "max_tokens": 256
}
```

### Health check

```sh
curl -s http://127.0.0.1:8080/health | jq '{queue, sessions, n_ctx}'
```

---

## 7. FAQ

**Can one user steal another’s conversation?**  
With API mode + gateway: no across users (server checks `user_id` owns
`conversation_id`). Without API mode: only if they guess the id — use the
gateway for public APIs.

**Why did my long chat lose history?**  
Either LRU evicted the whole conversation (`session_max` or
`session_max_per_user`), or the client sent more tokens than the per-lane cap.
The server trims oldest non-system messages in API mode but does not summarize.

**Can I run two users at full speed simultaneously?**  
No on one `cne_server` process — one decode at a time. Scale with multiple
instances behind a load balancer (each instance = one model copy; one gateway
or shared client-key store per deployment).

**MTP for API users?**  
Only for stateless, single-shot requests without `chat_id` / `conversation_id`.
Chat APIs should use `mtp: 0`.

**Where do client API keys live?**  
`gateway/api_keys.local.txt` (or env). Internal engine key:
`models/api_keys.txt` from `cne_setup`. See **docs/GATEWAY.md**.

---

## 8. Related docs

| Doc | Contents |
|---|---|
| **docs/GATEWAY.md** | `cne_gateway` setup, client vs internal keys, `gateway.json` |
| **docs/SETUP.md** | `cne_setup`, `server.json`, API mode fields |
| **docs/FEATURES.md** §11 | Endpoints, env knobs, session vs MTP |
| **docs/TESTING.md** | Live E2E matrix (HTTP, API, gateway, Qwen) |
| **README.md** | Product overview and quick start |
