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
| API keys, per-user quotas, fair eviction | **Not in server** — your API layer |
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

### No cross-user isolation (today)

`conversation_id` is an **unauthenticated global string**. Any client that
guesses another id could attach to that KV (low practical risk) or collide on
names. One user opening many chats can evict **any** least-recently-used slot,
including another user’s chat.

There is no `user_id`, per-tenant pool, or “reserved slots” in v1.

### Serial decode (not batched multi-user)

Two users never share one `llama_decode` batch. User B waits on the queue
while user A generates. `session_max` only parks KV between turns; it does not
parallelize compute.

---

## 3. Recommended stack for a public API

Do **not** expose `cne_server` directly on the internet. Put an API layer in
front.

```
Clients (JWT)  →  gateway :8090  →  cne_server :8080 (API mode, 127.0.0.1)
```

**Path B (recommended):** use the JWT gateway — **docs/GATEWAY.md**. End users
authenticate to the gateway; the gateway forwards to CNE with the internal API
key and `X-User-Id` from the JWT `sub`.

```
┌───────────────────────────────────────┐
│  cne_gateway (JWT, :8090)              │  ← Path B — docs/GATEWAY.md
│  - POST /v1/auth/token                 │
│  - validates JWT on /v1/chat/*          │
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
| **Authentication** | `cne_gateway` — JWT (`POST /v1/auth/token`) |
| **Stable id namespace** | Gateway → CNE `chat_id` as `{user}:{chat}` |
| **Per-user session limit** | `cne_server` — `session_max_per_user` |
| **Rate limits** | Gateway `CNE_GATEWAY_RPM` + nginx |
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
It issues JWTs, maps `X-Chat-Id` → CNE sessions, and never exposes the internal
API key.

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

### Phase 0 — Operate safely (**API**, now)

- [x] API key auth (`CNE_API_MODE`, `CNE_API_KEY` / `CNE_API_KEYS` / `api_keys_file`)
- [x] `X-User-Id` + server-owned `chat_id` → `conversation_id`
- [x] Per-user session cap (`session_max_per_user` / `CNE_SESSION_MAX_PER_USER`)
- [x] Message trim to per-lane context before generate
- [x] `/health` exposes `api.*`, `n_ctx_per_seq`, queue + sessions
- [x] Example nginx TLS/rate-limit config: `tools/nginx/cne_api.conf.example`
- [ ] TLS termination in production (deploy nginx/Caddy — see example config)
- [ ] Monitor `/health` queue depth in your orchestrator (poll `queue.waiting`)

### Phase 1 — Tenant-aware sessions (server)

- [x] `X-User-Id` on authenticated requests
- [x] Evict LRU **within user** before global eviction (`SessionStore`)
- [x] Reject `conversation_id` not owned by authenticated user
- [ ] Config: separate global lane count vs `session_max_per_user` docs in setup
- [ ] Tests: `session_tenant`, `server_api_live` (see **docs/TESTING.md**)

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
can go live today with a proxy.

---

## 6. Request contract (reference)

### Multi-turn chat (recommended for API)

```http
POST /v1/chat/completions
Content-Type: application/json
X-Conversation-Id: user-42:chat-9

{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "follow-up"}
  ],
  "max_tokens": 512,
  "stream": true
}
```

Or embed `"conversation_id": "user-42:chat-9"` in the JSON body.

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
Today, only if they guess the `conversation_id`. Use server-assigned ids and
auth (Phase 0–1).

**Why did my long chat lose history?**  
Either LRU evicted the whole conversation (too many chats for `session_max`),
or the client sent a longer prompt than the per-lane cap. Not automatic
compression.

**Can I run two users at full speed simultaneously?**  
No on one `cne_server` process — one decode at a time. Scale with multiple
instances + load balancer for throughput (each instance = one model copy).

**MTP for API users?**  
Only for stateless, single-shot requests without `conversation_id`. Chat APIs
should use `mtp: 0`.

---

## 8. Related docs

| Doc | Contents |
|---|---|
| **docs/SETUP.md** | `cne_setup`, `server.json`, `session_max` in config |
| **docs/FEATURES.md** §11 | Endpoints, env knobs, session vs MTP |
| **docs/TESTING.md** | `server_e2e_live`, `session_kv_live` |
| **README.md** | Product overview and quick start |
