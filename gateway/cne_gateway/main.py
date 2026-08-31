from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .auth import AuthError, RateLimiter, authenticate_api_key, load_api_keys
from .config import Settings, load_settings
from .proxy import extract_chat_id, parse_json_body, proxy_get, proxy_json, proxy_stream

_bearer = HTTPBearer(auto_error=False)


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or load_settings()
    client_keys: set[str] = set()
    key_to_user: dict[str, str] = {}
    load_api_keys(cfg.api_keys_file, client_keys, key_to_user)
    if not client_keys:
        raise SystemExit(
            f"no client API keys loaded (check {cfg.api_keys_file} or "
            "CNE_GATEWAY_API_KEY / CNE_GATEWAY_API_KEYS)"
        )

    limiter = RateLimiter(cfg.rpm_per_user)
    http_client: httpx.AsyncClient | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        nonlocal http_client
        http_client = httpx.AsyncClient(timeout=None)
        yield
        await http_client.aclose()

    app = FastAPI(title="CNE API Gateway", version="0.2.0", lifespan=lifespan)

    def get_client() -> httpx.AsyncClient:
        assert http_client is not None
        return http_client

    async def current_user(
        creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> str:
        if creds is None or creds.scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Bearer API key required")
        try:
            user_id = authenticate_api_key(
                creds.credentials, client_keys, key_to_user
            )
        except AuthError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
        if not limiter.allow(user_id):
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        return user_id

    @app.get("/health")
    async def health(client: httpx.AsyncClient = Depends(get_client)) -> JSONResponse:
        payload: dict = {
            "status": "ok",
            "gateway": {
                "auth": "api_key",
                "upstream": cfg.upstream,
                "client_keys": len(client_keys),
            },
        }
        try:
            upstream = await client.get(f"{cfg.upstream}/health", timeout=5.0)
            if upstream.status_code == 200:
                payload["cne"] = upstream.json()
            else:
                payload["cne"] = {"status": "error", "http": upstream.status_code}
        except httpx.HTTPError as exc:
            payload["cne"] = {"status": "unreachable", "error": str(exc)}
        return JSONResponse(payload)

    @app.get("/v1/models")
    async def models(
        _: str = Depends(current_user), client: httpx.AsyncClient = Depends(get_client)
    ) -> Response:
        return await proxy_get(client, cfg, "/v1/models")

    @app.post("/v1/chat/completions", response_model=None)
    async def chat(
        request: Request,
        user_id: str = Depends(current_user),
        client: httpx.AsyncClient = Depends(get_client),
    ):
        raw = await request.body()
        body = parse_json_body(raw)
        if "messages" not in body or not isinstance(body["messages"], list):
            raise HTTPException(status_code=400, detail="'messages' must be an array")
        chat_id = extract_chat_id(request, body)
        if body.get("stream"):
            return await proxy_stream(
                client, cfg, user_id, "/v1/chat/completions", body, chat_id
            )
        return await proxy_json(
            client, cfg, user_id, "/v1/chat/completions", body, chat_id
        )

    return app


def main() -> None:
    import uvicorn

    settings = load_settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
