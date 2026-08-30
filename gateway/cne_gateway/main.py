from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .auth import AuthError, RateLimiter, decode_token, issue_token, load_users, verify_password
from .config import Settings, load_settings
from .proxy import extract_chat_id, parse_json_body, proxy_get, proxy_json, proxy_stream

_bearer = HTTPBearer(auto_error=False)


class TokenRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or load_settings()
    users = load_users(cfg.users_file)
    limiter = RateLimiter(cfg.rpm_per_user)
    http_client: httpx.AsyncClient | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        nonlocal http_client
        http_client = httpx.AsyncClient(timeout=None)
        yield
        await http_client.aclose()

    app = FastAPI(title="CNE API Gateway", version="0.1.0", lifespan=lifespan)

    def get_client() -> httpx.AsyncClient:
        assert http_client is not None
        return http_client

    async def current_user(
        creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> str:
        if creds is None or creds.scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Bearer token required")
        try:
            user_id = decode_token(creds.credentials, cfg.jwt_secret)
        except AuthError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
        if not limiter.allow(user_id):
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        return user_id

    @app.get("/health")
    async def health(client: httpx.AsyncClient = Depends(get_client)) -> JSONResponse:
        payload: dict = {
            "status": "ok",
            "gateway": {"jwt": True, "upstream": cfg.upstream},
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

    @app.post("/v1/auth/token", response_model=TokenResponse)
    async def login(body: TokenRequest) -> TokenResponse:
        try:
            user_id = verify_password(users, body.username, body.password)
        except AuthError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
        token = issue_token(user_id, cfg.jwt_secret, cfg.jwt_ttl_s)
        return TokenResponse(access_token=token, expires_in=cfg.jwt_ttl_s)

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
