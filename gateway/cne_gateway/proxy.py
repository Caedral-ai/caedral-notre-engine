from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from .config import Settings


def upstream_headers(settings: Settings, user_id: str, chat_id: str | None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {settings.internal_api_key}",
        "X-User-Id": user_id,
        "Content-Type": "application/json",
    }
    if chat_id:
        headers["X-Chat-Id"] = chat_id
    return headers


def extract_chat_id(request: Request, body: dict[str, Any]) -> str | None:
    hdr = request.headers.get("x-chat-id") or request.headers.get("X-Chat-Id")
    if hdr:
        return hdr.strip() or None
    raw = body.get("chat_id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


async def proxy_json(
    client: httpx.AsyncClient,
    settings: Settings,
    user_id: str,
    path: str,
    body: dict[str, Any],
    chat_id: str | None,
) -> Response:
    url = f"{settings.upstream}{path}"
    headers = upstream_headers(settings, user_id, chat_id)
    if chat_id and "chat_id" not in body:
        body = {**body, "chat_id": chat_id}
    try:
        upstream = await client.post(url, headers=headers, json=body, timeout=None)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"upstream unreachable: {exc}") from exc
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


async def _stream_bytes(upstream: httpx.Response) -> AsyncIterator[bytes]:
    async for chunk in upstream.aiter_bytes():
        yield chunk


async def proxy_stream(
    client: httpx.AsyncClient,
    settings: Settings,
    user_id: str,
    path: str,
    body: dict[str, Any],
    chat_id: str | None,
) -> StreamingResponse:
    url = f"{settings.upstream}{path}"
    headers = upstream_headers(settings, user_id, chat_id)
    if chat_id and "chat_id" not in body:
        body = {**body, "chat_id": chat_id}
    try:
        req = client.build_request("POST", url, headers=headers, json=body)
        upstream = await client.send(req, stream=True)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"upstream unreachable: {exc}") from exc

    if upstream.status_code >= 400:
        data = await upstream.aread()
        await upstream.aclose()
        return Response(
            content=data,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    media = upstream.headers.get("content-type", "text/event-stream")
    return StreamingResponse(
        _stream_bytes(upstream),
        status_code=upstream.status_code,
        media_type=media,
    )


async def proxy_get(
    client: httpx.AsyncClient, settings: Settings, path: str
) -> Response:
    url = f"{settings.upstream}{path}"
    try:
        upstream = await client.get(url, timeout=30.0)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"upstream unreachable: {exc}") from exc
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


def parse_json_body(raw: bytes) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="expected JSON object")
    return data
