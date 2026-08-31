from __future__ import annotations

import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict

import os


class AuthError(Exception):
    def __init__(self, message: str, status: int = 401) -> None:
        super().__init__(message)
        self.status = status


def _trim(s: str) -> str:
    return s.strip()


def load_api_keys(
    path: Path,
    keys_out: set[str],
    key_to_user_out: dict[str, str],
) -> None:
    """Load client API keys. Each line: ``<api_key> <user_id>`` (user_id required)."""
    keys_out.clear()
    key_to_user_out.clear()

    def ingest(key: str, user: str) -> None:
        key, user = _trim(key), _trim(user)
        if not key or not user:
            return
        keys_out.add(key)
        key_to_user_out[key] = user

    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                ingest(parts[0], parts[1])
            elif len(parts) == 1:
                ingest(parts[0], parts[0])

    if single := os.environ.get("CNE_GATEWAY_API_KEY"):
        user = os.environ.get("CNE_GATEWAY_API_KEY_USER", single)
        ingest(single, user)

    if csv := os.environ.get("CNE_GATEWAY_API_KEYS"):
        for item in csv.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                key, user = item.split(":", 1)
                ingest(key, user)
            else:
                ingest(item, item)


def authenticate_api_key(
    token: str,
    keys: set[str],
    key_to_user: dict[str, str],
) -> str:
    if not keys:
        raise AuthError("no client API keys configured", status=503)
    if token not in keys:
        raise AuthError("invalid API key")
    user = key_to_user.get(token)
    if not user:
        raise AuthError("API key has no user mapping", status=503)
    return user


class RateLimiter:
    def __init__(self, rpm: int) -> None:
        self.rpm = rpm
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, user_id: str) -> bool:
        if self.rpm <= 0:
            return True
        now = time.time()
        window = 60.0
        q = self._hits[user_id]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= self.rpm:
            return False
        q.append(now)
        return True
