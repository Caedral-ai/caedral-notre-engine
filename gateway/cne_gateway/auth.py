from __future__ import annotations

import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict

import jwt


class AuthError(Exception):
    def __init__(self, message: str, status: int = 401) -> None:
        super().__init__(message)
        self.status = status


def load_users(path: Path) -> dict[str, str]:
    users: dict[str, str] = {}
    if not path.is_file():
        raise AuthError(f"users file not found: {path}", status=503)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        user, password = line.split(":", 1)
        user, password = user.strip(), password.strip()
        if user:
            users[user] = password
    if not users:
        raise AuthError(f"no users in {path}", status=503)
    return users


def verify_password(users: dict[str, str], username: str, password: str) -> str:
    expected = users.get(username)
    if expected is None or expected != password:
        raise AuthError("invalid username or password")
    return username


def issue_token(user_id: str, secret: str, ttl_s: int) -> str:
    now = int(time.time())
    payload = {"sub": user_id, "iat": now, "exp": now + ttl_s}
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_token(token: str, secret: str) -> str:
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise AuthError("invalid or expired token") from exc
    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub:
        raise AuthError("token missing subject")
    return sub


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
