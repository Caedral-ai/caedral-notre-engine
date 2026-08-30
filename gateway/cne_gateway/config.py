from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    jwt_secret: str
    jwt_ttl_s: int
    upstream: str
    internal_api_key: str
    users_file: Path
    rpm_per_user: int


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def load_settings() -> Settings:
    secret = os.environ.get("CNE_GATEWAY_JWT_SECRET", "")
    if not secret:
        raise SystemExit(
            "CNE_GATEWAY_JWT_SECRET is required (use a long random string)"
        )
    key = os.environ.get("CNE_INTERNAL_API_KEY", "")
    if not key:
        raise SystemExit(
            "CNE_INTERNAL_API_KEY is required (must match cne_server API key)"
        )
    users = os.environ.get("CNE_GATEWAY_USERS_FILE", "gateway/users.example.txt")
    return Settings(
        host=os.environ.get("CNE_GATEWAY_HOST", "127.0.0.1"),
        port=_env_int("CNE_GATEWAY_PORT", 8090),
        jwt_secret=secret,
        jwt_ttl_s=_env_int("CNE_GATEWAY_JWT_TTL_S", 86_400),
        upstream=os.environ.get("CNE_UPSTREAM", "http://127.0.0.1:8080").rstrip(
            "/"
        ),
        internal_api_key=key,
        users_file=Path(users),
        rpm_per_user=_env_int("CNE_GATEWAY_RPM", 120),
    )
