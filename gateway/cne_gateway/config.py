from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    upstream: str
    internal_api_key: str
    api_keys_file: Path
    rpm_per_user: int


def gateway_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _resolve_api_keys_file(raw: str) -> Path:
    """Resolve keys file whether cwd is repo root or gateway/."""
    p = Path(raw)
    if p.is_absolute():
        return p.resolve() if p.is_file() else p
    if p.is_file():
        return p.resolve()
    root = gateway_root()
    repo_root = root.parent
    for candidate in (
        Path.cwd() / p,
        repo_root / p,
        root / p,
        root / p.name,
    ):
        if candidate.is_file():
            return candidate.resolve()
    return p


def _resolve_config_path() -> Path | None:
    raw = os.environ.get("CNE_GATEWAY_CONFIG")
    if raw:
        p = Path(raw)
        if p.is_file():
            return p.resolve()
        root = gateway_root()
        for candidate in (Path.cwd() / p, root.parent / p, root / p.name, root / p):
            if candidate.is_file():
                return candidate.resolve()
        return p
    default = gateway_root() / "gateway.json"
    return default if default.is_file() else None


def _load_json_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid gateway config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"invalid gateway config {path}: root must be an object")
    return data


def _pick_str(file_cfg: dict[str, Any], json_key: str, env_name: str, default: str) -> str:
    raw = os.environ.get(env_name)
    if raw is not None and raw != "":
        return raw
    val = file_cfg.get(json_key)
    if val is not None:
        return str(val)
    return default


def _pick_int(file_cfg: dict[str, Any], json_key: str, env_name: str, default: int) -> int:
    raw = os.environ.get(env_name)
    if raw is not None and raw != "":
        return int(raw)
    val = file_cfg.get(json_key)
    if val is not None:
        return int(val)
    return default


def load_settings() -> Settings:
    cfg_path = _resolve_config_path()
    file_cfg = _load_json_config(cfg_path)

    internal_api_key = _pick_str(file_cfg, "internal_api_key", "CNE_INTERNAL_API_KEY", "")
    if not internal_api_key:
        hint = f" (set in {cfg_path} or CNE_INTERNAL_API_KEY)" if cfg_path else ""
        raise SystemExit(
            "CNE_INTERNAL_API_KEY is required (must match cne_server API key)" + hint
        )

    keys_raw = _pick_str(
        file_cfg, "api_keys_file", "CNE_GATEWAY_API_KEYS_FILE", "api_keys.example.txt"
    )
    settings = Settings(
        host=_pick_str(file_cfg, "host", "CNE_GATEWAY_HOST", "127.0.0.1"),
        port=_pick_int(file_cfg, "port", "CNE_GATEWAY_PORT", 8090),
        upstream=_pick_str(
            file_cfg, "upstream", "CNE_UPSTREAM", "http://127.0.0.1:8080"
        ).rstrip("/"),
        internal_api_key=internal_api_key,
        api_keys_file=_resolve_api_keys_file(keys_raw),
        rpm_per_user=_pick_int(file_cfg, "rpm_per_user", "CNE_GATEWAY_RPM", 120),
    )

    if cfg_path:
        print(
            f"[gateway] loaded {cfg_path}",
            file=sys.stderr,
        )
        print(
            f"[gateway] host={settings.host} port={settings.port} "
            f"upstream={settings.upstream} keys={settings.api_keys_file}",
            file=sys.stderr,
        )

    return settings
