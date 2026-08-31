from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from cne_gateway.auth import authenticate_api_key, load_api_keys
from cne_gateway.config import load_settings
from cne_gateway.main import create_app

from tests.helpers import make_settings


def test_api_key_maps_to_user(tmp_path: Path) -> None:
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("sk-alice alice\n", encoding="utf-8")
    keys: set[str] = set()
    mapping: dict[str, str] = {}
    load_api_keys(keys_file, keys, mapping)
    assert authenticate_api_key("sk-alice", keys, mapping) == "alice"
    with pytest.raises(Exception):
        authenticate_api_key("wrong", keys, mapping)


def test_api_key_proxy_chat(tmp_path: Path) -> None:
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("sk-alice alice\n", encoding="utf-8")
    settings = make_settings(keys_file)

    with respx.mock:
        route = respx.post("http://engine.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        with TestClient(create_app(settings)) as client:
            bad = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer wrong"},
                json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
            )
            assert bad.status_code == 401

            chat = client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer sk-alice",
                    "X-Chat-Id": "c1",
                },
                json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
            )
            assert chat.status_code == 200
            assert chat.json() == {"ok": True}
            assert route.called
            sent = route.calls[0].request
            assert sent.headers["authorization"] == "Bearer internal-key"
            assert sent.headers["x-user-id"] == "alice"
            import json

            assert json.loads(sent.content)["chat_id"] == "c1"


def test_load_settings_from_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("sk-alice alice\n", encoding="utf-8")
    cfg = tmp_path / "gateway.json"
    cfg.write_text(
        f"""{{
  "host": "0.0.0.0",
  "port": 9001,
  "upstream": "http://engine.test:8080",
  "internal_api_key": "secret-from-file",
  "api_keys_file": "{keys_file}",
  "rpm_per_user": 42
}}""",
        encoding="utf-8",
    )
    monkeypatch.delenv("CNE_INTERNAL_API_KEY", raising=False)
    monkeypatch.delenv("CNE_GATEWAY_HOST", raising=False)
    monkeypatch.delenv("CNE_GATEWAY_API_KEYS_FILE", raising=False)
    monkeypatch.delenv("CNE_GATEWAY_PORT", raising=False)
    monkeypatch.setenv("CNE_GATEWAY_CONFIG", str(cfg))
    s = load_settings()
    assert s.host == "0.0.0.0"
    assert s.port == 9001
    assert s.upstream == "http://engine.test:8080"
    assert s.internal_api_key == "secret-from-file"
    assert s.api_keys_file == keys_file.resolve()
    assert s.rpm_per_user == 42
    assert s.allow_thinking is False
    assert s.max_tokens_per_request == 0

    monkeypatch.setenv("CNE_GATEWAY_PORT", "7777")
    s2 = load_settings()
    assert s2.port == 7777


def test_chat_requires_api_key(tmp_path: Path) -> None:
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("sk-alice alice\n", encoding="utf-8")
    with TestClient(create_app(make_settings(keys_file))) as client:
        res = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert res.status_code == 401
