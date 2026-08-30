from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from cne_gateway.auth import decode_token, issue_token, verify_password
from cne_gateway.config import Settings
from cne_gateway.main import create_app


def make_settings(users_file: Path) -> Settings:
    return Settings(
        host="127.0.0.1",
        port=8090,
        jwt_secret="unit-test-secret",
        jwt_ttl_s=3600,
        upstream="http://engine.test",
        internal_api_key="internal-key",
        users_file=users_file,
        rpm_per_user=1000,
    )


def test_jwt_roundtrip() -> None:
    token = issue_token("alice", "unit-test-secret", 60)
    assert decode_token(token, "unit-test-secret") == "alice"


def test_verify_password(tmp_path: Path) -> None:
    users_file = tmp_path / "users.txt"
    users_file.write_text("alice:secret\n", encoding="utf-8")
    users = {"alice": "secret"}
    assert verify_password(users, "alice", "secret") == "alice"
    with pytest.raises(Exception):
        verify_password(users, "alice", "wrong")


def test_login_and_proxy_chat(tmp_path: Path) -> None:
    users_file = tmp_path / "users.txt"
    users_file.write_text("alice:secret\n", encoding="utf-8")
    settings = make_settings(users_file)

    with respx.mock:
        route = respx.post("http://engine.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        with TestClient(create_app(settings)) as client:
            bad = client.post("/v1/auth/token", json={"username": "alice", "password": "x"})
            assert bad.status_code == 401

            tok = client.post(
                "/v1/auth/token", json={"username": "alice", "password": "secret"}
            )
            assert tok.status_code == 200
            token = tok.json()["access_token"]

            chat = client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {token}", "X-Chat-Id": "c1"},
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


def test_chat_requires_jwt(tmp_path: Path) -> None:
    users_file = tmp_path / "users.txt"
    users_file.write_text("alice:secret\n", encoding="utf-8")
    with TestClient(create_app(make_settings(users_file))) as client:
        res = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert res.status_code == 401
