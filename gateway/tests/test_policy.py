from __future__ import annotations

import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from cne_gateway.config import load_settings
from cne_gateway.main import create_app
from cne_gateway.policy import PolicyError, apply_chat_policy

from tests.helpers import make_settings, policy_settings


@pytest.mark.parametrize(
    ("request_tokens", "expected"),
    [
        (None, 64),
        (32, 32),
        (64, 64),
        (512, 64),
    ],
)
def test_apply_chat_policy_max_tokens_cap(
    request_tokens: int | None, expected: int
) -> None:
    body: dict = {"messages": []}
    if request_tokens is not None:
        body["max_tokens"] = request_tokens
    out = apply_chat_policy(body, policy_settings(max_tokens_per_request=64))
    assert out["max_tokens"] == expected


def test_apply_chat_policy_max_tokens_unlimited() -> None:
    body = {"messages": [], "max_tokens": 4096}
    out = apply_chat_policy(body, policy_settings(max_tokens_per_request=0))
    assert out["max_tokens"] == 4096
    assert "max_tokens" not in apply_chat_policy({"messages": []}, policy_settings())


@pytest.mark.parametrize(
    "max_tokens",
    ["64", True, 0, -1],
)
def test_apply_chat_policy_rejects_invalid_max_tokens(max_tokens: object) -> None:
    with pytest.raises(PolicyError) as exc:
        apply_chat_policy(
            {"messages": [], "max_tokens": max_tokens},
            policy_settings(max_tokens_per_request=64),
        )
    assert exc.value.status == 400


def test_apply_chat_policy_does_not_mutate_input() -> None:
    body = {
        "messages": [],
        "max_tokens": 512,
        "chat_template_kwargs": {"foo": "bar"},
    }
    original = json.loads(json.dumps(body))
    apply_chat_policy(
        body, policy_settings(allow_thinking=True, max_tokens_per_request=64)
    )
    assert body == original


def test_apply_chat_policy_thinking_allowed() -> None:
    settings = policy_settings(allow_thinking=True)
    body = {"messages": [], "chat_template_kwargs": {"enable_thinking": True}}
    out = apply_chat_policy(body, settings)
    assert out["chat_template_kwargs"]["enable_thinking"] is True

    out2 = apply_chat_policy({"messages": []}, settings)
    assert "chat_template_kwargs" not in out2


def test_apply_chat_policy_thinking_disallowed() -> None:
    settings = policy_settings(allow_thinking=False)

    with pytest.raises(PolicyError) as exc:
        apply_chat_policy(
            {
                "messages": [],
                "chat_template_kwargs": {"enable_thinking": True},
            },
            settings,
        )
    assert exc.value.status == 403
    assert "thinking" in exc.value.detail

    out = apply_chat_policy({"messages": []}, settings)
    assert out["chat_template_kwargs"]["enable_thinking"] is False

    out2 = apply_chat_policy(
        {
            "messages": [],
            "chat_template_kwargs": {"enable_thinking": False, "foo": "bar"},
        },
        settings,
    )
    assert out2["chat_template_kwargs"] == {"enable_thinking": False, "foo": "bar"}


def test_load_policy_settings_from_json(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("sk-alice alice\n", encoding="utf-8")
    cfg = tmp_path / "gateway.json"
    cfg.write_text(
        f"""{{
  "internal_api_key": "secret",
  "api_keys_file": "{keys_file}",
  "allow_thinking": true,
  "max_tokens_per_request": 256
}}""",
        encoding="utf-8",
    )
    for name in (
        "CNE_INTERNAL_API_KEY",
        "CNE_GATEWAY_ALLOW_THINKING",
        "CNE_GATEWAY_MAX_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CNE_GATEWAY_CONFIG", str(cfg))

    settings = load_settings()
    assert settings.allow_thinking is True
    assert settings.max_tokens_per_request == 256


def test_load_policy_settings_env_overrides(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("sk-alice alice\n", encoding="utf-8")
    cfg = tmp_path / "gateway.json"
    cfg.write_text(
        f"""{{
  "internal_api_key": "secret",
  "api_keys_file": "{keys_file}",
  "allow_thinking": false,
  "max_tokens_per_request": 128
}}""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CNE_GATEWAY_CONFIG", str(cfg))
    monkeypatch.setenv("CNE_GATEWAY_ALLOW_THINKING", "1")
    monkeypatch.setenv("CNE_GATEWAY_MAX_TOKENS", "512")

    settings = load_settings()
    assert settings.allow_thinking is True
    assert settings.max_tokens_per_request == 512


def test_health_exposes_policy(keys_file) -> None:
    settings = make_settings(
        keys_file, allow_thinking=False, max_tokens_per_request=512
    )
    with TestClient(create_app(settings)) as client:
        res = client.get("/health")
    assert res.status_code == 200
    gateway = res.json()["gateway"]
    assert gateway["allow_thinking"] is False
    assert gateway["max_tokens_per_request"] == 512


def test_chat_allows_thinking_when_enabled(keys_file) -> None:
    settings = make_settings(keys_file, allow_thinking=True)

    with respx.mock:
        route = respx.post("http://engine.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        with TestClient(create_app(settings)) as client:
            res = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-alice"},
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            )
        assert res.status_code == 200
        body = json.loads(route.calls[0].request.content)
        assert body["chat_template_kwargs"]["enable_thinking"] is True


def test_chat_rejects_invalid_max_tokens(keys_file) -> None:
    settings = make_settings(keys_file, max_tokens_per_request=64)
    with TestClient(create_app(settings)) as client:
        res = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-alice"},
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": "lots",
            },
        )
    assert res.status_code == 400
    assert "max_tokens" in res.json()["detail"]


def test_chat_policy_applies_to_streaming(keys_file) -> None:
    settings = make_settings(keys_file, max_tokens_per_request=32)

    with respx.mock:
        route = respx.post("http://engine.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, content=b"data: {}\n\n")
        )
        with TestClient(create_app(settings)) as client:
            res = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-alice"},
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                    "max_tokens": 256,
                },
            )
        assert res.status_code == 200
        body = json.loads(route.calls[0].request.content)
        assert body["max_tokens"] == 32
        assert body["stream"] is True
