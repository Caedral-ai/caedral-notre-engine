from __future__ import annotations

from typing import Any

from .config import Settings


class PolicyError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(detail)


def apply_chat_policy(body: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Enforce gateway.json chat limits before forwarding to cne_server."""
    out = dict(body)

    if settings.max_tokens_per_request > 0:
        cap = settings.max_tokens_per_request
        raw = out.get("max_tokens")
        if raw is None:
            out["max_tokens"] = cap
        elif not isinstance(raw, int) or isinstance(raw, bool):
            raise PolicyError(400, "max_tokens must be a positive integer")
        elif raw < 1:
            raise PolicyError(400, "max_tokens must be >= 1")
        elif raw > cap:
            out["max_tokens"] = cap

    if not settings.allow_thinking:
        kwargs = out.get("chat_template_kwargs")
        if not isinstance(kwargs, dict):
            kwargs = {}
        else:
            kwargs = dict(kwargs)
        if kwargs.get("enable_thinking") is True:
            raise PolicyError(403, "thinking is disabled by gateway policy")
        kwargs["enable_thinking"] = False
        out["chat_template_kwargs"] = kwargs

    return out
