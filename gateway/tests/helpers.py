from __future__ import annotations

from pathlib import Path

from cne_gateway.config import Settings


def make_settings(
    keys_file: Path,
    *,
    allow_thinking: bool = False,
    max_tokens_per_request: int = 0,
) -> Settings:
    return Settings(
        host="127.0.0.1",
        port=8090,
        upstream="http://engine.test",
        internal_api_key="internal-key",
        api_keys_file=keys_file,
        rpm_per_user=1000,
        allow_thinking=allow_thinking,
        max_tokens_per_request=max_tokens_per_request,
    )


def policy_settings(
    *,
    allow_thinking: bool = False,
    max_tokens_per_request: int = 0,
) -> Settings:
    return Settings(
        host="127.0.0.1",
        port=8090,
        upstream="http://engine.test",
        internal_api_key="k",
        api_keys_file=Path("."),
        rpm_per_user=100,
        allow_thinking=allow_thinking,
        max_tokens_per_request=max_tokens_per_request,
    )
