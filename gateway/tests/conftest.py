from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def keys_file(tmp_path: Path) -> Path:
    path = tmp_path / "keys.txt"
    path.write_text("sk-alice alice\n", encoding="utf-8")
    return path
