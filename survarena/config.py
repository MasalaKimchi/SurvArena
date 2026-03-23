from __future__ import annotations

from pathlib import Path
from typing import Any


def read_yaml(path: Path) -> dict[str, Any]:
    import importlib

    yaml = importlib.import_module("yaml")

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
