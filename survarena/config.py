from __future__ import annotations

from pathlib import Path
from typing import Any


def read_yaml(path: Path) -> dict[str, Any]:
    import importlib

    yaml = importlib.import_module("yaml")

    return _resolve_extends(path=Path(path), yaml_module=yaml, seen=())


def _resolve_extends(*, path: Path, yaml_module: Any, seen: tuple[Path, ...]) -> dict[str, Any]:
    resolved_path = path.resolve()
    if resolved_path in seen:
        chain = " -> ".join(str(item) for item in (*seen, resolved_path))
        raise ValueError(f"YAML extends cycle detected: {chain}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml_module.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")

    parent_ref = data.pop("extends", None)
    if parent_ref is None:
        return data

    parent_path = Path(parent_ref)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    parent = _resolve_extends(path=parent_path, yaml_module=yaml_module, seen=(*seen, resolved_path))
    return _deep_merge(parent, data)


def _deep_merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = dict(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
