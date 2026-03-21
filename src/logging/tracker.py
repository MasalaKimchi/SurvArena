from __future__ import annotations

import gzip
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

import psutil


def current_memory_mb() -> float:
    process = psutil.Process()
    return float(process.memory_info().rss / (1024**2))


def peak_memory_mb() -> float:
    try:
        import resource

        peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if peak_rss <= 0:
            return current_memory_mb()
        if sys.platform == "darwin":
            return float(peak_rss / (1024**2))
        return float(peak_rss / 1024.0)
    except Exception:
        return current_memory_mb()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def canonical_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def payload_sha256(payload: Any) -> str:
    data = canonical_json_dumps(payload).encode("utf-8")
    return sha256(data).hexdigest()


def write_jsonl_gz(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json_dumps(record))
            handle.write("\n")
