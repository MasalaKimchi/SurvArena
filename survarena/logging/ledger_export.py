from __future__ import annotations

from datetime import datetime
from pathlib import Path

from survarena.logging.export_shared import RUN_LEDGER_COMPACT_SCHEMA_VERSION, RUN_LEDGER_SCHEMA_VERSION
from survarena.logging.tracker import write_json, write_jsonl_gz


def export_run_ledger(
    root: Path,
    run_records: list[dict],
    *,
    benchmark_id: str,
    output_dir: Path | None = None,
    write_full_ledger: bool = False,
) -> None:
    created_at = datetime.now().isoformat(timespec="seconds")
    normalized_records: list[dict[str, object]] = []
    for record in run_records:
        normalized = {
            "schema_version": RUN_LEDGER_SCHEMA_VERSION,
            **record,
        }
        metrics = normalized.get("metrics")
        status = normalized.get("status")
        retry_attempt = normalized.get("retry_attempt")
        if isinstance(metrics, dict):
            if status is None:
                status = metrics.get("status")
            if retry_attempt is None:
                retry_attempt = metrics.get("retry_attempt")
        normalized["status"] = status if status is not None else "unknown"
        try:
            normalized["retry_attempt"] = int(retry_attempt) if retry_attempt is not None else 0
        except (TypeError, ValueError):
            normalized["retry_attempt"] = 0
        normalized["failure"] = normalized.get("failure")
        normalized_records.append(normalized)
    if output_dir is None:
        compact_output = root / "results" / "runs" / f"{benchmark_id}_run_records_compact.jsonl.gz"
        compact_index_output = root / "results" / "runs" / f"{benchmark_id}_run_records_compact_index.json"
        output = root / "results" / "runs" / f"{benchmark_id}_run_records.jsonl.gz"
        index_output = root / "results" / "runs" / f"{benchmark_id}_run_records_index.json"
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        compact_output = output_dir / f"{benchmark_id}_run_records_compact.jsonl.gz"
        compact_index_output = output_dir / f"{benchmark_id}_run_records_compact_index.json"
        output = output_dir / f"{benchmark_id}_run_records.jsonl.gz"
        index_output = output_dir / f"{benchmark_id}_run_records_index.json"
    shared_manifest: dict[str, object] = {}
    if normalized_records:
        first_manifest = normalized_records[0].get("manifest")
        if isinstance(first_manifest, dict):
            for key, value in first_manifest.items():
                if all(
                    isinstance(record.get("manifest"), dict) and record["manifest"].get(key) == value
                    for record in normalized_records
                ):
                    shared_manifest[key] = value
    compact_records: list[dict[str, object]] = []
    for record in normalized_records:
        compact: dict[str, object] = {"schema_version": RUN_LEDGER_COMPACT_SCHEMA_VERSION}
        for key, value in record.items():
            if key == "schema_version":
                continue
            if key == "manifest" and isinstance(value, dict):
                unique_manifest = {mk: mv for mk, mv in value.items() if mk not in shared_manifest}
                if unique_manifest:
                    compact["manifest"] = unique_manifest
                continue
            compact[key] = value
        compact_records.append(compact)
    write_jsonl_gz(compact_output, compact_records)
    write_json(
        compact_index_output,
        {
            "schema_version": RUN_LEDGER_COMPACT_SCHEMA_VERSION,
            "benchmark_id": benchmark_id,
            "record_count": len(compact_records),
            "format": "jsonl.gz",
            "path": str(compact_output),
            "created_at": created_at,
            "manifest_shared": shared_manifest,
            "record_sections": [
                "schema_version",
                "manifest_shared",
                "manifest",
                "metrics",
                "backend_metadata",
                "hpo_metadata",
                "hpo_trials",
                "failure",
            ],
        },
    )
    if write_full_ledger:
        write_jsonl_gz(output, normalized_records)
        write_json(
            index_output,
            {
                "schema_version": RUN_LEDGER_SCHEMA_VERSION,
                "benchmark_id": benchmark_id,
                "record_count": len(normalized_records),
                "format": "jsonl.gz",
                "path": str(output),
                "created_at": created_at,
                "record_sections": [
                    "schema_version",
                    "manifest",
                    "metrics",
                    "backend_metadata",
                    "hpo_metadata",
                    "hpo_trials",
                    "failure",
                ],
            },
        )
