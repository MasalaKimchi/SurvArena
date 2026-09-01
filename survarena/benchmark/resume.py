from __future__ import annotations

import math
from pathlib import Path
from typing import Any


ResumeCompletionKey = tuple[str, str, str, int, str]

# M3-resume: the single default mode a legacy (pre-hpo_mode) row is allowed to
# satisfy. Legacy rows only ever came from the default no-HPO arm.
_LEGACY_DEFAULT_MODE = "no_hpo"


def is_missing_resume_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def resume_completion_key(
    row: dict[str, Any],
    *,
    primary_metric: str,
) -> tuple[ResumeCompletionKey | None, str | None]:
    if str(row.get("status", "")).lower() != "success":
        return None, "status is not success"

    for field in ("dataset_id", "method_id", "split_id"):
        if is_missing_resume_value(row.get(field)):
            return None, f"missing required field '{field}'"

    seed_value = row.get("seed")
    if is_missing_resume_value(seed_value):
        return None, "missing required field 'seed'"
    assert seed_value is not None
    try:
        seed = int(seed_value)
    except (TypeError, ValueError):
        return None, f"invalid seed value '{seed_value}'"

    metric_value = row.get(primary_metric)
    if is_missing_resume_value(metric_value):
        return None, f"missing required metric '{primary_metric}'"

    return (
        (
            str(row.get("dataset_id")),
            str(row.get("method_id")),
            str(row.get("split_id")),
            seed,
            str(row.get("hpo_mode", "")),
        ),
        None,
    )


def completed_resume_keys(
    fold_results_path: Path,
    *,
    primary_metric: str,
    comparison_modes: tuple[str, ...],
) -> set[ResumeCompletionKey]:
    if not fold_results_path.exists():
        return set()

    import pandas as pd

    completed_keys: set[ResumeCompletionKey] = set()
    existing = pd.read_csv(fold_results_path)
    for row in existing.to_dict(orient="records"):
        key, reason = resume_completion_key(row, primary_metric=primary_metric)
        if key is None:
            # D-06: success status alone is insufficient without required output integrity.
            if str(row.get("status", "")).lower() == "success":
                print(
                    "[resume][D-06] Ignoring ineligible success row: "
                    f"dataset_id={row.get('dataset_id')} method_id={row.get('method_id')} "
                    f"split_id={row.get('split_id')} seed={row.get('seed')} reason={reason}"
                )
            continue
        if key[4]:
            completed_keys.add(key)
        else:
            # M3-resume: a legacy row without an hpo_mode column predates dual-mode
            # runs and only ever represented the default (no-HPO) arm. Treat it as
            # completing the no_hpo mode ONLY -- never every comparison mode -- so a
            # requested hpo arm that was never executed is still scheduled to run.
            completed_keys.add((key[0], key[1], key[2], key[3], _LEGACY_DEFAULT_MODE))
    return completed_keys
