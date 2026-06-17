from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(slots=True)
class PredictorModelResult:
    method_id: str
    selection_score: float
    validation_metrics: dict[str, float]
    fit_time_sec: float
    selection_evaluations: int
    params: dict[str, Any]
    training_backend: str = "native"
    hpo_backend: str = "none"
    autogluon_presets: Any | None = None
    autogluon_best_model: str | None = None
    autogluon_model_count: int = 0
    autogluon_path: str | None = None
    bagging_folds: int = 0
    stack_levels: int = 0
    time_limit_sec: float | None = None
    retained_for_inference: bool = False
    status: str = "success"
    error: str | None = None
    error_type: str | None = None


def build_leaderboard(
    results: list[PredictorModelResult],
    *,
    eval_metric: str,
    model_test_metrics: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "method_id": result.method_id,
            "selection_metric": eval_metric,
            "selection_score": result.selection_score,
            "fit_time_sec": result.fit_time_sec,
            "training_backend": result.training_backend,
            "hpo_backend": result.hpo_backend,
            "autogluon_presets": result.autogluon_presets,
            "autogluon_best_model": result.autogluon_best_model,
            "autogluon_model_count": result.autogluon_model_count,
            "autogluon_path": result.autogluon_path,
            "bagging_folds": result.bagging_folds,
            "stack_levels": result.stack_levels,
            "selection_evaluations": result.selection_evaluations,
            "time_limit_sec": result.time_limit_sec,
            "retained_for_inference": result.retained_for_inference,
            "status": result.status,
            "error": result.error,
            "error_type": result.error_type,
            "params": result.params,
        }
        row.update(result.validation_metrics)
        row.update(model_test_metrics.get(result.method_id, {}))
        rows.append(row)

    leaderboard = pd.DataFrame(rows)
    leaderboard["_status_rank"] = leaderboard["status"].map({"success": 0, "failed": 1, "skipped": 2}).fillna(3)
    leaderboard = leaderboard.sort_values(
        by=["_status_rank", f"validation_{eval_metric}"],
        ascending=[True, False],
        na_position="last",
    ).drop(columns=["_status_rank"])
    leaderboard = leaderboard.reset_index(drop=True)
    leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1, dtype=int))
    return leaderboard


def selection_sort_key(result: PredictorModelResult) -> tuple[bool, float]:
    score = float(result.selection_score)
    if not np.isfinite(score):
        return (False, 0.0)
    return (True, score)


def append_budget_exhausted_results(
    *,
    results: list[PredictorModelResult],
    method_ids: list[str],
    training_backend_for_method: Callable[[str], str],
) -> None:
    for method_id in method_ids:
        results.append(
            PredictorModelResult(
                method_id=method_id,
                selection_score=float("nan"),
                validation_metrics={},
                fit_time_sec=0.0,
                selection_evaluations=0,
                params={},
                training_backend=training_backend_for_method(method_id),
                time_limit_sec=0.0,
                status="skipped",
                error="Global fit time budget exhausted before this model could be selected.",
                error_type="TimeLimitExceeded",
            )
        )


def training_backend_for_method(method_id: str, *, is_autogluon_method: Callable[[str], bool]) -> str:
    return "autogluon" if is_autogluon_method(method_id) else "native"


def hpo_backend_for_method(
    method_id: str,
    params: dict[str, Any],
    *,
    is_autogluon_method: Callable[[str], bool],
) -> str:
    if is_autogluon_method(method_id) and params.get("hyperparameter_tune_kwargs"):
        return "autogluon"
    return "none"


def attach_result_fit_metadata(result: PredictorModelResult, model: Any) -> None:
    metadata_getter = getattr(model, "autogluon_metadata", None)
    if not callable(metadata_getter):
        return
    metadata = dict(metadata_getter())
    result.autogluon_best_model = metadata.get("autogluon_best_model")
    result.autogluon_model_count = int(metadata.get("autogluon_model_count") or 0)
    result.autogluon_path = metadata.get("autogluon_path")
