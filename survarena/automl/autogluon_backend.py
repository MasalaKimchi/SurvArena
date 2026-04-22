from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tempfile

import numpy as np
import pandas as pd


_TARGET_COL = "__survarena_event_target__"


@dataclass(slots=True)
class AutoGluonFitMetadata:
    best_model: str | None = None
    model_count: int = 0
    path: str | None = None
    leaderboard: list[dict[str, Any]] = field(default_factory=list)


def _as_frame(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True).copy()
    return pd.DataFrame(np.asarray(X)).reset_index(drop=True)


def _training_frame(X: Any, event: np.ndarray) -> pd.DataFrame:
    frame = _as_frame(X)
    frame[_TARGET_COL] = np.asarray(event, dtype=int)
    return frame


def fit_autogluon_event_predictor(
    *,
    X_train: Any,
    event_train: np.ndarray,
    X_val: Any | None = None,
    event_val: np.ndarray | None = None,
    presets: str | list[str] | None = "medium",
    time_limit: float | None = None,
    hyperparameters: Any | None = None,
    hyperparameter_tune_kwargs: dict[str, Any] | None = None,
    num_bag_folds: int = 0,
    num_stack_levels: int = 0,
    refit_full: bool | str = False,
    path: str | Path | None = None,
    verbosity: int = 0,
) -> tuple[Any, AutoGluonFitMetadata]:
    from autogluon.tabular import TabularPredictor

    train_frame = _training_frame(X_train, event_train)
    tuning_frame = None
    if X_val is not None and event_val is not None and int(num_bag_folds) <= 0:
        tuning_frame = _training_frame(X_val, event_val)

    if path is None:
        path = Path(tempfile.mkdtemp(prefix="survarena_autogluon_"))
    else:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

    predictor = TabularPredictor(
        label=_TARGET_COL,
        problem_type="binary",
        eval_metric="roc_auc",
        path=str(path),
        verbosity=int(verbosity),
    )
    fit_kwargs: dict[str, Any] = {
        "train_data": train_frame,
        "presets": presets,
        "time_limit": time_limit,
        "hyperparameters": hyperparameters,
        "hyperparameter_tune_kwargs": hyperparameter_tune_kwargs,
        "num_bag_folds": int(num_bag_folds),
        "num_stack_levels": int(num_stack_levels),
    }
    if tuning_frame is not None:
        fit_kwargs["tuning_data"] = tuning_frame

    predictor.fit(**{key: value for key, value in fit_kwargs.items() if value is not None})
    if refit_full:
        try:
            predictor.refit_full(model="best")
        except Exception:
            # Refit is a quality improvement, not a fit contract for every AutoGluon setup.
            pass

    leaderboard_rows: list[dict[str, Any]] = []
    try:
        leaderboard_rows = predictor.leaderboard(silent=True).to_dict(orient="records")
    except Exception:
        leaderboard_rows = []

    best_model: str | None = None
    try:
        best_model = str(predictor.model_best)
    except Exception:
        best_model = None

    return predictor, AutoGluonFitMetadata(
        best_model=best_model,
        model_count=len(leaderboard_rows),
        path=str(path),
        leaderboard=leaderboard_rows,
    )


def predict_event_probability(predictor: Any, X: Any) -> np.ndarray:
    frame = _as_frame(X)
    proba = predictor.predict_proba(frame)
    if isinstance(proba, pd.DataFrame):
        if 1 in proba.columns:
            return proba[1].to_numpy(dtype=float)
        if "1" in proba.columns:
            return proba["1"].to_numpy(dtype=float)
        return proba.iloc[:, -1].to_numpy(dtype=float)
    array = np.asarray(proba, dtype=float)
    if array.ndim == 2:
        return array[:, -1]
    return array.reshape(-1)
