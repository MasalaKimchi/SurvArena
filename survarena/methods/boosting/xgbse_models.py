from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from survarena.methods.base import BaseSurvivalMethod, to_structured_y


def _as_xgbse_input(X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return np.asarray(X, dtype=np.float32)


def _has_validation_data(
    X_val: np.ndarray | pd.DataFrame | None,
    time_val: np.ndarray | None,
    event_val: np.ndarray | None,
) -> bool:
    return X_val is not None and time_val is not None and event_val is not None


def _n_rows(X: np.ndarray | pd.DataFrame) -> int:
    return len(X) if isinstance(X, pd.DataFrame) else int(np.asarray(X).shape[0])


def _validated_training_arrays(
    time: np.ndarray,
    event: np.ndarray,
    *,
    require_event: bool = True,
    time_name: str = "time_train",
    event_name: str = "event_train",
) -> tuple[np.ndarray, np.ndarray]:
    time_array = np.asarray(time, dtype=np.float64).reshape(-1)
    event_array = np.asarray(event, dtype=np.int32).reshape(-1)
    if time_array.shape[0] != event_array.shape[0]:
        raise ValueError(f"{time_name} and {event_name} must have the same length.")
    if not np.isfinite(time_array).all():
        raise ValueError(f"{time_name} must contain only finite values.")
    if np.any(time_array <= 0.0):
        raise ValueError(f"{time_name} must contain strictly positive survival times.")
    if require_event and not np.any(event_array.astype(bool)):
        raise ValueError("XGBSEKaplanNeighborsMethod requires at least one observed event.")
    return time_array, event_array


def _risk_time_bins(time: np.ndarray, event: np.ndarray, *, max_bins: int) -> np.ndarray:
    if max_bins <= 0:
        raise ValueError("risk_time_bins must be positive.")
    event_times = np.asarray(time, dtype=np.float64)[np.asarray(event, dtype=bool)]
    source_times = event_times if event_times.size else np.asarray(time, dtype=np.float64)
    source_times = source_times[np.isfinite(source_times) & (source_times > 0.0)]
    if source_times.size == 0:
        return np.asarray([1.0], dtype=np.float64)

    unique_times = np.unique(source_times)
    if unique_times.size <= max_bins:
        return unique_times.astype(np.float64)
    quantiles = np.linspace(0.0, 1.0, max_bins)
    return np.unique(np.quantile(unique_times, quantiles)).astype(np.float64)


def _clean_survival_curves(values: np.ndarray) -> np.ndarray:
    curves = np.asarray(values, dtype=np.float64)
    curves = np.nan_to_num(curves, nan=1.0, posinf=1.0, neginf=1e-8)
    curves = np.clip(curves, 1e-8, 1.0)
    return np.minimum.accumulate(curves, axis=1).astype(np.float64)


def _xgbse_xgb_params(params: dict[str, Any]) -> dict[str, Any]:
    xgb_params: dict[str, Any] = {
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "learning_rate": float(params["learning_rate"]),
        "max_depth": int(params["max_depth"]),
        "min_child_weight": float(params["min_child_weight"]),
        "subsample": float(params["subsample"]),
        "colsample_bynode": float(params["colsample_bynode"]),
        "lambda": float(params["reg_lambda"]),
        "alpha": float(params["reg_alpha"]),
        "max_bin": int(params["max_bin"]),
        "tree_method": str(params["tree_method"]),
        "device": str(params["device"]),
        "booster": str(params["booster"]),
        "nthread": 1,
    }
    if params.get("seed") is not None:
        xgb_params["seed"] = int(params["seed"])
    if xgb_params["booster"] == "dart":
        xgb_params.update(
            {
                "normalize_type": str(params["normalize_type"]),
                "rate_drop": float(params["rate_drop"]),
                "skip_drop": float(params["skip_drop"]),
            }
        )
    return xgb_params


def _import_xgbse_kaplan_neighbors() -> Any:
    try:
        from xgbse import XGBSEKaplanNeighbors
    except ImportError as exc:
        raise RuntimeError(
            "XGBSEKaplanNeighborsMethod requires optional package 'xgbse'. "
            "The latest xgbse release declares compatibility with xgboost>=2.1,<3; "
            "use a compatible environment before running this adapter."
        ) from exc
    return XGBSEKaplanNeighbors


class XGBSEKaplanNeighborsMethod(BaseSurvivalMethod):
    def __init__(
        self,
        learning_rate: float = 0.05,
        num_boost_round: int = 300,
        max_depth: int = 4,
        min_child_weight: float = 10.0,
        subsample: float = 0.8,
        colsample_bynode: float = 0.8,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        max_bin: int = 256,
        tree_method: str = "hist",
        device: str = "cpu",
        booster: str = "dart",
        normalize_type: str = "tree",
        rate_drop: float = 0.1,
        skip_drop: float = 0.0,
        n_neighbors: int = 30,
        radius: float | None = None,
        enable_categorical: bool = False,
        early_stopping_rounds: int | None = 25,
        risk_time_bins: int = 32,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            num_boost_round=num_boost_round,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bynode=colsample_bynode,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            max_bin=max_bin,
            tree_method=tree_method,
            device=device,
            booster=booster,
            normalize_type=normalize_type,
            rate_drop=rate_drop,
            skip_drop=skip_drop,
            n_neighbors=n_neighbors,
            radius=radius,
            enable_categorical=enable_categorical,
            early_stopping_rounds=early_stopping_rounds,
            risk_time_bins=risk_time_bins,
            seed=seed,
        )
        self.model = None
        self.risk_time_bins_: np.ndarray | None = None

    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | pd.DataFrame | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "XGBSEKaplanNeighborsMethod":
        XGBSEKaplanNeighbors = _import_xgbse_kaplan_neighbors()
        time_array, event_array = _validated_training_arrays(time_train, event_train)
        n_train = _n_rows(X_train)
        if n_train != time_array.shape[0]:
            raise ValueError("X_train, time_train, and event_train must have matching row counts.")

        has_validation = _has_validation_data(X_val, time_val, event_val)
        validation_data = None
        if has_validation:
            if _n_rows(X_val) != len(time_val) or len(time_val) != len(event_val):
                raise ValueError("X_val, time_val, and event_val must have matching row counts.")
            validation_time, validation_event = _validated_training_arrays(
                time_val,
                event_val,
                require_event=False,
                time_name="time_val",
                event_name="event_val",
            )
            validation_data = [(_as_xgbse_input(X_val), to_structured_y(validation_time, validation_event))]

        configured_n_neighbors = int(self.params["n_neighbors"])
        if configured_n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive.")
        radius = self.params["radius"]
        if radius is not None and float(radius) < 0.0:
            raise ValueError("radius must be non-negative when provided.")

        n_neighbors = max(1, min(configured_n_neighbors, n_train))
        self.model = XGBSEKaplanNeighbors(
            xgb_params=_xgbse_xgb_params(self.params),
            n_neighbors=n_neighbors,
            radius=None if radius is None else float(radius),
            enable_categorical=bool(self.params["enable_categorical"]),
        )
        self.model.fit(
            _as_xgbse_input(X_train),
            to_structured_y(time_array, event_array),
            validation_data=validation_data,
            num_boost_round=int(self.params["num_boost_round"]),
            early_stopping_rounds=(
                None if not has_validation or self.params["early_stopping_rounds"] is None
                else int(self.params["early_stopping_rounds"])
            ),
            verbose_eval=0,
        )
        self.risk_time_bins_ = _risk_time_bins(
            time_array,
            event_array,
            max_bins=int(self.params["risk_time_bins"]),
        )
        return self

    def predict_risk(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.risk_time_bins_ is None:
            raise RuntimeError("XGBSEKaplanNeighborsMethod must be fit before prediction.")
        survival = self._predict_survival_at_times(X, self.risk_time_bins_)
        if survival.shape[1] <= 1:
            return (1.0 - survival.mean(axis=1)).astype(np.float64)
        return (-np.trapz(survival, x=self.risk_time_bins_, axis=1)).astype(np.float64)

    def predict_survival(self, X: np.ndarray | pd.DataFrame, times: np.ndarray) -> np.ndarray:
        return self._predict_survival_at_times(X, np.asarray(times, dtype=np.float64))

    def _predict_survival_at_times(self, X: np.ndarray | pd.DataFrame, times: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("XGBSEKaplanNeighborsMethod must be fit before prediction.")
        eval_times = np.asarray(times, dtype=np.float64).reshape(-1)
        if not np.isfinite(eval_times).all():
            raise ValueError("Prediction times must contain only finite values.")
        n_samples = _n_rows(X)
        if eval_times.size == 0:
            return np.empty((n_samples, 0), dtype=np.float64)

        positive_mask = eval_times > 0.0
        curves_at_times = np.ones((n_samples, eval_times.size), dtype=np.float64)
        if not np.any(positive_mask):
            return curves_at_times

        unique_times, inverse = np.unique(eval_times[positive_mask], return_inverse=True)
        predictions = self.model.predict(_as_xgbse_input(X), time_bins=unique_times)
        curves = _clean_survival_curves(predictions.to_numpy(dtype=np.float64))
        curves_at_times[:, positive_mask] = curves[:, inverse]
        return _clean_survival_curves(curves_at_times)
