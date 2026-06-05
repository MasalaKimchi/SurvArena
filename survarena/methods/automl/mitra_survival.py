from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from survarena.automl.autogluon_backend import (
    AutoGluonFitMetadata,
    fit_autogluon_event_predictor,
    predict_event_probability,
)
from survarena.methods.base import BaseSurvivalMethod, SurvivalPredictions
from survarena.methods.foundation.readiness import ensure_foundation_runtime_ready, rewrite_foundation_runtime_error
from survarena.methods.foundation.tabpfn_survival import (
    _clean_horizon_event_probabilities,
    _kaplan_meier_survival_at,
)
from survarena.methods.survival_utils import fit_breslow_baseline_survival, predict_breslow_survival


class _AutoGluonEventRiskSurvivalBase(BaseSurvivalMethod):
    foundation_method_id = ""
    foundation_backbone = "AutoGluon"
    foundation_hyperparameter_key = "AG"
    foundation_training = "default"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.predictor_: Any | None = None
        self.fit_metadata_: AutoGluonFitMetadata | None = None
        self.baseline_event_times_: np.ndarray | None = None
        self.baseline_survival_: np.ndarray | None = None

    def fit(
        self,
        X_train: Any,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: Any | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "_AutoGluonEventRiskSurvivalBase":
        params = dict(self.params)
        method_id = self.foundation_method_id or self.__class__.__name__
        try:
            ensure_foundation_runtime_ready(method_id)
            self.predictor_, self.fit_metadata_ = fit_autogluon_event_predictor(
                X_train=X_train,
                event_train=event_train,
                X_val=X_val,
                event_val=event_val,
                presets=params.get("presets", "medium"),
                time_limit=params.get("time_limit"),
                hyperparameters=params.get("hyperparameters"),
                hyperparameter_tune_kwargs=params.get("hyperparameter_tune_kwargs"),
                num_bag_folds=int(params.get("num_bag_folds", 0)),
                num_stack_levels=int(params.get("num_stack_levels", 0)),
                refit_full=params.get("refit_full", False),
                path=params.get("path"),
                verbosity=int(params.get("verbosity", 0)),
            )
            train_risk = self.predict_risk(X_train)
            self.baseline_event_times_, self.baseline_survival_ = fit_breslow_baseline_survival(
                time_train=np.asarray(time_train, dtype=float),
                event_train=np.asarray(event_train, dtype=int),
                train_risk_scores=train_risk,
            )
            return self
        except Exception as exc:
            raise rewrite_foundation_runtime_error(method_id, exc) from exc

    def predict_risk(self, X: Any) -> np.ndarray:
        if self.predictor_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before prediction.")
        probabilities = predict_event_probability(self.predictor_, X)
        return np.asarray(probabilities, dtype=float)

    def predict_survival(self, X: Any, times: np.ndarray) -> np.ndarray:
        if self.baseline_event_times_ is None or self.baseline_survival_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before survival prediction.")
        return self._survival_from_risk(self.predict_risk(X), times)

    def _survival_from_risk(self, risk_scores: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.baseline_event_times_ is None or self.baseline_survival_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before survival prediction.")
        return predict_breslow_survival(
            risk_scores=risk_scores,
            times=np.asarray(times, dtype=float),
            baseline_event_times=self.baseline_event_times_,
            baseline_survival=self.baseline_survival_,
        )

    def predict_bundle(self, X: Any, times: np.ndarray) -> SurvivalPredictions:
        risk = self.predict_risk(X)
        return SurvivalPredictions(risk=risk, survival=self._survival_from_risk(risk, times))

    def autogluon_metadata(self) -> dict[str, Any]:
        if self.fit_metadata_ is None:
            return {}
        return {
            "autogluon_best_model": self.fit_metadata_.best_model,
            "autogluon_model_count": self.fit_metadata_.model_count,
            "autogluon_path": self.fit_metadata_.path,
            "autogluon_leaderboard": list(self.fit_metadata_.leaderboard),
        }

    def foundation_metadata(self) -> dict[str, Any]:
        hyperparameters = dict(self.params.get("hyperparameters", {}) or {})
        backbone_params = dict(hyperparameters.get(self.foundation_hyperparameter_key, {}) or {})
        return {
            "foundation_backbone": self.foundation_backbone,
            "foundation_backbone_task": "classification_event",
            "foundation_backbone_training": self.foundation_training,
            "foundation_time_limit_sec": self.params.get("time_limit"),
            "foundation_autogluon_hyperparameter_key": self.foundation_hyperparameter_key,
            "foundation_autogluon_backbone_params": backbone_params,
        }


class _MitraSurvivalMethod(_AutoGluonEventRiskSurvivalBase):
    foundation_method_id = "mitra_survival_frozen"
    foundation_backbone = "Mitra"
    foundation_hyperparameter_key = "MITRA"

    def __init__(self, **params: Any) -> None:
        mitra_params = dict(params.pop("mitra_params", {}) or {})
        mitra_params.setdefault("fine_tune", False)
        resolved = {
            **params,
            "presets": params.pop("presets", None),
            "hyperparameters": {"MITRA": mitra_params},
        }
        super().__init__(**resolved)

    def fit(
        self,
        X_train: Any,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: Any | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "_MitraSurvivalMethod":
        try:
            from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Mitra Survival requires AutoGluon's Mitra extra. "
                'Install it with `python -m pip install -e ".[foundation-mitra]"`.'
            ) from exc
        return super().fit(X_train, time_train, event_train, X_val, time_val, event_val)


class MitraSurvivalFrozenMethod(_MitraSurvivalMethod):
    foundation_training = "frozen"

    def __init__(self, **params: Any) -> None:
        mitra_params = dict(params.pop("mitra_params", {}) or {})
        mitra_params["fine_tune"] = False
        super().__init__(**params, mitra_params=mitra_params)

    def foundation_metadata(self) -> dict[str, Any]:
        metadata = super().foundation_metadata()
        hyperparameters = dict(self.params.get("hyperparameters", {}) or {})
        mitra_params = dict(hyperparameters.get("MITRA", {}) or {})
        metadata["foundation_mitra_fine_tune"] = bool(mitra_params.get("fine_tune", False))
        return metadata


class _AutoGluonFoundationSurvivalMethod(_AutoGluonEventRiskSurvivalBase):
    def __init__(self, **params: Any) -> None:
        backbone_params_key = f"{self.foundation_hyperparameter_key.lower().replace('-', '_')}_params"
        backbone_params = dict(params.pop(backbone_params_key, {}) or {})
        resolved = {
            **params,
            "presets": params.pop("presets", None),
            "hyperparameters": {self.foundation_hyperparameter_key: backbone_params},
        }
        super().__init__(**resolved)


class _AutoGluonHorizonSurvivalMethod(_AutoGluonFoundationSurvivalMethod):
    foundation_training = "default"

    def __init__(self, **params: Any) -> None:
        params.setdefault("horizon_quantiles", "0.25-0.5-0.75")
        params.setdefault("min_known_per_horizon", 20)
        params.setdefault("aggregate_risk", "mean_event_probability")
        super().__init__(**params)
        self.predictors_: list[Any | None] = []
        self.fit_metadata_by_horizon_: list[AutoGluonFitMetadata | None] = []
        self.horizon_times_: np.ndarray | None = None
        self.constant_event_probabilities_: np.ndarray | None = None
        self.used_fallback_: list[bool] = []

    @staticmethod
    def _parse_horizon_quantiles(value: str | list[float]) -> np.ndarray:
        if isinstance(value, str):
            quantiles = [float(part.strip()) for part in value.split("-") if part.strip()]
        else:
            quantiles = [float(part) for part in value]
        if not quantiles:
            raise ValueError("horizon_quantiles must contain at least one quantile.")
        array = np.asarray(quantiles, dtype=np.float64)
        if np.any((array <= 0.0) | (array >= 1.0)):
            raise ValueError("horizon_quantiles must be strictly between 0 and 1.")
        return np.unique(array)

    @staticmethod
    def _horizon_known_labels(
        *,
        time: np.ndarray,
        event: np.ndarray,
        horizon: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        time_np = np.asarray(time, dtype=np.float64)
        event_np = np.asarray(event).astype(bool)
        positive = (time_np <= float(horizon)) & event_np
        negative = time_np > float(horizon)
        known = positive | negative
        return known, positive[known].astype(np.int32)

    def fit(
        self,
        X_train: Any,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: Any | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "_AutoGluonHorizonSurvivalMethod":
        params = dict(self.params)
        method_id = self.foundation_method_id or self.__class__.__name__
        try:
            ensure_foundation_runtime_ready(method_id)
            time_train_np = np.asarray(time_train, dtype=np.float64)
            event_train_np = np.asarray(event_train, dtype=np.int32)
            if int(event_train_np.sum()) <= 0:
                raise ValueError(f"{self.foundation_backbone} survival training requires at least one observed event.")

            quantiles = self._parse_horizon_quantiles(params["horizon_quantiles"])
            event_times = time_train_np[event_train_np.astype(bool)]
            self.horizon_times_ = np.unique(np.quantile(event_times, quantiles).astype(np.float64))
            baseline_survival = _kaplan_meier_survival_at(time_train_np, event_train_np, self.horizon_times_)
            baseline_event_prob = 1.0 - baseline_survival

            self.predictors_ = []
            self.fit_metadata_by_horizon_ = []
            self.used_fallback_ = []
            constants: list[float] = []
            min_known = int(params["min_known_per_horizon"])
            per_horizon_time_limit = _per_horizon_time_limit(params.get("time_limit"), len(self.horizon_times_))

            for idx, horizon in enumerate(self.horizon_times_):
                known_mask, labels = self._horizon_known_labels(
                    time=time_train_np,
                    event=event_train_np,
                    horizon=float(horizon),
                )
                has_both_classes = np.unique(labels).size == 2
                if int(known_mask.sum()) < min_known or not has_both_classes:
                    self.predictors_.append(None)
                    self.fit_metadata_by_horizon_.append(None)
                    self.used_fallback_.append(True)
                    constants.append(float(baseline_event_prob[idx]))
                    continue

                X_val_horizon = None
                labels_val = None
                if X_val is not None and time_val is not None and event_val is not None:
                    val_known, candidate_labels_val = self._horizon_known_labels(
                        time=np.asarray(time_val, dtype=np.float64),
                        event=np.asarray(event_val, dtype=np.int32),
                        horizon=float(horizon),
                    )
                    if int(val_known.sum()) >= min_known and np.unique(candidate_labels_val).size == 2:
                        X_val_horizon = _slice_rows(X_val, val_known)
                        labels_val = candidate_labels_val

                predictor, metadata = fit_autogluon_event_predictor(
                    X_train=_slice_rows(X_train, known_mask),
                    event_train=labels,
                    X_val=X_val_horizon,
                    event_val=labels_val,
                    presets=params.get("presets", "medium"),
                    time_limit=per_horizon_time_limit,
                    hyperparameters=params.get("hyperparameters"),
                    hyperparameter_tune_kwargs=params.get("hyperparameter_tune_kwargs"),
                    num_bag_folds=int(params.get("num_bag_folds", 0)),
                    num_stack_levels=int(params.get("num_stack_levels", 0)),
                    refit_full=params.get("refit_full", False),
                    path=_horizon_path(params.get("path"), idx),
                    verbosity=int(params.get("verbosity", 0)),
                )
                self.predictors_.append(predictor)
                self.fit_metadata_by_horizon_.append(metadata)
                self.used_fallback_.append(False)
                constants.append(float(baseline_event_prob[idx]))

            self.constant_event_probabilities_ = _clean_horizon_event_probabilities(
                np.asarray(constants, dtype=np.float64)[None, :]
            )[0]
            return self
        except Exception as exc:
            raise rewrite_foundation_runtime_error(method_id, exc) from exc

    def _horizon_event_probabilities(self, X: Any) -> np.ndarray:
        if self.horizon_times_ is None or self.constant_event_probabilities_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before prediction.")
        columns: list[np.ndarray] = []
        row_count = _row_count(X)
        for predictor, fallback_prob in zip(self.predictors_, self.constant_event_probabilities_):
            if predictor is None:
                columns.append(np.full(row_count, float(fallback_prob), dtype=np.float64))
            else:
                columns.append(predict_event_probability(predictor, X))
        if not columns:
            return np.zeros((row_count, 0), dtype=np.float64)
        return _clean_horizon_event_probabilities(np.column_stack(columns))

    def predict_risk(self, X: Any) -> np.ndarray:
        horizon_event_probs = self._horizon_event_probabilities(X)
        return self._risk_from_horizon_event_probabilities(horizon_event_probs)

    def _risk_from_horizon_event_probabilities(self, horizon_event_probs: np.ndarray) -> np.ndarray:
        if str(self.params["aggregate_risk"]) == "last_event_probability":
            return horizon_event_probs[:, -1].astype(np.float64)
        return horizon_event_probs.mean(axis=1).astype(np.float64)

    def predict_survival(self, X: Any, times: np.ndarray) -> np.ndarray:
        if self.horizon_times_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before prediction.")
        horizon_event_probs = self._horizon_event_probabilities(X)
        return self._survival_from_horizon_event_probabilities(horizon_event_probs, times)

    def _survival_from_horizon_event_probabilities(
        self,
        horizon_event_probs: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        if self.horizon_times_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before prediction.")
        eval_times = np.asarray(times, dtype=np.float64).reshape(-1)
        rows: list[np.ndarray] = []
        for row in horizon_event_probs:
            event_prob_at_times = np.interp(
                eval_times,
                self.horizon_times_,
                row,
                left=0.0,
                right=float(row[-1]),
            )
            rows.append(1.0 - event_prob_at_times)
        survival = np.vstack(rows) if rows else np.empty((0, eval_times.size), dtype=np.float64)
        survival = np.nan_to_num(survival, nan=1.0, posinf=1.0, neginf=1e-8)
        survival = np.clip(survival, 1e-8, 1.0)
        return np.minimum.accumulate(survival, axis=1).astype(np.float64)

    def predict_bundle(self, X: Any, times: np.ndarray) -> SurvivalPredictions:
        horizon_event_probs = self._horizon_event_probabilities(X)
        return SurvivalPredictions(
            risk=self._risk_from_horizon_event_probabilities(horizon_event_probs),
            survival=self._survival_from_horizon_event_probabilities(horizon_event_probs, times),
        )

    def autogluon_metadata(self) -> dict[str, Any]:
        metadata_rows = [metadata for metadata in self.fit_metadata_by_horizon_ if metadata is not None]
        if not metadata_rows:
            return {}
        leaderboard_rows: list[dict[str, Any]] = []
        for idx, metadata in enumerate(self.fit_metadata_by_horizon_):
            if metadata is None:
                continue
            for row in metadata.leaderboard:
                leaderboard_rows.append({"horizon_index": idx, **dict(row)})
        return {
            "autogluon_best_model": ";".join(
                str(metadata.best_model) for metadata in metadata_rows if metadata.best_model is not None
            )
            or None,
            "autogluon_model_count": int(sum(metadata.model_count for metadata in metadata_rows)),
            "autogluon_path": ";".join(str(metadata.path) for metadata in metadata_rows if metadata.path),
            "autogluon_leaderboard": leaderboard_rows,
        }

    def foundation_metadata(self) -> dict[str, Any]:
        metadata = super().foundation_metadata()
        metadata["foundation_backbone_task"] = "censored_aware_horizon_classification"
        metadata["foundation_horizon_count"] = 0 if self.horizon_times_ is None else int(len(self.horizon_times_))
        metadata["foundation_horizon_fallback_count"] = int(sum(self.used_fallback_))
        metadata["foundation_horizon_time_limit_sec"] = _per_horizon_time_limit(
            self.params.get("time_limit"),
            0 if self.horizon_times_ is None else len(self.horizon_times_),
        )
        return metadata


def _slice_rows(X: Any, mask: np.ndarray) -> Any:
    if hasattr(X, "iloc"):
        return X.iloc[np.asarray(mask, dtype=bool)]
    return np.asarray(X)[np.asarray(mask, dtype=bool)]


def _row_count(X: Any) -> int:
    if hasattr(X, "shape"):
        return int(X.shape[0])
    return int(len(X))


def _horizon_path(path: Any | None, idx: int) -> Path | None:
    if path is None:
        return None
    return Path(path) / f"horizon_{idx}"


def _per_horizon_time_limit(time_limit: Any | None, horizon_count: int) -> float | None:
    if time_limit is None:
        return None
    resolved = float(time_limit)
    if horizon_count <= 0:
        return resolved
    return max(resolved / float(horizon_count), 1.0)


class TabMSurvivalMethod(_AutoGluonHorizonSurvivalMethod):
    foundation_method_id = "tabm_survival"
    foundation_backbone = "TabM"
    foundation_hyperparameter_key = "TABM"
    foundation_training = "fit"


class RealTabPFNV2SurvivalMethod(_AutoGluonHorizonSurvivalMethod):
    foundation_method_id = "realtabpfn_survival"
    foundation_backbone = "RealTabPFN-V2"
    foundation_hyperparameter_key = "REALTABPFN-V2"
    foundation_training = "in_context"
