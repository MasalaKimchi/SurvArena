from __future__ import annotations

from typing import Any

import numpy as np

from survarena.methods.base import BaseSurvivalMethod
from survarena.methods.foundation.readiness import ensure_foundation_runtime_ready, rewrite_foundation_runtime_error
from survarena.methods.foundation.tabpfn_survival import (
    _clean_horizon_event_probabilities,
    _kaplan_meier_survival_at,
)


class _DirectHorizonClassifierSurvivalMethod(BaseSurvivalMethod):
    method_id = ""
    foundation_backbone = "DirectFoundationClassifier"
    foundation_training = "frozen"

    def __init__(
        self,
        horizon_quantiles: str | list[float] = "0.25-0.5-0.75",
        min_known_per_horizon: int = 20,
        aggregate_risk: str = "mean_event_probability",
        seed: int | None = None,
        predict_batch_size: int = 32,
        **params: Any,
    ) -> None:
        super().__init__(
            horizon_quantiles=horizon_quantiles,
            min_known_per_horizon=min_known_per_horizon,
            aggregate_risk=aggregate_risk,
            seed=seed,
            predict_batch_size=predict_batch_size,
            **params,
        )
        self.models_: list[Any | None] = []
        self.horizon_times_: np.ndarray | None = None
        self.constant_event_probabilities_: np.ndarray | None = None
        self.used_fallback_: list[bool] = []

    def foundation_metadata(self) -> dict[str, Any]:
        return {
            "foundation_backbone": self.foundation_backbone,
            "foundation_backbone_task": "censored_aware_horizon_classification",
            "foundation_backbone_training": self.foundation_training,
            "foundation_horizon_count": 0 if self.horizon_times_ is None else int(len(self.horizon_times_)),
            "foundation_horizon_fallback_count": int(sum(self.used_fallback_)),
        }

    def _build_backbone(self) -> Any:
        raise NotImplementedError

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
        time_train: np.ndarray,
        event_train: np.ndarray,
        horizon: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        time = np.asarray(time_train, dtype=np.float64)
        event = np.asarray(event_train).astype(bool)
        positive = (time <= float(horizon)) & event
        negative = time > float(horizon)
        known = positive | negative
        return known, positive[known].astype(np.int32)

    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "_DirectHorizonClassifierSurvivalMethod":
        del X_val, time_val, event_val
        method_id = self.method_id or self.__class__.__name__
        ensure_foundation_runtime_ready(method_id)
        try:
            X_train_np = np.asarray(X_train, dtype=np.float32)
            time_train_np = np.asarray(time_train, dtype=np.float64)
            event_train_np = np.asarray(event_train, dtype=np.int32)
            if int(event_train_np.sum()) <= 0:
                raise ValueError(f"{self.foundation_backbone} survival training requires at least one observed event.")

            quantiles = self._parse_horizon_quantiles(self.params["horizon_quantiles"])
            event_times = time_train_np[event_train_np.astype(bool)]
            self.horizon_times_ = np.unique(np.quantile(event_times, quantiles).astype(np.float64))
            baseline_survival = _kaplan_meier_survival_at(time_train_np, event_train_np, self.horizon_times_)
            baseline_event_prob = 1.0 - baseline_survival

            self.models_ = []
            self.used_fallback_ = []
            constants: list[float] = []
            min_known = int(self.params["min_known_per_horizon"])
            for idx, horizon in enumerate(self.horizon_times_):
                known_mask, labels = self._horizon_known_labels(
                    time_train=time_train_np,
                    event_train=event_train_np,
                    horizon=float(horizon),
                )
                has_both_classes = np.unique(labels).size == 2
                if int(known_mask.sum()) < min_known or not has_both_classes:
                    self.models_.append(None)
                    self.used_fallback_.append(True)
                    constants.append(float(baseline_event_prob[idx]))
                    continue

                model = self._build_backbone()
                model.fit(X_train_np[known_mask], labels)
                self.models_.append(model)
                self.used_fallback_.append(False)
                constants.append(float(baseline_event_prob[idx]))

            self.constant_event_probabilities_ = _clean_horizon_event_probabilities(
                np.asarray(constants, dtype=np.float64)[None, :]
            )[0]
            return self
        except Exception as exc:
            raise rewrite_foundation_runtime_error(method_id, exc) from exc

    @staticmethod
    def _positive_class_probability(model: Any, X: np.ndarray, *, batch_size: int) -> np.ndarray:
        X_np = np.asarray(X, dtype=np.float32)
        resolved_batch_size = max(1, int(batch_size))
        if X_np.shape[0] > resolved_batch_size:
            chunks = [
                _DirectHorizonClassifierSurvivalMethod._positive_class_probability(
                    model,
                    X_np[start : start + resolved_batch_size],
                    batch_size=resolved_batch_size,
                )
                for start in range(0, X_np.shape[0], resolved_batch_size)
            ]
            return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)

        probabilities = np.asarray(model.predict_proba(X_np), dtype=np.float64)
        if probabilities.ndim == 1:
            return probabilities.reshape(-1)
        classes = np.asarray(getattr(model, "classes_", np.arange(probabilities.shape[1])))
        positive_positions = np.flatnonzero(classes.astype(str) == "1")
        if positive_positions.size:
            return probabilities[:, int(positive_positions[-1])]
        return probabilities[:, -1]

    def _horizon_event_probabilities(self, X: np.ndarray) -> np.ndarray:
        if self.horizon_times_ is None or self.constant_event_probabilities_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before prediction.")
        X_np = np.asarray(X, dtype=np.float32)
        columns: list[np.ndarray] = []
        batch_size = int(self.params.get("predict_batch_size", 32))
        for model, fallback_prob in zip(self.models_, self.constant_event_probabilities_):
            if model is None:
                columns.append(np.full(X_np.shape[0], float(fallback_prob), dtype=np.float64))
            else:
                columns.append(self._positive_class_probability(model, X_np, batch_size=batch_size))
        if not columns:
            return np.zeros((X_np.shape[0], 0), dtype=np.float64)
        return _clean_horizon_event_probabilities(np.column_stack(columns))

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        horizon_event_probs = self._horizon_event_probabilities(X)
        if str(self.params["aggregate_risk"]) == "last_event_probability":
            return horizon_event_probs[:, -1].astype(np.float64)
        return horizon_event_probs.mean(axis=1).astype(np.float64)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.horizon_times_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before prediction.")
        eval_times = np.asarray(times, dtype=np.float64).reshape(-1)
        horizon_event_probs = self._horizon_event_probabilities(X)
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


class TabICLHorizonSurvivalMethod(_DirectHorizonClassifierSurvivalMethod):
    method_id = "tabicl_survival"
    foundation_backbone = "TabICL"

    def __init__(
        self,
        n_estimators: int = 1,
        batch_size: int = 8,
        checkpoint_version: str = "tabicl-classifier-v1.1-0506.ckpt",
        device: str | None = None,
        use_amp: bool = False,
        allow_auto_download: bool = True,
        **params: Any,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            batch_size=batch_size,
            checkpoint_version=checkpoint_version,
            device=device,
            use_amp=use_amp,
            allow_auto_download=allow_auto_download,
            **params,
        )

    def foundation_metadata(self) -> dict[str, Any]:
        metadata = super().foundation_metadata()
        metadata["foundation_n_estimators"] = int(self.params["n_estimators"])
        return metadata

    def _build_backbone(self) -> Any:
        from tabicl import TabICLClassifier

        return TabICLClassifier(
            n_estimators=int(self.params["n_estimators"]),
            batch_size=int(self.params["batch_size"]),
            checkpoint_version=str(self.params["checkpoint_version"]),
            device=self.params.get("device"),
            use_amp=bool(self.params["use_amp"]),
            allow_auto_download=bool(self.params["allow_auto_download"]),
            random_state=self.params.get("seed"),
            verbose=False,
        )


class TabDPTHorizonSurvivalMethod(_DirectHorizonClassifierSurvivalMethod):
    method_id = "tabdpt_survival"
    foundation_backbone = "TabDPT"

    def __init__(
        self,
        inf_batch_size: int = 16,
        context_size: int = 512,
        temperature: float = 0.8,
        normalizer: str = "standard",
        feature_reduction: str = "pca",
        compile: bool = False,  # noqa: A002
        device: str | None = None,
        **params: Any,
    ) -> None:
        super().__init__(
            inf_batch_size=inf_batch_size,
            context_size=context_size,
            temperature=temperature,
            normalizer=normalizer,
            feature_reduction=feature_reduction,
            compile=compile,
            device=device,
            **params,
        )

    def foundation_metadata(self) -> dict[str, Any]:
        metadata = super().foundation_metadata()
        metadata["foundation_context_size"] = int(self.params["context_size"])
        return metadata

    def _build_backbone(self) -> Any:
        from tabdpt import TabDPTClassifier

        return _TabDPTPredictProbaAdapter(
            TabDPTClassifier(
                inf_batch_size=int(self.params["inf_batch_size"]),
                normalizer=str(self.params["normalizer"]),
                feature_reduction=str(self.params["feature_reduction"]),
                device=self.params.get("device"),
                compile=bool(self.params["compile"]),
                verbose=False,
            ),
            context_size=int(self.params["context_size"]),
            temperature=float(self.params["temperature"]),
            seed=self.params.get("seed"),
        )


class _TabDPTPredictProbaAdapter:
    def __init__(self, model: Any, *, context_size: int, temperature: float, seed: int | None) -> None:
        self.model = model
        self.context_size = context_size
        self.temperature = temperature
        self.seed = seed
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_TabDPTPredictProbaAdapter":
        self.model.fit(X, y)
        self.classes_ = np.asarray(getattr(self.model, "classes_", np.unique(y)))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(
            X,
            context_size=self.context_size,
            temperature=self.temperature,
            seed=self.seed,
        )
