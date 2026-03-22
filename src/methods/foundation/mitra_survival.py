from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd

from src.methods.base import BaseSurvivalMethod, to_structured_y


def _array_to_feature_frame(X: np.ndarray) -> pd.DataFrame:
    array = np.asarray(X, dtype=np.float32)
    columns = [f"f{i}" for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)


class MitraSurvivalMethod(BaseSurvivalMethod):
    def __init__(
        self,
        fine_tune: bool = False,
        presets: str = "medium",
        time_limit: int | None = None,
        path: str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            fine_tune=fine_tune,
            presets=presets,
            time_limit=time_limit,
            path=path,
            seed=seed,
        )
        self.backbone = None
        self.survival_head = None
        self.path_: str | None = None

    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "MitraSurvivalMethod":
        from autogluon.tabular import TabularPredictor
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        frame = _array_to_feature_frame(X_train)
        label = "__survarena_log_time__"
        train_frame = frame.copy()
        train_frame[label] = np.log1p(np.asarray(time_train, dtype=np.float32))

        self.path_ = self.params["path"] or mkdtemp(prefix="survarena_mitra_")
        self.backbone = TabularPredictor(
            label=label,
            problem_type="regression",
            eval_metric="rmse",
            path=str(Path(self.path_)),
        )

        fit_kwargs: dict[str, object] = {
            "train_data": train_frame,
            "hyperparameters": {"MITRA": {"fine_tune": bool(self.params["fine_tune"])}},
            "presets": str(self.params["presets"]),
            "verbosity": 0,
        }
        if self.params["time_limit"] is not None:
            fit_kwargs["time_limit"] = int(self.params["time_limit"])

        self.backbone.fit(**fit_kwargs)

        train_backbone_feature = self._backbone_features(frame)
        self.survival_head = CoxPHSurvivalAnalysis(alpha=0.0001)
        self.survival_head.fit(train_backbone_feature, to_structured_y(time_train, event_train))
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.backbone is None or self.survival_head is None:
            raise RuntimeError("MitraSurvivalMethod must be fit before prediction.")
        features = self._backbone_features(_array_to_feature_frame(X))
        return self.survival_head.predict(features)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.backbone is None or self.survival_head is None:
            raise RuntimeError("MitraSurvivalMethod must be fit before prediction.")
        features = self._backbone_features(_array_to_feature_frame(X))
        fns = self.survival_head.predict_survival_function(features)
        eval_times = np.asarray(times, dtype=np.float64)
        return np.vstack([fn(eval_times) for fn in fns])

    def _backbone_features(self, frame: pd.DataFrame) -> np.ndarray:
        if self.backbone is None:
            raise RuntimeError("Backbone must be fit before prediction.")
        predicted_log_time = np.asarray(self.backbone.predict(frame), dtype=np.float32).reshape(-1, 1)
        return predicted_log_time
