from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from survarena.data.preprocess import TabularPreprocessor
from survarena.methods.base import SurvivalPredictions
from survarena.methods.preprocessing import finalize_preprocessed_features


@dataclass(slots=True)
class BaggedModelMember:
    method_id: str
    model: Any
    preprocessor: TabularPreprocessor


class BaggedSurvivalEnsemble:
    def __init__(self, members: list[BaggedModelMember]) -> None:
        if not members:
            raise ValueError("BaggedSurvivalEnsemble requires at least one fitted member.")
        self.members = list(members)

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        member_predictions = [
            np.asarray(
                member.model.predict_risk(
                    finalize_preprocessed_features(member.method_id, member.preprocessor.transform(X))
                ),
                dtype=float,
            )
            for member in self.members
        ]
        return np.mean(np.stack(member_predictions, axis=0), axis=0)

    def predict_survival(self, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        evaluation_times = np.asarray(times, dtype=float)
        member_predictions = [
            np.asarray(
                member.model.predict_survival(
                    finalize_preprocessed_features(member.method_id, member.preprocessor.transform(X)),
                    evaluation_times,
                ),
                dtype=float,
            )
            for member in self.members
        ]
        return np.mean(np.stack(member_predictions, axis=0), axis=0)

    def predict_bundle(self, X: pd.DataFrame, times: np.ndarray) -> SurvivalPredictions:
        evaluation_times = np.asarray(times, dtype=float)
        risk_predictions: list[np.ndarray] = []
        survival_predictions: list[np.ndarray] = []
        for member in self.members:
            transformed = finalize_preprocessed_features(member.method_id, member.preprocessor.transform(X))
            predictions = member.model.predict_bundle(transformed, evaluation_times)
            risk_predictions.append(np.asarray(predictions.risk, dtype=float))
            survival_predictions.append(np.asarray(predictions.survival, dtype=float))
        return SurvivalPredictions(
            risk=np.mean(np.stack(risk_predictions, axis=0), axis=0),
            survival=np.mean(np.stack(survival_predictions, axis=0), axis=0),
        )

    def __len__(self) -> int:
        return len(self.members)
