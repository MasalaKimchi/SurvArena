from __future__ import annotations

from pathlib import Path

import importlib
import numpy as np
import pandas as pd

from src.api.predictor import SurvivalPredictor
from src.automl.presets import PresetConfig


class MockSurvivalMethod:
    def __init__(self, bias: float = 0.0, seed: int | None = None) -> None:
        self.bias = float(bias)
        self.seed = seed

    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "MockSurvivalMethod":
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return X.sum(axis=1).astype(float) + self.bias

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        risk = self.predict_risk(X)
        return np.exp(-np.outer(np.maximum(risk, 0.1), np.asarray(times, dtype=float)))


def test_predictor_tracks_multiple_fitted_models_and_roundtrips(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 1],
            "age": [61.0, 57.0, 70.0, 66.0],
            "stage": ["i", "ii", "ii", "iii"],
        }
    )

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a", "mock_b"), n_trials=0, inner_folds=2)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_prepare_inner_cv_cache(**kwargs) -> list[dict[str, np.ndarray]]:
        X_train = np.asarray([[0.0], [1.0]], dtype=float)
        X_val = np.asarray([[0.5], [1.5]], dtype=float)
        time_train = np.asarray([1.0, 2.0], dtype=float)
        event_train = np.asarray([1, 1], dtype=int)
        time_val = np.asarray([1.5, 2.5], dtype=float)
        event_val = np.asarray([1, 0], dtype=int)
        return [
            {
                "X_train": X_train,
                "X_val": X_val,
                "time_train": time_train,
                "event_train": event_train,
                "time_val": time_val,
                "event_val": event_val,
            }
        ]

    def fake_tune_hyperparameters(*, method_id: str, **kwargs) -> dict[str, object]:
        bias = 1.0 if method_id == "mock_a" else 2.0
        return {"best_params": {"bias": bias}, "best_score": bias, "n_trials_completed": 0}

    def fake_metric_bundle(self: SurvivalPredictor, **kwargs) -> dict[str, float]:
        score = float(np.mean(kwargs["risk_scores"]))
        return {
            "uno_c": score,
            "harrell_c": score,
            "ibs": 1.0 - min(score / 10.0, 0.9),
            "td_auc_25": score,
            "td_auc_50": score,
            "td_auc_75": score,
        }

    monkeypatch.setattr("src.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("src.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("src.api.predictor._prepare_inner_cv_cache", fake_prepare_inner_cv_cache)
    monkeypatch.setattr("src.api.predictor.tune_hyperparameters", fake_tune_hyperparameters)
    monkeypatch.setattr("src.api.predictor._method_registry", lambda: {"mock_a": MockSurvivalMethod, "mock_b": MockSurvivalMethod})
    monkeypatch.setattr(SurvivalPredictor, "_compute_metric_bundle_safe", fake_metric_bundle)

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(frame, test_data=frame, dataset_name="toy")

    assert predictor.best_method_id_ == "mock_b"
    assert predictor.model_names() == ["mock_a", "mock_b"]

    best_risk = predictor.predict_risk(frame)
    alt_risk = predictor.predict_risk(frame, model="mock_a")
    assert not np.allclose(best_risk, alt_risk)

    summary = predictor.fit_summary()
    assert summary["trained_models"] == ["mock_a", "mock_b"]
    assert set(summary["per_model_test_metrics"]) == {"mock_a", "mock_b"}

    saved_path = tmp_path / "toy" / "predictor.pkl"
    assert saved_path.exists()

    loaded = SurvivalPredictor.load(saved_path)
    np.testing.assert_allclose(loaded.predict_risk(frame, model="mock_b"), best_risk)


def test_predictor_reuses_metric_rows_from_tuning(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 1],
            "age": [61.0, 57.0, 70.0, 66.0],
            "stage": ["i", "ii", "ii", "iii"],
        }
    )

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a",), n_trials=1, inner_folds=2)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_prepare_inner_cv_cache(**kwargs) -> list[dict[str, np.ndarray]]:
        return [
            {
                "X_train": np.asarray([[0.0], [1.0]], dtype=float),
                "X_val": np.asarray([[0.5], [1.5]], dtype=float),
                "time_train": np.asarray([1.0, 2.0], dtype=float),
                "event_train": np.asarray([1, 1], dtype=int),
                "time_val": np.asarray([1.5, 2.5], dtype=float),
                "event_val": np.asarray([1, 0], dtype=int),
            }
        ]

    def fake_tune_hyperparameters(**kwargs) -> dict[str, object]:
        return {
            "best_params": {"bias": 1.0},
            "best_score": 0.8,
            "n_trials_completed": 1,
            "best_metric_rows": [
                {
                    "uno_c": 0.7,
                    "harrell_c": 0.8,
                    "ibs": 0.2,
                    "td_auc_25": 0.75,
                    "td_auc_50": 0.76,
                    "td_auc_75": 0.77,
                }
            ],
        }

    monkeypatch.setattr("src.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("src.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("src.api.predictor._prepare_inner_cv_cache", fake_prepare_inner_cv_cache)
    monkeypatch.setattr("src.api.predictor.tune_hyperparameters", fake_tune_hyperparameters)
    monkeypatch.setattr("src.api.predictor._method_registry", lambda: {"mock_a": MockSurvivalMethod})
    monkeypatch.setattr(
        SurvivalPredictor,
        "_cross_validated_metric_summary",
        lambda self, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected extra CV refit")),
    )

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(frame, dataset_name="toy")

    leaderboard = predictor.leaderboard()
    assert float(leaderboard.loc[0, "validation_harrell_c"]) == 0.8
    assert float(leaderboard.loc[0, "validation_td_auc_75"]) == 0.77


def test_public_package_exports_survival_predictor() -> None:
    survarena = importlib.import_module("survarena")

    assert survarena.SurvivalPredictor is SurvivalPredictor
