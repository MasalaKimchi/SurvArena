from __future__ import annotations

from pathlib import Path

import importlib
import numpy as np
import pandas as pd
import pytest

from survarena.api.predictor import SurvivalPredictor
from survarena.automl.presets import PresetConfig


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


class MockFrameAwareMethod:
    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.fit_input_type: type | None = None
        self.fit_columns: list[str] | None = None
        self.risk_input_type: type | None = None
        self.risk_columns: list[str] | None = None
        self.survival_input_type: type | None = None
        self.survival_columns: list[str] | None = None

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "MockFrameAwareMethod":
        self.fit_input_type = type(X_train)
        self.fit_columns = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None
        return self

    def predict_risk(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        self.risk_input_type = type(X)
        self.risk_columns = list(X.columns) if isinstance(X, pd.DataFrame) else None
        return np.linspace(0.2, 1.0, num=len(X), dtype=float)

    def predict_survival(self, X: pd.DataFrame | np.ndarray, times: np.ndarray) -> np.ndarray:
        self.survival_input_type = type(X)
        self.survival_columns = list(X.columns) if isinstance(X, pd.DataFrame) else None
        risk = np.maximum(self.predict_risk(X), 0.1)
        return np.exp(-np.outer(risk, np.asarray(times, dtype=float)))


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
        return PresetConfig(name="test", method_ids=("mock_a", "mock_b"), holdout_frac=0.25)

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

    def fake_select_hyperparameters(*, method_id: str, **kwargs) -> dict[str, object]:
        bias = 1.0 if method_id == "mock_a" else 2.0
        return {
            "best_params": {"bias": bias},
            "best_score": bias,
            "best_metric_rows": [
                {
                    "uno_c": bias,
                    "harrell_c": bias,
                    "ibs": 0.2,
                    "td_auc_25": bias,
                    "td_auc_50": bias,
                    "td_auc_75": bias,
                }
            ],
        }

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

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.prepare_validation_fold_cache", fake_prepare_inner_cv_cache)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr(
        "survarena.api.predictor._get_method_class",
        lambda method_id: {"mock_a": MockSurvivalMethod, "mock_b": MockSurvivalMethod}[method_id],
    )
    monkeypatch.setattr(SurvivalPredictor, "_compute_metric_bundle_safe", fake_metric_bundle)

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        retain_top_k_models=2,
        save_path=tmp_path,
    )
    predictor.fit(frame, tuning_data=frame, test_data=frame, dataset_name="toy")

    assert predictor.best_method_id_ == "mock_b"
    assert set(predictor.model_names()) == {"mock_a", "mock_b"}

    best_risk = predictor.predict_risk(frame)
    alt_risk = predictor.predict_risk(frame, model="mock_a")
    assert not np.allclose(best_risk, alt_risk)

    summary = predictor.fit_summary()
    assert set(summary["trained_models"]) == {"mock_a", "mock_b"}
    assert summary["validation_strategy"] == "tuning_data"
    assert set(summary["per_model_test_metrics"]) == {"mock_a", "mock_b"}

    saved_path = tmp_path / "toy" / "predictor.pkl"
    assert saved_path.exists()

    loaded = SurvivalPredictor.load(saved_path)
    np.testing.assert_allclose(loaded.predict_risk(frame, model="mock_b"), best_risk)


def test_predictor_retains_only_the_best_model_by_default(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 1],
            "age": [61.0, 57.0, 70.0, 66.0],
            "stage": ["i", "ii", "ii", "iii"],
        }
    )

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a", "mock_b"), holdout_frac=0.25)

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

    def fake_select_hyperparameters(*, method_id: str, **kwargs) -> dict[str, object]:
        bias = 1.0 if method_id == "mock_a" else 2.0
        return {
            "best_params": {"bias": bias},
            "best_score": bias,
            "best_metric_rows": [
                {
                    "uno_c": bias,
                    "harrell_c": bias,
                    "ibs": 0.2,
                    "td_auc_25": bias,
                    "td_auc_50": bias,
                    "td_auc_75": bias,
                }
            ],
        }

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.prepare_validation_fold_cache", fake_prepare_inner_cv_cache)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr(
        "survarena.api.predictor._get_method_class",
        lambda method_id: {"mock_a": MockSurvivalMethod, "mock_b": MockSurvivalMethod}[method_id],
    )

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(frame, tuning_data=frame, dataset_name="toy")

    assert predictor.best_method_id_ == "mock_b"
    assert predictor.model_names() == ["mock_b"]

    leaderboard = predictor.leaderboard().set_index("method_id")
    assert bool(leaderboard.loc["mock_b", "retained_for_inference"]) is True
    assert bool(leaderboard.loc["mock_a", "retained_for_inference"]) is False


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
        return PresetConfig(name="test", method_ids=("mock_a",), holdout_frac=0.25)

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

    def fake_select_hyperparameters(**kwargs) -> dict[str, object]:
        return {
            "best_params": {"bias": 1.0},
            "best_score": 0.8,
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

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.prepare_validation_fold_cache", fake_prepare_inner_cv_cache)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr("survarena.api.predictor._get_method_class", lambda method_id: {"mock_a": MockSurvivalMethod}[method_id])
    monkeypatch.setattr(
        SurvivalPredictor,
        "_fold_cache_metric_summary",
        lambda self, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected extra CV refit")),
    )

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(frame, tuning_data=frame, dataset_name="toy")

    leaderboard = predictor.leaderboard()
    assert float(leaderboard.loc[0, "validation_harrell_c"]) == 0.8
    assert float(leaderboard.loc[0, "validation_td_auc_75"]) == 0.77


def test_predictor_uses_automatic_holdout_when_tuning_data_is_absent(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "event": [1, 0, 1, 0, 1, 0],
            "age": [61.0, 57.0, 70.0, 66.0, 59.0, 63.0],
            "stage": ["i", "ii", "ii", "iii", "i", "iii"],
        }
    )
    fold_sizes: list[tuple[int, int]] = []

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a",), holdout_frac=0.5)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_select_hyperparameters(*, fold_cache: list[dict[str, np.ndarray]], **kwargs) -> dict[str, object]:
        fold_sizes.append((int(fold_cache[0]["X_train"].shape[0]), int(fold_cache[0]["X_val"].shape[0])))
        return {
            "best_params": {"bias": 1.0},
            "best_score": 0.8,
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

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr("survarena.api.predictor._get_method_class", lambda method_id: {"mock_a": MockSurvivalMethod}[method_id])

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(frame, dataset_name="toy", holdout_frac=0.5)

    summary = predictor.fit_summary()
    assert summary["validation_strategy"] == "auto_holdout"
    assert summary["validation_holdout_frac"] == 0.5
    assert summary["selection_train_rows"] == 3
    assert summary["validation_rows"] == 3
    assert fold_sizes == [(3, 3)]


def test_predictor_uses_bagged_oof_selection_when_num_bag_folds_enabled(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "event": [1, 0, 1, 0, 1, 0],
            "age": [61.0, 57.0, 70.0, 66.0, 59.0, 63.0],
            "stage": ["i", "ii", "ii", "iii", "i", "iii"],
        }
    )
    fold_shapes: list[list[tuple[int, int]]] = []

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a",), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_select_hyperparameters(*, fold_cache: list[dict[str, np.ndarray]], **kwargs) -> dict[str, object]:
        fold_shapes.append([(int(fold["X_train"].shape[0]), int(fold["X_val"].shape[0])) for fold in fold_cache])
        return {
            "best_params": {"bias": 1.0},
            "best_score": 0.8,
            "best_metric_rows": [
                {
                    "uno_c": 0.7,
                    "harrell_c": 0.8,
                    "ibs": 0.2,
                    "td_auc_25": 0.75,
                    "td_auc_50": 0.76,
                    "td_auc_75": 0.77,
                }
                for _ in fold_cache
            ],
        }

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr("survarena.api.predictor._get_method_class", lambda method_id: {"mock_a": MockSurvivalMethod}[method_id])

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(frame, dataset_name="toy", num_bag_folds=3, num_bag_sets=2)

    summary = predictor.fit_summary()
    assert summary["validation_strategy"] == "bagged_oof"
    assert summary["num_bag_folds"] == 3
    assert summary["num_bag_sets"] == 2
    assert summary["selection_train_rows"] == 4
    assert summary["validation_rows"] == 12
    assert len(fold_shapes) == 1
    assert fold_shapes[0] == [(4, 2)] * 6


def test_predictor_bagged_models_average_fold_members_for_inference(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "event": [1, 0, 1, 0, 1, 0],
            "age": [61.0, 57.0, 70.0, 66.0, 59.0, 63.0],
            "stage": ["i", "ii", "ii", "iii", "i", "iii"],
        }
    )
    fitted_member_ids: list[int] = []

    class AveragingMockMethod(MockSurvivalMethod):
        def fit(
            self,
            X_train: np.ndarray,
            time_train: np.ndarray,
            event_train: np.ndarray,
            X_val: np.ndarray | None = None,
            time_val: np.ndarray | None = None,
            event_val: np.ndarray | None = None,
        ) -> "AveragingMockMethod":
            self.member_id = len(fitted_member_ids) + 1
            fitted_member_ids.append(self.member_id)
            return self

        def predict_risk(self, X: np.ndarray) -> np.ndarray:
            return np.full(X.shape[0], float(self.member_id))

        def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
            return np.full((X.shape[0], len(times)), float(self.member_id))

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a",), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_select_hyperparameters(**kwargs) -> dict[str, object]:
        return {
            "best_params": {},
            "best_score": 0.8,
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

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr("survarena.api.predictor._get_method_class", lambda method_id: {"mock_a": AveragingMockMethod}[method_id])
    monkeypatch.setattr(SurvivalPredictor, "_persist_artifacts", lambda self, dataset_name, results: None)

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(frame, dataset_name="toy", num_bag_folds=3)

    summary = predictor.fit_summary()
    risk = predictor.predict_risk(frame)
    assert summary["validation_strategy"] == "bagged_oof"
    assert summary["trained_models"] == ["mock_a"]
    assert fitted_member_ids == [1, 2, 3]
    np.testing.assert_allclose(risk, np.full(len(frame), 2.0))


def test_predictor_bagged_model_round_trips(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "event": [1, 0, 1, 0, 1, 0],
            "age": [61.0, 57.0, 70.0, 66.0, 59.0, 63.0],
            "stage": ["i", "ii", "ii", "iii", "i", "iii"],
        }
    )

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a",), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {"bias": 1.0}}

    def fake_select_hyperparameters(**kwargs) -> dict[str, object]:
        return {
            "best_params": {"bias": 1.0},
            "best_score": 0.8,
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

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr("survarena.api.predictor._get_method_class", lambda method_id: {"mock_a": MockSurvivalMethod}[method_id])

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(frame, dataset_name="toy", num_bag_folds=2)

    saved_path = tmp_path / "toy" / "predictor.pkl"
    loaded = SurvivalPredictor.load(saved_path)

    np.testing.assert_allclose(loaded.predict_risk(frame), predictor.predict_risk(frame))


def test_predictor_num_bag_sets_requires_bagged_folds(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 0],
            "age": [61.0, 57.0, 70.0, 66.0],
            "stage": ["i", "ii", "ii", "iii"],
        }
    )

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a",), holdout_frac=0.25)

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )

    with pytest.raises(ValueError, match="num_bag_sets > 1 requires num_bag_folds >= 2"):
        predictor.fit(frame, dataset_name="toy", num_bag_sets=2)


def test_predictor_refit_full_uses_tuning_data_for_final_training(tmp_path: Path, monkeypatch) -> None:
    train_frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 1],
            "age": [61.0, 57.0, 70.0, 66.0],
            "stage": ["i", "ii", "ii", "iii"],
        }
    )
    tuning_frame = pd.DataFrame(
        {
            "time": [5.0, 6.0],
            "event": [0, 1],
            "age": [59.0, 63.0],
            "stage": ["i", "iii"],
        }
    )
    fit_row_counts: list[int] = []

    class RecordingMockMethod(MockSurvivalMethod):
        def fit(
            self,
            X_train: np.ndarray,
            time_train: np.ndarray,
            event_train: np.ndarray,
            X_val: np.ndarray | None = None,
            time_val: np.ndarray | None = None,
            event_val: np.ndarray | None = None,
        ) -> "RecordingMockMethod":
            fit_row_counts.append(int(X_train.shape[0]))
            return self

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a",), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_prepare_validation_fold_cache(**kwargs) -> list[dict[str, np.ndarray]]:
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

    def fake_select_hyperparameters(**kwargs) -> dict[str, object]:
        return {
            "best_params": {"bias": 1.0},
            "best_score": 0.8,
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

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.prepare_validation_fold_cache", fake_prepare_validation_fold_cache)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr(
        "survarena.api.predictor._get_method_class",
        lambda method_id: {"mock_a": RecordingMockMethod}[method_id],
    )
    monkeypatch.setattr(SurvivalPredictor, "_persist_artifacts", lambda self, dataset_name, results: None)

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(train_frame, tuning_data=tuning_frame, dataset_name="toy", refit_full=True)

    summary = predictor.fit_summary()
    assert fit_row_counts == [6]
    assert summary["refit_full"] is True
    assert summary["final_train_rows"] == 6


def test_predictor_refit_full_false_keeps_explicit_tuning_rows_out_of_final_training(tmp_path: Path, monkeypatch) -> None:
    train_frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 1],
            "age": [61.0, 57.0, 70.0, 66.0],
            "stage": ["i", "ii", "ii", "iii"],
        }
    )
    tuning_frame = pd.DataFrame(
        {
            "time": [5.0, 6.0],
            "event": [0, 1],
            "age": [59.0, 63.0],
            "stage": ["i", "iii"],
        }
    )
    fit_row_counts: list[int] = []

    class RecordingMockMethod(MockSurvivalMethod):
        def fit(
            self,
            X_train: np.ndarray,
            time_train: np.ndarray,
            event_train: np.ndarray,
            X_val: np.ndarray | None = None,
            time_val: np.ndarray | None = None,
            event_val: np.ndarray | None = None,
        ) -> "RecordingMockMethod":
            fit_row_counts.append(int(X_train.shape[0]))
            return self

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a",), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_prepare_validation_fold_cache(**kwargs) -> list[dict[str, np.ndarray]]:
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

    def fake_select_hyperparameters(**kwargs) -> dict[str, object]:
        return {
            "best_params": {"bias": 1.0},
            "best_score": 0.8,
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

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.prepare_validation_fold_cache", fake_prepare_validation_fold_cache)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr(
        "survarena.api.predictor._get_method_class",
        lambda method_id: {"mock_a": RecordingMockMethod}[method_id],
    )
    monkeypatch.setattr(SurvivalPredictor, "_persist_artifacts", lambda self, dataset_name, results: None)

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(train_frame, tuning_data=tuning_frame, dataset_name="toy", refit_full=False)

    summary = predictor.fit_summary()
    assert fit_row_counts == [4]
    assert summary["refit_full"] is False
    assert summary["final_train_rows"] == 4


def test_predictor_fit_level_autogluon_kwargs_are_normalized(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 1],
            "age": [61.0, 57.0, 70.0, 66.0],
            "stage": ["i", "ii", "ii", "iii"],
        }
    )
    recorded_timeout: list[float | None] = []

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a",), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_prepare_validation_fold_cache(**kwargs) -> list[dict[str, np.ndarray]]:
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

    def fake_select_hyperparameters(**kwargs) -> dict[str, object]:
        recorded_timeout.append(kwargs.get("method_cfg", {}).get("default_params", {}).get("time_limit"))
        return {
            "best_params": {"bias": 1.0},
            "best_score": 0.8,
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

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.prepare_validation_fold_cache", fake_prepare_validation_fold_cache)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr("survarena.api.predictor._get_method_class", lambda method_id: {"mock_a": MockSurvivalMethod}[method_id])

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(
        frame,
        tuning_data=frame,
        dataset_name="toy",
        hyperparameter_tune_kwargs={"num_trials": 5, "timeout": 12.0},
    )

    summary = predictor.fit_summary()
    assert recorded_timeout == [None]
    assert summary["hyperparameter_tune_kwargs"] == {"num_trials": 5, "timeout_seconds": 12.0}


def test_predictor_time_limit_skips_candidates_when_budget_is_exhausted(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 1],
            "age": [61.0, 57.0, 70.0, 66.0],
            "stage": ["i", "ii", "ii", "iii"],
        }
    )
    assigned_budgets = iter([1.5, 0.0])
    selection_calls: list[str] = []

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a", "mock_b"), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_prepare_validation_fold_cache(**kwargs) -> list[dict[str, np.ndarray]]:
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

    def fake_select_hyperparameters(*, method_id: str, **kwargs) -> dict[str, object]:
        selection_calls.append(method_id)
        return {
            "best_params": {"bias": 1.0},
            "best_score": 0.8,
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

    def fake_next_method_time_limit(self: SurvivalPredictor, **kwargs) -> float | None:
        return next(assigned_budgets)

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.prepare_validation_fold_cache", fake_prepare_validation_fold_cache)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr(
        "survarena.api.predictor._get_method_class",
        lambda method_id: {"mock_a": MockSurvivalMethod, "mock_b": MockSurvivalMethod}[method_id],
    )
    monkeypatch.setattr(SurvivalPredictor, "_next_method_time_limit", fake_next_method_time_limit)

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(frame, tuning_data=frame, dataset_name="toy", time_limit=3.0)

    summary = predictor.fit_summary()
    assert summary["time_limit_sec"] == 3.0
    assert summary["selection_time_budget_sec"] == pytest.approx(2.4)
    assert summary["trained_models"] == ["mock_a"]
    assert selection_calls == ["mock_a"]

    leaderboard = predictor.leaderboard().set_index("method_id")
    assert bool(leaderboard.loc["mock_a", "retained_for_inference"]) is True
    assert leaderboard.loc["mock_b", "status"] == "skipped"
    assert float(leaderboard.loc["mock_b", "time_limit_sec"]) == 0.0


def test_predictor_time_limit_prioritizes_refitting_the_best_model(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 1],
            "age": [61.0, 57.0, 70.0, 66.0],
            "stage": ["i", "ii", "ii", "iii"],
        }
    )
    assigned_budgets = iter([1.0, 1.0])

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a", "mock_b"), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_prepare_validation_fold_cache(**kwargs) -> list[dict[str, np.ndarray]]:
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

    def fake_select_hyperparameters(*, method_id: str, **kwargs) -> dict[str, object]:
        bias = 1.0 if method_id == "mock_a" else 2.0
        return {
            "best_params": {"bias": bias},
            "best_score": bias,
            "best_metric_rows": [
                {
                    "uno_c": bias,
                    "harrell_c": bias,
                    "ibs": 0.2,
                    "td_auc_25": bias,
                    "td_auc_50": bias,
                    "td_auc_75": bias,
                }
            ],
        }

    def fake_next_method_time_limit(self: SurvivalPredictor, **kwargs) -> float | None:
        return next(assigned_budgets)

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.prepare_validation_fold_cache", fake_prepare_validation_fold_cache)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr(
        "survarena.api.predictor._get_method_class",
        lambda method_id: {"mock_a": MockSurvivalMethod, "mock_b": MockSurvivalMethod}[method_id],
    )
    monkeypatch.setattr(SurvivalPredictor, "_next_method_time_limit", fake_next_method_time_limit)
    monkeypatch.setattr(SurvivalPredictor, "_remaining_fit_time", lambda self, fit_started_at, time_limit: 0.0)

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(frame, tuning_data=frame, dataset_name="toy", time_limit=3.0)

    assert predictor.best_method_id_ == "mock_b"
    assert predictor.model_names() == ["mock_b"]

    leaderboard = predictor.leaderboard().set_index("method_id")
    assert bool(leaderboard.loc["mock_b", "retained_for_inference"]) is True
    assert bool(leaderboard.loc["mock_a", "retained_for_inference"]) is False


@pytest.mark.parametrize("method_id", ["catboost_cox", "catboost_survival_aft"])
def test_predictor_preserves_native_categorical_frames_for_catboost_method(
    tmp_path: Path,
    monkeypatch,
    method_id: str,
) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 0],
            "age": [61.0, 57.0, 70.0, 66.0],
            "stage": ["i", "ii", "ii", "iii"],
        }
    )

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=(method_id,), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_prepare_validation_fold_cache(**kwargs) -> list[dict[str, object]]:
        return [
            {
                "X_train": pd.DataFrame({"age": [61.0, 57.0], "stage": ["i", "ii"]}),
                "X_val": pd.DataFrame({"age": [70.0, 66.0], "stage": ["ii", "iii"]}),
                "time_train": np.asarray([1.0, 2.0], dtype=float),
                "event_train": np.asarray([1, 0], dtype=int),
                "time_val": np.asarray([3.0, 4.0], dtype=float),
                "event_val": np.asarray([1, 0], dtype=int),
            }
        ]

    def fake_select_hyperparameters(**kwargs) -> dict[str, object]:
        return {
            "best_params": {},
            "best_score": 0.8,
            "best_metric_rows": [
                {
                    "uno_c": 0.8,
                    "harrell_c": 0.8,
                    "ibs": 0.2,
                    "td_auc_25": 0.8,
                    "td_auc_50": 0.8,
                    "td_auc_75": 0.8,
                }
            ],
        }

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.prepare_validation_fold_cache", fake_prepare_validation_fold_cache)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr("survarena.api.predictor._get_method_class", lambda method_id: MockFrameAwareMethod)

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="fast",
        save_path=tmp_path,
    )
    predictor.fit(frame, tuning_data=frame, dataset_name="toy")

    fitted = predictor.fitted_models_[method_id]
    assert fitted.fit_input_type is pd.DataFrame
    assert fitted.fit_columns == ["age", "stage"]

    predictor.predict_risk(frame)
    predictor.predict_survival(frame)

    assert fitted.risk_input_type is pd.DataFrame
    assert fitted.risk_columns == ["age", "stage"]
    assert fitted.survival_input_type is pd.DataFrame
    assert fitted.survival_columns == ["age", "stage"]


@pytest.mark.parametrize("method_id", ["catboost_cox", "catboost_survival_aft"])
def test_bagged_predictor_preserves_native_categorical_frames_for_catboost_method(
    tmp_path: Path,
    monkeypatch,
    method_id: str,
) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "event": [1, 0, 1, 0, 1, 0],
            "age": [61.0, 57.0, 70.0, 66.0, 59.0, 63.0],
            "stage": ["i", "ii", "ii", "iii", "i", "iii"],
        }
    )

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=(method_id,), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_select_hyperparameters(**kwargs) -> dict[str, object]:
        return {
            "best_params": {},
            "best_score": 0.8,
            "best_metric_rows": [
                {
                    "uno_c": 0.8,
                    "harrell_c": 0.8,
                    "ibs": 0.2,
                    "td_auc_25": 0.8,
                    "td_auc_50": 0.8,
                    "td_auc_75": 0.8,
                }
            ],
        }

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr("survarena.api.predictor._get_method_class", lambda method_id: MockFrameAwareMethod)

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="fast",
        save_path=tmp_path,
    )
    predictor.fit(frame, dataset_name="toy", num_bag_folds=2)

    fitted = predictor.fitted_models_[method_id]
    members = fitted.members
    assert len(members) == 2
    assert all(member.model.fit_input_type is pd.DataFrame for member in members)
    assert all(member.model.fit_columns == ["age", "stage"] for member in members)

    predictor.predict_risk(frame)
    predictor.predict_survival(frame)

    assert all(member.model.risk_input_type is pd.DataFrame for member in members)
    assert all(member.model.risk_columns == ["age", "stage"] for member in members)
    assert all(member.model.survival_input_type is pd.DataFrame for member in members)
    assert all(member.model.survival_columns == ["age", "stage"] for member in members)


def test_public_package_exports_survival_predictor() -> None:
    survarena = importlib.import_module("survarena")

    assert survarena.SurvivalPredictor is SurvivalPredictor
    assert callable(survarena.compare_survival_models)
