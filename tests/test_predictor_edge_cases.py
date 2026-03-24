from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from survarena.api.predictor import SurvivalPredictor
from survarena.automl.presets import PresetConfig
from survarena.evaluation.metrics import MetricBundle


def test_predictor_save_requires_artifact_dir_when_no_path_is_provided() -> None:
    predictor = SurvivalPredictor(label_time="time", label_event="event")

    with pytest.raises(RuntimeError, match="No artifact directory is available"):
        predictor.save()


def test_predictor_load_rejects_unsupported_serialization_versions(tmp_path: Path) -> None:
    predictor = SurvivalPredictor(label_time="time", label_event="event")
    output_path = tmp_path / "predictor.pkl"
    predictor.save(output_path)

    manifest_path = tmp_path / "predictor_manifest.json"
    manifest_path.write_text(
        json.dumps({"serialization_version": 99}, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Unsupported predictor serialization version 99"):
        SurvivalPredictor.load(output_path)


def test_predictor_fit_surfaces_when_all_candidate_models_fail(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 0],
            "age": [61.0, 57.0, 70.0, 66.0],
        }
    )

    monkeypatch.setattr(
        "survarena.api.predictor.resolve_preset",
        lambda *args, **kwargs: PresetConfig(name="test", method_ids=("mock_a", "mock_b"), n_trials=0, holdout_frac=0.25),
    )
    monkeypatch.setattr("survarena.api.predictor.read_yaml", lambda path: {"default_params": {}})
    monkeypatch.setattr(
        "survarena.api.predictor.prepare_validation_fold_cache",
        lambda **kwargs: [
            {
                "X_train": np.asarray([[0.0], [1.0]], dtype=float),
                "X_val": np.asarray([[0.5], [1.5]], dtype=float),
                "time_train": np.asarray([1.0, 2.0], dtype=float),
                "event_train": np.asarray([1, 0], dtype=int),
                "time_val": np.asarray([1.5, 2.5], dtype=float),
                "event_val": np.asarray([1, 0], dtype=int),
            }
        ],
    )

    def fake_tune_hyperparameters(*, method_id: str, **kwargs) -> dict[str, object]:
        raise RuntimeError(f"{method_id} exploded")

    monkeypatch.setattr("survarena.api.predictor.tune_hyperparameters", fake_tune_hyperparameters)

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )

    with pytest.raises(RuntimeError, match="All candidate models failed during fitting"):
        predictor.fit(frame, tuning_data=frame, dataset_name="toy")


def test_compute_metric_bundle_safe_clips_inputs_after_training_support_error(monkeypatch) -> None:
    predictor = SurvivalPredictor(label_time="time", label_event="event")
    calls: list[dict[str, np.ndarray | tuple[float, float, float]]] = []

    def fake_compute_survival_metrics(**kwargs) -> MetricBundle:
        calls.append(kwargs)
        if len(calls) == 1:
            raise ValueError("largest observed training event time point")
        return MetricBundle(uno_c=0.71, harrell_c=0.72, ibs=0.2, td_auc_25=0.73, td_auc_50=0.74, td_auc_75=0.75)

    monkeypatch.setattr("survarena.api.predictor.compute_survival_metrics", fake_compute_survival_metrics)

    metrics = predictor._compute_metric_bundle_safe(
        train_time=np.asarray([1.0, 3.0, 5.0]),
        train_event=np.asarray([1, 1, 0]),
        test_time=np.asarray([2.0, 4.0]),
        test_event=np.asarray([1, 0]),
        risk_scores=np.asarray([0.2, 0.4]),
        survival_probs=np.asarray([[0.9, 0.8, 0.7], [0.95, 0.85, 0.75]]),
        survival_times=np.asarray([1.0, 2.0, 4.0]),
    )

    assert metrics["uno_c"] == 0.71
    assert len(calls) == 2
    np.testing.assert_allclose(calls[1]["test_time"], np.asarray([2.0]))
    np.testing.assert_allclose(calls[1]["risk_scores"], np.asarray([0.2]))
    np.testing.assert_allclose(calls[1]["survival_times"], np.asarray([1.0, 2.0]))
    assert calls[1]["survival_probs"].shape == (1, 2)
    assert max(calls[1]["horizons"]) < 3.0


def test_compute_metric_bundle_safe_falls_back_to_harrell_only_when_no_rows_are_supported(monkeypatch) -> None:
    predictor = SurvivalPredictor(label_time="time", label_event="event")

    monkeypatch.setattr(
        "survarena.api.predictor.compute_survival_metrics",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("largest observed training event time point")),
    )
    monkeypatch.setattr("survarena.api.predictor.compute_harrell_c_index", lambda **kwargs: 0.61)

    metrics = predictor._compute_metric_bundle_safe(
        train_time=np.asarray([1.0, 2.0]),
        train_event=np.asarray([1, 1]),
        test_time=np.asarray([3.0, 4.0]),
        test_event=np.asarray([1, 0]),
        risk_scores=np.asarray([0.2, 0.4]),
        survival_probs=np.asarray([[0.9, 0.8], [0.95, 0.85]]),
        survival_times=np.asarray([3.0, 4.0]),
    )

    assert metrics["harrell_c"] == 0.61
    assert math.isnan(metrics["uno_c"])
    assert math.isnan(metrics["ibs"])
    assert math.isnan(metrics["td_auc_25"])
