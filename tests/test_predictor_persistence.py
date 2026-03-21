from __future__ import annotations

from pathlib import Path

from src.api.predictor import SurvivalPredictor


def test_predictor_save_writes_pickle_and_manifest(tmp_path: Path) -> None:
    predictor = SurvivalPredictor(label_time="time", label_event="event")
    output_path = tmp_path / "predictor.pkl"

    saved_path = predictor.save(output_path)

    manifest_path = tmp_path / "predictor_manifest.json"
    assert saved_path == output_path
    assert output_path.exists()
    assert manifest_path.exists()


def test_predictor_load_round_trips_unfitted_predictor(tmp_path: Path) -> None:
    predictor = SurvivalPredictor(
        label_time="duration",
        label_event="observed",
        eval_metric="uno_c",
        enable_foundation_models=True,
    )
    output_path = tmp_path / "predictor.pkl"
    predictor.save(output_path)

    loaded = SurvivalPredictor.load(output_path)

    assert loaded.label_time == "duration"
    assert loaded.label_event == "observed"
    assert loaded.eval_metric == "uno_c"
    assert loaded.enable_foundation_models is True
