from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest
from survarena.config import read_yaml
from survarena.data.io import read_tabular_data
import numpy as np
from survarena.data.loaders import load_dataset
from survarena.data.preprocess import TabularPreprocessor
from survarena.data.profiling import infer_feature_metadata
from survarena.data.robustness import apply_label_noise, apply_robustness_track, resolve_robustness_tracks
from survarena.data.splitters import SplitDefinition
from survarena.data.user_dataset import load_user_dataset
from survarena.automl.validation import (
    ValidationPlan,
    build_refit_dataset,
    build_validation_plan,
    default_holdout_frac,
    prepare_validation_fold_cache,
)
from survarena.data.schema import DatasetMetadata, SurvivalDataset
from survarena.methods.preprocessing import method_uses_native_categorical_features, method_uses_scaled_numeric_features


# --- test_io_config.py ---


def test_read_tabular_data_returns_dataframe_copy() -> None:
    frame = pd.DataFrame({"age": [61, 57], "stage": ["i", "ii"]})

    loaded = read_tabular_data(frame)

    assert loaded.equals(frame)
    assert loaded is not frame


def test_read_tabular_data_supports_csv_and_parquet(tmp_path: Path) -> None:
    frame = pd.DataFrame({"age": [61, 57], "stage": ["i", "ii"]})
    csv_path = tmp_path / "toy.csv"
    parquet_path = tmp_path / "toy.parquet"
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)

    csv_loaded = read_tabular_data(csv_path)
    parquet_loaded = read_tabular_data(parquet_path)

    pd.testing.assert_frame_equal(csv_loaded, frame)
    pd.testing.assert_frame_equal(parquet_loaded, frame)


def test_read_tabular_data_rejects_unknown_file_extensions(tmp_path: Path) -> None:
    path = tmp_path / "toy.tsv"
    path.write_text("age\tstage\n61\ti\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        read_tabular_data(path)


def test_read_yaml_loads_mapping_payload(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("dataset_id: toy\nnum_trials: 4\n", encoding="utf-8")

    loaded = read_yaml(path)

    assert loaded == {"dataset_id": "toy", "num_trials": 4}


# --- test_dataset_loaders.py ---


@pytest.mark.parametrize(
    ("dataset_id", "shape", "event_sum"),
    [
        ("aids", (1151, 11), 96),
        ("gbsg2", (686, 8), 299),
        ("flchain", (7874, 9), 2169),
        ("whas500", (500, 14), 215),
    ],
)
def test_load_sksurv_dataset_matches_documented_event_counts(
    dataset_id: str,
    shape: tuple[int, int],
    event_sum: int,
) -> None:
    dataset = load_dataset(dataset_id, repo_root=Path(__file__).resolve().parents[1])

    assert dataset.metadata.dataset_id == dataset_id
    assert dataset.metadata.source == "scikit-survival"
    assert dataset.X.shape == shape
    assert dataset.time.shape == (shape[0],)
    assert np.all(dataset.time > 0.0)
    assert dataset.event.shape == (shape[0],)
    assert int(dataset.event.sum()) == event_sum


def test_load_nwtco_pycox_dataset_matches_documented_shape() -> None:
    dataset = load_dataset("nwtco", repo_root=Path(__file__).resolve().parents[1])

    assert dataset.metadata.dataset_id == "nwtco"
    assert dataset.metadata.source == "pycox"
    assert dataset.X.shape == (4028, 6)
    assert dataset.time.shape == (4028,)
    assert np.all(dataset.time > 0.0)
    assert dataset.event.shape == (4028,)
    assert int(dataset.event.sum()) == 571


def test_load_local_file_dataset_from_config(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs" / "datasets"
    configs_dir.mkdir(parents=True)
    data_path = tmp_path / "data" / "processed" / "toy.csv"
    data_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3"],
            "time": [5.0, 10.0, 12.0],
            "event": [1, 0, 1],
            "gene_a": [0.1, 0.2, 0.3],
            "gene_b": [2.0, 1.5, 1.0],
        }
    ).to_csv(data_path, index=False)
    (configs_dir / "toy_local.yaml").write_text(
        "\n".join(
            [
                "dataset_id: toy_local",
                "name: Toy local file",
                "source: local_file",
                "local_path: data/processed/toy.csv",
                "time_col: time",
                "event_col: event",
                "id_col: patient_id",
                "primary_metric: uno_c",
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_dataset("toy_local", repo_root=tmp_path)

    assert dataset.metadata.dataset_id == "toy_local"
    assert dataset.metadata.source == "local_file"
    assert dataset.X.shape == (3, 2)
    assert dataset.time.tolist() == [5.0, 10.0, 12.0]
    assert dataset.event.tolist() == [1, 0, 1]


def test_load_local_file_dataset_missing_file_has_actionable_error(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs" / "datasets"
    configs_dir.mkdir(parents=True)
    (configs_dir / "missing_local.yaml").write_text(
        "\n".join(
            [
                "dataset_id: missing_local",
                "source: local_file",
                "local_path: data/processed/missing.csv",
                "time_col: time",
                "event_col: event",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Run the documented preparation command"):
        load_dataset("missing_local", repo_root=tmp_path)


# --- test_preprocess.py ---


def test_preprocessor_treats_boolean_and_low_cardinality_numeric_as_categorical() -> None:
    frame = pd.DataFrame(
        {
            "age": np.linspace(40.0, 79.0, 40),
            "stage": np.tile([0.0, 1.0, 2.0, 3.0], 10),
            "marker": np.arange(40, dtype=float),
            "is_male": [True, False] * 20,
        }
    )

    preprocessor = TabularPreprocessor(scale_numeric=False)
    transformed = preprocessor.fit_transform(frame)

    assert preprocessor.numeric_columns == ["age", "marker"]
    assert preprocessor.categorical_columns == ["stage", "is_male"]
    assert "stage_0.0" in transformed.columns
    assert "is_male_True" in transformed.columns
    np.testing.assert_allclose(transformed["age"].to_numpy(), frame["age"].to_numpy())


def test_native_preprocessor_preserves_categorical_frame_for_catboost() -> None:
    frame = pd.DataFrame(
        {
            "age": np.linspace(40.0, 79.0, 40),
            "stage": np.tile([0.0, 1.0, 1.0, 2.0], 10),
            "flag": [True, False] * 20,
        }
    )

    preprocessor = TabularPreprocessor(scale_numeric=False, categorical_encoding="native")
    transformed = preprocessor.fit_transform(frame)

    assert transformed["age"].dtype.kind == "f"
    assert transformed["stage"].head(4).tolist() == ["0.0", "1.0", "1.0", "2.0"]
    assert transformed["flag"].head(4).tolist() == ["True", "False", "True", "False"]


def test_preprocessor_rejects_dense_one_hot_expansion_above_budget() -> None:
    frame = pd.DataFrame({"zip_code": [f"z{i}" for i in range(6)]})

    with pytest.raises(ValueError, match="Dense one-hot preprocessing would create 6 features"):
        TabularPreprocessor(max_dense_features=5).fit(frame)


def test_feature_metadata_marks_low_cardinality_numeric_as_categorical() -> None:
    frame = pd.DataFrame(
        {
            "age": np.linspace(40.0, 79.0, 40),
            "stage": np.tile([0.0, 1.0, 2.0, 3.0], 10),
            "marker": np.arange(40, dtype=float),
            "is_male": [True, False] * 20,
        }
    )

    metadata = {feature.name: feature.inferred_type for feature in infer_feature_metadata(frame)}

    assert metadata == {
        "age": "numerical",
        "stage": "categorical",
        "marker": "numerical",
        "is_male": "boolean",
    }


# --- test_robustness_tracks.py ---


def _split() -> SplitDefinition:
    return SplitDefinition(
        split_id="s0",
        seed=11,
        repeat=0,
        fold=0,
        train_idx=np.asarray([0, 1, 2, 3], dtype=int),
        test_idx=np.asarray([4, 5], dtype=int),
    )


def test_resolve_robustness_tracks_returns_baseline_when_disabled() -> None:
    tracks = resolve_robustness_tracks(None, dataset_id="d", feature_columns=["x"], seed_pool=[11])
    assert len(tracks) == 1
    assert tracks[0].track_id == "baseline"


def test_covariate_noise_track_changes_test_rows_only() -> None:
    tracks = resolve_robustness_tracks(
        {"enabled": True, "tracks": ["covariate_noise"], "severity_levels": [0.2]},
        dataset_id="d",
        feature_columns=["x", "y"],
        seed_pool=[11],
    )
    track = [t for t in tracks if t.kind == "covariate_noise"][0]
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "y": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})
    out = apply_robustness_track(X, track=track, split=_split(), seed=11)
    assert np.allclose(out.iloc[:4].to_numpy(), X.iloc[:4].to_numpy())
    assert not np.allclose(out.iloc[4:].to_numpy(), X.iloc[4:].to_numpy())


def test_label_noise_flips_subset_of_test_labels() -> None:
    tracks = resolve_robustness_tracks(
        {"enabled": True, "tracks": ["label_noise"], "severity_levels": [1.0]},
        dataset_id="d",
        feature_columns=["x"],
        seed_pool=[11],
    )
    track = [t for t in tracks if t.kind == "label_noise"][0]
    event = np.asarray([0, 0, 1, 1, 0, 1], dtype=int)
    noisy = apply_label_noise(event, track=track, split=_split(), seed=11)
    assert np.array_equal(noisy[:4], event[:4])
    assert np.array_equal(noisy[4:], 1 - event[4:])


# --- test_user_dataset.py ---


def test_load_user_dataset_populates_feature_metadata_and_diagnostics() -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 0, 0],
            "age": [61, 57, 70, 66],
            "patient_id": ["a1", "a2", "a3", "a4"],
            "visit_date": ["2024-01-01", "2024-01-03", "2024-01-05", "2024-01-07"],
            "notes": [
                "progression observed with several complications",
                "stable follow-up with no major changes recorded",
                "stable follow-up with no major changes recorded",
                "stable follow-up with no major changes recorded",
            ],
        }
    )

    dataset = load_user_dataset(frame, time_col="time", event_col="event", dataset_id="toy")

    assert dataset.metadata.feature_types == ["numerical", "categorical", "datetime", "text"]
    assert dataset.metadata.diagnostics is not None
    assert dataset.metadata.diagnostics.n_events == 1
    assert "patient_id" in dataset.metadata.diagnostics.id_like_features
    assert any("Very few observed events" in warning for warning in dataset.metadata.diagnostics.warnings)
    assert any(feature.inferred_type == "datetime" for feature in dataset.metadata.feature_metadata)
    assert any(feature.inferred_type == "text" for feature in dataset.metadata.feature_metadata)


def test_load_user_dataset_preserves_binary_string_event_labels() -> None:
    frame = pd.DataFrame(
        {
            "time": [1, 2, 3],
            "event": ["dead", "alive", "dead"],
            "x1": [0.2, 0.5, 0.7],
        }
    )

    dataset = load_user_dataset(frame, time_col="time", event_col="event")

    assert dataset.event.tolist() == [1, 0, 1]


def test_load_user_dataset_rejects_missing_or_nonbinary_numeric_event_labels() -> None:
    missing_frame = pd.DataFrame(
        {
            "time": [1, 2, 3],
            "event": [1, None, 0],
            "x1": [0.2, 0.5, 0.7],
        }
    )
    nonbinary_frame = pd.DataFrame(
        {
            "time": [1, 2, 3],
            "event": [1.0, 0.5, 0.0],
            "x1": [0.2, 0.5, 0.7],
        }
    )

    try:
        load_user_dataset(missing_frame, time_col="time", event_col="event")
    except ValueError as exc:
        assert "must not contain missing event indicators" in str(exc)
    else:
        raise AssertionError("Expected missing numeric event labels to raise a ValueError.")

    try:
        load_user_dataset(nonbinary_frame, time_col="time", event_col="event")
    except ValueError as exc:
        assert "must contain only binary event indicators" in str(exc)
    else:
        raise AssertionError("Expected non-binary numeric event labels to raise a ValueError.")


# --- test_validation.py ---


def _dataset(frame: pd.DataFrame, *, time: list[float], event: list[int]) -> SurvivalDataset:
    return SurvivalDataset(
        metadata=DatasetMetadata(dataset_id="toy", name="toy", source="unit_test"),
        X=frame.reset_index(drop=True),
        time=np.asarray(time, dtype=float),
        event=np.asarray(event, dtype=int),
    )


@pytest.mark.parametrize(
    ("n_rows", "expected"),
    [
        (10, 0.2),
        (499, 0.2),
        (500, 0.15),
        (4_999, 0.15),
        (5_000, 0.1),
        (24_999, 0.1),
        (25_000, 0.05),
    ],
)
def test_default_holdout_frac_uses_documented_thresholds(n_rows: int, expected: float) -> None:
    assert default_holdout_frac(n_rows) == expected


def test_build_validation_plan_aligns_tuning_columns_by_name() -> None:
    training = _dataset(
        pd.DataFrame({"age": [61, 57, 70, 66], "stage": ["i", "ii", "ii", "iii"]}),
        time=[1.0, 2.0, 3.0, 4.0],
        event=[1, 0, 1, 0],
    )
    tuning = _dataset(
        pd.DataFrame({"stage": ["ii", "i"], "age": [59, 63]}),
        time=[5.0, 6.0],
        event=[1, 0],
    )

    plan = build_validation_plan(training, tuning_dataset=tuning, seed=7)

    assert plan.source == "tuning_data"
    assert list(plan.validation_X.columns) == ["age", "stage"]
    assert plan.validation_X.to_dict(orient="list") == {"age": [59, 63], "stage": ["ii", "i"]}


def test_build_validation_plan_rejects_tuning_feature_mismatch() -> None:
    training = _dataset(
        pd.DataFrame({"age": [61, 57, 70, 66], "stage": ["i", "ii", "ii", "iii"]}),
        time=[1.0, 2.0, 3.0, 4.0],
        event=[1, 0, 1, 0],
    )
    tuning = _dataset(
        pd.DataFrame({"age": [59, 63], "grade": ["a", "b"]}),
        time=[5.0, 6.0],
        event=[1, 0],
    )

    with pytest.raises(ValueError, match="missing columns: \\['stage'\\].*extra columns: \\['grade'\\]"):
        build_validation_plan(training, tuning_dataset=tuning, seed=7)


def test_build_validation_plan_rejects_small_auto_holdout_dataset() -> None:
    dataset = _dataset(
        pd.DataFrame({"age": [61, 57, 70]}),
        time=[1.0, 2.0, 3.0],
        event=[1, 0, 1],
    )

    with pytest.raises(ValueError, match="Need at least 4 rows"):
        build_validation_plan(dataset, seed=0)


def test_build_validation_plan_rejects_low_count_event_classes() -> None:
    dataset = _dataset(
        pd.DataFrame({"age": [61, 57, 70, 66]}),
        time=[1.0, 2.0, 3.0, 4.0],
        event=[1, 0, 0, 0],
    )

    with pytest.raises(ValueError, match="requires at least two samples in each event class"):
        build_validation_plan(dataset, seed=0)


@pytest.mark.parametrize("holdout_frac", [-0.1, 0.0, 1.0, 1.2])
def test_build_validation_plan_rejects_invalid_holdout_fraction(holdout_frac: float) -> None:
    dataset = _dataset(
        pd.DataFrame({"age": [61, 57, 70, 66, 59, 63, 68, 55]}),
        time=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        event=[1, 0, 1, 0, 1, 0, 1, 0],
    )

    with pytest.raises(ValueError, match="holdout_frac must be between 0 and 1"):
        build_validation_plan(dataset, seed=0, holdout_frac=holdout_frac)


def test_build_validation_plan_creates_stratified_auto_holdout() -> None:
    dataset = _dataset(
        pd.DataFrame({"row_id": list(range(10)), "age": [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]}),
        time=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        event=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    )

    plan = build_validation_plan(dataset, seed=11, holdout_frac=0.2)

    assert plan.source == "auto_holdout"
    assert plan.holdout_frac == 0.2
    assert len(plan.train_X) == 8
    assert len(plan.validation_X) == 2
    assert set(plan.train_X["row_id"]).isdisjoint(set(plan.validation_X["row_id"]))
    assert int(plan.validation_event.sum()) == 1
    assert int((plan.validation_event == 0).sum()) == 1


def test_build_refit_dataset_with_explicit_tuning_data_respects_refit_full() -> None:
    training = _dataset(
        pd.DataFrame({"age": [61, 57, 70], "stage": ["i", "ii", "iii"]}),
        time=[1.0, 2.0, 3.0],
        event=[1, 0, 1],
    )
    tuning = _dataset(
        pd.DataFrame({"stage": ["iii", "i"], "age": [66, 59]}),
        time=[4.0, 5.0],
        event=[0, 1],
    )

    refit_full = build_refit_dataset(training, validation_plan=None, tuning_dataset=tuning, refit_full=True)
    refit_train_only = build_refit_dataset(training, validation_plan=None, tuning_dataset=tuning, refit_full=False)

    assert refit_full.X.to_dict(orient="list") == {
        "age": [61, 57, 70, 66, 59],
        "stage": ["i", "ii", "iii", "iii", "i"],
    }
    np.testing.assert_allclose(refit_full.time, np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]))
    np.testing.assert_array_equal(refit_full.event, np.asarray([1, 0, 1, 0, 1]))
    assert refit_train_only.X.to_dict(orient="list") == training.X.to_dict(orient="list")
    np.testing.assert_allclose(refit_train_only.time, training.time)
    np.testing.assert_array_equal(refit_train_only.event, training.event)


def test_prepare_validation_fold_cache_applies_method_specific_numeric_scaling() -> None:
    plan = ValidationPlan(
        source="tuning_data",
        holdout_frac=None,
        train_X=pd.DataFrame({"age": [20.0, 40.0], "stage": ["i", "ii"]}),
        train_time=np.asarray([1.0, 2.0], dtype=float),
        train_event=np.asarray([1, 0], dtype=int),
        validation_X=pd.DataFrame({"age": [30.0], "stage": ["ii"]}),
        validation_time=np.asarray([1.5], dtype=float),
        validation_event=np.asarray([1], dtype=int),
    )

    rsf_fold = prepare_validation_fold_cache(method_id="rsf", plan=plan)[0]
    extra_trees_fold = prepare_validation_fold_cache(method_id="extra_survival_trees", plan=plan)[0]
    gradient_boosting_fold = prepare_validation_fold_cache(method_id="gradient_boosting_survival", plan=plan)[0]
    catboost_fold = prepare_validation_fold_cache(method_id="catboost_cox", plan=plan)[0]
    catboost_aft_fold = prepare_validation_fold_cache(method_id="catboost_survival_aft", plan=plan)[0]
    xgboost_fold = prepare_validation_fold_cache(method_id="xgboost_cox", plan=plan)[0]
    xgboost_aft_fold = prepare_validation_fold_cache(method_id="xgboost_aft", plan=plan)[0]
    componentwise_fold = prepare_validation_fold_cache(method_id="componentwise_gradient_boosting", plan=plan)[0]

    assert method_uses_scaled_numeric_features("coxph") is True
    assert method_uses_scaled_numeric_features("componentwise_gradient_boosting") is True
    assert method_uses_scaled_numeric_features("rsf") is False
    assert method_uses_scaled_numeric_features("extra_survival_trees") is False
    assert method_uses_scaled_numeric_features("gradient_boosting_survival") is False
    assert method_uses_scaled_numeric_features("xgboost_cox") is False
    assert method_uses_scaled_numeric_features("xgboost_aft") is False
    assert method_uses_scaled_numeric_features("mitra_survival_frozen") is False
    assert method_uses_scaled_numeric_features("tabicl_survival") is True
    assert method_uses_scaled_numeric_features("tabm_survival") is False
    assert method_uses_scaled_numeric_features("tabdpt_survival") is True
    assert method_uses_scaled_numeric_features("catboost_cox") is False
    assert method_uses_scaled_numeric_features("catboost_survival_aft") is False
    assert method_uses_native_categorical_features("mitra_survival_frozen") is True
    assert method_uses_native_categorical_features("tabicl_survival") is False
    assert method_uses_native_categorical_features("tabm_survival") is True
    assert method_uses_native_categorical_features("tabdpt_survival") is False
    assert method_uses_native_categorical_features("catboost_cox") is True
    assert method_uses_native_categorical_features("catboost_survival_aft") is True
    np.testing.assert_allclose(rsf_fold["X_train"][:, 0], np.asarray([20.0, 40.0]))
    np.testing.assert_allclose(extra_trees_fold["X_train"][:, 0], np.asarray([20.0, 40.0]))
    np.testing.assert_allclose(gradient_boosting_fold["X_train"][:, 0], np.asarray([20.0, 40.0]))
    np.testing.assert_allclose(xgboost_fold["X_train"][:, 0], np.asarray([20.0, 40.0]))
    np.testing.assert_allclose(xgboost_aft_fold["X_train"][:, 0], np.asarray([20.0, 40.0]))
    assert isinstance(catboost_fold["X_train"], pd.DataFrame)
    np.testing.assert_allclose(catboost_fold["X_train"]["age"].to_numpy(dtype=float), np.asarray([20.0, 40.0]))
    assert catboost_fold["X_train"]["stage"].tolist() == ["i", "ii"]
    assert isinstance(catboost_aft_fold["X_train"], pd.DataFrame)
    np.testing.assert_allclose(catboost_aft_fold["X_train"]["age"].to_numpy(dtype=float), np.asarray([20.0, 40.0]))
    assert catboost_aft_fold["X_train"]["stage"].tolist() == ["i", "ii"]
    assert not np.allclose(componentwise_fold["X_train"][:, 0], rsf_fold["X_train"][:, 0])
    assert np.isclose(float(componentwise_fold["X_train"][:, 0].mean()), 0.0)
