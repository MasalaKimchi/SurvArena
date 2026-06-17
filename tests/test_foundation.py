from __future__ import annotations

import sys
import types
from pathlib import Path
import numpy as np
from survarena.api.predictor import SurvivalPredictor
from survarena.automl.presets import resolve_preset
from survarena.config import read_yaml
from survarena.methods.foundation.catalog import available_foundation_model_specs
from survarena.methods.foundation.readiness import FoundationRuntimeStatus, foundation_runtime_status


# --- test_foundation.py ---

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_foundation_model_catalog_exposes_current_and_planned_backbones() -> None:
    predictor = SurvivalPredictor(label_time="time", label_event="event")

    catalog = predictor.foundation_model_catalog()

    assert "tabpfn_survival" in catalog["method_id"].tolist()
    assert "mitra_survival_frozen" in catalog["method_id"].tolist()
    assert "tabicl_survival" in catalog["method_id"].tolist()
    assert "tabm_survival" in catalog["method_id"].tolist()
    assert not set(catalog["method_id"]).intersection(
        {
            "tabpfn_discrete_hazard_survival",
            "tabicl_discrete_hazard_survival",
            "tabm_discrete_hazard_survival",
            "realtabpfn_discrete_hazard_survival",
        }
    )
    assert "dependency_installed" in catalog.columns
    assert "runtime_ready" in catalog.columns
    assert "install_extra" in catalog.columns
    implemented = dict(zip(catalog["method_id"], catalog["implemented"], strict=False))
    assert implemented["tabpfn_survival"] is True
    assert implemented["mitra_survival_frozen"] is True
    assert implemented["tabicl_survival"] is True
    assert implemented["tabm_survival"] is True


def test_tabpfn_method_config_uses_discrete_hazard_adapter() -> None:
    spec = next(item for item in available_foundation_model_specs() if item.method_id == "tabpfn_survival")
    method_cfg = read_yaml(REPO_ROOT / "configs" / "methods" / "tabpfn_survival.yaml")

    assert spec.supports_finetune is False
    assert method_cfg["default_params"]["model_version"] == "v2.5"
    assert method_cfg["default_params"]["n_intervals"] == 5
    assert method_cfg["default_params"]["max_stacked_rows"] == 50000
    assert "backbone_training" not in method_cfg["default_params"]
    assert "backbone_task" not in method_cfg["default_params"]
    assert "n_estimators_final_inference" not in method_cfg["default_params"]


def test_direct_foundation_method_configs_use_discrete_hazard_adapters() -> None:
    from survarena.methods.foundation.discrete_hazard import TabICLSurvivalMethod
    from survarena.methods.registry import get_method_class

    tabicl_cfg = read_yaml(REPO_ROOT / "configs" / "methods" / "tabicl_survival.yaml")

    assert get_method_class("tabicl_survival") is TabICLSurvivalMethod
    assert tabicl_cfg["default_params"]["n_intervals"] == 5
    assert tabicl_cfg["default_params"]["max_stacked_rows"] == 50000
    assert tabicl_cfg["default_params"]["predict_batch_size"] is None
    assert "time_limit" not in tabicl_cfg["default_params"]


def test_manuscript_config_includes_foundation_track() -> None:
    benchmark_cfg = read_yaml(REPO_ROOT / "configs" / "benchmark" / "manuscript_v1.yaml")
    overrides = benchmark_cfg["hpo"]["method_overrides"]

    assert benchmark_cfg["primary_metric"] == "uno_c"
    assert benchmark_cfg["comparison_modes"] == ["no_hpo"]
    assert benchmark_cfg["outer_folds"] == 5
    assert benchmark_cfg["outer_repeats"] == 3
    assert "tabpfn_survival" in benchmark_cfg["methods"]
    assert "tabpfn_survival_classifier" not in benchmark_cfg["methods"]
    assert "tabpfn_survival_regressor" not in benchmark_cfg["methods"]
    assert "mitra_survival_frozen" not in benchmark_cfg["methods"]
    assert {"tabicl_survival", "tabm_survival", "realtabpfn_survival"}.issubset(benchmark_cfg["methods"])
    assert "mitra_survival_finetune" not in overrides
    assert overrides["tabpfn_survival"]["default_params"]["n_intervals"] == 5
    assert overrides["tabicl_survival"]["default_params"]["n_intervals"] == 5
    assert overrides["realtabpfn_survival"]["default_params"]["max_stacked_rows"] == 9500
    assert benchmark_cfg["profile"] == "manuscript"
    assert set(benchmark_cfg["datasets"]) == {"support", "metabric", "nwtco", "aids", "gbsg2", "flchain", "whas500"}
    assert "Mitra is excluded" in benchmark_cfg["notes"]


def test_manuscript_hpo_config_uses_explicit_hpo_track() -> None:
    no_hpo_cfg = read_yaml(REPO_ROOT / "configs" / "benchmark" / "manuscript_v1.yaml")
    hpo_cfg = read_yaml(REPO_ROOT / "configs" / "benchmark" / "manuscript_hpo_v1.yaml")

    assert hpo_cfg["benchmark_id"] == "manuscript_hpo_v1"
    assert hpo_cfg["profile"] == "manuscript"
    assert hpo_cfg["comparison_modes"] == ["hpo"]
    assert hpo_cfg["hpo"]["enabled"] is True
    assert hpo_cfg["hpo"]["max_trials"] == 30
    assert hpo_cfg["hpo"]["timeout_seconds"] == 1800
    assert hpo_cfg["hpo"]["sampler"] == "tpe"
    assert hpo_cfg["hpo"]["pruner"] == "median"
    assert hpo_cfg["hpo"]["n_startup_trials"] == 10
    assert hpo_cfg["datasets"] == no_hpo_cfg["datasets"]
    assert hpo_cfg["methods"] == no_hpo_cfg["methods"]


def test_foundation_runtime_status_reports_install_command_for_missing_dependency(monkeypatch) -> None:
    spec = next(item for item in available_foundation_model_specs() if item.method_id == "tabpfn_survival")

    monkeypatch.setattr("survarena.methods.foundation.readiness._has_dependency", lambda module_name: False)

    status = foundation_runtime_status(spec)

    assert status.runtime_ready is False
    assert status.dependency_installed is False
    assert status.install_extra == "foundation-tabpfn"
    assert status.install_command == 'python -m pip install -e ".[foundation-tabpfn]"'
    assert status.blocked_reason is not None


def test_foundation_runtime_status_warns_when_tabpfn_auth_is_missing(monkeypatch) -> None:
    spec = next(item for item in available_foundation_model_specs() if item.method_id == "tabpfn_survival")

    monkeypatch.setattr("survarena.methods.foundation.readiness._has_dependency", lambda module_name: True)
    monkeypatch.setattr("survarena.methods.foundation.readiness._huggingface_auth_configured", lambda: False)

    status = foundation_runtime_status(spec)

    assert status.runtime_ready is True
    assert status.auth_configured is False
    assert status.warning_reason is not None
    assert "hf auth login" in status.warning_reason


def test_tabpfn_survival_supports_explicit_model_versions(monkeypatch) -> None:
    from survarena.methods.foundation.tabpfn_survival import TabPFNSurvivalMethod

    class FakeBackbone:
        init_kwargs: dict[str, object] | None = None
        created_version: object | None = None

        def __init__(self, **kwargs) -> None:
            FakeBackbone.init_kwargs = kwargs

        @classmethod
        def create_default_for_version(cls, version: object, **kwargs):
            cls.created_version = version
            return cls(**kwargs)

        def fit(self, X: np.ndarray, y: np.ndarray) -> "FakeBackbone":
            self.classes_ = np.asarray([0, 1])
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            X_np = np.asarray(X, dtype=np.float32)
            positive = np.clip(0.5 + 0.1 * X_np[:, 0], 0.0, 1.0)
            return np.column_stack([1.0 - positive, positive])

    fake_tabpfn = types.ModuleType("tabpfn")
    fake_tabpfn.TabPFNClassifier = FakeBackbone
    fake_constants = types.ModuleType("tabpfn.constants")
    fake_constants.ModelVersion = types.SimpleNamespace(V2="v2", V2_5="v2.5")

    monkeypatch.setitem(sys.modules, "tabpfn", fake_tabpfn)
    monkeypatch.setitem(sys.modules, "tabpfn.constants", fake_constants)

    method = TabPFNSurvivalMethod(model_version="v2.5", n_estimators=4)
    estimator = method._build_backbone()

    assert FakeBackbone.created_version == "v2.5"
    assert FakeBackbone.init_kwargs is not None
    assert FakeBackbone.init_kwargs["n_estimators"] == 4
    assert estimator is not None


def test_foundation_preset_adds_tabpfn_when_dependency_exists(monkeypatch) -> None:
    monkeypatch.setattr(
        "survarena.automl.presets._foundation_runtime_status",
        lambda spec: FoundationRuntimeStatus(
            method_id=spec.method_id,
            dependency_module=spec.dependency_module,
            install_extra=spec.install_extra,
            dependency_installed=True,
            runtime_ready=True,
            requires_hf_auth=spec.requires_hf_auth,
            auth_configured=True,
            install_command=None,
        ),
    )

    preset = resolve_preset(
        "foundation",
        n_rows=1000,
        n_features=50,
        event_count=200,
        event_fraction=0.2,
    )

    assert "tabpfn_survival" in preset.method_ids


def test_tabpfn_survival_frozen_path_fit_predicts_with_fake_backbone(monkeypatch) -> None:
    from survarena.methods.foundation.tabpfn_survival import TabPFNSurvivalMethod

    class FakeBackbone:
        fit_labels: list[np.ndarray] = []

        def __init__(self, **kwargs) -> None:
            self.kwargs = dict(kwargs)

        @classmethod
        def create_default_for_version(cls, version: object, **kwargs):
            return cls(model_version=version, **kwargs)

        def fit(self, X: np.ndarray, y: np.ndarray) -> "FakeBackbone":
            self.offset_ = float(np.mean(y))
            self.classes_ = np.asarray([0, 1])
            FakeBackbone.fit_labels.append(np.asarray(y, dtype=np.int32))
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            X_np = np.asarray(X, dtype=np.float32)
            logits = X_np[:, 0] + self.offset_
            positive = 1.0 / (1.0 + np.exp(-logits))
            return np.column_stack([1.0 - positive, positive])

    fake_tabpfn = types.ModuleType("tabpfn")
    fake_tabpfn.TabPFNClassifier = FakeBackbone
    fake_constants = types.ModuleType("tabpfn.constants")
    fake_constants.ModelVersion = types.SimpleNamespace(V2="v2", V2_5="v2.5")

    monkeypatch.setitem(sys.modules, "tabpfn", fake_tabpfn)
    monkeypatch.setitem(sys.modules, "tabpfn.constants", fake_constants)
    monkeypatch.setattr(
        "survarena.methods.foundation.discrete_hazard.ensure_foundation_runtime_ready",
        lambda method_id, checkpoint_path=None: None,
    )

    X = np.asarray(
        [
            [-1.0, 0.0],
            [-0.5, 0.1],
            [0.0, 0.2],
            [0.3, 0.3],
            [0.7, 0.4],
            [1.0, 0.5],
            [1.2, 0.6],
            [1.5, 0.7],
        ],
        dtype=np.float32,
    )
    time = np.asarray([2.0, 3.0, 4.0, 6.0, 7.0, 9.0, 11.0, 13.0], dtype=np.float64)
    event = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int32)

    method = TabPFNSurvivalMethod(
        n_estimators=1,
        horizon_quantiles=[0.25, 0.5, 0.75],
        min_rows_per_interval=1,
        device="cpu",
        seed=13,
    )
    method.fit(X, time, event)
    risk = method.predict_risk(X[:4])
    survival = method.predict_survival(X[:4], np.asarray([1.0, 5.0, 8.0, 12.0]))

    assert len(FakeBackbone.fit_labels) == 1
    assert risk.shape == (4,)
    assert survival.shape == (4, 4)
    assert np.isfinite(risk).all()
    assert np.isfinite(survival).all()
    assert ((survival >= 0.0) & (survival <= 1.0)).all()
    assert (np.diff(survival, axis=1) <= 1e-8).all()
    assert method.foundation_metadata()["foundation_backbone_task"] == (
        "censored_aware_pooled_discrete_time_hazard_classification"
    )


def test_tabpfn_survival_batches_prediction_with_fake_backbone() -> None:
    from survarena.methods.foundation.inference import positive_class_probability_with_backoff

    class FakeBackbone:
        calls: list[int] = []
        classes_ = np.asarray([0, 1])

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            self.calls.append(len(X))
            positive = np.linspace(0.1, 0.9, len(X), dtype=np.float64)
            return np.column_stack([1.0 - positive, positive])

    model = FakeBackbone()
    X = np.zeros((5, 2), dtype=np.float32)

    probabilities = positive_class_probability_with_backoff(model, X, batch_size=2)

    assert model.calls == [2, 2, 1]
    assert probabilities.shape == (5,)
    assert np.isfinite(probabilities).all()


def test_foundation_prediction_batching_backs_off_and_reuses_safe_size() -> None:
    from survarena.methods.foundation.inference import positive_class_probability_with_backoff

    class FakeBackbone:
        classes_ = np.asarray([0, 1])

        def __init__(self) -> None:
            self.calls: list[int] = []

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            self.calls.append(len(X))
            if len(X) > 3:
                raise RuntimeError("out of memory")
            positive = np.full(len(X), 0.75, dtype=np.float64)
            return np.column_stack([1.0 - positive, positive])

    model = FakeBackbone()
    X = np.zeros((9, 2), dtype=np.float32)

    probabilities = positive_class_probability_with_backoff(model, X, batch_size=None)
    second_probabilities = positive_class_probability_with_backoff(model, X[:5], batch_size=None)

    assert model.calls == [9, 4, 2, 2, 2, 2, 1, 2, 2, 1]
    np.testing.assert_array_equal(probabilities, np.full(9, 0.75))
    np.testing.assert_array_equal(second_probabilities, np.full(5, 0.75))


def test_direct_discrete_hazard_foundation_adapter_fit_predicts_with_fake_backbone(monkeypatch) -> None:
    from survarena.methods.foundation.discrete_hazard import TabICLDiscreteHazardSurvivalMethod

    class FakeClassifier:
        fit_y: np.ndarray | None = None
        predict_count = 0

        def __init__(self, **kwargs) -> None:
            self.kwargs = dict(kwargs)

        def fit(self, X: np.ndarray, y: np.ndarray) -> "FakeClassifier":
            self.classes_ = np.asarray([0, 1])
            self.offset_ = float(np.mean(y))
            FakeClassifier.fit_y = np.asarray(y, dtype=np.int32)
            return self

        def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
            FakeClassifier.predict_count += 1
            X_np = np.asarray(X, dtype=np.float32)
            positive = np.clip(0.15 + 0.15 * X_np[:, 0] + 0.1 * X_np[:, -4] + self.offset_, 0.0, 1.0)
            return np.column_stack([1.0 - positive, positive])

    fake_tabicl = types.ModuleType("tabicl")
    fake_tabicl.TabICLClassifier = FakeClassifier
    monkeypatch.setitem(sys.modules, "tabicl", fake_tabicl)
    monkeypatch.setattr(
        "survarena.methods.foundation.discrete_hazard.ensure_foundation_runtime_ready", lambda method_id, **kwargs: None
    )

    X = np.asarray(
        [
            [-1.0, 0.0],
            [-0.5, 0.1],
            [0.0, 0.2],
            [0.3, 0.3],
            [0.7, 0.4],
            [1.0, 0.5],
            [1.2, 0.6],
            [1.5, 0.7],
        ],
        dtype=np.float32,
    )
    time = np.asarray([2.0, 3.0, 4.0, 6.0, 7.0, 9.0, 11.0, 13.0], dtype=np.float64)
    event = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int32)

    method = TabICLDiscreteHazardSurvivalMethod(
        n_estimators=1,
        horizon_quantiles=[0.25, 0.5],
        min_rows_per_interval=1,
    )
    method.fit(X, time, event)
    FakeClassifier.predict_count = 0
    risk = method.predict_risk(X[:3])
    survival = method.predict_survival(X[:3], np.asarray([1.0, 5.0, 10.0]))
    separate_predict_count = FakeClassifier.predict_count
    FakeClassifier.predict_count = 0
    predictions = method.predict_bundle(X[:3], np.asarray([1.0, 5.0, 10.0]))

    assert FakeClassifier.fit_y is not None
    assert FakeClassifier.fit_y.sum() >= 1
    assert risk.shape == (3,)
    assert survival.shape == (3, 3)
    assert np.isfinite(risk).all()
    assert np.isfinite(survival).all()
    assert (np.diff(survival, axis=1) <= 1e-8).all()
    np.testing.assert_array_equal(predictions.risk, risk)
    np.testing.assert_array_equal(predictions.survival, survival)
    assert separate_predict_count == 4
    assert FakeClassifier.predict_count == 2
    metadata = method.foundation_metadata()
    assert metadata["foundation_backbone_task"] == "censored_aware_pooled_discrete_time_hazard_classification"
    assert metadata["foundation_sample_weight_supported"] is False
    assert metadata["foundation_sample_weight_requested"] == "normalized"
    assert metadata["foundation_sample_weight_applied"] is False


def test_direct_discrete_hazard_applies_sample_weight_when_backbone_supports_it(monkeypatch) -> None:
    from survarena.methods.foundation.discrete_hazard import TabICLDiscreteHazardSurvivalMethod

    class WeightedFakeClassifier:
        received_sample_weight: np.ndarray | None = None

        def __init__(self, **kwargs) -> None:
            self.kwargs = dict(kwargs)

        def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "WeightedFakeClassifier":
            self.classes_ = np.asarray([0, 1])
            WeightedFakeClassifier.received_sample_weight = None if sample_weight is None else np.asarray(sample_weight)
            return self

        def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
            positive = np.full(np.asarray(X).shape[0], 0.2, dtype=float)
            return np.column_stack([1.0 - positive, positive])

    fake_tabicl = types.ModuleType("tabicl")
    fake_tabicl.TabICLClassifier = WeightedFakeClassifier
    monkeypatch.setitem(sys.modules, "tabicl", fake_tabicl)
    monkeypatch.setattr(
        "survarena.methods.foundation.discrete_hazard.ensure_foundation_runtime_ready", lambda method_id, **kwargs: None
    )

    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    time = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    event = np.asarray([1, 0, 1, 0], dtype=np.int32)
    method = TabICLDiscreteHazardSurvivalMethod(
        horizon_quantiles=[0.25, 0.5, 0.75],
        min_rows_per_interval=1,
    )

    method.fit(X, time, event)

    assert WeightedFakeClassifier.received_sample_weight is not None
    metadata = method.foundation_metadata()
    assert metadata["foundation_sample_weight_supported"] is True
    assert metadata["foundation_sample_weight_applied"] is True


# --- test_presets.py ---


def test_resolve_preset_skips_high_capacity_models_for_low_event_data() -> None:
    preset = resolve_preset(
        "best",
        n_rows=120,
        n_features=24,
        event_count=8,
        event_fraction=8 / 120,
        enable_foundation_models=True,
    )

    assert "coxph" in preset.method_ids
    assert "coxnet" in preset.method_ids
    assert "rsf" in preset.method_ids
    assert "deepsurv" not in preset.method_ids
    assert "deepsurv_moco" not in preset.method_ids
    assert "tabpfn_survival" not in preset.method_ids
    assert any("only 8 observed events" in note for note in preset.portfolio_notes)


def test_resolve_preset_skips_foundation_models_for_unsupported_feature_shapes() -> None:
    preset = resolve_preset(
        "medium",
        n_rows=1000,
        n_features=40,
        event_count=180,
        event_fraction=0.18,
        high_cardinality_feature_count=3,
        has_datetime_features=True,
        enable_foundation_models=True,
    )

    assert "tabpfn_survival" not in preset.method_ids
    assert any("high-cardinality categorical features" in note for note in preset.portfolio_notes)
    assert any("datetime-aware feature handling" in note for note in preset.portfolio_notes)


def test_foundation_preset_requests_foundation_models_without_extra_flag() -> None:
    from pytest import MonkeyPatch

    monkeypatch = MonkeyPatch()
    monkeypatch.setattr(
        "survarena.automl.presets._foundation_runtime_status",
        lambda spec: FoundationRuntimeStatus(
            method_id=spec.method_id,
            dependency_module=spec.dependency_module,
            install_extra=spec.install_extra,
            dependency_installed=True,
            runtime_ready=True,
            requires_hf_auth=spec.requires_hf_auth,
            auth_configured=True,
            install_command=None,
        ),
    )
    preset = resolve_preset(
        "foundation",
        n_rows=500,
        n_features=30,
        event_count=150,
        event_fraction=0.3,
    )
    monkeypatch.undo()

    assert preset.method_ids == (
        "coxph",
        "tabpfn_survival",
        "mitra_survival_frozen",
        "tabicl_survival",
        "tabm_survival",
        "realtabpfn_survival",
    )


def test_foundation_preset_reports_when_no_current_adapter_is_eligible() -> None:
    preset = resolve_preset(
        "foundation",
        n_rows=50_000,
        n_features=2_500,
        event_count=2_000,
        event_fraction=0.04,
    )

    assert preset.method_ids == ("coxph",)
    assert any(
        "No currently implemented foundation-model adapters were eligible" in note for note in preset.portfolio_notes
    )


def test_all_preset_runs_full_portfolio_and_auto_adds_foundation_models() -> None:
    from pytest import MonkeyPatch

    monkeypatch = MonkeyPatch()
    monkeypatch.setattr(
        "survarena.automl.presets._foundation_runtime_status",
        lambda spec: FoundationRuntimeStatus(
            method_id=spec.method_id,
            dependency_module=spec.dependency_module,
            install_extra=spec.install_extra,
            dependency_installed=True,
            runtime_ready=True,
            requires_hf_auth=spec.requires_hf_auth,
            auth_configured=True,
            install_command=None,
        ),
    )
    preset = resolve_preset(
        "all",
        n_rows=500,
        n_features=30,
        event_count=150,
        event_fraction=0.3,
    )
    monkeypatch.undo()

    assert preset.method_ids == (
        "coxph",
        "coxnet",
        "rsf",
        "deepsurv",
        "deepsurv_moco",
        "tabpfn_survival",
        "mitra_survival_frozen",
        "tabicl_survival",
        "tabm_survival",
        "realtabpfn_survival",
    )


def test_resolve_preset_surfaces_foundation_runtime_warnings(monkeypatch) -> None:
    monkeypatch.setattr(
        "survarena.automl.presets._foundation_runtime_status",
        lambda spec: FoundationRuntimeStatus(
            method_id=spec.method_id,
            dependency_module=spec.dependency_module,
            install_extra=spec.install_extra,
            dependency_installed=True,
            runtime_ready=True,
            requires_hf_auth=spec.requires_hf_auth,
            auth_configured=False if spec.method_id == "tabpfn_survival" else True,
            install_command=None,
            warning_reason="Run `hf auth login` first." if spec.method_id == "tabpfn_survival" else None,
        ),
    )

    preset = resolve_preset(
        "foundation",
        n_rows=500,
        n_features=30,
        event_count=150,
        event_fraction=0.3,
    )

    assert "tabpfn_survival" in preset.method_ids
    assert any("hf auth login" in note for note in preset.portfolio_notes)
