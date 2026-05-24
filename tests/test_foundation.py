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

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_foundation_model_catalog_exposes_current_and_planned_backbones() -> None:
    predictor = SurvivalPredictor(label_time="time", label_event="event")

    catalog = predictor.foundation_model_catalog()

    assert "tabpfn_survival" in catalog["method_id"].tolist()
    assert "mitra_survival_frozen" in catalog["method_id"].tolist()
    assert "tabicl_survival" in catalog["method_id"].tolist()
    assert "dependency_installed" in catalog.columns
    assert "runtime_ready" in catalog.columns
    assert "install_extra" in catalog.columns
    implemented = dict(zip(catalog["method_id"], catalog["implemented"], strict=False))
    assert implemented["tabpfn_survival"] is True
    assert implemented["mitra_survival_frozen"] is True
    assert implemented["tabicl_survival"] is False


def test_tabpfn_method_config_uses_horizon_adapter_only() -> None:
    spec = next(item for item in available_foundation_model_specs() if item.method_id == "tabpfn_survival")
    method_cfg = read_yaml(REPO_ROOT / "configs" / "methods" / "tabpfn_survival.yaml")

    assert spec.supports_finetune is False
    assert method_cfg["default_params"]["model_version"] == "v2.5"
    assert method_cfg["default_params"]["horizon_quantiles"] == "0.25-0.5-0.75"
    assert "backbone_training" not in method_cfg["default_params"]
    assert "backbone_task" not in method_cfg["default_params"]
    assert "n_estimators_final_inference" not in method_cfg["default_params"]


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
    assert "mitra_survival_frozen" in benchmark_cfg["methods"]
    assert "mitra_survival_finetune" not in overrides
    assert overrides["tabpfn_survival"]["default_params"]["horizon_quantiles"] == "0.25-0.5-0.75"
    assert overrides["mitra_survival_frozen"]["default_params"]["time_limit"] == 120
    assert benchmark_cfg["profile"] == "manuscript"
    assert set(benchmark_cfg["datasets"]) == {"support", "metabric", "nwtco", "aids", "gbsg2", "flchain", "whas500"}
    assert "full backbone fine-tuning remains excluded" in benchmark_cfg["notes"]


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


def test_tabpfn_horizon_labels_exclude_unknown_censored_rows() -> None:
    from survarena.methods.foundation.tabpfn_survival import TabPFNSurvivalMethod

    known, labels = TabPFNSurvivalMethod._horizon_known_labels(
        time_train=np.asarray([2.0, 3.0, 5.0, 7.0, 9.0]),
        event_train=np.asarray([0, 1, 0, 1, 0]),
        horizon=5.0,
    )

    np.testing.assert_array_equal(known, np.asarray([False, True, False, True, True]))
    np.testing.assert_array_equal(labels, np.asarray([1, 0, 0], dtype=np.int32))


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
        "survarena.methods.foundation.tabpfn_survival.ensure_foundation_runtime_ready",
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
        min_known_per_horizon=1,
        device="cpu",
        seed=13,
    )
    method.fit(X, time, event)
    risk = method.predict_risk(X[:4])
    survival = method.predict_survival(X[:4], np.asarray([1.0, 5.0, 8.0, 12.0]))

    assert len(FakeBackbone.fit_labels) == 3
    assert risk.shape == (4,)
    assert survival.shape == (4, 4)
    assert np.isfinite(risk).all()
    assert np.isfinite(survival).all()
    assert ((survival >= 0.0) & (survival <= 1.0)).all()
    assert (np.diff(survival, axis=1) <= 1e-8).all()
    assert method.foundation_metadata()["foundation_backbone_task"] == "censored_aware_horizon_classification"
