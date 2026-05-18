from __future__ import annotations

import subprocess
import sys
import types
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from survarena.api.predictor import SurvivalPredictor
from survarena.automl.presets import resolve_preset
from survarena.config import read_yaml
from survarena.methods.foundation.catalog import available_foundation_model_specs
from survarena.methods.foundation.readiness import FoundationRuntimeStatus, foundation_runtime_status

REPO_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _torch_backend_probe() -> tuple[bool, str | None]:
    completed = subprocess.run(
        [sys.executable, "-c", "import torch; import torchsurv.loss.cox"],
        check=False,
        capture_output=True,
        cwd=str(REPO_ROOT),
        text=True,
    )
    if completed.returncode == 0:
        return True, None
    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    detail = stderr or stdout or f"subprocess exited with code {completed.returncode}"
    return False, detail


def test_foundation_model_catalog_exposes_current_and_planned_backbones() -> None:
    predictor = SurvivalPredictor(label_time="time", label_event="event")

    catalog = predictor.foundation_model_catalog()

    assert "tabpfn_survival" in catalog["method_id"].tolist()
    assert "mitra_survival" in catalog["method_id"].tolist()
    assert "tabicl_survival" in catalog["method_id"].tolist()
    assert "dependency_installed" in catalog.columns
    assert "runtime_ready" in catalog.columns
    assert "install_extra" in catalog.columns
    implemented = dict(zip(catalog["method_id"], catalog["implemented"], strict=False))
    assert implemented["tabpfn_survival"] is True
    assert implemented["mitra_survival"] is True
    assert implemented["tabicl_survival"] is False


def test_tabpfn_method_config_does_not_search_unsupported_finetuning() -> None:
    spec = next(item for item in available_foundation_model_specs() if item.method_id == "tabpfn_survival")
    method_cfg = read_yaml(REPO_ROOT / "configs" / "methods" / "tabpfn_survival.yaml")

    assert spec.supports_finetune is False
    assert method_cfg["search_space"]["backbone_training"]["choices"] == ["frozen"]
    assert method_cfg["default_params"]["model_version"] == "v2.5"
    assert method_cfg["default_params"]["backbone_task"] == "classification_event"
    assert "n_estimators_finetune" not in method_cfg["search_space"]


def test_tabpfn_frozen_smoke_config_forces_bounded_defaults() -> None:
    benchmark_cfg = read_yaml(REPO_ROOT / "configs" / "benchmark" / "tabpfn_frozen_smoke.yaml")
    override = benchmark_cfg["hpo"]["method_overrides"]["tabpfn_survival"]

    assert benchmark_cfg["comparison_modes"] == ["no_hpo"]
    assert benchmark_cfg["hpo"]["enabled"] is False
    assert benchmark_cfg["datasets"][0] == "whas500"
    assert benchmark_cfg["methods"] == ["coxph", "rsf", "tabpfn_survival"]
    assert override["search_space"] is None
    assert override["default_params"] == {
        "model_version": "v2.5",
        "backbone_task": "classification_event",
        "backbone_training": "frozen",
        "n_estimators": 1,
        "n_estimators_final_inference": 1,
        "hidden_layers": "64",
        "dropout": 0.0,
        "max_epochs": 25,
        "patience": 5,
    }


def test_foundation_elo_config_uses_budgeted_method_variants() -> None:
    benchmark_cfg = read_yaml(REPO_ROOT / "configs" / "benchmark" / "foundation_elo_v1.yaml")
    overrides = benchmark_cfg["hpo"]["method_overrides"]

    assert benchmark_cfg["primary_metric"] == "uno_c"
    assert benchmark_cfg["comparison_modes"] == ["no_hpo"]
    assert benchmark_cfg["outer_folds"] == 3
    assert benchmark_cfg["outer_repeats"] == 3
    assert "tabpfn_survival_classifier" in benchmark_cfg["methods"]
    assert "tabpfn_survival_regressor" in benchmark_cfg["methods"]
    assert "mitra_survival_frozen" in benchmark_cfg["methods"]
    assert "mitra_survival_finetune" not in overrides
    assert overrides["tabpfn_survival_classifier"]["default_params"]["backbone_task"] == "classification_event"
    assert overrides["tabpfn_survival_regressor"]["default_params"]["backbone_task"] == "regression_time"
    assert overrides["mitra_survival_frozen"]["default_params"]["time_limit"] == 120
    assert "fine-tuning is intentionally excluded" in benchmark_cfg["notes"]


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
            return self

        def get_embeddings(self, X: np.ndarray, data_source: str = "test") -> np.ndarray:
            X = np.asarray(X, dtype=np.float32)
            return np.stack([X, X + 1.0], axis=0)

    fake_tabpfn = types.ModuleType("tabpfn")
    fake_tabpfn.TabPFNClassifier = FakeBackbone
    fake_tabpfn.TabPFNRegressor = FakeBackbone
    fake_constants = types.ModuleType("tabpfn.constants")
    fake_constants.ModelVersion = types.SimpleNamespace(V2="v2", V2_5="v2.5")

    monkeypatch.setitem(sys.modules, "tabpfn", fake_tabpfn)
    monkeypatch.setitem(sys.modules, "tabpfn.constants", fake_constants)

    method = TabPFNSurvivalMethod(model_version="v2.5", n_estimators=4)
    estimator = method._build_backbone(n_estimators=4, fit_mode="batched")

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
    torch_ready, probe_detail = _torch_backend_probe()
    if not torch_ready:
        pytest.skip(f"Torch foundation probe failed in this environment: {probe_detail}")

    from survarena.methods.foundation.tabpfn_survival import TabPFNSurvivalMethod

    class FakeBackbone:
        init_kwargs: list[dict[str, object]] = []

        def __init__(self, **kwargs) -> None:
            self.kwargs = dict(kwargs)
            FakeBackbone.init_kwargs.append(self.kwargs)

        @classmethod
        def create_default_for_version(cls, version: object, **kwargs):
            return cls(model_version=version, **kwargs)

        def fit(self, X: np.ndarray, y: np.ndarray) -> "FakeBackbone":
            self.X_fit_ = np.asarray(X, dtype=np.float32)
            self.y_fit_ = np.asarray(y)
            return self

        def get_embeddings(self, X: np.ndarray, data_source: str = "test") -> np.ndarray:
            X_np = np.asarray(X, dtype=np.float32)
            base = np.column_stack(
                [
                    X_np[:, 0],
                    X_np[:, 1],
                    X_np[:, 2],
                    X_np.mean(axis=1),
                ]
            ).astype(np.float32)
            n_estimators = int(self.kwargs.get("n_estimators", 1))
            return np.stack([base + (0.01 * i) for i in range(n_estimators)], axis=0)

    fake_tabpfn = types.ModuleType("tabpfn")
    fake_tabpfn.TabPFNClassifier = FakeBackbone
    fake_tabpfn.TabPFNRegressor = FakeBackbone
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
            [0.10, 1.00, 0.20],
            [0.20, 0.90, 0.10],
            [0.35, 0.80, 0.30],
            [0.40, 0.70, 0.25],
            [0.55, 0.60, 0.40],
            [0.65, 0.50, 0.35],
            [0.70, 0.40, 0.50],
            [0.80, 0.30, 0.45],
            [0.85, 0.25, 0.60],
            [0.95, 0.20, 0.55],
            [1.05, 0.15, 0.70],
            [1.10, 0.10, 0.65],
        ],
        dtype=np.float32,
    )
    time = np.asarray([4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 20, 22], dtype=np.float64)
    event = np.asarray([1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)

    method = TabPFNSurvivalMethod(
        backbone_training="frozen",
        n_estimators=1,
        n_estimators_final_inference=1,
        hidden_layers="64",
        dropout=0.0,
        max_epochs=2,
        patience=1,
        device="cpu",
        seed=13,
    )

    method.fit(X, time, event)
    risk = method.predict_risk(X[:4])
    survival = method.predict_survival(X[:4], np.asarray([5.0, 10.0, 15.0]))

    assert FakeBackbone.init_kwargs[-1]["n_estimators"] == 1
    assert method.finetuned_estimator_ is None
    assert risk.shape == (4,)
    assert survival.shape == (4, 3)
    assert np.isfinite(risk).all()
    assert np.isfinite(survival).all()
    assert ((survival >= 0.0) & (survival <= 1.0)).all()


def test_tabpfn_survival_regressor_variant_uses_regression_surrogate() -> None:
    from survarena.methods.foundation.tabpfn_survival import TabPFNSurvivalRegressorMethod
    from survarena.methods.registry import get_method_class

    method = TabPFNSurvivalRegressorMethod()
    target = method._build_surrogate_target(
        np.asarray([1.0, 3.0, 7.0]),
        np.asarray([1, 0, 1]),
    )

    assert get_method_class("tabpfn_survival_regressor") is TabPFNSurvivalRegressorMethod
    assert method.params["backbone_task"] == "regression_time"
    assert method.params["backbone_training"] == "frozen"
    np.testing.assert_allclose(target, np.log1p([1.0, 3.0, 7.0]).astype(np.float32))
    assert method.foundation_metadata()["foundation_backbone_task"] == "regression_time"


def test_tabpfn_survival_classifier_variant_uses_event_surrogate() -> None:
    from survarena.methods.foundation.tabpfn_survival import TabPFNSurvivalClassifierMethod
    from survarena.methods.registry import get_method_class

    method = TabPFNSurvivalClassifierMethod()
    target = method._build_surrogate_target(
        np.asarray([1.0, 3.0, 7.0]),
        np.asarray([1, 0, 1]),
    )

    assert get_method_class("tabpfn_survival_classifier") is TabPFNSurvivalClassifierMethod
    assert method.params["backbone_task"] == "classification_event"
    assert method.params["backbone_training"] == "frozen"
    np.testing.assert_array_equal(target, np.asarray([1, 0, 1], dtype=np.int32))
    assert method.foundation_metadata()["foundation_backbone_task"] == "classification_event"


def test_tabpfn_embedding_extraction_supports_tensor_outputs_without_optional_kwarg() -> None:
    torch_ready, probe_detail = _torch_backend_probe()
    if not torch_ready:
        pytest.skip(f"Torch backend probe failed in this environment: {probe_detail}")

    import torch
    from survarena.methods.foundation.tabpfn_survival import TabPFNSurvivalMethod

    class FakeExecutor:
        def __init__(self) -> None:
            self.feature_schema_list = [[object()]]
            self.ensemble_configs = [[object()]]

        def use_torch_inference_mode(self, use_inference: bool = False) -> None:
            return None

        def iter_outputs(self, X, *, autocast: bool):  # noqa: ANN001
            for _ in X:
                yield torch.ones((4, 1, 10), dtype=torch.float32), []

    class FakeEstimator:
        def __init__(self) -> None:
            self.executor_ = FakeExecutor()
            self.use_autocast_ = False

    method = TabPFNSurvivalMethod(aggregate_estimators="mean")
    method.finetuned_estimator_ = FakeEstimator()
    method.device_ = torch.device("cpu")

    embeddings = method._extract_batch_embeddings(
        [torch.zeros((4, 3), dtype=torch.float32), torch.zeros((4, 3), dtype=torch.float32)]
    )

    assert embeddings.shape == (4, 10)
    assert torch.allclose(embeddings, torch.ones((4, 10), dtype=torch.float32))
