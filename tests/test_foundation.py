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
from survarena.methods.foundation.catalog import available_foundation_model_specs
from survarena.methods.foundation.readiness import FoundationRuntimeStatus, foundation_runtime_status

REPO_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _torch_backend_probe() -> tuple[bool, str | None]:
    completed = subprocess.run(
        [sys.executable, "-c", "import torch"],
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
    assert "tabicl_survival" in catalog["method_id"].tolist()
    assert "dependency_installed" in catalog.columns
    assert "runtime_ready" in catalog.columns
    assert "install_extra" in catalog.columns
    implemented = dict(zip(catalog["method_id"], catalog["implemented"], strict=False))
    assert implemented["tabpfn_survival"] is True
    assert implemented["tabicl_survival"] is False


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
