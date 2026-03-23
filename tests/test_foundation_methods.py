from __future__ import annotations

import sys
import types

import numpy as np

from src.automl.presets import resolve_preset
from src.methods.foundation.mitra_survival import MitraSurvivalMethod
from src.methods.foundation.tabpfn_survival import TabPFNSurvivalMethod


def test_tabpfn_survival_supports_explicit_model_versions(monkeypatch) -> None:
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


def test_mitra_survival_passes_finetune_controls_to_autogluon(monkeypatch) -> None:
    class FakeTabularPredictor:
        init_kwargs: dict[str, object] | None = None
        fit_kwargs: dict[str, object] | None = None

        def __init__(self, **kwargs) -> None:
            FakeTabularPredictor.init_kwargs = kwargs

        def fit(self, **kwargs) -> "FakeTabularPredictor":
            FakeTabularPredictor.fit_kwargs = kwargs
            return self

        def predict(self, frame) -> np.ndarray:
            values = np.asarray(frame.to_numpy(dtype=np.float32), dtype=np.float32)
            return values.sum(axis=1) * 0.25 + 0.5

    fake_autogluon = types.ModuleType("autogluon")
    fake_tabular = types.ModuleType("autogluon.tabular")
    fake_tabular.TabularPredictor = FakeTabularPredictor

    monkeypatch.setitem(sys.modules, "autogluon", fake_autogluon)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", fake_tabular)

    method = MitraSurvivalMethod(
        backbone_training="finetune",
        fine_tune_steps=123,
    )
    method.fit(
        X_train=np.asarray([[0.1, 0.2], [0.4, 0.5], [0.2, 0.3]], dtype=float),
        time_train=np.asarray([1.0, 2.0, 3.0], dtype=float),
        event_train=np.asarray([1, 1, 0], dtype=int),
    )

    assert FakeTabularPredictor.init_kwargs is not None
    assert FakeTabularPredictor.init_kwargs["problem_type"] == "regression"
    assert FakeTabularPredictor.fit_kwargs is not None
    assert FakeTabularPredictor.fit_kwargs["hyperparameters"]["MITRA"]["fine_tune"] is True
    assert FakeTabularPredictor.fit_kwargs["hyperparameters"]["MITRA"]["fine_tune_steps"] == 123

    risk = method.predict_risk(np.asarray([[0.5, 0.2], [0.1, 0.9]], dtype=float))
    survival = method.predict_survival(
        np.asarray([[0.5, 0.2], [0.1, 0.9]], dtype=float),
        np.asarray([1.0, 2.0], dtype=float),
    )

    assert risk.shape == (2,)
    assert survival.shape == (2, 2)
    assert np.all(survival > 0.0)
    assert np.all(survival <= 1.0)


def test_foundation_preset_adds_mitra_when_dependency_exists(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.automl.presets._has_dependency",
        lambda module_name: module_name in {"tabpfn", "autogluon.tabular"},
    )

    preset = resolve_preset(
        "foundation",
        n_rows=1000,
        n_features=50,
        event_count=200,
        event_fraction=0.2,
    )

    assert "tabpfn_survival" in preset.method_ids
    assert "mitra_survival" in preset.method_ids


def test_tabpfn_embedding_extraction_supports_tensor_outputs_without_optional_kwarg() -> None:
    import torch

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
