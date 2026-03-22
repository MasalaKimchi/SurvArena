from __future__ import annotations

import sys
import types

import numpy as np

from src.automl.presets import resolve_preset
from src.methods.foundation.tabpfn_survival import TabPFNSurvivalMethod


class _FakeCoxPHSurvivalAnalysis:
    def __init__(self, alpha: float = 0.0001) -> None:
        self.alpha = alpha
        self.fitted_X = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_FakeCoxPHSurvivalAnalysis":
        self.fitted_X = np.asarray(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X).sum(axis=1)

    def predict_survival_function(self, X: np.ndarray) -> list[object]:
        risks = self.predict(X)

        class _Fn:
            def __init__(self, risk: float) -> None:
                self.risk = max(float(risk), 0.1)

            def __call__(self, times: np.ndarray) -> np.ndarray:
                return np.exp(-self.risk * np.asarray(times, dtype=float))

        return [_Fn(risk) for risk in risks]


def test_tabpfn_survival_supports_explicit_model_versions(monkeypatch) -> None:
    class FakeRegressor:
        init_kwargs: dict[str, object] | None = None
        created_version: object | None = None

        def __init__(self, **kwargs) -> None:
            FakeRegressor.init_kwargs = kwargs

        @classmethod
        def create_default_for_version(cls, version: object, **kwargs):
            cls.created_version = version
            return cls(**kwargs)

        def fit(self, X: np.ndarray, y: np.ndarray) -> "FakeRegressor":
            return self

        def get_embeddings(self, X: np.ndarray, data_source: str = "test") -> np.ndarray:
            X = np.asarray(X, dtype=np.float32)
            return np.stack([X, X + 1.0], axis=0)

    fake_tabpfn = types.ModuleType("tabpfn")
    fake_tabpfn.TabPFNRegressor = FakeRegressor
    fake_constants = types.ModuleType("tabpfn.constants")
    fake_constants.ModelVersion = types.SimpleNamespace(V2="v2", V2_5="v2.5")
    fake_linear_model = types.ModuleType("sksurv.linear_model")
    fake_linear_model.CoxPHSurvivalAnalysis = _FakeCoxPHSurvivalAnalysis

    monkeypatch.setitem(sys.modules, "tabpfn", fake_tabpfn)
    monkeypatch.setitem(sys.modules, "tabpfn.constants", fake_constants)
    monkeypatch.setitem(sys.modules, "sksurv.linear_model", fake_linear_model)

    method = TabPFNSurvivalMethod(model_version="v2.5", n_estimators=4)
    method.fit(
        X_train=np.asarray([[0.1, 0.2], [0.4, 0.5]], dtype=float),
        time_train=np.asarray([1.0, 2.0], dtype=float),
        event_train=np.asarray([1, 1], dtype=int),
    )

    assert FakeRegressor.created_version == "v2.5"
    assert FakeRegressor.init_kwargs is not None
    assert FakeRegressor.init_kwargs["n_estimators"] == 4


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
