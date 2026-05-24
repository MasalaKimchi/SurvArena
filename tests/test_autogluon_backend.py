from __future__ import annotations

from types import ModuleType

import numpy as np
import pandas as pd

from survarena.automl.autogluon_backend import fit_autogluon_event_predictor, predict_event_probability
from survarena.methods.automl.mitra_survival import MitraSurvivalFrozenMethod
from survarena.methods.registry import get_method_class, registered_method_ids


class FakeTabularPredictor:
    init_kwargs: dict[str, object] | None = None
    fit_kwargs: dict[str, object] | None = None

    def __init__(self, **kwargs) -> None:
        FakeTabularPredictor.init_kwargs = kwargs
        self.model_best = "WeightedEnsemble_L2"
        self.path = kwargs.get("path")

    def fit(self, **kwargs) -> "FakeTabularPredictor":
        FakeTabularPredictor.fit_kwargs = kwargs
        return self

    def leaderboard(self, silent: bool = True) -> pd.DataFrame:
        return pd.DataFrame(
            [{"model": "WeightedEnsemble_L2", "score_val": 0.75, "fit_time": 0.1}]
        )

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({0: np.full(len(frame), 0.25), 1: np.full(len(frame), 0.75)})


def test_autogluon_backend_passes_fit_controls_and_predicts_event_probability(monkeypatch, tmp_path) -> None:
    fake_module = ModuleType("autogluon.tabular")
    fake_module.TabularPredictor = FakeTabularPredictor
    monkeypatch.setitem(__import__("sys").modules, "autogluon", ModuleType("autogluon"))
    monkeypatch.setitem(__import__("sys").modules, "autogluon.tabular", fake_module)

    predictor, metadata = fit_autogluon_event_predictor(
        X_train=pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
        event_train=np.asarray([0, 1, 0, 1]),
        X_val=pd.DataFrame({"x": [4.0, 5.0]}),
        event_val=np.asarray([1, 0]),
        presets="best",
        time_limit=12.0,
        hyperparameter_tune_kwargs={"num_trials": 2},
        num_bag_folds=0,
        num_stack_levels=1,
        refit_full=False,
        path=tmp_path / "ag",
    )

    assert FakeTabularPredictor.init_kwargs["problem_type"] == "binary"
    assert FakeTabularPredictor.fit_kwargs is not None
    assert FakeTabularPredictor.fit_kwargs["presets"] == "best"
    assert FakeTabularPredictor.fit_kwargs["time_limit"] == 12.0
    assert FakeTabularPredictor.fit_kwargs["hyperparameter_tune_kwargs"] == {"num_trials": 2}
    assert "tuning_data" in FakeTabularPredictor.fit_kwargs
    assert metadata.best_model == "WeightedEnsemble_L2"
    assert metadata.model_count == 1
    np.testing.assert_allclose(predict_event_probability(predictor, pd.DataFrame({"x": [6.0, 7.0]})), [0.75, 0.75])


def test_mitra_survival_frozen_method_forces_mitra_hyperparameters(monkeypatch, tmp_path) -> None:
    fake_module = ModuleType("autogluon.tabular")
    fake_module.TabularPredictor = FakeTabularPredictor
    fake_sklearn_interface = ModuleType("autogluon.tabular.models.mitra.sklearn_interface")
    fake_sklearn_interface.MitraClassifier = object
    monkeypatch.setitem(__import__("sys").modules, "autogluon", ModuleType("autogluon"))
    monkeypatch.setitem(__import__("sys").modules, "autogluon.tabular", fake_module)
    monkeypatch.setitem(__import__("sys").modules, "autogluon.tabular.models", ModuleType("autogluon.tabular.models"))
    monkeypatch.setitem(__import__("sys").modules, "autogluon.tabular.models.mitra", ModuleType("autogluon.tabular.models.mitra"))
    monkeypatch.setitem(
        __import__("sys").modules,
        "autogluon.tabular.models.mitra.sklearn_interface",
        fake_sklearn_interface,
    )

    assert "mitra_survival_frozen" in registered_method_ids()
    assert get_method_class("mitra_survival_frozen") is MitraSurvivalFrozenMethod

    model = MitraSurvivalFrozenMethod(path=tmp_path / "mitra", hyperparameters=None, mitra_params={"fine_tune": True})
    model.fit(
        pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([1, 0, 1, 0]),
    )

    assert FakeTabularPredictor.fit_kwargs is not None
    assert FakeTabularPredictor.fit_kwargs["hyperparameters"] == {"MITRA": {"fine_tune": False}}
    np.testing.assert_allclose(model.predict_risk(pd.DataFrame({"x": [5.0, 6.0]})), [0.75, 0.75])
    assert model.predict_survival(pd.DataFrame({"x": [5.0, 6.0]}), np.asarray([1.0, 2.0, 3.0])).shape == (2, 3)


def test_mitra_survival_frozen_variant_forces_frozen_policy() -> None:
    frozen = MitraSurvivalFrozenMethod(time_limit=12, mitra_params={"ag.max_memory_usage_ratio": 1.1})

    assert get_method_class("mitra_survival_frozen") is MitraSurvivalFrozenMethod
    assert frozen.params["hyperparameters"] == {"MITRA": {"ag.max_memory_usage_ratio": 1.1, "fine_tune": False}}
    assert frozen.foundation_metadata()["foundation_backbone_training"] == "frozen"
