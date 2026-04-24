from __future__ import annotations

from types import ModuleType

import numpy as np
import pandas as pd

from survarena.automl.autogluon_backend import fit_autogluon_event_predictor, predict_event_probability
from survarena.methods.automl.autogluon_survival import AutoGluonSurvivalMethod
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


def test_autogluon_survival_method_is_registered_and_exposes_survival_predictions(monkeypatch, tmp_path) -> None:
    fake_module = ModuleType("autogluon.tabular")
    fake_module.TabularPredictor = FakeTabularPredictor
    monkeypatch.setitem(__import__("sys").modules, "autogluon", ModuleType("autogluon"))
    monkeypatch.setitem(__import__("sys").modules, "autogluon.tabular", fake_module)

    assert "autogluon_survival" in registered_method_ids()
    assert get_method_class("autogluon_survival") is AutoGluonSurvivalMethod

    model = AutoGluonSurvivalMethod(path=tmp_path / "ag")
    model.fit(
        pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([1, 0, 1, 0]),
    )

    risk = model.predict_risk(pd.DataFrame({"x": [5.0, 6.0]}))
    survival = model.predict_survival(pd.DataFrame({"x": [5.0, 6.0]}), np.asarray([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(risk, [0.75, 0.75])
    assert survival.shape == (2, 3)
    assert model.autogluon_metadata()["autogluon_best_model"] == "WeightedEnsemble_L2"
