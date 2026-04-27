from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from survarena.methods.base import BaseSurvivalMethod
from survarena.methods.survival_utils import (
    fit_breslow_baseline_survival,
    normalize_aft_distribution_name,
    predict_aft_survival,
    predict_breslow_survival,
)


def _cox_signed_target(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    signed_time = np.asarray(time, dtype=np.float64).copy()
    signed_time[~np.asarray(event, dtype=bool)] *= -1.0
    return signed_time


def _as_float32_array(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float32)


def _as_tabular_input(X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return _as_float32_array(X)


def _catboost_cat_features(X: np.ndarray | pd.DataFrame) -> list[str] | list[int]:
    if not isinstance(X, pd.DataFrame):
        return []
    numeric_columns = set(X.select_dtypes(include=[np.number, "bool"]).columns.tolist())
    return [column for column in X.columns if column not in numeric_columns]


def _aft_bounds(
    time: np.ndarray,
    event: np.ndarray,
    *,
    censored_upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(time, dtype=np.float64).copy()
    upper = lower.copy()
    upper[~np.asarray(event, dtype=bool)] = censored_upper
    return lower, upper


def _xgboost_aft_matrix(
    X: np.ndarray | pd.DataFrame,
    *,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
):
    import xgboost as xgb

    matrix = xgb.DMatrix(_as_tabular_input(X))
    matrix.set_float_info("label_lower_bound", np.asarray(lower_bound, dtype=np.float32))
    matrix.set_float_info("label_upper_bound", np.asarray(upper_bound, dtype=np.float32))
    return matrix


def _cox_training_target(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    return _cox_signed_target(np.asarray(time, dtype=np.float64), np.asarray(event, dtype=np.int32))


def _seed_param(params: dict[str, Any]) -> int | None:
    return None if params.get("seed") is None else int(params["seed"])


def _early_stopping_rounds(params: dict[str, Any], *, has_validation: bool) -> int | None:
    if not has_validation or params.get("early_stopping_rounds") is None:
        return None
    return int(params["early_stopping_rounds"])


def _xgboost_common_tree_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "learning_rate": float(params["learning_rate"]),
        "max_depth": int(params["max_depth"]),
        "min_child_weight": float(params["min_child_weight"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "max_bin": int(params["max_bin"]),
        "tree_method": str(params["tree_method"]),
        "device": str(params["device"]),
    }


def _xgboost_sklearn_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        **_xgboost_common_tree_params(params),
        "reg_lambda": float(params["reg_lambda"]),
        "reg_alpha": float(params["reg_alpha"]),
        "n_jobs": 1,
        "random_state": _seed_param(params),
        "verbosity": 0,
    }


def _xgboost_train_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        **_xgboost_common_tree_params(params),
        "lambda": float(params["reg_lambda"]),
        "alpha": float(params["reg_alpha"]),
        "verbosity": 0,
        "seed": _seed_param(params),
        "nthread": 1,
    }


def _catboost_regressor_params(params: dict[str, Any], *, loss_function: str, eval_metric: str) -> dict[str, Any]:
    return {
        "loss_function": loss_function,
        "eval_metric": eval_metric,
        "learning_rate": float(params["learning_rate"]),
        "iterations": int(params["iterations"]),
        "depth": int(params["depth"]),
        "l2_leaf_reg": float(params["l2_leaf_reg"]),
        "random_strength": float(params["random_strength"]),
        "rsm": float(params["rsm"]),
        "bootstrap_type": "Bernoulli",
        "subsample": float(params["subsample"]),
        "min_data_in_leaf": int(params["min_data_in_leaf"]),
        "thread_count": -1,
        "random_seed": _seed_param(params),
        "verbose": False,
        "allow_writing_files": False,
    }


def _catboost_fit_kwargs(
    params: dict[str, Any],
    *,
    has_validation: bool,
    eval_set: tuple[np.ndarray | pd.DataFrame | None, np.ndarray] | None,
    cat_features: list[str] | list[int],
) -> dict[str, Any]:
    fit_kwargs: dict[str, Any] = {"use_best_model": has_validation}
    if has_validation:
        fit_kwargs["eval_set"] = eval_set
        early_stopping_rounds = _early_stopping_rounds(params, has_validation=has_validation)
        if early_stopping_rounds is not None:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
    if cat_features:
        fit_kwargs["cat_features"] = cat_features
    return fit_kwargs


def _has_validation_data(
    X_val: np.ndarray | pd.DataFrame | None,
    time_val: np.ndarray | None,
    event_val: np.ndarray | None,
) -> bool:
    return X_val is not None and time_val is not None and event_val is not None


class _BaseCalibratedCoxBoostingMethod(BaseSurvivalMethod):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.model = None
        self.baseline_event_times_: np.ndarray | None = None
        self.baseline_survival_: np.ndarray | None = None

    def _fit_breslow_baseline(
        self,
        *,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
    ) -> None:
        train_risk = self.predict_risk(X_train)
        self.baseline_event_times_, self.baseline_survival_ = fit_breslow_baseline_survival(
            time_train=np.asarray(time_train, dtype=np.float64),
            event_train=np.asarray(event_train, dtype=np.int32),
            train_risk_scores=train_risk,
        )

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.baseline_event_times_ is None or self.baseline_survival_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before prediction.")
        return predict_breslow_survival(
            risk_scores=self.predict_risk(X),
            times=np.asarray(times, dtype=np.float64),
            baseline_event_times=self.baseline_event_times_,
            baseline_survival=self.baseline_survival_,
        )


class _BaseAFTBoostingMethod(BaseSurvivalMethod):
    _distribution_param: str
    _scale_param: str

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.model = None

    def _predict_location(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return -self._predict_location(X)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        return predict_aft_survival(
            location_scores=self._predict_location(X),
            times=np.asarray(times, dtype=np.float64),
            distribution=str(self.params[self._distribution_param]),
            scale=float(self.params[self._scale_param]),
        )


class XGBoostCoxMethod(_BaseCalibratedCoxBoostingMethod):
    def __init__(
        self,
        learning_rate: float = 0.05,
        n_estimators: int = 300,
        max_depth: int = 4,
        min_child_weight: float = 1.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        max_bin: int = 256,
        tree_method: str = "hist",
        device: str = "cpu",
        early_stopping_rounds: int | None = 25,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            max_bin=max_bin,
            tree_method=tree_method,
            device=device,
            early_stopping_rounds=early_stopping_rounds,
            seed=seed,
        )

    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "XGBoostCoxMethod":
        import xgboost as xgb

        has_validation = _has_validation_data(X_val, time_val, event_val)
        self.model = xgb.XGBRegressor(
            objective="survival:cox",
            eval_metric="cox-nloglik",
            **_xgboost_sklearn_params(self.params),
            n_estimators=int(self.params["n_estimators"]),
            early_stopping_rounds=_early_stopping_rounds(self.params, has_validation=has_validation),
        )

        fit_kwargs: dict[str, Any] = {"verbose": False}
        if has_validation:
            fit_kwargs["eval_set"] = [
                (
                    _as_float32_array(X_val),
                    _cox_training_target(time_val, event_val),
                )
            ]

        self.model.fit(
            _as_float32_array(X_train),
            _cox_training_target(time_train, event_train),
            **fit_kwargs,
        )
        self._fit_breslow_baseline(X_train=X_train, time_train=time_train, event_train=event_train)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("XGBoostCoxMethod must be fit before prediction.")
        return self.model.predict(_as_float32_array(X), output_margin=True).astype(np.float64)


class XGBoostAFTMethod(_BaseAFTBoostingMethod):
    _distribution_param = "aft_loss_distribution"
    _scale_param = "aft_loss_distribution_scale"

    def __init__(
        self,
        learning_rate: float = 0.05,
        n_estimators: int = 300,
        max_depth: int = 4,
        min_child_weight: float = 1.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        max_bin: int = 256,
        tree_method: str = "hist",
        device: str = "cpu",
        aft_loss_distribution: str = "normal",
        aft_loss_distribution_scale: float = 1.0,
        early_stopping_rounds: int | None = 25,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            max_bin=max_bin,
            tree_method=tree_method,
            device=device,
            aft_loss_distribution=aft_loss_distribution,
            aft_loss_distribution_scale=aft_loss_distribution_scale,
            early_stopping_rounds=early_stopping_rounds,
            seed=seed,
        )

    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "XGBoostAFTMethod":
        import xgboost as xgb

        has_validation = _has_validation_data(X_val, time_val, event_val)
        train_lower, train_upper = _aft_bounds(time_train, event_train, censored_upper=np.inf)
        dtrain = _xgboost_aft_matrix(X_train, lower_bound=train_lower, upper_bound=train_upper)
        evals = [(dtrain, "train")]
        if has_validation:
            val_lower, val_upper = _aft_bounds(time_val, event_val, censored_upper=np.inf)
            dval = _xgboost_aft_matrix(X_val, lower_bound=val_lower, upper_bound=val_upper)
            evals.append((dval, "valid"))

        self.model = xgb.train(
            params={
                "objective": "survival:aft",
                "eval_metric": "aft-nloglik",
                **_xgboost_train_params(self.params),
                "aft_loss_distribution": normalize_aft_distribution_name(self.params["aft_loss_distribution"]),
                "aft_loss_distribution_scale": float(self.params["aft_loss_distribution_scale"]),
            },
            dtrain=dtrain,
            num_boost_round=int(self.params["n_estimators"]),
            evals=evals,
            early_stopping_rounds=_early_stopping_rounds(self.params, has_validation=has_validation),
            verbose_eval=False,
        )
        return self

    def _predict_location(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        if self.model is None:
            raise RuntimeError("XGBoostAFTMethod must be fit before prediction.")
        dmatrix = xgb.DMatrix(_as_tabular_input(X))
        return self.model.predict(dmatrix, output_margin=True).astype(np.float64)


class CatBoostCoxMethod(_BaseCalibratedCoxBoostingMethod):
    def __init__(
        self,
        learning_rate: float = 0.05,
        iterations: int = 300,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
        rsm: float = 1.0,
        subsample: float = 0.8,
        min_data_in_leaf: int = 1,
        early_stopping_rounds: int | None = 25,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            iterations=iterations,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_strength=random_strength,
            rsm=rsm,
            subsample=subsample,
            min_data_in_leaf=min_data_in_leaf,
            early_stopping_rounds=early_stopping_rounds,
            seed=seed,
        )
        self.cat_features_: list[str] | list[int] = []

    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "CatBoostCoxMethod":
        from catboost import CatBoostRegressor

        X_train_input = _as_tabular_input(X_train)
        X_val_input = None if X_val is None else _as_tabular_input(X_val)
        has_validation = _has_validation_data(X_val, time_val, event_val)
        self.cat_features_ = _catboost_cat_features(X_train_input)
        self.model = CatBoostRegressor(
            **_catboost_regressor_params(self.params, loss_function="Cox", eval_metric="Cox"),
        )

        eval_set = None
        if has_validation:
            eval_set = (
                X_val_input,
                _cox_training_target(time_val, event_val),
            )
        fit_kwargs = _catboost_fit_kwargs(
            self.params,
            has_validation=has_validation,
            eval_set=eval_set,
            cat_features=self.cat_features_,
        )

        self.model.fit(
            X_train_input,
            _cox_training_target(time_train, event_train),
            **fit_kwargs,
        )
        self._fit_breslow_baseline(X_train=X_train, time_train=time_train, event_train=event_train)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("CatBoostCoxMethod must be fit before prediction.")
        return self.model.predict(_as_tabular_input(X), prediction_type="RawFormulaVal").astype(np.float64)


class CatBoostSurvivalAFTMethod(_BaseAFTBoostingMethod):
    _distribution_param = "dist"
    _scale_param = "scale"

    def __init__(
        self,
        learning_rate: float = 0.05,
        iterations: int = 300,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
        rsm: float = 1.0,
        subsample: float = 0.8,
        min_data_in_leaf: int = 1,
        dist: str = "Normal",
        scale: float = 1.0,
        early_stopping_rounds: int | None = 25,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            iterations=iterations,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_strength=random_strength,
            rsm=rsm,
            subsample=subsample,
            min_data_in_leaf=min_data_in_leaf,
            dist=dist,
            scale=scale,
            early_stopping_rounds=early_stopping_rounds,
            seed=seed,
        )
        self.cat_features_: list[str] | list[int] = []

    def _resolved_dist(self) -> str:
        mapping = {"normal": "Normal", "logistic": "Logistic", "extreme": "Extreme"}
        return mapping[normalize_aft_distribution_name(self.params["dist"])]

    def _loss_spec(self) -> str:
        return f"SurvivalAft:dist={self._resolved_dist()};scale={float(self.params['scale'])}"

    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "CatBoostSurvivalAFTMethod":
        from catboost import CatBoostRegressor

        X_train_input = _as_tabular_input(X_train)
        X_val_input = None if X_val is None else _as_tabular_input(X_val)
        has_validation = _has_validation_data(X_val, time_val, event_val)
        self.cat_features_ = _catboost_cat_features(X_train_input)
        train_lower, train_upper = _aft_bounds(time_train, event_train, censored_upper=-1.0)
        loss_spec = self._loss_spec()
        self.model = CatBoostRegressor(
            **_catboost_regressor_params(self.params, loss_function=loss_spec, eval_metric=loss_spec),
        )

        eval_set = None
        if has_validation:
            val_lower, val_upper = _aft_bounds(time_val, event_val, censored_upper=-1.0)
            eval_set = (X_val_input, np.column_stack([val_lower, val_upper]))
        fit_kwargs = _catboost_fit_kwargs(
            self.params,
            has_validation=has_validation,
            eval_set=eval_set,
            cat_features=self.cat_features_,
        )

        self.model.fit(
            X_train_input,
            np.column_stack([train_lower, train_upper]),
            **fit_kwargs,
        )
        return self

    def _predict_location(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("CatBoostSurvivalAFTMethod must be fit before prediction.")
        return self.model.predict(_as_tabular_input(X), prediction_type="RawFormulaVal").astype(np.float64)
