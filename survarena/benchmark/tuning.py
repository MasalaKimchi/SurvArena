from __future__ import annotations

from typing import Any, Callable


_RUNTIME_ONLY_METHOD_PARAMS = {"seed"}


def method_param_suggestions(trial: Any, method_cfg: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in method_cfg.get("search_space", {}).items():
        spec_type = spec["type"]
        if spec_type == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        elif spec_type == "float":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif spec_type == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif spec_type == "int_or_none":
            use_none = trial.suggest_categorical(f"{name}_is_none", [True, False])
            params[name] = None if use_none else trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        else:
            raise ValueError(f"Unsupported search spec type: {spec_type}")
    if not params:
        params = dict(method_cfg.get("default_params", {}))
    return params


def resolve_runtime_method_params(params: dict[str, Any], *, seed: int) -> dict[str, Any]:
    resolved = dict(params)
    resolved["seed"] = int(seed)
    return resolved


def prepare_inner_cv_cache(
    *,
    method_id: str,
    X_train: Any,
    time_train: Any,
    event_train: Any,
    inner_folds: int,
    seed: int,
) -> list[dict[str, Any]]:
    import importlib

    from survarena.data.preprocess import TabularPreprocessor
    from survarena.methods.preprocessing import finalize_preprocessed_features, method_preprocessor_kwargs

    StratifiedKFold = importlib.import_module("sklearn.model_selection").StratifiedKFold

    fold_cache: list[dict[str, Any]] = []
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)

    for train_idx, val_idx in skf.split(X_train, event_train):
        pre = TabularPreprocessor(**method_preprocessor_kwargs(method_id))
        X_train_fold = finalize_preprocessed_features(method_id, pre.fit_transform(X_train.iloc[train_idx]))
        X_val_fold = finalize_preprocessed_features(method_id, pre.transform(X_train.iloc[val_idx]))
        fold_cache.append(
            {
                "X_train": X_train_fold,
                "X_val": X_val_fold,
                "time_train": time_train[train_idx],
                "event_train": event_train[train_idx],
                "time_val": time_train[val_idx],
                "event_val": event_train[val_idx],
            }
        )
    return fold_cache


def _metric_optimization_direction(primary_metric: str) -> str:
    if primary_metric in {"harrell_c", "uno_c"}:
        return "maximize"
    raise ValueError(f"Unsupported primary metric for tuning direction: {primary_metric}")


def _searchable_default_params(method_cfg: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(method_cfg.get("default_params", {}))
    return {key: value for key, value in defaults.items() if key not in _RUNTIME_ONLY_METHOD_PARAMS}


def _is_better_score(candidate: float, reference: float, *, direction: str) -> bool:
    if direction == "maximize":
        return candidate > reference
    if direction == "minimize":
        return candidate < reference
    raise ValueError(f"Unsupported optimization direction: {direction}")


def _inner_cv_evaluate(
    *,
    method_id: str,
    params: dict[str, Any],
    fold_cache: list[dict[str, Any]],
    primary_metric: str,
    metric_bundle_callback: Callable[[dict[str, Any], Any, Any], dict[str, float]] | None = None,
) -> dict[str, Any]:
    import numpy as np

    from survarena.evaluation.metrics import compute_primary_metric_score
    from survarena.methods.registry import get_method_class

    method_cls = get_method_class(method_id)
    scores: list[float] = []
    metric_rows: list[dict[str, float]] = []
    for fold_data in fold_cache:
        model = method_cls(**params)
        model.fit(
            fold_data["X_train"],
            fold_data["time_train"],
            fold_data["event_train"],
            fold_data["X_val"],
            fold_data["time_val"],
            fold_data["event_val"],
        )
        risk = model.predict_risk(fold_data["X_val"])
        score = compute_primary_metric_score(
            primary_metric=primary_metric,
            train_time=fold_data["time_train"],
            train_event=fold_data["event_train"],
            eval_time=fold_data["time_val"],
            eval_event=fold_data["event_val"],
            eval_risk_scores=risk,
        )
        scores.append(float(score))
        if metric_bundle_callback is not None:
            metric_rows.append(metric_bundle_callback(fold_data, model, risk))

    result: dict[str, Any] = {"primary_score": float(np.mean(scores))}
    if metric_bundle_callback is not None:
        result["metric_rows"] = metric_rows
    return result


def select_hyperparameters(
    *,
    method_id: str,
    method_cfg: dict[str, Any],
    fold_cache: list[dict[str, Any]],
    primary_metric: str,
    n_trials: int,
    seed: int,
    timeout_seconds: float | None = None,
    quiet: bool = False,
    metric_bundle_callback: Callable[[dict[str, Any], Any, Any], dict[str, float]] | None = None,
) -> dict[str, Any]:
    from survarena.utils.quiet import quiet_training_output

    with quiet_training_output(enabled=quiet):
        defaults = _searchable_default_params(method_cfg)
        direction = _metric_optimization_direction(primary_metric)
        default_eval = _inner_cv_evaluate(
            method_id=method_id,
            params=resolve_runtime_method_params(defaults, seed=seed),
            fold_cache=fold_cache,
            primary_metric=primary_metric,
            metric_bundle_callback=metric_bundle_callback,
        )
        default_score = float(default_eval["primary_score"])
        default_metric_rows = default_eval.get("metric_rows")
        return {
            "best_params": defaults,
            "best_score": default_score,
            "n_trials_completed": 0,
            "best_metric_rows": default_metric_rows,
        }


tune_hyperparameters = select_hyperparameters
