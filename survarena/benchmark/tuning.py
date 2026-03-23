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

    StratifiedKFold = importlib.import_module("sklearn.model_selection").StratifiedKFold

    fold_cache: list[dict[str, Any]] = []
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)

    for train_idx, val_idx in skf.split(X_train, event_train):
        pre = TabularPreprocessor(scale_numeric=(method_id != "rsf"))
        X_train_fold = pre.fit_transform(X_train.iloc[train_idx]).to_numpy()
        X_val_fold = pre.transform(X_train.iloc[val_idx]).to_numpy()
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
    from survarena.methods.registry import method_registry

    registry = method_registry()
    scores: list[float] = []
    metric_rows: list[dict[str, float]] = []
    for fold_data in fold_cache:
        model = registry[method_id](**params)
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


def tune_hyperparameters(
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
    import importlib

    optuna = importlib.import_module("optuna")

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
        if not method_cfg.get("search_space") or n_trials <= 0:
            return {
                "best_params": defaults,
                "best_score": default_score,
                "n_trials_completed": 0,
                "best_metric_rows": default_metric_rows,
            }

        def objective(trial: Any) -> float:
            params = method_param_suggestions(trial, method_cfg)
            trial_eval = _inner_cv_evaluate(
                method_id=method_id,
                params=resolve_runtime_method_params(params, seed=seed),
                fold_cache=fold_cache,
                primary_metric=primary_metric,
                metric_bundle_callback=metric_bundle_callback,
            )
            metric_rows = trial_eval.get("metric_rows")
            if metric_rows is not None:
                trial.set_user_attr("metric_rows", metric_rows)
            return float(trial_eval["primary_score"])

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_seconds,
            show_progress_bar=False,
        )

    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        return {
            "best_params": defaults,
            "best_score": default_score,
            "n_trials_completed": 0,
            "best_metric_rows": default_metric_rows,
        }

    best_trial = study.best_trial
    best_trial_score = float(study.best_value)
    if _is_better_score(default_score, best_trial_score, direction=direction):
        return {
            "best_params": defaults,
            "best_score": default_score,
            "n_trials_completed": len(completed_trials),
            "best_metric_rows": default_metric_rows,
        }

    allowed_keys = set(method_cfg.get("search_space", {}).keys())
    best_trial_params = dict(best_trial.params if best_trial.params else {})
    filtered_trial_params = {k: v for k, v in best_trial_params.items() if k in allowed_keys}
    best_params = dict(defaults)
    best_params.update(filtered_trial_params)
    return {
        "best_params": best_params,
        "best_score": best_trial_score,
        "n_trials_completed": len(completed_trials),
        "best_metric_rows": best_trial.user_attrs.get("metric_rows", default_metric_rows),
    }
