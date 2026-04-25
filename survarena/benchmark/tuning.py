from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from survarena.evaluation.statistics import metric_direction


_RUNTIME_ONLY_METHOD_PARAMS = {"seed"}


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


def _searchable_default_params(method_cfg: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(method_cfg.get("default_params", {}))
    return {key: value for key, value in defaults.items() if key not in _RUNTIME_ONLY_METHOD_PARAMS}


def _metric_direction_for_optimization(primary_metric: str) -> str:
    return metric_direction(primary_metric)


def _build_hpo_metadata(
    *,
    resolved_hpo: dict[str, Any],
    enabled: bool,
    backend: str,
    status: str,
    realized_trial_count: int,
    started_at: str | None = None,
    finished_at: str | None = None,
    best_trial_number: int | None = None,
    best_trial_score: float | None = None,
) -> dict[str, Any]:
    realized = int(realized_trial_count)
    metadata: dict[str, Any] = {
        "enabled": bool(enabled),
        "backend": str(backend),
        "status": str(status),
        "started_at": started_at,
        "finished_at": finished_at,
        "realized_trial_count": realized,
        # Backward-compatible alias during transition.
        "trial_count": realized,
        "requested_max_trials": int(resolved_hpo["max_trials"]),
        "requested_timeout_seconds": resolved_hpo["timeout_seconds"],
        "requested_sampler": str(resolved_hpo["sampler"]),
        "requested_pruner": str(resolved_hpo["pruner"]),
        "max_trials": int(resolved_hpo["max_trials"]),
        "timeout_seconds": resolved_hpo["timeout_seconds"],
        "sampler": str(resolved_hpo["sampler"]),
        "pruner": str(resolved_hpo["pruner"]),
        "n_startup_trials": int(resolved_hpo["n_startup_trials"]),
    }
    if best_trial_number is not None:
        metadata["best_trial_number"] = int(best_trial_number)
    if best_trial_score is not None:
        metadata["best_trial_score"] = float(best_trial_score)
    return metadata


def _parse_hpo_config(method_cfg: dict[str, Any], hpo_config: dict[str, Any] | None) -> dict[str, Any]:
    base = {
        "enabled": bool(method_cfg.get("search_space")) and False,
        "max_trials": 20,
        "timeout_seconds": None,
        "sampler": "tpe",
        "pruner": "median",
        "n_startup_trials": 8,
    }
    cfg = dict(base)
    if hpo_config:
        cfg.update({k: v for k, v in hpo_config.items() if v is not None})
    cfg["enabled"] = bool(cfg.get("enabled", False)) and bool(method_cfg.get("search_space"))
    cfg["max_trials"] = max(int(cfg.get("max_trials", 20)), 1)
    cfg["n_startup_trials"] = max(int(cfg.get("n_startup_trials", 8)), 1)
    timeout_seconds = cfg.get("timeout_seconds")
    cfg["timeout_seconds"] = None if timeout_seconds is None else max(float(timeout_seconds), 0.0)
    cfg["sampler"] = str(cfg.get("sampler", "tpe")).lower()
    cfg["pruner"] = str(cfg.get("pruner", "median")).lower()
    return cfg


def _suggest_param(trial: Any, name: str, spec: dict[str, Any]) -> Any:
    param_type = str(spec.get("type", "")).lower()
    if param_type == "categorical":
        choices = list(spec.get("choices", []))
        if not choices:
            raise ValueError(f"Search space for '{name}' has empty categorical choices.")
        return trial.suggest_categorical(name, choices)
    if param_type == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        return trial.suggest_int(name, low, high, log=bool(spec.get("log", False)))
    if param_type == "int_or_none":
        low = int(spec["low"])
        high = int(spec["high"])
        values = [None] + list(range(low, high + 1))
        return trial.suggest_categorical(name, values)
    if param_type == "float":
        low = float(spec["low"])
        high = float(spec["high"])
        return trial.suggest_float(name, low, high, log=bool(spec.get("log", False)))
    raise ValueError(f"Unsupported search space type '{param_type}' for '{name}'.")


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
    seed: int,
    hpo_config: dict[str, Any] | None = None,
    quiet: bool = False,
    metric_bundle_callback: Callable[[dict[str, Any], Any, Any], dict[str, float]] | None = None,
    evaluate_defaults_when_disabled: bool = True,
) -> dict[str, Any]:
    from survarena.utils.quiet import quiet_training_output

    with quiet_training_output(enabled=quiet):
        defaults = _searchable_default_params(method_cfg)
        resolved_hpo = _parse_hpo_config(method_cfg, hpo_config)
        default_score = float("nan")
        default_metric_rows = None
        should_evaluate_defaults = bool(resolved_hpo["enabled"]) or bool(evaluate_defaults_when_disabled)
        if should_evaluate_defaults:
            default_eval = _inner_cv_evaluate(
                method_id=method_id,
                params=resolve_runtime_method_params(defaults, seed=seed),
                fold_cache=fold_cache,
                primary_metric=primary_metric,
                metric_bundle_callback=metric_bundle_callback,
            )
            default_score = float(default_eval["primary_score"])
            default_metric_rows = default_eval.get("metric_rows")
        default_result = {
            "best_params": defaults,
            "best_score": default_score,
            "best_metric_rows": default_metric_rows,
            "hpo_metadata": _build_hpo_metadata(
                resolved_hpo=resolved_hpo,
                enabled=bool(resolved_hpo["enabled"]),
                backend="none",
                status="disabled",
                realized_trial_count=0,
            ),
            "hpo_trials": [],
        }
        if not resolved_hpo["enabled"]:
            return default_result

        search_space = dict(method_cfg.get("search_space", {}))
        if not search_space:
            return default_result

        try:
            import optuna
            from optuna.pruners import MedianPruner, NopPruner
            from optuna.samplers import RandomSampler, TPESampler
        except ModuleNotFoundError:
            default_result["hpo_metadata"]["status"] = "optuna_missing"
            return default_result

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        maximize = _metric_direction_for_optimization(primary_metric) == "maximize"
        sampler_name = str(resolved_hpo["sampler"])
        if sampler_name == "random":
            sampler = RandomSampler(seed=seed)
        else:
            sampler = TPESampler(seed=seed, n_startup_trials=int(resolved_hpo["n_startup_trials"]))
        pruner_name = str(resolved_hpo["pruner"])
        pruner = MedianPruner(n_startup_trials=int(resolved_hpo["n_startup_trials"])) if pruner_name == "median" else NopPruner()

        started_at = datetime.utcnow().isoformat(timespec="seconds")

        def _objective(trial: Any) -> float:
            sampled_params = dict(defaults)
            for name, spec in search_space.items():
                sampled_params[name] = _suggest_param(trial, name, dict(spec))
            result = _inner_cv_evaluate(
                method_id=method_id,
                params=resolve_runtime_method_params(sampled_params, seed=seed),
                fold_cache=fold_cache,
                primary_metric=primary_metric,
                metric_bundle_callback=metric_bundle_callback,
            )
            score = float(result["primary_score"])
            if not score == score:
                return float("-inf") if maximize else float("inf")
            return score

        study = optuna.create_study(
            direction="maximize" if maximize else "minimize",
            sampler=sampler,
            pruner=pruner,
        )
        study.optimize(
            _objective,
            n_trials=int(resolved_hpo["max_trials"]),
            timeout=resolved_hpo["timeout_seconds"],
        )
        finished_at = datetime.utcnow().isoformat(timespec="seconds")

        if study.best_trial is None:
            default_result["hpo_metadata"] = _build_hpo_metadata(
                resolved_hpo=resolved_hpo,
                enabled=True,
                backend="optuna",
                status="no_valid_trial",
                started_at=started_at,
                finished_at=finished_at,
                realized_trial_count=int(len(study.trials)),
            )
            return default_result

        selected = dict(defaults)
        selected.update(dict(study.best_trial.params))
        best_score = float(study.best_value)
        best_metric_rows = None
        if metric_bundle_callback is not None:
            best_eval = _inner_cv_evaluate(
                method_id=method_id,
                params=resolve_runtime_method_params(selected, seed=seed),
                fold_cache=fold_cache,
                primary_metric=primary_metric,
                metric_bundle_callback=metric_bundle_callback,
            )
            best_score = float(best_eval["primary_score"])
            best_metric_rows = best_eval.get("metric_rows")
        trial_rows: list[dict[str, Any]] = []
        for trial in study.trials:
            trial_rows.append(
                {
                    "trial_number": int(trial.number),
                    "state": str(getattr(trial.state, "name", trial.state)),
                    "value": None if trial.value is None else float(trial.value),
                    "params": dict(trial.params),
                    "datetime_start": None if trial.datetime_start is None else trial.datetime_start.isoformat(),
                    "datetime_complete": None if trial.datetime_complete is None else trial.datetime_complete.isoformat(),
                }
            )
        return {
            "best_params": selected,
            "best_score": best_score,
            "best_metric_rows": best_metric_rows,
            "hpo_metadata": _build_hpo_metadata(
                resolved_hpo={
                    **resolved_hpo,
                    "sampler": sampler_name,
                    "pruner": pruner_name,
                },
                enabled=True,
                backend="optuna",
                status="success",
                started_at=started_at,
                finished_at=finished_at,
                realized_trial_count=int(len(study.trials)),
                best_trial_number=int(study.best_trial.number),
                best_trial_score=float(study.best_value),
            ),
            "hpo_trials": trial_rows,
        }
