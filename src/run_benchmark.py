from __future__ import annotations

import argparse
import traceback
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Callable


_RUNTIME_ONLY_METHOD_PARAMS = {"seed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SurvArena benchmark.")
    parser.add_argument(
        "--benchmark-config",
        type=str,
        default="configs/benchmark/standard_v1.yaml",
        help="Path to benchmark YAML config.",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset override.")
    parser.add_argument("--method", type=str, default=None, help="Optional method override.")
    parser.add_argument("--limit-seeds", type=int, default=None, help="Use first N seeds only.")
    parser.add_argument("--n-trials", type=int, default=None, help="Optuna trials override.")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without fitting models.")
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    import importlib

    yaml = importlib.import_module("yaml")

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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


def _metric_optimization_direction(primary_metric: str) -> str:
    if primary_metric in {"harrell_c", "uno_c"}:
        return "maximize"
    raise ValueError(f"Unsupported primary metric for tuning direction: {primary_metric}")


def _searchable_default_params(method_cfg: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(method_cfg.get("default_params", {}))
    return {key: value for key, value in defaults.items() if key not in _RUNTIME_ONLY_METHOD_PARAMS}


def _resolve_runtime_method_params(params: dict[str, Any], *, seed: int) -> dict[str, Any]:
    resolved = dict(params)
    resolved["seed"] = int(seed)
    return resolved


def _is_better_score(candidate: float, reference: float, *, direction: str) -> bool:
    if direction == "maximize":
        return candidate > reference
    if direction == "minimize":
        return candidate < reference
    raise ValueError(f"Unsupported optimization direction: {direction}")


def _prepare_inner_cv_cache(
    *,
    method_id: str,
    X_train: Any,
    time_train: Any,
    event_train: Any,
    inner_folds: int,
    seed: int,
) -> list[dict[str, Any]]:
    import importlib

    from src.data.preprocess import TabularPreprocessor

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


def _inner_cv_evaluate(
    *,
    method_id: str,
    params: dict[str, Any],
    fold_cache: list[dict[str, Any]],
    primary_metric: str,
    metric_bundle_callback: Callable[[dict[str, Any], Any, Any], dict[str, float]] | None = None,
) -> dict[str, Any]:
    import numpy as np

    from src.evaluation.metrics import compute_primary_metric_score

    method_registry = _method_registry()
    scores: list[float] = []
    metric_rows: list[dict[str, float]] = []
    for fold_data in fold_cache:
        model = method_registry[method_id](**params)
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


def _inner_cv_primary_metric(
    *,
    method_id: str,
    params: dict[str, Any],
    fold_cache: list[dict[str, Any]],
    primary_metric: str,
) -> float:
    evaluation = _inner_cv_evaluate(
        method_id=method_id,
        params=params,
        fold_cache=fold_cache,
        primary_metric=primary_metric,
    )
    return float(evaluation["primary_score"])


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

    from src.utils.quiet import quiet_training_output

    with quiet_training_output(enabled=quiet):
        defaults = _searchable_default_params(method_cfg)
        direction = _metric_optimization_direction(primary_metric)
        default_eval = _inner_cv_evaluate(
            method_id=method_id,
            params=_resolve_runtime_method_params(defaults, seed=seed),
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
                params=_resolve_runtime_method_params(params, seed=seed),
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


def evaluate_split(
    *,
    benchmark_id: str,
    dataset_id: str,
    method_id: str,
    split: Any,
    X: Any,
    time: Any,
    event: Any,
    method_cfg: dict[str, Any],
    inner_folds: int,
    n_trials: int,
    timeout_seconds: float | None,
    primary_metric: str,
    horizons_quantiles: tuple[float, float, float],
    benchmark_cfg_hash: str,
) -> dict[str, Any]:
    import numpy as np

    from src.data.preprocess import TabularPreprocessor
    from src.evaluation.metrics import compute_survival_metrics, horizons_from_train_event_times
    from src.logging.manifest import RunManifest
    from src.logging.tracker import payload_sha256, peak_memory_mb as peak_process_memory_mb
    from src.utils.seeds import set_global_seed
    from src.utils.time import timer

    method_registry = _method_registry()
    run_id = f"{dataset_id}_{method_id}_{split.split_id}_seed{split.seed}"
    started_at = perf_counter()
    split_indices_hash = payload_sha256(
        {
            "train_idx": split.train_idx.tolist(),
            "test_idx": split.test_idx.tolist(),
            "val_idx": split.val_idx.tolist() if split.val_idx is not None else None,
        }
    )
    method_cfg_hash = payload_sha256(method_cfg)

    try:
        set_global_seed(split.seed)
        X_train = X.iloc[split.train_idx]
        t_train = time[split.train_idx]
        e_train = event[split.train_idx]
        X_test = X.iloc[split.test_idx]
        t_test = time[split.test_idx]
        e_test = event[split.test_idx]

        fold_cache = _prepare_inner_cv_cache(
            method_id=method_id,
            X_train=X_train,
            time_train=t_train,
            event_train=e_train,
            inner_folds=inner_folds,
            seed=split.seed,
        )

        with timer() as tune_timer:
            tuning_result = tune_hyperparameters(
                method_id=method_id,
                method_cfg=method_cfg,
                fold_cache=fold_cache,
                primary_metric=primary_metric,
                n_trials=n_trials,
                seed=split.seed,
                timeout_seconds=timeout_seconds,
            )
        tuning_sec = tune_timer.elapsed
        best_params = dict(tuning_result["best_params"])
        validation_score = float(tuning_result["best_score"])
        n_trials_completed = int(tuning_result["n_trials_completed"])

        pre = TabularPreprocessor(scale_numeric=(method_id != "rsf"))
        X_train_proc = pre.fit_transform(X_train)
        X_test_proc = pre.transform(X_test)

        model = method_registry[method_id](**_resolve_runtime_method_params(best_params, seed=split.seed))
        with timer() as fit_timer:
            model.fit(X_train_proc.to_numpy(), t_train, e_train)
        fit_time_sec = fit_timer.elapsed

        eval_times = np.linspace(
            max(1e-8, float(np.percentile(t_train, 5))),
            max(float(np.percentile(t_train, 95)), max(1e-8, float(np.percentile(t_train, 5)) + 1e-8)),
            50,
        )
        with timer() as infer_timer:
            risk_scores = model.predict_risk(X_test_proc.to_numpy())
            surv_probs = model.predict_survival(X_test_proc.to_numpy(), eval_times)
        infer_time_sec = infer_timer.elapsed

        horizons = horizons_from_train_event_times(t_train, e_train, horizons_quantiles)
        metrics = compute_survival_metrics(
            train_time=t_train,
            train_event=e_train,
            test_time=t_test,
            test_event=e_test,
            risk_scores=risk_scores,
            survival_probs=surv_probs,
            survival_times=eval_times,
            horizons=horizons,
        ).to_dict()

        peak_memory_mb = peak_process_memory_mb()
        runtime_sec = tuning_sec + fit_time_sec + infer_time_sec

        manifest = RunManifest(
            run_id=run_id,
            benchmark_id=benchmark_id,
            dataset_id=dataset_id,
            method_id=method_id,
            split_id=split.split_id,
            seed=split.seed,
            hyperparameters=best_params,
            preprocessing_config={
                "numeric_imputer": "median",
                "categorical_imputer": "most_frequent",
                "numeric_scaling": method_id != "rsf",
                "categorical_encoding": "one_hot",
            },
            runtime_seconds=runtime_sec,
            peak_memory_mb=peak_memory_mb,
            status="success",
            benchmark_config_hash=benchmark_cfg_hash,
            method_config_hash=method_cfg_hash,
            split_indices_hash=split_indices_hash,
        )
        run_payload = {
            "manifest": manifest.to_dict(),
            "metrics": {
                "run_id": run_id,
                "dataset_id": dataset_id,
                "method_id": method_id,
                "seed": split.seed,
                "split_id": split.split_id,
                "primary_metric": primary_metric,
                "validation_score": validation_score,
                "test_metrics": metrics,
                "tuning_time_sec": tuning_sec,
                "runtime_seconds": runtime_sec,
                "fit_time_sec": fit_time_sec,
                "infer_time_sec": infer_time_sec,
                "peak_memory_mb": peak_memory_mb,
                "n_trials_requested": n_trials,
                "n_trials_completed": n_trials_completed,
                "tuning_timeout_seconds": timeout_seconds,
                "status": "success",
                "best_params": best_params,
            },
            "failure": None,
        }
        return {
            "run_payload": run_payload,
            "benchmark_id": benchmark_id,
            "dataset_id": dataset_id,
            "method_id": method_id,
            "split_id": split.split_id,
            "seed": split.seed,
            "primary_metric": primary_metric,
            "validation_score": validation_score,
            **metrics,
            "tuning_time_sec": tuning_sec,
            "runtime_sec": runtime_sec,
            "fit_time_sec": fit_time_sec,
            "infer_time_sec": infer_time_sec,
            "peak_memory_mb": peak_memory_mb,
            "n_trials_requested": n_trials,
            "n_trials_completed": n_trials_completed,
            "status": "success",
        }
    except Exception as exc:  # noqa: BLE001
        tb_str = traceback.format_exc()
        elapsed_before_failure = perf_counter() - started_at
        peak_memory_mb = peak_process_memory_mb()
        manifest = RunManifest(
            run_id=run_id,
            benchmark_id=benchmark_id,
            dataset_id=dataset_id,
            method_id=method_id,
            split_id=split.split_id,
            seed=split.seed,
            hyperparameters={},
            preprocessing_config={},
            runtime_seconds=elapsed_before_failure,
            peak_memory_mb=peak_memory_mb,
            status="failed",
            benchmark_config_hash=benchmark_cfg_hash,
            method_config_hash=method_cfg_hash,
            split_indices_hash=split_indices_hash,
            notes=str(exc),
        )
        run_payload = {
            "manifest": manifest.to_dict(),
            "metrics": {
                "run_id": run_id,
                "dataset_id": dataset_id,
                "method_id": method_id,
                "seed": split.seed,
                "split_id": split.split_id,
                "primary_metric": primary_metric,
                "status": "failed",
                "failure_type": type(exc).__name__,
                "exception_message": str(exc),
                "elapsed_time_before_failure": elapsed_before_failure,
                "peak_memory_mb": peak_memory_mb,
            },
            "failure": {
                "traceback": tb_str,
            },
        }
        return {
            "run_payload": run_payload,
            "benchmark_id": benchmark_id,
            "dataset_id": dataset_id,
            "method_id": method_id,
            "split_id": split.split_id,
            "seed": split.seed,
            "primary_metric": primary_metric,
            "validation_score": np.nan,
            "uno_c": np.nan,
            "harrell_c": np.nan,
            "ibs": np.nan,
            "td_auc_25": np.nan,
            "td_auc_50": np.nan,
            "td_auc_75": np.nan,
            "tuning_time_sec": np.nan,
            "runtime_sec": elapsed_before_failure,
            "fit_time_sec": np.nan,
            "infer_time_sec": np.nan,
            "peak_memory_mb": peak_memory_mb,
            "n_trials_requested": n_trials,
            "n_trials_completed": 0,
            "status": "failed",
        }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    try:
        benchmark_cfg = read_yaml(repo_root / args.benchmark_config)
    except ModuleNotFoundError as exc:
        if args.dry_run:
            print("Dry run completed with missing optional dependency.")
            print(f"missing_module={exc.name}")
            print("Install requirements.txt before full benchmark execution.")
            return
        raise
    benchmark_id = benchmark_cfg["benchmark_id"]
    primary_metric = str(benchmark_cfg.get("primary_metric", "harrell_c"))
    datasets = [args.dataset] if args.dataset else list(benchmark_cfg["datasets"])
    methods = [args.method] if args.method else list(benchmark_cfg["methods"])
    method_registry = _method_registry()

    seeds = list(benchmark_cfg["seeds"])
    if args.limit_seeds is not None:
        seeds = seeds[: args.limit_seeds]
    if not seeds:
        raise ValueError("Seed list cannot be empty.")

    split_strategy = str(benchmark_cfg["split_strategy"])
    requested_outer_repeats = int(benchmark_cfg.get("outer_repeats", 1))
    if split_strategy == "repeated_nested_cv":
        if args.limit_seeds is not None:
            outer_repeats = min(requested_outer_repeats, len(seeds))
        else:
            outer_repeats = requested_outer_repeats
        if outer_repeats > len(seeds):
            raise ValueError(
                f"Benchmark requests {outer_repeats} outer repeats but only {len(seeds)} seeds are available."
            )
    else:
        outer_repeats = requested_outer_repeats

    n_trials = int(args.n_trials) if args.n_trials is not None else int(benchmark_cfg["tuning"]["n_trials"])
    timeout_seconds = benchmark_cfg.get("tuning", {}).get("timeout_seconds")
    timeout_seconds = None if timeout_seconds is None else float(timeout_seconds)
    if args.dry_run:
        print("Dry run complete.")
        print(f"benchmark_id={benchmark_id}")
        print(f"datasets={datasets}")
        print(f"methods={methods}")
        print(f"seeds={seeds}")
        print(f"outer_repeats={outer_repeats}")
        print(f"n_trials={n_trials}")
        print(f"timeout_seconds={timeout_seconds}")
        print(f"primary_metric={primary_metric}")
        return

    from src.data.loaders import load_dataset
    from src.data.splitters import load_or_create_splits
    from src.logging.export import (
        create_experiment_dir,
        export_fold_results,
        export_leaderboard,
        export_overall_summary,
        export_run_ledger,
        export_seed_summary,
    )
    from src.logging.tracker import payload_sha256, write_json

    all_records: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    method_cfg_cache = {method_id: read_yaml(repo_root / "configs" / "methods" / f"{method_id}.yaml") for method_id in methods}
    benchmark_cfg_hash = payload_sha256(benchmark_cfg)
    experiment_dir = create_experiment_dir(repo_root)
    write_json(
        experiment_dir / "experiment_manifest.json",
        {
            "benchmark_id": benchmark_id,
            "datasets": datasets,
            "methods": methods,
            "seeds": seeds,
            "n_trials": n_trials,
            "timeout_seconds": timeout_seconds,
            "primary_metric": primary_metric,
            "benchmark_config_hash": benchmark_cfg_hash,
            "output_dir": str(experiment_dir),
        },
    )

    for dataset_id in datasets:
        dataset = load_dataset(dataset_id, repo_root)
        task_id = f"{dataset_id}_{benchmark_id}"
        splits = load_or_create_splits(
            root=repo_root,
            task_id=task_id,
            split_strategy=benchmark_cfg["split_strategy"],
            n_samples=len(dataset.X),
            event=dataset.event,
            seeds=seeds,
            outer_folds=int(benchmark_cfg.get("outer_folds", 5)),
            outer_repeats=outer_repeats,
        )

        filtered_splits = [s for s in splits if s.seed in seeds]
        horizons_q = tuple(float(x) for x in benchmark_cfg.get("time_horizons_quantiles", [0.25, 0.5, 0.75]))

        for method_id in methods:
            if method_id not in method_registry:
                raise ValueError(f"Unknown method_id '{method_id}'. Registered: {sorted(method_registry.keys())}")
            method_cfg = method_cfg_cache[method_id]
            for split in filtered_splits:
                record = evaluate_split(
                    benchmark_id=benchmark_id,
                    dataset_id=dataset_id,
                    method_id=method_id,
                    split=split,
                    X=dataset.X,
                    time=dataset.time,
                    event=dataset.event,
                    method_cfg=method_cfg,
                    inner_folds=int(benchmark_cfg.get("inner_folds", 3)),
                    n_trials=n_trials,
                    timeout_seconds=timeout_seconds,
                    primary_metric=primary_metric,
                    horizons_quantiles=horizons_q,  # type: ignore[arg-type]
                    benchmark_cfg_hash=benchmark_cfg_hash,
                )
                run_records.append(record.pop("run_payload"))
                all_records.append(record)
                print(
                    f"[{record['status']}] {dataset_id}/{method_id}/{split.split_id}/seed{split.seed} "
                    f"{primary_metric}={record.get(primary_metric)}"
                )

    frame = export_fold_results(repo_root, all_records, output_dir=experiment_dir, file_prefix=benchmark_id)
    seed_summary = export_seed_summary(repo_root, frame, output_dir=experiment_dir, file_prefix=benchmark_id)
    export_overall_summary(repo_root, frame, output_dir=experiment_dir, file_prefix=benchmark_id)
    export_leaderboard(
        repo_root,
        seed_summary,
        primary_metric=primary_metric,
        output_dir=experiment_dir,
        file_prefix=benchmark_id,
    )
    export_run_ledger(repo_root, run_records, benchmark_id=benchmark_id, output_dir=experiment_dir)
    print(f"Benchmark run complete. Outputs saved to: {experiment_dir}")


@lru_cache(maxsize=1)
def _method_registry() -> dict[str, Any]:
    from src.methods.classical.coxnet import CoxNetMethod
    from src.methods.classical.coxph import CoxPHMethod
    from src.methods.deep.deepsurv import DeepSurvMethod
    from src.methods.deep.deepsurv_moco import DeepSurvMomentumMethod
    from src.methods.foundation.mitra_survival import MitraSurvivalMethod
    from src.methods.foundation.tabpfn_survival import TabPFNSurvivalMethod
    from src.methods.tree.rsf import RSFMethod

    return {
        "coxph": CoxPHMethod,
        "coxnet": CoxNetMethod,
        "rsf": RSFMethod,
        "deepsurv": DeepSurvMethod,
        "deepsurv_moco": DeepSurvMomentumMethod,
        "mitra_survival": MitraSurvivalMethod,
        "tabpfn_survival": TabPFNSurvivalMethod,
    }


if __name__ == "__main__":
    main()
