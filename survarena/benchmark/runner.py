from __future__ import annotations

import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

from survarena.benchmark.tuning import prepare_inner_cv_cache, resolve_runtime_method_params, tune_hyperparameters
from survarena.config import read_yaml
from survarena.methods.registry import method_registry


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

    from survarena.data.preprocess import TabularPreprocessor
    from survarena.evaluation.metrics import compute_survival_metrics, horizons_from_train_event_times
    from survarena.logging.manifest import RunManifest
    from survarena.logging.tracker import payload_sha256, peak_memory_mb as peak_process_memory_mb
    from survarena.utils.seeds import set_global_seed
    from survarena.utils.time import timer

    registry = method_registry()
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

        fold_cache = prepare_inner_cv_cache(
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

        model = registry[method_id](**resolve_runtime_method_params(best_params, seed=split.seed))
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


def run_benchmark(
    *,
    repo_root: Path,
    benchmark_cfg: dict[str, Any],
    dataset_override: str | None = None,
    method_override: str | None = None,
    limit_seeds: int | None = None,
    n_trials_override: int | None = None,
    dry_run: bool = False,
) -> None:
    from survarena.data.loaders import load_dataset
    from survarena.data.splitters import load_or_create_splits
    from survarena.logging.export import (
        create_experiment_dir,
        export_fold_results,
        export_leaderboard,
        export_overall_summary,
        export_run_ledger,
        export_seed_summary,
    )
    from survarena.logging.tracker import payload_sha256, write_json

    benchmark_id = benchmark_cfg["benchmark_id"]
    primary_metric = str(benchmark_cfg.get("primary_metric", "harrell_c"))
    datasets = [dataset_override] if dataset_override else list(benchmark_cfg["datasets"])
    methods = [method_override] if method_override else list(benchmark_cfg["methods"])

    seeds = list(benchmark_cfg["seeds"])
    if limit_seeds is not None:
        seeds = seeds[:limit_seeds]
    if not seeds:
        raise ValueError("Seed list cannot be empty.")

    split_strategy = str(benchmark_cfg["split_strategy"])
    requested_outer_repeats = int(benchmark_cfg.get("outer_repeats", 1))
    if split_strategy == "repeated_nested_cv":
        if limit_seeds is not None:
            outer_repeats = min(requested_outer_repeats, len(seeds))
        else:
            outer_repeats = requested_outer_repeats
        if outer_repeats > len(seeds):
            raise ValueError(
                f"Benchmark requests {outer_repeats} outer repeats but only {len(seeds)} seeds are available."
            )
    else:
        outer_repeats = requested_outer_repeats

    n_trials = int(n_trials_override) if n_trials_override is not None else int(benchmark_cfg["tuning"]["n_trials"])
    timeout_seconds = benchmark_cfg.get("tuning", {}).get("timeout_seconds")
    timeout_seconds = None if timeout_seconds is None else float(timeout_seconds)

    if dry_run:
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

    registry = method_registry()
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

        filtered_splits = [split for split in splits if split.seed in seeds]
        horizons_q = tuple(float(x) for x in benchmark_cfg.get("time_horizons_quantiles", [0.25, 0.5, 0.75]))

        for method_id in methods:
            if method_id not in registry:
                raise ValueError(f"Unknown method_id '{method_id}'. Registered: {sorted(registry.keys())}")
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
