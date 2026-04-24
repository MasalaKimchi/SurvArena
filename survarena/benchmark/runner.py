from __future__ import annotations

import math
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

from survarena.benchmark.tuning import prepare_inner_cv_cache, resolve_runtime_method_params, select_hyperparameters
from survarena.logging.export import MANUSCRIPT_METRIC_COLUMNS
from survarena.config import read_yaml
from survarena.methods.registry import get_method_class, registered_method_ids


_CANONICAL_PROFILES = ("smoke", "standard", "manuscript")
_DUAL_HPO_MODE_ORDER: tuple[str, str] = ("no_hpo", "hpo")
_PROFILE_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "smoke": ("outer_repeats",),
    "standard": ("outer_folds", "outer_repeats"),
    "manuscript": ("outer_folds", "outer_repeats"),
}


def _is_missing_resume_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _resume_completion_key(
    row: dict[str, Any],
    *,
    primary_metric: str,
) -> tuple[tuple[str, str, str, int, str] | None, str | None]:
    if str(row.get("status", "")).lower() != "success":
        return None, "status is not success"

    for field in ("dataset_id", "method_id", "split_id"):
        if _is_missing_resume_value(row.get(field)):
            return None, f"missing required field '{field}'"

    seed_value = row.get("seed")
    if _is_missing_resume_value(seed_value):
        return None, "missing required field 'seed'"
    try:
        seed = int(seed_value)
    except (TypeError, ValueError):
        return None, f"invalid seed value '{seed_value}'"

    metric_value = row.get(primary_metric)
    if _is_missing_resume_value(metric_value):
        return None, f"missing required metric '{primary_metric}'"

    return (
        (
            str(row.get("dataset_id")),
            str(row.get("method_id")),
            str(row.get("split_id")),
            seed,
            str(row.get("hpo_mode", "")),
        ),
        None,
    )


def _require_int(cfg: dict[str, Any], key: str) -> int:
    value = cfg.get(key)
    if value is None:
        raise ValueError(f"Missing required deterministic fields: {key}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Deterministic field '{key}' must be an integer. Received: {value!r}") from exc


def validate_benchmark_profile_contract(benchmark_cfg: dict[str, Any]) -> None:
    profile = str(benchmark_cfg.get("profile", "")).lower()
    if profile not in _CANONICAL_PROFILES:
        raise ValueError(
            f"Invalid profile '{profile}'. "
            f"Allowed profiles: {', '.join(_CANONICAL_PROFILES)}."
        )

    required_keys = {"split_strategy", "seeds", *(_PROFILE_REQUIRED_KEYS.get(profile, ()))}
    missing = sorted(key for key in required_keys if key not in benchmark_cfg)
    if missing:
        raise ValueError(
            "Missing required deterministic fields: "
            + ", ".join(missing)
            + "."
        )

    split_strategy = str(benchmark_cfg.get("split_strategy"))
    if split_strategy != "repeated_nested_cv":
        raise ValueError(
            f"Profile '{profile}' requires split_strategy='repeated_nested_cv'. "
            f"Received: '{split_strategy}'."
        )

    seeds = benchmark_cfg.get("seeds")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError(
            "Deterministic field 'seeds' must be a non-empty list of integers."
        )

    outer_repeats = _require_int(benchmark_cfg, "outer_repeats")
    if profile == "smoke" and outer_repeats != 1:
        raise ValueError(
            f"Profile '{profile}' requires outer_repeats=1. Received: {outer_repeats}."
        )

    if profile in {"standard", "manuscript"}:
        outer_folds = _require_int(benchmark_cfg, "outer_folds")
        if outer_folds < 3:
            raise ValueError(
                f"Profile '{profile}' requires outer_folds>=3 for deterministic comparability. "
                f"Received: {outer_folds}."
            )
        if outer_repeats < 3:
            raise ValueError(
                f"Profile '{profile}' requires outer_repeats>=3 for deterministic comparability. "
                f"Received: {outer_repeats}."
            )


def _normalize_hpo_budget_telemetry(
    *,
    hpo_metadata: dict[str, Any],
    hpo_cfg: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(hpo_metadata)
    realized_trial_count = int(normalized.get("realized_trial_count", normalized.get("trial_count", 0)))
    normalized["realized_trial_count"] = realized_trial_count
    normalized["trial_count"] = realized_trial_count
    normalized["requested_max_trials"] = int(normalized.get("requested_max_trials", hpo_cfg.get("max_trials", 20)))
    timeout_value = normalized.get("requested_timeout_seconds", hpo_cfg.get("timeout_seconds"))
    normalized["requested_timeout_seconds"] = None if timeout_value is None else float(timeout_value)
    normalized["requested_sampler"] = str(normalized.get("requested_sampler", hpo_cfg.get("sampler", "tpe"))).lower()
    normalized["requested_pruner"] = str(normalized.get("requested_pruner", hpo_cfg.get("pruner", "median"))).lower()
    return normalized


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
    timeout_seconds: float | None,
    primary_metric: str,
    horizons_quantiles: tuple[float, float, float],
    decision_thresholds: tuple[float, ...],
    benchmark_cfg_hash: str,
    autogluon_cfg: dict[str, Any] | None = None,
    hpo_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import numpy as np

    from survarena.data.preprocess import TabularPreprocessor
    from survarena.evaluation.metrics import compute_survival_metrics, horizons_from_train_event_times
    from survarena.logging.manifest import RunManifest
    from survarena.logging.tracker import payload_sha256, peak_memory_mb as peak_process_memory_mb
    from survarena.methods.preprocessing import (
        finalize_preprocessed_features,
        method_preprocessing_summary,
        method_preprocessor_kwargs,
    )
    from survarena.utils.seeds import set_global_seed
    from survarena.utils.time import timer

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
            method_cfg_for_selection = _method_cfg_with_autogluon_defaults(method_cfg, autogluon_cfg)
            selection_result = select_hyperparameters(
                method_id=method_id,
                method_cfg=method_cfg_for_selection,
                fold_cache=fold_cache,
                primary_metric=primary_metric,
                seed=split.seed,
                hpo_config=hpo_cfg,
            )
        tuning_sec = tune_timer.elapsed
        best_params = dict(selection_result["best_params"])
        validation_score = float(selection_result["best_score"])
        hpo_metadata = dict(selection_result.get("hpo_metadata", {}))
        hpo_trials = list(selection_result.get("hpo_trials", []))

        pre = TabularPreprocessor(**method_preprocessor_kwargs(method_id))
        X_train_proc = finalize_preprocessed_features(method_id, pre.fit_transform(X_train))
        X_test_proc = finalize_preprocessed_features(method_id, pre.transform(X_test))

        model = get_method_class(method_id)(**resolve_runtime_method_params(best_params, seed=split.seed))
        with timer() as fit_timer:
            model.fit(X_train_proc, t_train, e_train)
        fit_time_sec = fit_timer.elapsed

        eval_times = np.linspace(
            max(1e-8, float(np.percentile(t_train, 5))),
            max(float(np.percentile(t_train, 95)), max(1e-8, float(np.percentile(t_train, 5)) + 1e-8)),
            50,
        )
        with timer() as infer_timer:
            risk_scores = model.predict_risk(X_test_proc)
            surv_probs = model.predict_survival(X_test_proc, eval_times)
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
            decision_thresholds=decision_thresholds,
        ).to_dict()

        peak_memory_mb = peak_process_memory_mb()
        runtime_sec = tuning_sec + fit_time_sec + infer_time_sec
        autogluon_metadata = _autogluon_metadata(model)
        training_backend = "autogluon" if method_id == "autogluon_survival" else "native"
        hpo_backend = str(hpo_metadata.get("backend", "none"))
        if method_id == "autogluon_survival" and best_params.get("hyperparameter_tune_kwargs"):
            hpo_backend = "autogluon"

        manifest = RunManifest(
            run_id=run_id,
            benchmark_id=benchmark_id,
            dataset_id=dataset_id,
            method_id=method_id,
            split_id=split.split_id,
            seed=split.seed,
            hyperparameters=best_params,
            preprocessing_config=method_preprocessing_summary(method_id),
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
                "training_backend": training_backend,
                "hpo_backend": hpo_backend,
                "autogluon_presets": best_params.get("presets") if method_id == "autogluon_survival" else None,
                "autogluon_best_model": autogluon_metadata.get("autogluon_best_model"),
                "autogluon_model_count": autogluon_metadata.get("autogluon_model_count"),
                "autogluon_path": autogluon_metadata.get("autogluon_path"),
                "bagging_folds": best_params.get("num_bag_folds", 0) if method_id == "autogluon_survival" else 0,
                "stack_levels": best_params.get("num_stack_levels", 0) if method_id == "autogluon_survival" else 0,
                "tuning_timeout_seconds": timeout_seconds,
                "status": "success",
                "best_params": best_params,
                "hpo_status": hpo_metadata.get("status", "disabled"),
                "hpo_trial_count": hpo_metadata.get("trial_count", 0),
            },
            "backend_metadata": {
                "autogluon_leaderboard": autogluon_metadata.get("autogluon_leaderboard", []),
            },
            "hpo_metadata": hpo_metadata,
            "hpo_trials": hpo_trials,
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
            "training_backend": training_backend,
            "hpo_backend": hpo_backend,
            "autogluon_presets": best_params.get("presets") if method_id == "autogluon_survival" else None,
            "autogluon_best_model": autogluon_metadata.get("autogluon_best_model"),
            "autogluon_model_count": autogluon_metadata.get("autogluon_model_count"),
            "autogluon_path": autogluon_metadata.get("autogluon_path"),
            "bagging_folds": best_params.get("num_bag_folds", 0) if method_id == "autogluon_survival" else 0,
            "stack_levels": best_params.get("num_stack_levels", 0) if method_id == "autogluon_survival" else 0,
            "hpo_status": hpo_metadata.get("status", "disabled"),
            "hpo_trial_count": hpo_metadata.get("trial_count", 0),
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
            **{metric: np.nan for metric in MANUSCRIPT_METRIC_COLUMNS},
            "tuning_time_sec": np.nan,
            "runtime_sec": elapsed_before_failure,
            "fit_time_sec": np.nan,
            "infer_time_sec": np.nan,
            "peak_memory_mb": peak_memory_mb,
            "training_backend": "autogluon" if method_id == "autogluon_survival" else "native",
            "hpo_backend": "none",
            "autogluon_presets": None,
            "autogluon_best_model": None,
            "autogluon_model_count": 0,
            "autogluon_path": None,
            "bagging_folds": 0,
            "stack_levels": 0,
            "hpo_status": "failed",
            "hpo_trial_count": 0,
            "status": "failed",
        }


def _autogluon_metadata(model: Any) -> dict[str, Any]:
    metadata_getter = getattr(model, "autogluon_metadata", None)
    if callable(metadata_getter):
        return dict(metadata_getter())
    return {}


def _method_cfg_with_autogluon_defaults(method_cfg: dict[str, Any], autogluon_cfg: dict[str, Any] | None) -> dict[str, Any]:
    if method_cfg.get("method_id") != "autogluon_survival":
        return method_cfg
    merged = dict(method_cfg)
    defaults = dict(method_cfg.get("default_params", {}))
    if autogluon_cfg:
        defaults.update(
            {
                "presets": autogluon_cfg.get("presets", defaults.get("presets", "medium")),
                "time_limit": autogluon_cfg.get("time_limit_seconds", defaults.get("time_limit")),
                "hyperparameter_tune_kwargs": autogluon_cfg.get(
                    "hyperparameter_tune_kwargs",
                    defaults.get("hyperparameter_tune_kwargs"),
                ),
                "num_bag_folds": autogluon_cfg.get("num_bag_folds", defaults.get("num_bag_folds", 0)),
                "num_stack_levels": autogluon_cfg.get("num_stack_levels", defaults.get("num_stack_levels", 0)),
                "refit_full": autogluon_cfg.get("refit_full", defaults.get("refit_full", False)),
            }
        )
    merged["default_params"] = defaults
    return merged


def run_benchmark(
    *,
    repo_root: Path,
    benchmark_cfg: dict[str, Any],
    dataset_override: str | None = None,
    method_override: str | None = None,
    limit_seeds: int | None = None,
    dry_run: bool = False,
    output_dir: Path | None = None,
    resume: bool = False,
    max_retries: int = 0,
    regenerate_splits: bool = False,
) -> None:
    import numpy as np

    from survarena.data.loaders import load_dataset
    from survarena.data.robustness import apply_label_noise, apply_robustness_track, resolve_robustness_tracks
    from survarena.data.splitters import load_or_create_splits
    from survarena.logging.export import (
        create_experiment_dir,
        export_dataset_curation_table,
        export_fold_results,
        export_hpo_trials,
        export_leaderboard,
        export_manuscript_comparison,
        export_overall_summary,
        export_run_ledger,
        export_seed_summary,
    )
    from survarena.logging.tracker import payload_sha256, write_json

    validate_benchmark_profile_contract(benchmark_cfg)

    benchmark_id = benchmark_cfg["benchmark_id"]
    primary_metric = str(benchmark_cfg.get("primary_metric", "harrell_c"))
    profile = str(benchmark_cfg.get("profile", "custom")).lower()
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

    if profile in {"standard", "manuscript"} and (len(seeds) < 3 or int(benchmark_cfg.get("outer_folds", 5)) < 3):
        print(
            f"[warning] profile='{profile}' is typically underpowered with seeds={len(seeds)} "
            f"and outer_folds={int(benchmark_cfg.get('outer_folds', 5))}."
        )

    autogluon_cfg = dict(benchmark_cfg.get("autogluon", {}))
    timeout_seconds = autogluon_cfg.get("time_limit_seconds")
    timeout_seconds = None if timeout_seconds is None else float(timeout_seconds)
    hpo_cfg = dict(benchmark_cfg.get("hpo", {}))
    hpo_cfg.setdefault("enabled", False)
    hpo_cfg.setdefault("max_trials", 20)
    hpo_cfg.setdefault("timeout_seconds", None)
    hpo_cfg.setdefault("sampler", "tpe")
    hpo_cfg.setdefault("pruner", "median")
    hpo_cfg.setdefault("n_startup_trials", 8)
    decision_thresholds = tuple(
        float(x) for x in benchmark_cfg.get("decision_curve", {}).get("thresholds", [0.2])
    )

    if dry_run:
        print("Dry run complete.")
        print(f"benchmark_id={benchmark_id}")
        print(f"profile={profile}")
        print(f"datasets={datasets}")
        print(f"methods={methods}")
        print(f"seeds={seeds}")
        print(f"outer_repeats={outer_repeats}")
        print(f"timeout_seconds={timeout_seconds}")
        print(f"autogluon={autogluon_cfg}")
        print(f"hpo={hpo_cfg}")
        print(f"decision_thresholds={list(decision_thresholds)}")
        print(f"primary_metric={primary_metric}")
        return

    registered_methods = set(registered_method_ids())
    unknown_methods = [m for m in methods if m not in registered_methods]
    if unknown_methods:
        raise ValueError(
            f"Unknown method_id(s) {unknown_methods}. Registered: {sorted(registered_methods)}"
        )
    all_records: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    hpo_trial_rows: list[dict[str, Any]] = []
    dataset_curation_rows: list[dict[str, Any]] = []
    method_cfg_cache = {
        method_id: read_yaml(repo_root / "configs" / "methods" / f"{method_id}.yaml") for method_id in methods
    }
    benchmark_cfg_hash = payload_sha256(benchmark_cfg)
    experiment_dir = output_dir if output_dir is not None else create_experiment_dir(repo_root)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    completed_keys: set[tuple[str, str, str, int, str]] = set()
    if resume:
        existing_fold_results = experiment_dir / f"{benchmark_id}_fold_results.csv"
        if existing_fold_results.exists():
            import pandas as pd

            existing = pd.read_csv(existing_fold_results)
            for row in existing.to_dict(orient="records"):
                key, reason = _resume_completion_key(row, primary_metric=primary_metric)
                if key is None:
                    # D-06: success status alone is insufficient without required output integrity.
                    if str(row.get("status", "")).lower() == "success":
                        print(
                            "[resume][D-06] Ignoring ineligible success row: "
                            f"dataset_id={row.get('dataset_id')} method_id={row.get('method_id')} "
                            f"split_id={row.get('split_id')} seed={row.get('seed')} reason={reason}"
                        )
                    continue
                if key[4]:
                    completed_keys.add(key)
                else:
                    # Backward compatibility: pre-dual-mode artifacts had no hpo_mode.
                    # Treat a successful legacy row as completed for both execution modes.
                    for legacy_mode in _DUAL_HPO_MODE_ORDER:
                        completed_keys.add((key[0], key[1], key[2], key[3], legacy_mode))
    write_json(
        experiment_dir / "experiment_manifest.json",
        {
            "benchmark_id": benchmark_id,
            "profile": profile,
            "datasets": datasets,
            "methods": methods,
            "seeds": seeds,
            "timeout_seconds": timeout_seconds,
            "autogluon": autogluon_cfg,
            "hpo": hpo_cfg,
            "decision_curve_thresholds": list(decision_thresholds),
            "primary_metric": primary_metric,
            "benchmark_config_hash": benchmark_cfg_hash,
            "output_dir": str(experiment_dir),
            "resume": bool(resume),
            "max_retries": int(max_retries),
        },
    )

    for dataset_id in datasets:
        dataset = load_dataset(dataset_id, repo_root)
        dataset_curation_rows.append(
            {
                "dataset_id": dataset_id,
                "n_rows": int(len(dataset.X)),
                "n_features": int(dataset.X.shape[1]),
                "n_events": int(dataset.event.sum()),
                "event_rate": float(dataset.event.mean()),
                "censoring_rate": float(1.0 - dataset.event.mean()),
                "time_min": float(dataset.time.min()),
                "time_median": float(np.median(dataset.time)),
                "time_max": float(dataset.time.max()),
                "feature_types": getattr(dataset.metadata, "feature_types", {}),
            }
        )
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
            regenerate_on_mismatch=bool(regenerate_splits),
        )

        filtered_splits = [split for split in splits if split.seed in seeds]
        horizons_q = tuple(float(x) for x in benchmark_cfg.get("time_horizons_quantiles", [0.25, 0.5, 0.75]))

        robustness_tracks = resolve_robustness_tracks(
            benchmark_cfg.get("robustness", {}),
            dataset_id=dataset_id,
            feature_columns=list(dataset.X.columns),
            seed_pool=seeds,
        )
        for method_id in methods:
            method_cfg = method_cfg_cache[method_id]
            for split in filtered_splits:
                for track in robustness_tracks:
                    track_dataset_id = f"{dataset_id}__{track.track_id}"
                    track_split_id = f"{split.split_id}__{track.track_id}"
                    X_track = apply_robustness_track(
                        dataset.X,
                        track=track,
                        split=split,
                        seed=split.seed,
                    )
                    event_track = apply_label_noise(
                        dataset.event,
                        track=track,
                        split=split,
                        seed=split.seed,
                    )
                    parity_key = f"{track_dataset_id}|{track_split_id}|{int(split.seed)}|{method_id}"
                    for hpo_mode in _DUAL_HPO_MODE_ORDER:
                        key = (track_dataset_id, method_id, track_split_id, int(split.seed), hpo_mode)
                        if key in completed_keys:
                            continue
                        mode_hpo_cfg = dict(hpo_cfg)
                        mode_hpo_cfg["enabled"] = hpo_mode == "hpo"
                        attempt = 0
                        while True:
                            record = evaluate_split(
                                benchmark_id=benchmark_id,
                                dataset_id=track_dataset_id,
                                method_id=method_id,
                                split=split,
                                X=X_track,
                                time=dataset.time,
                                event=event_track,
                                method_cfg=method_cfg,
                                inner_folds=int(benchmark_cfg.get("inner_folds", 3)),
                                timeout_seconds=timeout_seconds,
                                primary_metric=primary_metric,
                                horizons_quantiles=horizons_q,  # type: ignore[arg-type]
                                decision_thresholds=decision_thresholds,
                                benchmark_cfg_hash=benchmark_cfg_hash,
                                autogluon_cfg=autogluon_cfg,
                                hpo_cfg=mode_hpo_cfg,
                            )
                            run_payload = record.pop("run_payload")
                            run_payload["manifest"]["split_id"] = track_split_id
                            run_payload["metrics"]["split_id"] = track_split_id
                            run_payload["metrics"]["robustness_track_id"] = track.track_id
                            run_payload["metrics"]["hpo_mode"] = hpo_mode
                            run_payload["metrics"]["parity_key"] = parity_key
                            run_payload["metrics"]["parity_eligible"] = True
                            run_payload["metrics"]["retry_attempt"] = int(attempt)
                            run_payload["status"] = str(record.get("status", run_payload["metrics"].get("status", "failed")))
                            run_payload["retry_attempt"] = int(attempt)
                            hpo_metadata = _normalize_hpo_budget_telemetry(
                                hpo_metadata=dict(run_payload.get("hpo_metadata", {})),
                                hpo_cfg=mode_hpo_cfg,
                            )
                            realized_trial_count = int(hpo_metadata["realized_trial_count"])
                            run_payload["hpo_metadata"] = hpo_metadata
                            run_payload["metrics"]["requested_max_trials"] = hpo_metadata["requested_max_trials"]
                            run_payload["metrics"]["requested_timeout_seconds"] = hpo_metadata["requested_timeout_seconds"]
                            run_payload["metrics"]["requested_sampler"] = hpo_metadata["requested_sampler"]
                            run_payload["metrics"]["requested_pruner"] = hpo_metadata["requested_pruner"]
                            run_payload["metrics"]["realized_trial_count"] = realized_trial_count
                            run_payload["metrics"]["hpo_trial_count"] = realized_trial_count
                            for trial in list(run_payload.get("hpo_trials", [])):
                                hpo_trial_rows.append(
                                    {
                                        "benchmark_id": benchmark_id,
                                        "dataset_id": track_dataset_id,
                                        "method_id": method_id,
                                        "split_id": track_split_id,
                                        "seed": int(split.seed),
                                        "track_id": track.track_id,
                                        "hpo_mode": hpo_mode,
                                        "parity_key": parity_key,
                                        "hpo_status": hpo_metadata.get("status", "disabled"),
                                        **trial,
                                    }
                                )
                            run_records.append(run_payload)
                            record["dataset_id"] = track_dataset_id
                            record["split_id"] = track_split_id
                            record["hpo_mode"] = hpo_mode
                            record["parity_key"] = parity_key
                            record["parity_eligible"] = True
                            record["robustness_track_id"] = track.track_id
                            record["retry_attempt"] = int(attempt)
                            record["requested_max_trials"] = hpo_metadata["requested_max_trials"]
                            record["requested_timeout_seconds"] = hpo_metadata["requested_timeout_seconds"]
                            record["requested_sampler"] = hpo_metadata["requested_sampler"]
                            record["requested_pruner"] = hpo_metadata["requested_pruner"]
                            record["realized_trial_count"] = realized_trial_count
                            all_records.append(record)
                            if record["status"] == "success" or attempt >= max_retries:
                                print(
                                    f"[{record['status']}] [{hpo_mode}] "
                                    f"{track_dataset_id}/{method_id}/{track_split_id}/seed{split.seed} "
                                    f"{primary_metric}={record.get(primary_metric)}"
                                )
                                break
                            attempt += 1

    parity_modes: dict[str, set[str]] = {}
    for run_payload in run_records:
        metrics = run_payload.get("metrics", {})
        status = str(run_payload.get("status", metrics.get("status", ""))).lower()
        parity_key = str(metrics.get("parity_key", ""))
        hpo_mode = str(metrics.get("hpo_mode", ""))
        if status == "success" and parity_key and hpo_mode:
            parity_modes.setdefault(parity_key, set()).add(hpo_mode)

    for run_payload in run_records:
        metrics = run_payload.get("metrics", {})
        parity_key = str(metrics.get("parity_key", ""))
        modes = parity_modes.get(parity_key, set())
        has_both_modes = "no_hpo" in modes and "hpo" in modes
        if has_both_modes:
            metrics["parity_eligible"] = True
            metrics["comparison_ineligible"] = False
            metrics["parity_reason"] = None
            metrics["missing_modes"] = []
        else:
            missing_modes = [mode for mode in _DUAL_HPO_MODE_ORDER if mode not in modes]
            metrics["parity_eligible"] = False
            metrics["comparison_ineligible"] = True
            metrics["parity_reason"] = "missing_counterpart_mode"
            metrics["missing_modes"] = missing_modes

    for row in all_records:
        parity_key = str(row.get("parity_key", ""))
        modes = parity_modes.get(parity_key, set())
        has_both_modes = "no_hpo" in modes and "hpo" in modes
        if has_both_modes:
            row["parity_eligible"] = True
            row["comparison_ineligible"] = False
            row["parity_reason"] = None
            row["missing_modes"] = []
        else:
            missing_modes = [mode for mode in _DUAL_HPO_MODE_ORDER if mode not in modes]
            row["parity_eligible"] = False
            row["comparison_ineligible"] = True
            row["parity_reason"] = "missing_counterpart_mode"
            row["missing_modes"] = missing_modes

    frame = export_fold_results(repo_root, all_records, output_dir=experiment_dir, file_prefix=benchmark_id)
    seed_summary = export_seed_summary(repo_root, frame, output_dir=experiment_dir, file_prefix=benchmark_id)
    export_overall_summary(repo_root, frame, output_dir=experiment_dir, file_prefix=benchmark_id)
    leaderboard = export_leaderboard(
        repo_root,
        seed_summary,
        primary_metric=primary_metric,
        output_dir=experiment_dir,
        file_prefix=benchmark_id,
    )
    exports_cfg = dict(benchmark_cfg.get("exports") or {})
    export_manuscript_comparison(
        repo_root,
        leaderboard,
        primary_metric=primary_metric,
        fold_results=frame,
        output_dir=experiment_dir,
        file_prefix=benchmark_id,
        artifact_layout=str(exports_cfg.get("manuscript_artifact_layout", "full")),
    )
    export_dataset_curation_table(repo_root, dataset_curation_rows, benchmark_id=benchmark_id, output_dir=experiment_dir)
    export_run_ledger(repo_root, run_records, benchmark_id=benchmark_id, output_dir=experiment_dir)
    export_hpo_trials(repo_root, hpo_trial_rows, benchmark_id=benchmark_id, output_dir=experiment_dir)
    print(f"Benchmark run complete. Outputs saved to: {experiment_dir}")
