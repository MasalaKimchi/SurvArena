from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import pickle
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from survarena.automl.bagging import BaggedModelMember, BaggedSurvivalEnsemble
from survarena.automl.presets import PresetConfig, resolve_preset
from survarena.automl.validation import (
    bagging_row_summary,
    build_bagging_folds,
    build_refit_dataset,
    build_validation_plan,
    prepare_resampled_fold_cache,
    prepare_validation_fold_cache,
)
from survarena.benchmark.tuning import (
    resolve_runtime_method_params as _resolve_runtime_method_params,
    select_hyperparameters,
)
from survarena.config import read_yaml
from survarena.data.io import read_tabular_data
from survarena.data.preprocess import TabularPreprocessor
from survarena.data.schema import SurvivalDataset
from survarena.data.user_dataset import load_user_dataset
from survarena.evaluation.metrics import (
    MetricBundle,
    compute_harrell_c_index,
    compute_survival_metrics,
    horizons_from_train_event_times,
)
from survarena.logging.tracker import write_json
from survarena.methods.foundation.catalog import foundation_model_catalog
from survarena.methods.foundation.readiness import foundation_runtime_status
from survarena.methods.preprocessing import finalize_preprocessed_features, method_preprocessor_kwargs
from survarena.methods.registry import get_method_class, registered_method_ids
from survarena.utils.quiet import quiet_training_output


_PREDICTOR_SERIALIZATION_VERSION = 1
_SELECTION_TIME_BUDGET_RATIO = 0.8


def _configure_plotting_cache() -> None:
    cache_root = Path("/tmp") / "survarena_mpl_cache"
    (cache_root / "mplconfig").mkdir(parents=True, exist_ok=True)
    (cache_root / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_root / "mplconfig")
    os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg")


def _get_method_class(method_id: str) -> Any:
    return get_method_class(method_id)


@dataclass(slots=True)
class PredictorModelResult:
    method_id: str
    selection_score: float
    validation_metrics: dict[str, float]
    fit_time_sec: float
    selection_evaluations: int
    params: dict[str, Any]
    training_backend: str = "native"
    hpo_backend: str = "none"
    autogluon_presets: Any | None = None
    autogluon_best_model: str | None = None
    autogluon_model_count: int = 0
    autogluon_path: str | None = None
    bagging_folds: int = 0
    stack_levels: int = 0
    time_limit_sec: float | None = None
    retained_for_inference: bool = False
    status: str = "success"
    error: str | None = None
    error_type: str | None = None


class SurvivalPredictor:
    def __init__(
        self,
        *,
        label_time: str,
        label_event: str,
        eval_metric: str = "uno_c",
        presets: str = "all",
        included_models: list[str] | None = None,
        excluded_models: list[str] | None = None,
        retain_top_k_models: int | None = 1,
        random_state: int = 0,
        save_path: str | Path | None = None,
        verbose: bool = False,
        enable_foundation_models: bool = False,
    ) -> None:
        self.label_time = label_time
        self.label_event = label_event
        self.eval_metric = eval_metric
        self.presets = presets
        self.included_models = included_models
        self.excluded_models = excluded_models
        self.retain_top_k_models = self._validate_retain_top_k_models(retain_top_k_models)
        self.random_state = int(random_state)
        self.save_path = Path(save_path) if save_path is not None else None
        self.verbose = bool(verbose)
        self.enable_foundation_models = bool(enable_foundation_models)

        self.dataset_: SurvivalDataset | None = None
        self.test_dataset_: SurvivalDataset | None = None
        self.preset_config_: PresetConfig | None = None
        self.leaderboard_: pd.DataFrame | None = None
        self.model_results_: list[PredictorModelResult] = []
        self.best_method_id_: str | None = None
        self.best_params_: dict[str, Any] | None = None
        self.best_model_: Any = None
        self.best_preprocessor_: TabularPreprocessor | None = None
        self.survival_times_: np.ndarray | None = None
        self.test_metrics_: dict[str, float] | None = None
        self.artifact_dir_: Path | None = None
        self.fitted_models_: dict[str, Any] = {}
        self.model_preprocessors_: dict[str, TabularPreprocessor | None] = {}
        self.model_survival_times_: dict[str, np.ndarray] = {}
        self.model_test_metrics_: dict[str, dict[str, float]] = {}
        self._ensure_runtime_state_defaults()

    def _ensure_runtime_state_defaults(self) -> None:
        if not hasattr(self, "retain_top_k_models"):
            self.retain_top_k_models: int | None = 1
        if not hasattr(self, "validation_strategy_"):
            self.validation_strategy_: str | None = None
        if not hasattr(self, "validation_holdout_frac_"):
            self.validation_holdout_frac_: float | None = None
        if not hasattr(self, "validation_rows_"):
            self.validation_rows_: int | None = None
        if not hasattr(self, "selection_train_rows_"):
            self.selection_train_rows_: int | None = None
        if not hasattr(self, "fit_time_limit_sec_"):
            self.fit_time_limit_sec_: float | None = None
        if not hasattr(self, "selection_time_budget_sec_"):
            self.selection_time_budget_sec_: float | None = None
        if not hasattr(self, "fit_elapsed_sec_"):
            self.fit_elapsed_sec_: float | None = None
        if not hasattr(self, "refit_full_"):
            self.refit_full_: bool = True
        if not hasattr(self, "final_train_rows_"):
            self.final_train_rows_: int | None = None
        if not hasattr(self, "hyperparameter_tune_kwargs_"):
            self.hyperparameter_tune_kwargs_: dict[str, Any] | None = None
        if not hasattr(self, "refit_dataset_"):
            self.refit_dataset_: SurvivalDataset | None = None
        if not hasattr(self, "num_bag_folds_"):
            self.num_bag_folds_: int = 0
        if not hasattr(self, "num_bag_sets_"):
            self.num_bag_sets_: int = 1

    def _reset_fit_state(self) -> None:
        self.dataset_ = None
        self.test_dataset_ = None
        self.preset_config_ = None
        self.leaderboard_ = None
        self.model_results_ = []
        self.best_method_id_ = None
        self.best_params_ = None
        self.best_model_ = None
        self.best_preprocessor_ = None
        self.survival_times_ = None
        self.test_metrics_ = None
        self.artifact_dir_ = None
        self.fitted_models_ = {}
        self.model_preprocessors_ = {}
        self.model_survival_times_ = {}
        self.model_test_metrics_ = {}
        self.validation_strategy_ = None
        self.validation_holdout_frac_ = None
        self.validation_rows_ = None
        self.selection_train_rows_ = None
        self.fit_time_limit_sec_ = None
        self.selection_time_budget_sec_ = None
        self.fit_elapsed_sec_ = None
        self.refit_full_ = True
        self.final_train_rows_ = None
        self.hyperparameter_tune_kwargs_ = None
        self.refit_dataset_ = None
        self.num_bag_folds_ = 0
        self.num_bag_sets_ = 1

    def fit(
        self,
        train_data: pd.DataFrame | str | Path,
        *,
        tuning_data: pd.DataFrame | str | Path | None = None,
        test_data: pd.DataFrame | str | Path | None = None,
        dataset_name: str = "user_dataset",
        id_col: str | None = None,
        drop_columns: list[str] | None = None,
        holdout_frac: float | None = None,
        time_limit: float | None = None,
        hyperparameter_tune_kwargs: dict[str, Any] | None = None,
        refit_full: bool = True,
        num_bag_folds: int = 0,
        num_bag_sets: int = 1,
    ) -> "SurvivalPredictor":
        self._reset_fit_state()
        fit_started_at = perf_counter()
        dataset = load_user_dataset(
            train_data,
            time_col=self.label_time,
            event_col=self.label_event,
            dataset_id=dataset_name,
            dataset_name=dataset_name,
            id_col=id_col,
            drop_columns=drop_columns,
        )
        self.dataset_ = dataset
        tuning_dataset = (
            load_user_dataset(
                tuning_data,
                time_col=self.label_time,
                event_col=self.label_event,
                dataset_id=f"{dataset_name}_tuning",
                dataset_name=f"{dataset_name}_tuning",
                id_col=id_col,
                drop_columns=drop_columns,
            )
            if tuning_data is not None
            else None
        )
        self.preset_config_ = resolve_preset(
            self.presets,
            n_rows=len(dataset.X),
            n_features=dataset.X.shape[1],
            event_count=dataset.metadata.diagnostics.n_events if dataset.metadata.diagnostics is not None else None,
            event_fraction=dataset.metadata.diagnostics.event_rate if dataset.metadata.diagnostics is not None else None,
            high_cardinality_feature_count=(
                len(dataset.metadata.diagnostics.high_cardinality_features)
                if dataset.metadata.diagnostics is not None
                else 0
            ),
            has_datetime_features="datetime" in dataset.metadata.feature_types,
            has_text_features="text" in dataset.metadata.feature_types,
            included_models=self.included_models,
            excluded_models=self.excluded_models,
            enable_foundation_models=self.enable_foundation_models,
        )
        resolved_time_limit = self._validate_time_limit(time_limit)
        resolved_tuning_controls = self._resolve_hyperparameter_tune_kwargs(hyperparameter_tune_kwargs)
        selection_time_budget = (
            None if resolved_time_limit is None else resolved_time_limit * _SELECTION_TIME_BUDGET_RATIO
        )
        self.fit_time_limit_sec_ = resolved_time_limit
        self.selection_time_budget_sec_ = selection_time_budget
        self.hyperparameter_tune_kwargs_ = resolved_tuning_controls
        self.refit_full_ = bool(refit_full)
        self.num_bag_folds_ = self._validate_num_bag_folds(num_bag_folds)
        self.num_bag_sets_ = self._validate_num_bag_sets(num_bag_sets, num_bag_folds=self.num_bag_folds_)

        validation_plan: Any = None
        selection_folds: list[Any] | None
        if self.num_bag_folds_ >= 2:
            selection_folds = build_bagging_folds(
                dataset,
                num_bag_folds=self.num_bag_folds_,
                num_bag_sets=self.num_bag_sets_,
                seed=self.random_state,
            )
            self.validation_strategy_ = "bagged_oof"
            self.validation_holdout_frac_ = None
            self.selection_train_rows_, self.validation_rows_ = bagging_row_summary(selection_folds)
        else:
            selection_folds = None
            validation_plan = build_validation_plan(
                dataset,
                tuning_dataset=tuning_dataset,
                holdout_frac=holdout_frac if holdout_frac is not None else self.preset_config_.holdout_frac,
                seed=self.random_state,
            )
            self.validation_strategy_ = validation_plan.source
            self.validation_holdout_frac_ = validation_plan.holdout_frac
            self.selection_train_rows_ = int(len(validation_plan.train_X))
            self.validation_rows_ = int(len(validation_plan.validation_X))
        refit_dataset = build_refit_dataset(
            dataset,
            validation_plan=validation_plan,
            tuning_dataset=tuning_dataset,
            refit_full=self.refit_full_,
        )
        self.refit_dataset_ = refit_dataset
        self.final_train_rows_ = int(len(refit_dataset.X))

        repo_root = Path(__file__).resolve().parents[2]
        method_cfg_cache = {
            method_id: read_yaml(repo_root / "configs" / "methods" / f"{method_id}.yaml")
            for method_id in self.preset_config_.method_ids
        }
        fit_level_tuning_timeout = self._resolve_tuning_timeout_seconds(resolved_tuning_controls)

        results: list[PredictorModelResult] = []
        method_ids = list(self.preset_config_.method_ids)
        for method_index, method_id in enumerate(method_ids):
            started_at = perf_counter()
            method_cfg = method_cfg_cache[method_id]
            method_time_limit = self._next_method_time_limit(
                fit_started_at=fit_started_at,
                selection_time_budget=selection_time_budget,
                remaining_methods=len(method_ids) - method_index,
            )
            method_time_limit = self._merge_time_limits(method_time_limit, fit_level_tuning_timeout)
            if method_time_limit is not None and method_time_limit <= 0.0:
                self._append_budget_exhausted_results(
                    results=results,
                    method_ids=method_ids[method_index:],
                )
                break
            try:
                method_cfg = self._method_cfg_with_autogluon_controls(
                    method_id=method_id,
                    method_cfg=method_cfg,
                    time_limit=method_time_limit,
                    tune_kwargs=resolved_tuning_controls,
                )
                if selection_folds is not None:
                    fold_cache = prepare_resampled_fold_cache(method_id=method_id, folds=selection_folds)
                else:
                    fold_cache = prepare_validation_fold_cache(
                        method_id=method_id,
                        plan=validation_plan,
                    )
                selection_result = select_hyperparameters(
                    method_id=method_id,
                    method_cfg=method_cfg,
                    fold_cache=fold_cache,
                    primary_metric=self.eval_metric,
                    seed=self.random_state,
                    quiet=not self.verbose,
                    metric_bundle_callback=self._collect_fold_metric_bundle,
                )
                metric_rows = selection_result.get("best_metric_rows")
                validation_metrics = (
                    self._summarize_metric_rows(metric_rows)
                    if metric_rows
                    else self._fold_cache_metric_summary(
                        method_id=method_id,
                        params=dict(selection_result["best_params"]),
                        fold_cache=fold_cache,
                    )
                )
                results.append(
                    PredictorModelResult(
                        method_id=method_id,
                        selection_score=float(validation_metrics[f"validation_{self.eval_metric}"]),
                        validation_metrics=validation_metrics,
                        fit_time_sec=float(perf_counter() - started_at),
                        selection_evaluations=1,
                        params=dict(selection_result["best_params"]),
                        training_backend=self._training_backend_for_method(method_id),
                        hpo_backend=self._hpo_backend_for_method(method_id, dict(selection_result["best_params"])),
                        autogluon_presets=dict(selection_result["best_params"]).get("presets"),
                        bagging_folds=int(dict(selection_result["best_params"]).get("num_bag_folds", 0) or 0),
                        stack_levels=int(dict(selection_result["best_params"]).get("num_stack_levels", 0) or 0),
                        time_limit_sec=method_time_limit,
                    )
                )
            except Exception as exc:
                results.append(
                    PredictorModelResult(
                        method_id=method_id,
                        selection_score=float("nan"),
                        validation_metrics={},
                        fit_time_sec=float(perf_counter() - started_at),
                        selection_evaluations=0,
                        params={},
                        training_backend=self._training_backend_for_method(method_id),
                        time_limit_sec=method_time_limit,
                        status="failed",
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
                )

        successful = [result for result in results if result.status == "success"]
        if not successful:
            errors = {result.method_id: result.error for result in results}
            raise RuntimeError(f"All candidate models failed during fitting: {errors}")

        self.model_results_ = results
        best_result = max(successful, key=lambda result: result.selection_score)
        refit_order = [best_result] + [
            result
            for result in sorted(successful, key=lambda result: result.selection_score, reverse=True)
            if result.method_id != best_result.method_id
        ]
        self._fit_successful_models(
            dataset=refit_dataset,
            results=refit_order,
            fit_started_at=fit_started_at,
            time_limit=resolved_time_limit,
            best_method_id=best_result.method_id,
        )
        self.best_method_id_ = best_result.method_id
        self.best_params_ = dict(best_result.params)
        self.best_model_ = self.fitted_models_[best_result.method_id]
        self.best_preprocessor_ = self.model_preprocessors_[best_result.method_id]
        self.survival_times_ = self.model_survival_times_[best_result.method_id]

        if test_data is not None:
            self.test_dataset_ = load_user_dataset(
                test_data,
                time_col=self.label_time,
                event_col=self.label_event,
                dataset_id=f"{dataset_name}_test",
                dataset_name=f"{dataset_name}_test",
                id_col=id_col,
                drop_columns=drop_columns,
            )
            self._evaluate_fitted_models(self.test_dataset_)
            self.test_metrics_ = dict(self.model_test_metrics_.get(self.best_method_id_, {}))

        self.fit_elapsed_sec_ = float(perf_counter() - fit_started_at)
        self.leaderboard_ = self._build_leaderboard(results)
        self._persist_artifacts(dataset_name, results)
        return self

    def leaderboard(self) -> pd.DataFrame:
        if self.leaderboard_ is None:
            raise RuntimeError("Call fit() before requesting the leaderboard.")
        return self.leaderboard_.copy()

    def model_names(self) -> list[str]:
        retained_model_ids = {result.method_id for result in self.model_results_ if result.retained_for_inference}
        if retained_model_ids:
            return [result.method_id for result in self.model_results_ if result.method_id in retained_model_ids]
        return list(self.fitted_models_.keys())

    def foundation_model_catalog(self) -> pd.DataFrame:
        implemented_method_ids = set(registered_method_ids())
        rows: list[dict[str, Any]] = []
        for spec in foundation_model_catalog():
            runtime_status = foundation_runtime_status(spec)
            rows.append(
                {
                    "method_id": spec.method_id,
                    "backbone": spec.backbone,
                    "provider": spec.provider,
                    "status": spec.status,
                    "implemented": spec.method_id in implemented_method_ids,
                    "task_support": list(spec.task_support),
                    "supports_finetune": spec.supports_finetune,
                    "supports_pretrained_weights": spec.supports_pretrained_weights,
                    "dependency_installed": runtime_status.dependency_installed,
                    "runtime_ready": runtime_status.runtime_ready,
                    "requires_hf_auth": runtime_status.requires_hf_auth,
                    "auth_configured": runtime_status.auth_configured,
                    "install_extra": runtime_status.install_extra,
                    "install_command": runtime_status.install_command,
                    "blocked_reason": runtime_status.blocked_reason,
                    "warning_reason": runtime_status.warning_reason,
                    "notes": spec.notes,
                }
            )
        return pd.DataFrame(rows)

    def predict_risk(self, data: pd.DataFrame | str | Path, *, model: str | None = None) -> np.ndarray:
        frame, model_id, _ = self._prepare_prediction_inputs(data, model=model)
        return self._predict_model_risk(model_id, frame)

    def predict_survival(
        self,
        data: pd.DataFrame | str | Path,
        times: np.ndarray | list[float] | None = None,
        *,
        model: str | None = None,
    ) -> pd.DataFrame:
        frame, model_id, default_times = self._prepare_prediction_inputs(data, model=model)
        survival_times = np.asarray(times, dtype=float) if times is not None else default_times
        survival = self._predict_model_survival(model_id, frame, survival_times)
        columns = [f"t_{time:.6g}" for time in survival_times]
        return pd.DataFrame(survival, columns=columns, index=frame.index)

    def save(self, path: str | Path | None = None) -> Path:
        output_path = Path(path) if path is not None else self._default_predictor_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        write_json(self._predictor_manifest_path(output_path), self._serialization_manifest(output_path))
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "SurvivalPredictor":
        path = Path(path)
        with path.open("rb") as handle:
            predictor = pickle.load(handle)
        if not isinstance(predictor, cls):
            raise TypeError(f"Serialized object at '{path}' is not a {cls.__name__}.")
        predictor._ensure_runtime_state_defaults()
        manifest_path = predictor._predictor_manifest_path(path)
        if manifest_path.exists():
            manifest = read_yaml(manifest_path)
            expected_version = int(manifest.get("serialization_version", _PREDICTOR_SERIALIZATION_VERSION))
            if expected_version != _PREDICTOR_SERIALIZATION_VERSION:
                raise RuntimeError(
                    f"Unsupported predictor serialization version {expected_version}; "
                    f"expected {_PREDICTOR_SERIALIZATION_VERSION}."
                )
        return predictor

    def plot_kaplan_meier_comparison(
        self,
        data: pd.DataFrame | str | Path | None = None,
        *,
        n_groups: int = 2,
        ax: Any | None = None,
        title: str | None = None,
        show_predicted: bool = True,
        save_path: str | Path | None = None,
    ) -> Any:
        if n_groups < 2:
            raise ValueError("n_groups must be at least 2.")
        dataset = self._resolve_labeled_dataset(data)
        best_model_id = self._resolve_model_id("best")
        risk_scores = self._predict_model_risk(best_model_id, dataset.X)
        survival_times = self._require_survival_times()
        survival = self._predict_model_survival(best_model_id, dataset.X, survival_times)

        self._prepare_matplotlib_env()

        import matplotlib.pyplot as plt
        from lifelines import KaplanMeierFitter

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        groups = self._risk_groups(risk_scores, n_groups=n_groups)
        for group_id in sorted(groups.unique()):
            mask = groups == group_id
            label_prefix = self._risk_group_label(group_id, n_groups)
            kmf = KaplanMeierFitter()
            kmf.fit(dataset.time[mask], dataset.event[mask].astype(bool), label=f"{label_prefix} empirical KM")
            kmf.plot_survival_function(ax=ax, ci_show=False)
            if show_predicted:
                mean_survival = survival[mask].mean(axis=0)
                ax.plot(
                    survival_times,
                    mean_survival,
                    linestyle="--",
                    linewidth=2,
                    label=f"{label_prefix} predicted mean",
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Survival probability")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title or f"Kaplan-Meier comparison: {self.best_method_id_}")
        ax.legend()

        output_path = Path(save_path) if save_path is not None else None
        if output_path is None and self.artifact_dir_ is not None:
            output_path = self.artifact_dir_ / "kaplan_meier_comparison.png"
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            ax.figure.savefig(output_path, bbox_inches="tight")
        return ax

    def fit_summary(self) -> dict[str, Any]:
        if self.best_method_id_ is None or self.preset_config_ is None:
            raise RuntimeError("Call fit() before requesting the fit summary.")
        summary: dict[str, Any] = {
            "best_method_id": self.best_method_id_,
            "best_params": dict(self.best_params_ or {}),
            "selection_metric": self.eval_metric,
            "selection_metric_column": f"validation_{self.eval_metric}",
            "validation_strategy": self.validation_strategy_,
            "validation_holdout_frac": self.validation_holdout_frac_,
            "num_bag_folds": self.num_bag_folds_,
            "num_bag_sets": self.num_bag_sets_,
            "selection_train_rows": self.selection_train_rows_,
            "validation_rows": self.validation_rows_,
            "refit_full": self.refit_full_,
            "final_train_rows": self.final_train_rows_,
            "hyperparameter_tune_kwargs": dict(self.hyperparameter_tune_kwargs_ or {}),
            "time_limit_sec": self.fit_time_limit_sec_,
            "selection_time_budget_sec": self.selection_time_budget_sec_,
            "fit_elapsed_sec": self.fit_elapsed_sec_,
            "preset": self.preset_config_.name,
            "portfolio": list(self.preset_config_.method_ids),
            "portfolio_notes": list(self.preset_config_.portfolio_notes),
            "trained_models": self.model_names(),
            "retain_top_k_models": self.retain_top_k_models,
            "foundation_models_enabled": self.enable_foundation_models,
            "foundation_model_catalog": self.foundation_model_catalog().to_dict(orient="records"),
        }
        if self.dataset_ is not None and self.dataset_.metadata.diagnostics is not None:
            summary["dataset_diagnostics"] = self.dataset_.metadata.diagnostics.to_dict()
        if self.test_metrics_ is not None:
            summary["test_metrics"] = dict(self.test_metrics_)
        if self.model_test_metrics_:
            summary["per_model_test_metrics"] = {
                method_id: dict(metrics) for method_id, metrics in self.model_test_metrics_.items()
            }
        if self.artifact_dir_ is not None:
            summary["artifact_dir"] = str(self.artifact_dir_)
        return summary

    def _build_leaderboard(self, results: list[PredictorModelResult]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for result in results:
            row: dict[str, Any] = {
                "method_id": result.method_id,
                "selection_metric": self.eval_metric,
                "selection_score": result.selection_score,
                "fit_time_sec": result.fit_time_sec,
                "training_backend": result.training_backend,
                "hpo_backend": result.hpo_backend,
                "autogluon_presets": result.autogluon_presets,
                "autogluon_best_model": result.autogluon_best_model,
                "autogluon_model_count": result.autogluon_model_count,
                "autogluon_path": result.autogluon_path,
                "bagging_folds": result.bagging_folds,
                "stack_levels": result.stack_levels,
                "selection_evaluations": result.selection_evaluations,
                "time_limit_sec": result.time_limit_sec,
                "retained_for_inference": result.retained_for_inference,
                "status": result.status,
                "error": result.error,
                "error_type": result.error_type,
                "params": result.params,
            }
            row.update(result.validation_metrics)
            row.update(self.model_test_metrics_.get(result.method_id, {}))
            rows.append(row)

        leaderboard = pd.DataFrame(rows)
        leaderboard["_status_rank"] = leaderboard["status"].map({"success": 0, "failed": 1, "skipped": 2}).fillna(3)
        leaderboard = leaderboard.sort_values(
            by=["_status_rank", f"validation_{self.eval_metric}"],
            ascending=[True, False],
            na_position="last",
        ).drop(columns=["_status_rank"])
        leaderboard = leaderboard.reset_index(drop=True)
        leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1, dtype=int))
        return leaderboard

    def _fold_cache_metric_summary(
        self,
        *,
        method_id: str,
        params: dict[str, Any],
        fold_cache: list[dict[str, Any]],
    ) -> dict[str, float]:
        method_cls = _get_method_class(method_id)
        bundle_rows: list[dict[str, float]] = []
        with quiet_training_output(enabled=not self.verbose):
            for fold_data in fold_cache:
                model = method_cls(**_resolve_runtime_method_params(params, seed=self.random_state))
                model.fit(
                    fold_data["X_train"],
                    fold_data["time_train"],
                    fold_data["event_train"],
                    fold_data["X_val"],
                    fold_data["time_val"],
                    fold_data["event_val"],
                )
                eval_times = self._default_survival_times(fold_data["time_train"], fold_data["event_train"])
                risk_scores = model.predict_risk(fold_data["X_val"])
                survival_probs = model.predict_survival(fold_data["X_val"], eval_times)
                bundle_rows.append(
                    self._compute_metric_bundle_safe(
                        train_time=fold_data["time_train"],
                        train_event=fold_data["event_train"],
                        test_time=fold_data["time_val"],
                        test_event=fold_data["event_val"],
                        risk_scores=risk_scores,
                        survival_probs=survival_probs,
                        survival_times=eval_times,
                    )
                )

        bundle_frame = pd.DataFrame(bundle_rows)
        return self._summarize_metric_rows(bundle_frame.to_dict(orient="records"))

    def _collect_fold_metric_bundle(
        self,
        fold_data: dict[str, Any],
        model: Any,
        risk_scores: np.ndarray,
    ) -> dict[str, float]:
        eval_times = self._default_survival_times(fold_data["time_train"], fold_data["event_train"])
        survival_probs = model.predict_survival(fold_data["X_val"], eval_times)
        return self._compute_metric_bundle_safe(
            train_time=fold_data["time_train"],
            train_event=fold_data["event_train"],
            test_time=fold_data["time_val"],
            test_event=fold_data["event_val"],
            risk_scores=risk_scores,
            survival_probs=survival_probs,
            survival_times=eval_times,
        )

    def _summarize_metric_rows(self, metric_rows: list[dict[str, float]]) -> dict[str, float]:
        bundle_frame = pd.DataFrame(metric_rows)
        metric_summary = {
            f"validation_{name}": float(bundle_frame[name].mean()) if name in bundle_frame.columns else float("nan")
            for name in MetricBundle.__annotations__.keys()
        }
        metric_summary["validation_primary_metric"] = float(metric_summary[f"validation_{self.eval_metric}"])
        return metric_summary

    def _prepare_prediction_inputs(
        self,
        data: pd.DataFrame | str | Path,
        *,
        model: str | None,
    ) -> tuple[pd.DataFrame, str, np.ndarray]:
        model_id = self._resolve_model_id(model)
        frame = self._read_features(data)
        return frame, model_id, self.model_survival_times_[model_id]

    def _predict_model_risk(self, model_id: str, frame: pd.DataFrame) -> np.ndarray:
        model = self.fitted_models_[model_id]
        if isinstance(model, BaggedSurvivalEnsemble):
            return model.predict_risk(frame)
        preprocessor = self.model_preprocessors_[model_id]
        if preprocessor is None:
            raise RuntimeError(f"Preprocessor is unavailable for model '{model_id}'.")
        transformed = finalize_preprocessed_features(model_id, preprocessor.transform(frame))
        return np.asarray(model.predict_risk(transformed), dtype=float)

    def _predict_model_survival(self, model_id: str, frame: pd.DataFrame, survival_times: np.ndarray) -> np.ndarray:
        model = self.fitted_models_[model_id]
        if isinstance(model, BaggedSurvivalEnsemble):
            return model.predict_survival(frame, survival_times)
        preprocessor = self.model_preprocessors_[model_id]
        if preprocessor is None:
            raise RuntimeError(f"Preprocessor is unavailable for model '{model_id}'.")
        transformed = finalize_preprocessed_features(model_id, preprocessor.transform(frame))
        return np.asarray(model.predict_survival(transformed, survival_times), dtype=float)

    def _read_features(self, data: pd.DataFrame | str | Path) -> pd.DataFrame:
        frame = read_tabular_data(data)
        removable = [col for col in (self.label_time, self.label_event) if col in frame.columns]
        return frame.drop(columns=removable, errors="ignore").reset_index(drop=True)

    def _evaluate_fitted_models(self, dataset: SurvivalDataset) -> None:
        train_reference = self._train_reference_dataset()
        self.model_test_metrics_ = {}
        for method_id in self.model_names():
            risk = self._predict_model_risk(method_id, dataset.X)
            survival_times = self.model_survival_times_[method_id]
            survival = self._predict_model_survival(method_id, dataset.X, survival_times)
            metrics = self._compute_metric_bundle_safe(
                train_time=train_reference.time,
                train_event=train_reference.event,
                test_time=dataset.time,
                test_event=dataset.event,
                risk_scores=risk,
                survival_probs=survival,
                survival_times=survival_times,
            )
            self.model_test_metrics_[method_id] = {f"test_{name}": float(value) for name, value in metrics.items()}

    def _resolve_labeled_dataset(self, data: pd.DataFrame | str | Path | None) -> SurvivalDataset:
        if data is None:
            if self.test_dataset_ is not None:
                return self.test_dataset_
            if self.dataset_ is not None:
                return self.dataset_
            raise RuntimeError("No fitted dataset is available.")
        return load_user_dataset(
            data,
            time_col=self.label_time,
            event_col=self.label_event,
            dataset_id="plot_dataset",
            dataset_name="plot_dataset",
        )

    def _risk_groups(self, risk_scores: np.ndarray, *, n_groups: int) -> pd.Series:
        labels = list(range(n_groups))
        ranked = pd.Series(np.asarray(risk_scores, dtype=float)).rank(method="first")
        return pd.qcut(ranked, q=n_groups, labels=labels)

    def _risk_group_label(self, group_id: int, n_groups: int) -> str:
        if n_groups == 2:
            return "High risk" if int(group_id) == n_groups - 1 else "Low risk"
        return f"Risk group {int(group_id) + 1}"

    def _prepare_matplotlib_env(self) -> None:
        _configure_plotting_cache()

    def _require_survival_times(self) -> np.ndarray:
        if self.survival_times_ is None:
            raise RuntimeError("Survival time grid is unavailable before fit().")
        return self.survival_times_

    def _resolve_model_id(self, model: str | None) -> str:
        if model in {None, "best"}:
            if self.best_method_id_ is None:
                raise RuntimeError("Call fit() before requesting predictions.")
            return self.best_method_id_
        if model not in self.fitted_models_:
            raise ValueError(f"Unknown model '{model}'. Available models: {sorted(self.fitted_models_)}")
        return model

    def _train_reference_dataset(self) -> SurvivalDataset:
        if self.refit_dataset_ is not None:
            return self.refit_dataset_
        if self.dataset_ is not None:
            return self.dataset_
        raise RuntimeError("No training dataset is available.")

    def _fit_successful_models(
        self,
        *,
        dataset: SurvivalDataset,
        results: list[PredictorModelResult],
        fit_started_at: float,
        time_limit: float | None,
        best_method_id: str,
    ) -> None:
        self.fitted_models_ = {}
        self.model_preprocessors_ = {}
        self.model_survival_times_ = {}
        self.model_test_metrics_ = {}
        retention_limit = len(results) if self.retain_top_k_models is None else min(len(results), self.retain_top_k_models)
        retained_count = 0
        for result in results:
            if retained_count >= retention_limit:
                break
            if result.method_id != best_method_id and self._remaining_fit_time(fit_started_at, time_limit) <= 0.0:
                break
            method_cls = _get_method_class(result.method_id)
            if self.num_bag_folds_ >= 2:
                model = self._fit_bagged_model(
                    method_cls=method_cls,
                    method_id=result.method_id,
                    params=result.params,
                    dataset=dataset,
                )
                preprocessor = None
            else:
                model, preprocessor = self._fit_single_model(
                    method_cls=method_cls,
                    method_id=result.method_id,
                    params=result.params,
                    dataset=dataset,
                )
            self.fitted_models_[result.method_id] = model
            self._attach_result_fit_metadata(result, model)
            self.model_preprocessors_[result.method_id] = preprocessor
            self.model_survival_times_[result.method_id] = self._default_survival_times(dataset.time, dataset.event)
            result.retained_for_inference = True
            retained_count += 1

    def _fit_single_model(
        self,
        *,
        method_cls: Any,
        method_id: str,
        params: dict[str, Any],
        dataset: SurvivalDataset,
    ) -> tuple[Any, TabularPreprocessor]:
        preprocessor = TabularPreprocessor(**method_preprocessor_kwargs(method_id))
        X_train_proc = finalize_preprocessed_features(method_id, preprocessor.fit_transform(dataset.X))
        model = method_cls(**_resolve_runtime_method_params(params, seed=self.random_state))
        with quiet_training_output(enabled=not self.verbose):
            model.fit(X_train_proc, dataset.time, dataset.event)
        return model, preprocessor

    def _fit_bagged_model(
        self,
        *,
        method_cls: Any,
        method_id: str,
        params: dict[str, Any],
        dataset: SurvivalDataset,
    ) -> BaggedSurvivalEnsemble:
        bagging_folds = build_bagging_folds(
            dataset,
            num_bag_folds=self.num_bag_folds_,
            num_bag_sets=self.num_bag_sets_,
            seed=self.random_state,
        )
        members: list[BaggedModelMember] = []
        for member_index, fold in enumerate(bagging_folds):
            preprocessor = TabularPreprocessor(**method_preprocessor_kwargs(method_id))
            X_train_proc = finalize_preprocessed_features(method_id, preprocessor.fit_transform(fold.train_X))
            X_validation_proc = finalize_preprocessed_features(method_id, preprocessor.transform(fold.validation_X))
            model = method_cls(
                **_resolve_runtime_method_params(params, seed=self.random_state + member_index)
            )
            with quiet_training_output(enabled=not self.verbose):
                model.fit(
                    X_train_proc,
                    fold.train_time,
                    fold.train_event,
                    X_validation_proc,
                    fold.validation_time,
                    fold.validation_event,
                )
            members.append(BaggedModelMember(method_id=method_id, model=model, preprocessor=preprocessor))
        return BaggedSurvivalEnsemble(members)

    def _validate_time_limit(self, time_limit: float | None) -> float | None:
        if time_limit is None:
            return None
        resolved = float(time_limit)
        if resolved <= 0.0:
            raise ValueError("time_limit must be positive when provided.")
        return resolved

    def _validate_num_bag_folds(self, num_bag_folds: int) -> int:
        resolved = int(num_bag_folds)
        if resolved < 0:
            raise ValueError("num_bag_folds must be >= 0.")
        if resolved == 1:
            raise ValueError("num_bag_folds must be 0 or >= 2.")
        return resolved

    def _validate_num_bag_sets(self, num_bag_sets: int, *, num_bag_folds: int) -> int:
        resolved = int(num_bag_sets)
        if resolved < 1:
            raise ValueError("num_bag_sets must be >= 1.")
        if resolved > 1 and num_bag_folds <= 0:
            raise ValueError("num_bag_sets > 1 requires num_bag_folds >= 2.")
        return resolved

    def _validate_retain_top_k_models(self, retain_top_k_models: int | None) -> int | None:
        if retain_top_k_models is None:
            return None
        resolved = int(retain_top_k_models)
        if resolved < 1:
            raise ValueError("retain_top_k_models must be >= 1 or None.")
        return resolved

    def _resolve_hyperparameter_tune_kwargs(
        self,
        hyperparameter_tune_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if hyperparameter_tune_kwargs is None:
            return None
        if not isinstance(hyperparameter_tune_kwargs, dict):
            raise TypeError("hyperparameter_tune_kwargs must be a dictionary when provided.")

        supported_keys = {"num_trials", "timeout", "timeout_seconds"}
        unexpected_keys = sorted(set(hyperparameter_tune_kwargs) - supported_keys)
        if unexpected_keys:
            raise ValueError(
                "Unsupported hyperparameter_tune_kwargs keys: "
                f"{unexpected_keys}. Supported keys: {sorted(supported_keys)}"
            )
        if "timeout" in hyperparameter_tune_kwargs and "timeout_seconds" in hyperparameter_tune_kwargs:
            raise ValueError("Specify only one of 'timeout' or 'timeout_seconds' in hyperparameter_tune_kwargs.")

        normalized: dict[str, Any] = {}
        num_trials = hyperparameter_tune_kwargs.get("num_trials")
        if num_trials is not None:
            resolved_num_trials = int(num_trials)
            if resolved_num_trials < 0:
                raise ValueError("hyperparameter_tune_kwargs num_trials must be >= 0.")
            normalized["num_trials"] = resolved_num_trials

        timeout_seconds = hyperparameter_tune_kwargs.get("timeout_seconds", hyperparameter_tune_kwargs.get("timeout"))
        if timeout_seconds is not None:
            resolved_timeout = float(timeout_seconds)
            if resolved_timeout <= 0.0:
                raise ValueError("hyperparameter_tune_kwargs timeout must be positive.")
            normalized["timeout_seconds"] = resolved_timeout

        return normalized

    def _resolve_tuning_timeout_seconds(self, fit_tune_kwargs: dict[str, Any] | None) -> float | None:
        if fit_tune_kwargs is None:
            return None
        timeout_seconds = fit_tune_kwargs.get("timeout_seconds")
        return None if timeout_seconds is None else float(timeout_seconds)

    def _remaining_fit_time(self, fit_started_at: float, time_limit: float | None) -> float:
        if time_limit is None:
            return float("inf")
        return max(0.0, float(time_limit) - float(perf_counter() - fit_started_at))

    def _next_method_time_limit(
        self,
        *,
        fit_started_at: float,
        selection_time_budget: float | None,
        remaining_methods: int,
    ) -> float | None:
        if selection_time_budget is None:
            return None
        if remaining_methods <= 0:
            return 0.0
        remaining_budget = self._remaining_fit_time(fit_started_at, selection_time_budget)
        if remaining_budget <= 0.0:
            return 0.0
        return remaining_budget / float(remaining_methods)

    def _merge_time_limits(self, first: float | None, second: float | None) -> float | None:
        limits = [float(limit) for limit in (first, second) if limit is not None]
        if not limits:
            return None
        return min(limits)

    def _append_budget_exhausted_results(
        self,
        *,
        results: list[PredictorModelResult],
        method_ids: list[str],
    ) -> None:
        for method_id in method_ids:
            results.append(
                PredictorModelResult(
                    method_id=method_id,
                    selection_score=float("nan"),
                    validation_metrics={},
                    fit_time_sec=0.0,
                    selection_evaluations=0,
                    params={},
                    training_backend=self._training_backend_for_method(method_id),
                    time_limit_sec=0.0,
                    status="skipped",
                    error="Global fit time budget exhausted before this model could be selected.",
                    error_type="TimeLimitExceeded",
                )
            )

    def _default_predictor_path(self) -> Path:
        if self.artifact_dir_ is None:
            raise RuntimeError("No artifact directory is available. Provide a save path or call fit() first.")
        return self.artifact_dir_ / "predictor.pkl"

    def _predictor_manifest_path(self, output_path: Path) -> Path:
        return output_path.with_name(f"{output_path.stem}_manifest.json")

    def _serialization_manifest(self, output_path: Path) -> dict[str, Any]:
        return {
            "serialization_version": _PREDICTOR_SERIALIZATION_VERSION,
            "class_name": type(self).__name__,
            "module": type(self).__module__,
            "path": str(output_path),
            "best_method_id": self.best_method_id_,
            "trained_models": self.model_names(),
            "eval_metric": self.eval_metric,
            "retain_top_k_models": self.retain_top_k_models,
        }

    def _training_backend_for_method(self, method_id: str) -> str:
        return "autogluon" if method_id == "autogluon_survival" else "native"

    def _hpo_backend_for_method(self, method_id: str, params: dict[str, Any]) -> str:
        if method_id == "autogluon_survival" and params.get("hyperparameter_tune_kwargs"):
            return "autogluon"
        return "none"

    def _method_cfg_with_autogluon_controls(
        self,
        *,
        method_id: str,
        method_cfg: dict[str, Any],
        time_limit: float | None,
        tune_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if method_id != "autogluon_survival":
            return method_cfg
        resolved = dict(method_cfg)
        defaults = dict(method_cfg.get("default_params", {}))
        if time_limit is not None:
            defaults["time_limit"] = float(time_limit)
        if tune_kwargs and tune_kwargs.get("num_trials", 0) > 0:
            defaults["hyperparameter_tune_kwargs"] = {"num_trials": int(tune_kwargs["num_trials"])}
        defaults["num_bag_folds"] = self.num_bag_folds_
        defaults["num_stack_levels"] = 1 if self.num_bag_folds_ >= 2 else 0
        defaults["refit_full"] = self.refit_full_
        defaults["verbosity"] = 2 if self.verbose else 0
        resolved["default_params"] = defaults
        return resolved

    def _attach_result_fit_metadata(self, result: PredictorModelResult, model: Any) -> None:
        metadata_getter = getattr(model, "autogluon_metadata", None)
        if not callable(metadata_getter):
            return
        metadata = dict(metadata_getter())
        result.autogluon_best_model = metadata.get("autogluon_best_model")
        result.autogluon_model_count = int(metadata.get("autogluon_model_count") or 0)
        result.autogluon_path = metadata.get("autogluon_path")

    def _default_survival_times(self, time: np.ndarray, event: np.ndarray) -> np.ndarray:
        event_times = time[event.astype(bool)]
        if event_times.size == 0:
            return np.linspace(1.0, 10.0, 25)
        max_supported = float(np.max(event_times))
        lower = max(1e-8, float(np.percentile(event_times, 5)))
        upper = min(max_supported - 1e-8, float(np.percentile(event_times, 95)))
        upper = max(lower + 1e-8, upper)
        return np.linspace(lower, upper, 50)

    def _compute_metric_bundle_safe(
        self,
        *,
        train_time: np.ndarray,
        train_event: np.ndarray,
        test_time: np.ndarray,
        test_event: np.ndarray,
        risk_scores: np.ndarray,
        survival_probs: np.ndarray,
        survival_times: np.ndarray,
    ) -> dict[str, float]:
        horizons = horizons_from_train_event_times(train_time, train_event)
        try:
            metrics = compute_survival_metrics(
                train_time=train_time,
                train_event=train_event,
                test_time=test_time,
                test_event=test_event,
                risk_scores=risk_scores,
                survival_probs=survival_probs,
                survival_times=survival_times,
                horizons=horizons,
            )
            return metrics.to_dict()
        except ValueError as exc:
            message = str(exc)
            if "largest observed training event time point" not in message:
                raise

            train_event_times = np.asarray(train_time)[np.asarray(train_event).astype(bool)]
            max_supported = float(np.max(train_event_times)) if train_event_times.size else float(np.max(train_time))
            mask = np.asarray(test_time) <= max_supported
            if mask.any():
                clipped_survival_mask = np.asarray(survival_times) <= (max_supported - 1e-8)
                if not clipped_survival_mask.any():
                    clipped_survival_mask = np.zeros_like(np.asarray(survival_times), dtype=bool)
                    clipped_survival_mask[0] = True
                metrics = compute_survival_metrics(
                    train_time=train_time,
                    train_event=train_event,
                    test_time=np.asarray(test_time)[mask],
                    test_event=np.asarray(test_event)[mask],
                    risk_scores=np.asarray(risk_scores)[mask],
                    survival_probs=np.asarray(survival_probs)[mask][:, clipped_survival_mask],
                    survival_times=np.asarray(survival_times)[clipped_survival_mask],
                    horizons=tuple(min(float(h), max_supported - 1e-8) for h in horizons),
                )
                return metrics.to_dict()

            harrell = compute_harrell_c_index(
                eval_time=np.asarray(test_time),
                eval_event=np.asarray(test_event),
                eval_risk_scores=np.asarray(risk_scores),
            )
            return {
                "uno_c": float("nan"),
                "harrell_c": float(harrell),
                "ibs": float("nan"),
                "td_auc_25": float("nan"),
                "td_auc_50": float("nan"),
                "td_auc_75": float("nan"),
            }

    def _persist_artifacts(self, dataset_name: str, results: list[PredictorModelResult]) -> None:
        artifact_root = self.save_path or Path("results") / "predictor"
        artifact_dir = artifact_root / dataset_name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir_ = artifact_dir

        if self.leaderboard_ is not None:
            self.leaderboard_.to_csv(artifact_dir / "leaderboard.csv", index=False)
        payload = {
            "config": {
                "label_time": self.label_time,
                "label_event": self.label_event,
                "eval_metric": self.eval_metric,
                "presets": self.presets,
                "retain_top_k_models": self.retain_top_k_models,
                "random_state": self.random_state,
                "verbose": self.verbose,
                "enable_foundation_models": self.enable_foundation_models,
                "validation_strategy": self.validation_strategy_,
                "holdout_frac": self.validation_holdout_frac_,
                "num_bag_folds": self.num_bag_folds_,
                "num_bag_sets": self.num_bag_sets_,
                "selection_train_rows": self.selection_train_rows_,
                "validation_rows": self.validation_rows_,
                "refit_full": self.refit_full_,
                "final_train_rows": self.final_train_rows_,
                "hyperparameter_tune_kwargs": self.hyperparameter_tune_kwargs_,
                "time_limit": self.fit_time_limit_sec_,
                "selection_time_budget_sec": self.selection_time_budget_sec_,
                "fit_elapsed_sec": self.fit_elapsed_sec_,
            },
            "best_method_id": self.best_method_id_,
            "best_params": self.best_params_ or {},
            "portfolio_notes": list(self.preset_config_.portfolio_notes) if self.preset_config_ is not None else [],
            "dataset_diagnostics": (
                self.dataset_.metadata.diagnostics.to_dict()
                if self.dataset_ is not None and self.dataset_.metadata.diagnostics is not None
                else None
            ),
            "test_metrics": self.test_metrics_,
            "trained_models": self.model_names(),
            "per_model_test_metrics": self.model_test_metrics_,
            "results": [asdict(result) for result in results],
        }
        write_json(artifact_dir / "fit_summary.json", payload)
        self.save(artifact_dir / "predictor.pkl")
