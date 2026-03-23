from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import pickle
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from survarena.automl.presets import PresetConfig, resolve_preset
from survarena.automl.validation import build_validation_plan, prepare_validation_fold_cache
from survarena.benchmark.tuning import (
    resolve_runtime_method_params as _resolve_runtime_method_params,
    tune_hyperparameters,
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
from survarena.methods.registry import method_registry as _method_registry, registered_method_ids
from survarena.utils.quiet import quiet_training_output


_PREDICTOR_SERIALIZATION_VERSION = 1


class _PredictorUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "src":
            module = "survarena"
        elif module.startswith("src."):
            module = f"survarena.{module.removeprefix('src.')}"
        return super().find_class(module, name)


def _configure_plotting_cache() -> None:
    cache_root = Path("/tmp") / "survarena_mpl_cache"
    (cache_root / "mplconfig").mkdir(parents=True, exist_ok=True)
    (cache_root / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_root / "mplconfig")
    os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg")


@dataclass(slots=True)
class PredictorModelResult:
    method_id: str
    selection_score: float
    validation_metrics: dict[str, float]
    fit_time_sec: float
    n_trials_completed: int
    params: dict[str, Any]
    status: str = "success"
    error: str | None = None
    error_type: str | None = None


class SurvivalPredictor:
    def __init__(
        self,
        *,
        label_time: str,
        label_event: str,
        eval_metric: str = "harrell_c",
        presets: str = "all",
        num_trials: int | None = None,
        included_models: list[str] | None = None,
        excluded_models: list[str] | None = None,
        random_state: int = 0,
        save_path: str | Path | None = None,
        verbose: bool = False,
        enable_foundation_models: bool = False,
    ) -> None:
        self.label_time = label_time
        self.label_event = label_event
        self.eval_metric = eval_metric
        self.presets = presets
        self.num_trials = num_trials
        self.included_models = included_models
        self.excluded_models = excluded_models
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
        self.model_preprocessors_: dict[str, TabularPreprocessor] = {}
        self.model_survival_times_: dict[str, np.ndarray] = {}
        self.model_test_metrics_: dict[str, dict[str, float]] = {}
        self._ensure_runtime_state_defaults()

    def _ensure_runtime_state_defaults(self) -> None:
        if not hasattr(self, "validation_strategy_"):
            self.validation_strategy_: str | None = None
        if not hasattr(self, "validation_holdout_frac_"):
            self.validation_holdout_frac_: float | None = None
        if not hasattr(self, "validation_rows_"):
            self.validation_rows_: int | None = None
        if not hasattr(self, "selection_train_rows_"):
            self.selection_train_rows_: int | None = None

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
    ) -> "SurvivalPredictor":
        self._reset_fit_state()
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

        repo_root = Path(__file__).resolve().parents[2]
        method_cfg_cache = {
            method_id: read_yaml(repo_root / "configs" / "methods" / f"{method_id}.yaml")
            for method_id in self.preset_config_.method_ids
        }

        results: list[PredictorModelResult] = []
        for method_id in self.preset_config_.method_ids:
            started_at = perf_counter()
            method_cfg = method_cfg_cache[method_id]
            try:
                fold_cache = prepare_validation_fold_cache(
                    method_id=method_id,
                    plan=validation_plan,
                )
                tuning_result = tune_hyperparameters(
                    method_id=method_id,
                    method_cfg=method_cfg,
                    fold_cache=fold_cache,
                    primary_metric=self.eval_metric,
                    n_trials=self.num_trials if self.num_trials is not None else self.preset_config_.n_trials,
                    seed=self.random_state,
                    timeout_seconds=None,
                    quiet=not self.verbose,
                    metric_bundle_callback=self._collect_fold_metric_bundle,
                )
                metric_rows = tuning_result.get("best_metric_rows")
                validation_metrics = (
                    self._summarize_metric_rows(metric_rows)
                    if metric_rows
                    else self._fold_cache_metric_summary(
                        method_id=method_id,
                        params=dict(tuning_result["best_params"]),
                        fold_cache=fold_cache,
                    )
                )
                results.append(
                    PredictorModelResult(
                        method_id=method_id,
                        selection_score=float(validation_metrics[f"validation_{self.eval_metric}"]),
                        validation_metrics=validation_metrics,
                        fit_time_sec=float(perf_counter() - started_at),
                        n_trials_completed=int(tuning_result["n_trials_completed"]),
                        params=dict(tuning_result["best_params"]),
                    )
                )
            except Exception as exc:
                results.append(
                    PredictorModelResult(
                        method_id=method_id,
                        selection_score=float("nan"),
                        validation_metrics={},
                        fit_time_sec=float(perf_counter() - started_at),
                        n_trials_completed=0,
                        params={},
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
        self._fit_successful_models(dataset=dataset, results=successful)

        best_result = max(successful, key=lambda result: result.selection_score)
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

        self.leaderboard_ = self._build_leaderboard(results)
        self._persist_artifacts(dataset_name, results)
        return self

    def leaderboard(self) -> pd.DataFrame:
        if self.leaderboard_ is None:
            raise RuntimeError("Call fit() before requesting the leaderboard.")
        return self.leaderboard_.copy()

    def model_names(self) -> list[str]:
        return list(self.fitted_models_.keys())

    def foundation_model_catalog(self) -> pd.DataFrame:
        implemented_method_ids = set(registered_method_ids())
        rows: list[dict[str, Any]] = []
        for spec in foundation_model_catalog():
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
                    "notes": spec.notes,
                }
            )
        return pd.DataFrame(rows)

    def predict_risk(self, data: pd.DataFrame | str | Path, *, model: str | None = None) -> np.ndarray:
        dataset, resolved_model, _ = self._prepare_prediction_inputs(data, model=model)
        return resolved_model.predict_risk(dataset.to_numpy())

    def predict_survival(
        self,
        data: pd.DataFrame | str | Path,
        times: np.ndarray | list[float] | None = None,
        *,
        model: str | None = None,
    ) -> pd.DataFrame:
        dataset, resolved_model, default_times = self._prepare_prediction_inputs(data, model=model)
        survival_times = np.asarray(times, dtype=float) if times is not None else default_times
        survival = resolved_model.predict_survival(dataset.to_numpy(), survival_times)
        columns = [f"t_{time:.6g}" for time in survival_times]
        return pd.DataFrame(survival, columns=columns, index=dataset.index)

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
            predictor = _PredictorUnpickler(handle).load()
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
        X_eval = self._require_preprocessor().transform(dataset.X)
        risk_scores = self.best_model_.predict_risk(X_eval.to_numpy())
        survival_times = self._require_survival_times()
        survival = self.best_model_.predict_survival(X_eval.to_numpy(), survival_times)

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
            "selection_train_rows": self.selection_train_rows_,
            "validation_rows": self.validation_rows_,
            "preset": self.preset_config_.name,
            "portfolio": list(self.preset_config_.method_ids),
            "portfolio_notes": list(self.preset_config_.portfolio_notes),
            "trained_models": self.model_names(),
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
                "n_trials_completed": result.n_trials_completed,
                "status": result.status,
                "error": result.error,
                "error_type": result.error_type,
                "params": result.params,
            }
            row.update(result.validation_metrics)
            row.update(self.model_test_metrics_.get(result.method_id, {}))
            rows.append(row)

        leaderboard = pd.DataFrame(rows)
        leaderboard["_status_rank"] = leaderboard["status"].map({"success": 0, "failed": 1}).fillna(2)
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
        method_registry = _method_registry()
        bundle_rows: list[dict[str, float]] = []
        with quiet_training_output(enabled=not self.verbose):
            for fold_data in fold_cache:
                model = method_registry[method_id](**_resolve_runtime_method_params(params, seed=self.random_state))
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
            f"validation_{name}": float(bundle_frame[name].mean())
            for name in MetricBundle.__annotations__.keys()
        }
        metric_summary["validation_primary_metric"] = float(metric_summary[f"validation_{self.eval_metric}"])
        return metric_summary

    def _prepare_prediction_inputs(
        self,
        data: pd.DataFrame | str | Path,
        *,
        model: str | None,
    ) -> tuple[pd.DataFrame, Any, np.ndarray]:
        model_id = self._resolve_model_id(model)
        frame = self._read_features(data)
        transformed = self.model_preprocessors_[model_id].transform(frame)
        return transformed, self.fitted_models_[model_id], self.model_survival_times_[model_id]

    def _read_features(self, data: pd.DataFrame | str | Path) -> pd.DataFrame:
        frame = read_tabular_data(data)
        removable = [col for col in (self.label_time, self.label_event) if col in frame.columns]
        return frame.drop(columns=removable, errors="ignore").reset_index(drop=True)

    def _evaluate_dataset(self, dataset: SurvivalDataset) -> dict[str, float]:
        X_eval = self._require_preprocessor().transform(dataset.X)
        risk = self.best_model_.predict_risk(X_eval.to_numpy())
        survival_times = self._require_survival_times()
        survival = self.best_model_.predict_survival(X_eval.to_numpy(), survival_times)
        metrics = self._compute_metric_bundle_safe(
            train_time=self.dataset_.time,
            train_event=self.dataset_.event,
            test_time=dataset.time,
            test_event=dataset.event,
            risk_scores=risk,
            survival_probs=survival,
            survival_times=survival_times,
        )
        return {f"test_{name}": float(value) for name, value in metrics.items()}

    def _evaluate_fitted_models(self, dataset: SurvivalDataset) -> None:
        self.model_test_metrics_ = {}
        for method_id in self.model_names():
            X_eval = self.model_preprocessors_[method_id].transform(dataset.X)
            risk = self.fitted_models_[method_id].predict_risk(X_eval.to_numpy())
            survival_times = self.model_survival_times_[method_id]
            survival = self.fitted_models_[method_id].predict_survival(X_eval.to_numpy(), survival_times)
            metrics = self._compute_metric_bundle_safe(
                train_time=self.dataset_.time,
                train_event=self.dataset_.event,
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

    def _require_preprocessor(self) -> TabularPreprocessor:
        if self.best_preprocessor_ is None:
            raise RuntimeError("Preprocessor is unavailable before fit().")
        return self.best_preprocessor_

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

    def _fit_successful_models(
        self,
        *,
        dataset: SurvivalDataset,
        results: list[PredictorModelResult],
    ) -> None:
        method_registry = _method_registry()
        self.fitted_models_ = {}
        self.model_preprocessors_ = {}
        self.model_survival_times_ = {}
        self.model_test_metrics_ = {}
        for result in results:
            preprocessor = TabularPreprocessor(scale_numeric=(result.method_id != "rsf"))
            X_train_proc = preprocessor.fit_transform(dataset.X)
            model = method_registry[result.method_id](
                **_resolve_runtime_method_params(result.params, seed=self.random_state)
            )
            with quiet_training_output(enabled=not self.verbose):
                model.fit(X_train_proc.to_numpy(), dataset.time, dataset.event)
            self.fitted_models_[result.method_id] = model
            self.model_preprocessors_[result.method_id] = preprocessor
            self.model_survival_times_[result.method_id] = self._default_survival_times(dataset.time, dataset.event)

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
        }

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
                "num_trials": self.num_trials,
                "random_state": self.random_state,
                "verbose": self.verbose,
                "enable_foundation_models": self.enable_foundation_models,
                "validation_strategy": self.validation_strategy_,
                "holdout_frac": self.validation_holdout_frac_,
                "selection_train_rows": self.selection_train_rows_,
                "validation_rows": self.validation_rows_,
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
