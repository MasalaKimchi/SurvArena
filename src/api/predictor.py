from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.automl.presets import PresetConfig, resolve_preset
from src.data.preprocess import TabularPreprocessor
from src.data.schema import SurvivalDataset
from src.data.user_dataset import load_user_dataset
from src.evaluation.metrics import compute_survival_metrics, horizons_from_train_event_times
from src.logging.tracker import write_json
from src.run_benchmark import (
    _method_registry,
    _prepare_inner_cv_cache,
    _resolve_runtime_method_params,
    read_yaml,
    tune_hyperparameters,
)


@dataclass(slots=True)
class PredictorModelResult:
    method_id: str
    validation_score: float
    fit_time_sec: float
    n_trials_completed: int
    params: dict[str, Any]
    status: str = "success"
    error: str | None = None


class SurvivalPredictor:
    def __init__(
        self,
        *,
        label_time: str,
        label_event: str,
        eval_metric: str = "harrell_c",
        presets: str = "medium",
        num_trials: int | None = None,
        included_models: list[str] | None = None,
        excluded_models: list[str] | None = None,
        random_state: int = 0,
        save_path: str | Path | None = None,
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

        self.dataset_: SurvivalDataset | None = None
        self.preset_config_: PresetConfig | None = None
        self.leaderboard_: pd.DataFrame | None = None
        self.best_method_id_: str | None = None
        self.best_params_: dict[str, Any] | None = None
        self.best_model_: Any = None
        self.best_preprocessor_: TabularPreprocessor | None = None
        self.survival_times_: np.ndarray | None = None
        self.test_metrics_: dict[str, float] | None = None
        self.artifact_dir_: Path | None = None

    def fit(
        self,
        train_data: pd.DataFrame | str | Path,
        *,
        test_data: pd.DataFrame | str | Path | None = None,
        dataset_name: str = "user_dataset",
        id_col: str | None = None,
        drop_columns: list[str] | None = None,
    ) -> "SurvivalPredictor":
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
        self.preset_config_ = resolve_preset(
            self.presets,
            n_rows=len(dataset.X),
            n_features=dataset.X.shape[1],
            included_models=self.included_models,
            excluded_models=self.excluded_models,
        )

        repo_root = Path(__file__).resolve().parents[2]
        method_registry = _method_registry()
        method_cfg_cache = {
            method_id: read_yaml(repo_root / "configs" / "methods" / f"{method_id}.yaml")
            for method_id in self.preset_config_.method_ids
        }

        results: list[PredictorModelResult] = []
        for method_id in self.preset_config_.method_ids:
            started_at = perf_counter()
            method_cfg = method_cfg_cache[method_id]
            try:
                fold_cache = _prepare_inner_cv_cache(
                    method_id=method_id,
                    X_train=dataset.X,
                    time_train=dataset.time,
                    event_train=dataset.event,
                    inner_folds=self.preset_config_.inner_folds,
                    seed=self.random_state,
                )
                tuning_result = tune_hyperparameters(
                    method_id=method_id,
                    method_cfg=method_cfg,
                    fold_cache=fold_cache,
                    primary_metric=self.eval_metric,
                    n_trials=self.num_trials if self.num_trials is not None else self.preset_config_.n_trials,
                    seed=self.random_state,
                    timeout_seconds=None,
                )
                results.append(
                    PredictorModelResult(
                        method_id=method_id,
                        validation_score=float(tuning_result["best_score"]),
                        fit_time_sec=float(perf_counter() - started_at),
                        n_trials_completed=int(tuning_result["n_trials_completed"]),
                        params=dict(tuning_result["best_params"]),
                    )
                )
            except Exception as exc:
                results.append(
                    PredictorModelResult(
                        method_id=method_id,
                        validation_score=float("nan"),
                        fit_time_sec=float(perf_counter() - started_at),
                        n_trials_completed=0,
                        params={},
                        status="failed",
                        error=str(exc),
                    )
                )

        successful = [result for result in results if result.status == "success"]
        if not successful:
            errors = {result.method_id: result.error for result in results}
            raise RuntimeError(f"All candidate models failed during fitting: {errors}")

        leaderboard = pd.DataFrame(
            [
                {
                    "method_id": result.method_id,
                    "validation_score": result.validation_score,
                    "fit_time_sec": result.fit_time_sec,
                    "n_trials_completed": result.n_trials_completed,
                    "status": result.status,
                    "error": result.error,
                    "params": result.params,
                }
                for result in results
            ]
        )
        leaderboard["_status_rank"] = leaderboard["status"].map({"success": 0, "failed": 1}).fillna(2)
        leaderboard = leaderboard.sort_values(
            by=["_status_rank", "validation_score"],
            ascending=[True, False],
            na_position="last",
        ).drop(columns=["_status_rank"])
        leaderboard = leaderboard.reset_index(drop=True)
        self.leaderboard_ = leaderboard

        best_result = max(successful, key=lambda result: result.validation_score)
        self.best_method_id_ = best_result.method_id
        self.best_params_ = dict(best_result.params)
        self.best_preprocessor_ = TabularPreprocessor(scale_numeric=(best_result.method_id != "rsf"))
        X_train_proc = self.best_preprocessor_.fit_transform(dataset.X)
        self.best_model_ = method_registry[best_result.method_id](
            **_resolve_runtime_method_params(best_result.params, seed=self.random_state)
        )
        self.best_model_.fit(X_train_proc.to_numpy(), dataset.time, dataset.event)
        self.survival_times_ = self._default_survival_times(dataset.time, dataset.event)

        if test_data is not None:
            test_dataset = load_user_dataset(
                test_data,
                time_col=self.label_time,
                event_col=self.label_event,
                dataset_id=f"{dataset_name}_test",
                dataset_name=f"{dataset_name}_test",
                id_col=id_col,
                drop_columns=drop_columns,
            )
            self.test_metrics_ = self._evaluate_dataset(test_dataset)

        self._persist_artifacts(dataset_name, results)
        return self

    def leaderboard(self) -> pd.DataFrame:
        if self.leaderboard_ is None:
            raise RuntimeError("Call fit() before requesting the leaderboard.")
        return self.leaderboard_.copy()

    def predict_risk(self, data: pd.DataFrame | str | Path) -> np.ndarray:
        dataset = self._prepare_inference_frame(data)
        return self.best_model_.predict_risk(dataset.to_numpy())

    def predict_survival(
        self,
        data: pd.DataFrame | str | Path,
        times: np.ndarray | list[float] | None = None,
    ) -> pd.DataFrame:
        dataset = self._prepare_inference_frame(data)
        survival_times = np.asarray(times, dtype=float) if times is not None else self._require_survival_times()
        survival = self.best_model_.predict_survival(dataset.to_numpy(), survival_times)
        columns = [f"t_{time:.6g}" for time in survival_times]
        return pd.DataFrame(survival, columns=columns, index=dataset.index)

    def fit_summary(self) -> dict[str, Any]:
        if self.best_method_id_ is None or self.preset_config_ is None:
            raise RuntimeError("Call fit() before requesting the fit summary.")
        summary: dict[str, Any] = {
            "best_method_id": self.best_method_id_,
            "best_params": dict(self.best_params_ or {}),
            "eval_metric": self.eval_metric,
            "preset": self.preset_config_.name,
            "portfolio": list(self.preset_config_.method_ids),
        }
        if self.test_metrics_ is not None:
            summary["test_metrics"] = dict(self.test_metrics_)
        if self.artifact_dir_ is not None:
            summary["artifact_dir"] = str(self.artifact_dir_)
        return summary

    def _prepare_inference_frame(self, data: pd.DataFrame | str | Path) -> pd.DataFrame:
        if self.best_preprocessor_ is None or self.best_model_ is None:
            raise RuntimeError("Call fit() before requesting predictions.")
        frame = self._read_features(data)
        return self.best_preprocessor_.transform(frame)

    def _read_features(self, data: pd.DataFrame | str | Path) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            frame = data.copy()
        else:
            path = Path(data)
            if path.suffix.lower() == ".csv":
                frame = pd.read_csv(path)
            elif path.suffix.lower() in {".parquet", ".pq"}:
                frame = pd.read_parquet(path)
            else:
                raise ValueError(f"Unsupported file format '{path.suffix}'. Expected CSV or Parquet.")

        removable = [col for col in (self.label_time, self.label_event) if col in frame.columns]
        return frame.drop(columns=removable, errors="ignore").reset_index(drop=True)

    def _evaluate_dataset(self, dataset: SurvivalDataset) -> dict[str, float]:
        if self.best_preprocessor_ is None or self.best_model_ is None:
            raise RuntimeError("Call fit() before evaluation.")
        X_eval = self.best_preprocessor_.transform(dataset.X)
        risk = self.best_model_.predict_risk(X_eval.to_numpy())
        survival_times = self._require_survival_times()
        survival = self.best_model_.predict_survival(X_eval.to_numpy(), survival_times)
        metrics = compute_survival_metrics(
            train_time=self.dataset_.time,
            train_event=self.dataset_.event,
            test_time=dataset.time,
            test_event=dataset.event,
            risk_scores=risk,
            survival_probs=survival,
            survival_times=survival_times,
            horizons=horizons_from_train_event_times(self.dataset_.time, self.dataset_.event),
        )
        return metrics.to_dict()

    def _require_survival_times(self) -> np.ndarray:
        if self.survival_times_ is None:
            raise RuntimeError("Survival time grid is unavailable before fit().")
        return self.survival_times_

    def _default_survival_times(self, time: np.ndarray, event: np.ndarray) -> np.ndarray:
        event_times = time[event.astype(bool)]
        if event_times.size == 0:
            return np.linspace(1.0, 10.0, 25)
        lower = max(1e-8, float(np.percentile(event_times, 5)))
        upper = max(lower + 1e-8, float(np.percentile(event_times, 95)))
        return np.linspace(lower, upper, 50)

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
            },
            "best_method_id": self.best_method_id_,
            "best_params": self.best_params_ or {},
            "test_metrics": self.test_metrics_,
            "results": [asdict(result) for result in results],
        }
        write_json(artifact_dir / "fit_summary.json", payload)
