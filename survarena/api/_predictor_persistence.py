from __future__ import annotations

from dataclasses import asdict
import pickle
from pathlib import Path
from typing import Any, TypeVar

from survarena.config import read_yaml
from survarena.logging.tracker import write_json


PREDICTOR_SERIALIZATION_VERSION = 1

PredictorT = TypeVar("PredictorT")


def default_predictor_path(artifact_dir: Path | None) -> Path:
    if artifact_dir is None:
        raise RuntimeError("No artifact directory is available. Provide a save path or call fit() first.")
    return artifact_dir / "predictor.pkl"


def predictor_manifest_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_manifest.json")


def serialization_manifest(predictor: Any, output_path: Path) -> dict[str, Any]:
    return {
        "serialization_version": PREDICTOR_SERIALIZATION_VERSION,
        "class_name": type(predictor).__name__,
        "module": type(predictor).__module__,
        "path": str(output_path),
        "best_method_id": predictor.best_method_id_,
        "trained_models": predictor.model_names(),
        "eval_metric": predictor.eval_metric,
        "retain_top_k_models": predictor.retain_top_k_models,
    }


def save_predictor(predictor: Any, path: str | Path | None = None) -> Path:
    output_path = Path(path) if path is not None else default_predictor_path(predictor.artifact_dir_)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(predictor, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(predictor_manifest_path(output_path), serialization_manifest(predictor, output_path))
    return output_path


def load_predictor(cls: type[PredictorT], path: str | Path) -> PredictorT:
    resolved_path = Path(path)
    with resolved_path.open("rb") as handle:
        predictor = pickle.load(handle)
    if not isinstance(predictor, cls):
        raise TypeError(f"Serialized object at '{resolved_path}' is not a {cls.__name__}.")
    predictor._ensure_runtime_state_defaults()
    manifest_path = predictor_manifest_path(resolved_path)
    if manifest_path.exists():
        manifest = read_yaml(manifest_path)
        expected_version = int(manifest.get("serialization_version", PREDICTOR_SERIALIZATION_VERSION))
        if expected_version != PREDICTOR_SERIALIZATION_VERSION:
            raise RuntimeError(
                f"Unsupported predictor serialization version {expected_version}; "
                f"expected {PREDICTOR_SERIALIZATION_VERSION}."
            )
    return predictor


def persist_artifacts(predictor: Any, dataset_name: str, results: list[Any]) -> None:
    artifact_root = predictor.save_path or Path("results") / "predictor"
    artifact_dir = artifact_root / dataset_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    predictor.artifact_dir_ = artifact_dir

    if predictor.leaderboard_ is not None:
        predictor.leaderboard_.to_csv(artifact_dir / "leaderboard.csv", index=False)
    payload = {
        "config": {
            "label_time": predictor.label_time,
            "label_event": predictor.label_event,
            "eval_metric": predictor.eval_metric,
            "presets": predictor.presets,
            "retain_top_k_models": predictor.retain_top_k_models,
            "random_state": predictor.random_state,
            "verbose": predictor.verbose,
            "enable_foundation_models": predictor.enable_foundation_models,
            "validation_strategy": predictor.validation_strategy_,
            "holdout_frac": predictor.validation_holdout_frac_,
            "num_bag_folds": predictor.num_bag_folds_,
            "num_bag_sets": predictor.num_bag_sets_,
            "selection_train_rows": predictor.selection_train_rows_,
            "validation_rows": predictor.validation_rows_,
            "refit_full": predictor.refit_full_,
            "final_train_rows": predictor.final_train_rows_,
            "hyperparameter_tune_kwargs": predictor.hyperparameter_tune_kwargs_,
            "time_limit": predictor.fit_time_limit_sec_,
            "selection_time_budget_sec": predictor.selection_time_budget_sec_,
            "fit_elapsed_sec": predictor.fit_elapsed_sec_,
        },
        "best_method_id": predictor.best_method_id_,
        "best_params": predictor.best_params_ or {},
        "portfolio_notes": (
            list(predictor.preset_config_.portfolio_notes) if predictor.preset_config_ is not None else []
        ),
        "dataset_diagnostics": (
            predictor.dataset_.metadata.diagnostics.to_dict()
            if predictor.dataset_ is not None and predictor.dataset_.metadata.diagnostics is not None
            else None
        ),
        "test_metrics": predictor.test_metrics_,
        "trained_models": predictor.model_names(),
        "per_model_test_metrics": predictor.model_test_metrics_,
        "results": [asdict(result) for result in results],
    }
    write_json(artifact_dir / "fit_summary.json", payload)
    save_predictor(predictor, artifact_dir / "predictor.pkl")
