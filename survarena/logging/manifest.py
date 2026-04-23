from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from survarena.utils.env import get_environment_snapshot, get_package_versions

RUN_MANIFEST_SCHEMA_VERSION = "2.0"


@dataclass(slots=True)
class RunManifest:
    run_id: str
    benchmark_id: str
    dataset_id: str
    method_id: str
    split_id: str
    seed: int
    hyperparameters: dict[str, Any]
    preprocessing_config: dict[str, Any]
    runtime_seconds: float
    peak_memory_mb: float
    status: str
    benchmark_config_hash: str
    method_config_hash: str
    split_indices_hash: str
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        env = get_environment_snapshot()
        env["package_versions"] = get_package_versions(
            [
                "numpy",
                "pandas",
                "scikit-learn",
                "scikit-survival",
                "lifelines",
                "pycox",
                "xgboost",
                "catboost",
                "autogluon.tabular",
            ]
        )
        return {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "benchmark_id": self.benchmark_id,
            "dataset_id": self.dataset_id,
            "method_id": self.method_id,
            "split_id": self.split_id,
            "seed": self.seed,
            "hyperparameters": self.hyperparameters,
            "preprocessing_config": self.preprocessing_config,
            "runtime_seconds": self.runtime_seconds,
            "peak_memory_mb": self.peak_memory_mb,
            "status": self.status,
            "benchmark_config_hash": self.benchmark_config_hash,
            "method_config_hash": self.method_config_hash,
            "split_indices_hash": self.split_indices_hash,
            "notes": self.notes,
            **env,
        }
