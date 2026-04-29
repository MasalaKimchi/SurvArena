from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from survarena.logging.export import export_run_diagnostics, export_runtime_failure_summary


def _synthetic_fold_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "benchmark_id": "bench",
                "dataset_id": "toy",
                "method_id": "coxph",
                "hpo_mode": "no_hpo",
                "seed": 11,
                "split_id": "repeat_0_fold_0__base",
                "status": "success",
                "validation_score": 0.7,
                "uno_c": 0.71,
                "harrell_c": 0.72,
                "ibs": 0.2,
                "td_auc_25": 0.68,
                "td_auc_50": 0.69,
                "td_auc_75": 0.7,
                "runtime_sec": 1.2,
            },
            {
                "benchmark_id": "bench",
                "dataset_id": "toy",
                "method_id": "rsf",
                "hpo_mode": "no_hpo",
                "seed": 11,
                "split_id": "repeat_0_fold_0__base",
                "status": "failed",
                "validation_score": np.nan,
                "uno_c": np.nan,
                "harrell_c": np.nan,
                "ibs": np.nan,
                "td_auc_25": np.nan,
                "td_auc_50": np.nan,
                "td_auc_75": np.nan,
                "runtime_sec": 0.4,
            },
            {
                "benchmark_id": "bench",
                "dataset_id": "toy",
                "method_id": "coxnet",
                "hpo_mode": "hpo",
                "seed": 11,
                "split_id": "repeat_0_fold_1__base",
                "status": "success",
                "validation_score": 0.6,
                "uno_c": np.nan,
                "harrell_c": 0.61,
                "ibs": 0.25,
                "td_auc_25": np.nan,
                "td_auc_50": np.nan,
                "td_auc_75": np.nan,
                "runtime_sec": 2.0,
            },
        ]
    )


def test_runtime_failure_summary_exports_csv_and_markdown(tmp_path: Path) -> None:
    run_records = [
        {
            "metrics": {
                "dataset_id": "toy",
                "method_id": "rsf",
                "hpo_mode": "no_hpo",
                "seed": 11,
                "split_id": "repeat_0_fold_0__base",
                "failure_type": "ImportError",
                "exception_message": "No module named sksurv",
            },
            "failure": {"traceback": "ModuleNotFoundError: No module named sksurv"},
        }
    ]

    summary = export_runtime_failure_summary(
        tmp_path,
        benchmark_id="bench",
        fold_results=_synthetic_fold_results(),
        run_records=run_records,
        output_dir=tmp_path,
        file_prefix="toy",
    )

    assert (tmp_path / "toy_runtime_failure_summary.csv").exists()
    assert (tmp_path / "toy_runtime_failure_summary.md").exists()
    by_method = summary.set_index("method_id")
    assert by_method.loc["coxph", "failure_category"] == "success"
    assert by_method.loc["rsf", "failure_category"] == "dependency_missing"
    assert by_method.loc["rsf", "n_crashed"] == 1
    assert by_method.loc["coxnet", "failure_category"] == "missing_metrics"
    assert by_method.loc["coxnet", "n_missing_metrics"] == 1
    assert by_method.loc["coxnet", "missing_metric_columns"] == "uno_c, td_auc_25, td_auc_50, td_auc_75"
    assert by_method.loc["coxph", "runtime_sec_mean"] == 1.2

    markdown = (tmp_path / "toy_runtime_failure_summary.md").read_text(encoding="utf-8")
    assert "Runtime and Failure Summary" in markdown
    assert "Successful rows with missing metrics: 1" in markdown


def test_run_diagnostics_writes_runtime_failure_summary(tmp_path: Path) -> None:
    export_run_diagnostics(
        tmp_path,
        benchmark_id="bench",
        fold_results=_synthetic_fold_results(),
        dataset_curation_rows=[],
        hpo_trial_rows=[],
        output_dir=tmp_path,
        file_prefix="toy",
    )

    assert (tmp_path / "toy_run_diagnostics.csv").exists()
    assert (tmp_path / "toy_runtime_failure_summary.csv").exists()
    assert (tmp_path / "toy_runtime_failure_summary.md").exists()
