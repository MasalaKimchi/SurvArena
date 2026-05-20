from __future__ import annotations

import pandas as pd

from survarena.logging.export import export_coverage_matrix, export_hpo_budget_summary


def test_coverage_matrix_exports_csv_only(tmp_path) -> None:
    fold_results = pd.DataFrame(
        [
            {
                "benchmark_id": "smoke",
                "dataset_id": "whas500__base",
                "method_id": "coxph",
                "hpo_mode": "no_hpo",
                "seed": 11,
                "split_id": "repeat_0_fold_0__base",
                "status": "success",
                "runtime_sec": 1.25,
                "uno_c": 0.72,
                "harrell_c": 0.70,
                "robustness_track_id": "base",
            },
            {
                "benchmark_id": "smoke",
                "dataset_id": "whas500__base",
                "method_id": "coxnet",
                "hpo_mode": "hpo",
                "seed": 11,
                "split_id": "repeat_0_fold_1__base",
                "status": "success",
                "runtime_sec": 1.0,
                "uno_c": None,
                "harrell_c": 0.69,
            },
            {
                "benchmark_id": "smoke",
                "dataset_id": "whas500__base",
                "method_id": "rsf",
                "hpo_mode": "no_hpo",
                "seed": 11,
                "split_id": "repeat_0_fold_2__base",
                "status": "failed",
                "runtime_sec": 0.5,
                "failure_type": "ValueError",
                "exception_message": "bad split",
                "uno_c": None,
            },
        ]
    )

    coverage = export_coverage_matrix(
        tmp_path,
        fold_results,
        primary_metric="uno_c",
        output_dir=tmp_path,
        file_prefix="coxph",
    )

    assert (tmp_path / "coxph_coverage_matrix.csv").exists()
    assert not (tmp_path / "coxph_coverage_matrix.md").exists()
    by_method = coverage.set_index("method_id")
    assert by_method.loc["coxph", "coverage_status"] == "success"
    assert by_method.loc["coxnet", "coverage_status"] == "missing_metric"
    assert by_method.loc["rsf", "coverage_status"] == "failed"
    assert by_method["repeat"].tolist() == [0, 0, 0]
    assert by_method.loc["coxph", "fold"] == 0
    assert by_method.loc["coxnet", "failure_reason"] == "missing_primary_metric"
    assert by_method.loc["rsf", "failure_reason"] == "ValueError"
    assert by_method["artifact_path"].tolist() == ["coxph_fold_results.csv"] * 3


def test_hpo_budget_summary_reports_requested_vs_realized_trials(tmp_path) -> None:
    fold_results = pd.DataFrame(
        [
            {
                "benchmark_id": "cloud",
                "dataset_id": "whas500",
                "method_id": "fast_survival_svm",
                "hpo_mode": "hpo",
                "requested_max_trials": 50,
                "realized_trial_count": 42,
                "requested_timeout_seconds": 2400,
                "requested_sampler": "tpe",
                "requested_pruner": "median",
                "hpo_budget_tier": "reduced",
                "hpo_config_target": 100,
                "hpo_cap_reason": "runtime cap",
            },
            {
                "benchmark_id": "cloud",
                "dataset_id": "whas500",
                "method_id": "fast_survival_svm",
                "hpo_mode": "hpo",
                "requested_max_trials": 50,
                "realized_trial_count": 50,
                "requested_timeout_seconds": 2400,
                "requested_sampler": "tpe",
                "requested_pruner": "median",
                "hpo_budget_tier": "reduced",
                "hpo_config_target": 100,
                "hpo_cap_reason": "runtime cap",
            },
        ]
    )

    summary = export_hpo_budget_summary(tmp_path, fold_results, output_dir=tmp_path, file_prefix="combined")

    row = summary.iloc[0]
    assert (tmp_path / "combined_hpo_budget_summary.csv").exists()
    assert row["requested_trials_total"] == 100
    assert row["realized_trials_total"] == 92
    assert row["trial_realization_rate"] == 0.92
    assert row["hpo_config_target"] == 100
    assert bool(row["hpo_capped"]) is True
    assert row["hpo_cap_reason"] == "runtime cap"
