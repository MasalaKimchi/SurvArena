from __future__ import annotations

import pandas as pd

from survarena.logging.export import export_coverage_matrix


def test_coverage_matrix_exports_csv_and_markdown(tmp_path) -> None:
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
    assert (tmp_path / "coxph_coverage_matrix.md").exists()
    by_method = coverage.set_index("method_id")
    assert by_method.loc["coxph", "coverage_status"] == "success"
    assert by_method.loc["coxnet", "coverage_status"] == "missing_metric"
    assert by_method.loc["rsf", "coverage_status"] == "failed"
    assert by_method["repeat"].tolist() == [0, 0, 0]
    assert by_method.loc["coxph", "fold"] == 0
    assert by_method.loc["coxnet", "failure_reason"] == "missing_primary_metric"
    assert by_method.loc["rsf", "failure_reason"] == "ValueError"
    assert by_method["artifact_path"].tolist() == ["coxph_fold_results.csv"] * 3
    assert "Coverage Matrix" in (tmp_path / "coxph_coverage_matrix.md").read_text(encoding="utf-8")
