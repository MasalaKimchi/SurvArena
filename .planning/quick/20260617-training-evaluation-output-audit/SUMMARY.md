# Training, Evaluation, And Output Audit Summary

## Changes

- Consolidated benchmark-run diagnostics so runtime failure summaries and HPO budget summaries are embedded in `<model_name>_run_diagnostics.csv`.
- Stopped default benchmark runs from emitting separate `<model_name>_hpo_budget_summary.csv` and `<model_name>_runtime_failure_summary.csv` files.
- Kept `export_hpo_budget_summary` and `export_runtime_failure_summary` available for explicit standalone exports.
- Updated artifact navigator/README expectations and tests to match the documented core CSV contract.

## Logic Audit Notes

- The benchmark loop performs no-HPO as a direct default fit and HPO as inner-CV selection followed by outer-train refit, matching `docs/protocol.md`.
- Prediction bundling is called once per outer test fold and reused for all survival metrics.
- Metric computation derives horizons from training event times, clamps them to IPCW support, deduplicates clipped horizons for AUC/Brier calls, and keeps calibration/net-benefit as horizon-specific derived metrics.

## Validation

- `pytest tests/test_benchmark.py::test_benchmark_run_emits_compact_artifact_links tests/test_benchmark.py::test_hpo_budget_summary_reports_requested_vs_realized_trials tests/test_benchmark.py::test_runtime_failure_summary_exports_csv_only tests/test_benchmark.py::test_run_diagnostics_embeds_runtime_failure_summary tests/test_benchmark.py::test_runtime_failure_summary_classifies_foundation_readiness_failures tests/test_evaluation.py`
- `python -m ruff check survarena tests scripts`
- `pytest`
