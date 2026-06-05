---
status: complete
completed_at: 2026-06-05T17:24:00Z
---

Implemented compact manuscript Elo metric-suite exports.

Changes:
- Metric-suite runs now write one aggregate CSV per analytical table: `elo_ratings.csv`, `pairwise_win_rates.csv`, `rank_summary.csv`, `coverage_summary.csv`, `method_summary.csv`, and `manuscript_fold_results_success.csv`.
- The fold-results source table is canonical and non-duplicated, retaining all requested metrics in wide format.
- Per-metric PNG figures are still emitted because figures are naturally separate presentation artifacts.
- Single-metric `build_outputs()` behavior remains compatible with existing per-metric CSV names.
- Added `scripts/compact_existing_elo_results.py` to migrate already-produced Elo folders into the compact layout.
- Migrated existing clinical and genomics no-HPO Elo results. Clinical archived 108 metric-sliced CSVs and moved 18 figures; genomics archived 144 metric-sliced CSVs and moved 24 figures.

Verification:
- `python -m pytest tests/test_evaluation.py::test_manuscript_elo_metric_suite_writes_index_with_multiple_metrics`
- `python -m pytest tests/test_evaluation.py`
- `ruff check scripts/build_manuscript_elo.py tests/test_evaluation.py`
- `ruff check scripts/build_manuscript_elo.py scripts/compact_existing_elo_results.py tests/test_evaluation.py`
- `python -m compileall scripts/compact_existing_elo_results.py scripts/build_manuscript_elo.py`
