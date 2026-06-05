---
status: complete
completed: 2026-06-05
---

# Summary

Updated manuscript Elo generation so the default command builds a metric-specific suite rather than only `uno_c`.

## Completed

- Added default Elo outputs for discrimination, Brier/IBS, time-dependent AUC, calibration absolute-error, and net-benefit metrics.
- Added `metric_suite_index.csv` for the generated metric suite.
- Kept single-metric `--metric` mode compatible with the previous output shape.
- Derived `calibration_slope_abs_error_*` and `calibration_intercept_abs_error_*` from legacy raw calibration columns in old fold artifacts.
- Removed raw calibration slope/intercept from comparable metric discovery and manuscript metric defaults.
- Vectorized `pairwise_win_rate`, reducing one manuscript metric pairwise table from about 22 seconds to about 0.09 seconds.
- Updated project Markdown to describe the 27-method clinical matrix, the retained `results/manuscript_grade/clinical_no_hpo/elo/` evidence path, and the metric-specific Elo suite.

## Verification

- `pytest tests/test_evaluation.py`
- `pytest tests/test_benchmark.py tests/test_evaluation.py`
- `ruff check survarena tests scripts configs`
- `python -m compileall scripts/build_manuscript_elo.py survarena/evaluation/_ranking.py survarena/logging/export_shared.py`
- `python scripts/build_manuscript_elo.py --bootstrap 0`
- `python scripts/build_manuscript_elo.py --metric uno_c --bootstrap 0 --no-asset`
