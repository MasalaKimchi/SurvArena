---
status: complete
---

# Purge Old Metrics

Completed metric-contract cleanup:
- Removed raw calibration slope/intercept fields from exported metric bundles.
- Removed legacy decision-curve aliases and threshold-specific net-benefit metric exports.
- Restricted comparable metric direction discovery to the retained 25/50/75 manuscript suite.
- Updated docs and tests to reflect the retained metric contract.
- Scrubbed local `results/` CSVs and stale figures so old metric columns/rows no longer remain.

Verification:
- `python -m pytest tests/test_evaluation.py -q`
- `python -m ruff check survarena/evaluation/metrics.py survarena/evaluation/_metric_stats.py survarena/logging/export_shared.py tests/test_evaluation.py`
- CSV scan found no old metric columns or metric rows under `results/`.
