# Training, Evaluation, And Output Audit

## Intent

Audit benchmark training, evaluation metric computation, and result persistence for correctness, redundant work, and artifact sprawl. Keep changes scoped to evidence-backed cleanup and preserve public behavior unless the documented storage contract or user request explicitly calls for tighter aggregation.

## Passes

### Pass: Benchmark Output Consolidation
Current behavior: Benchmark runs may emit fold results, leaderboard, diagnostics, and additional summary CSVs for HPO budgets and runtime failures.
Structural improvement: Keep benchmark-run outputs aligned with the documented core artifact contract by aggregating detailed diagnostic rows into the main run diagnostics table.
Validation check: Targeted benchmark/export tests, then full lint and test suite.
Migration split: None; standalone export helpers should remain available for compatibility unless proven unused.

### Pass: Training And Metric Logic Audit
Current behavior: Training, HPO/no-HPO selection, prediction bundling, and metric computation flow through `survarena/benchmark/runner.py`, `survarena/benchmark/tuning.py`, and `survarena/evaluation/metrics.py`.
Structural improvement: Remove redundant computation only where correctness is clear; otherwise document residual findings in the final summary.
Validation check: Metric, benchmark, and tuning tests plus full suite.
Migration split: Any statistical methodology change beyond bug fixes should become a separate protocol migration.
