---
status: complete
completed: 2026-06-17
task: repo-organization-cleanup
---

# Summary

Implemented behavior-preserving organization and cleanup improvements:

- Extracted CLI command execution into `survarena.commands.handlers` while preserving `survarena.cli:main` and monkeypatch-compatible facade dependencies.
- Extracted pure benchmark HPO/config helpers into `survarena.benchmark.hpo_config` with a runner-local AutoGluon wrapper to preserve existing patch points.
- Extracted predictor result dataclass, leaderboard shaping, selection sorting, budget-skip rows, backend labels, and metadata attachment into `survarena.api._predictor_results`.
- Added compact Elo input support to `scripts/build_manuscript_elo.py` so retained `elo/manuscript_fold_results_success.csv` bundles can rebuild comparisons without raw `dataset_model` trees.
- Extended `scripts/compact_existing_elo_results.py` with dry-run local pruning controls for result cruft, optional logs/local runs, and optional raw dataset-model inputs.
- Added `scripts/clean_local_artifacts.sh` as a dry-run local cache/artifact cleanup helper.
- Documented environment-freeze guidance and result retention policy.
- Tightened `.gitignore` for alternate virtualenvs and planning scratch artifacts.

Validation:

- `python -m pytest tests/test_api.py -k "cli or benchmark_plan_cli or benchmark_doctor_cli or benchmark_run_cli or benchmark_report_cli"`: passed.
- `python -m pytest tests/test_benchmark.py -k "hpo or run_benchmark or evaluate_split or resume or artifact or execution"`: passed.
- `python scripts/build_manuscript_elo.py --input-dir results/manuscript_grade/clinical_no_hpo/elo --list-metrics`: passed.
- `python scripts/compact_existing_elo_results.py --root results --prune-local-artifacts`: previewed local-only removable candidates.
- `python scripts/compact_existing_elo_results.py --root results --prune-local-artifacts --apply-prune`: removed 12 local-only cruft/empty result paths.
- `./scripts/clean_local_artifacts.sh`: previewed local caches/artifacts.
- `ruff check survarena tests scripts`: passed.
- `python -m compileall survarena`: passed.
- `python -m pytest`: 209 passed, 6 skipped.
