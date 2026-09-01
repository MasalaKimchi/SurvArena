# Phase 1: Truthful Green Baseline - Existing Patterns

**Mapped:** 2026-08-31
**Scope:** Test contracts, verification tooling, split compatibility, and status documentation

## Closest Analogs

| New or changed responsibility | Closest existing analog | Pattern to preserve | Required deviation |
|---|---|---|---|
| Artifact failure contract | `survarena/benchmark/runner.py::_save_model_artifacts` and the successful artifact test in `tests/test_benchmark.py` | Persist predictions before fitted state; record a JSON artifact manifest; include `hpo_mode` in the directory identity | Test the current structured `model_artifact_status="failed"` contract instead of expecting a run-fatal exception or the obsolete `prediction_only` label |
| Legacy resume identity | `survarena/benchmark/resume.py::completed_resume_keys` and `test_resume_export_merges_existing_and_new_fold_rows` | Completion keys include dataset, method, split, seed, and arm; only successful rows with the primary metric qualify | Add an explicit dual-arm legacy test: an arm-less row suppresses `no_hpo` only and still schedules `hpo` |
| Process scheduling test | `survarena/benchmark/runner.py::_execute_run_units` | Keep run units process-isolated because they mutate global RNG state | Patch `ProcessPoolExecutor`, whose fake `map` executes synchronously, rather than observing child-process mutations in parent memory |
| Parquet capability branches | `survarena/core/results/store.py::export_parquet` | Import `pyarrow` lazily and raise an actionable `RuntimeError` if unavailable | Exercise a real installed export and a forced import failure independently of ambient dependencies |
| Markdown report rendering | Simple deterministic serialization helpers in `survarena/logging/` | Normalize values before serialization and produce stable output | Replace pandas' optional `to_markdown` path with an internal pipe-table renderer that escapes pipes and newlines |
| Split mismatch rejection | `survarena/data/splitters.py::load_or_create_splits` mismatch branch | Fail closed unless `regenerate_on_mismatch=True`; reuse only exact manifest payload matches | Add a sorted field-level diff, manifest path, and literal `--regenerate-splits` guidance without changing split identity |
| Verification commands | `README.md` development commands and `.github/workflows/ci.yml` tiered jobs | Use the same commands locally and in CI; keep cheap gates separate from heavy tests | Add an explicitly scoped mypy job and semantic smoke assertions; do not claim whole-package type coverage |
| Evidence status | `PROJECT_STATE.md`, `docs/test_status.md`, `docs/ci_and_reproducibility.md` | Separate implemented code from executed evidence | Replace stale constrained-environment wording with dated local evidence and retain historical-result invalidation |

## File Ownership by Plan

- Plan 01 owns `pyproject.toml`, the four failing test contracts, and only the narrow production seam needed for deterministic testing.
- Plan 02 owns the publication-audit renderer, split mismatch diagnostics, and their targeted tests.
- Plan 03 owns CI/workflow and maintained status documentation, then records the integrated verification result.

## Constraints

- Do not change metric definitions, statistical comparisons, benchmark rosters, or generated result evidence.
- Do not replace process isolation with threads.
- Do not make split regeneration automatic.
- Do not add `tabulate` solely to render the release audit.
- Do not add broad mypy ignores or claim all of `survarena/` is statically verified.

---
*Phase: 01-truthful-green-baseline*
*Pattern map completed: 2026-08-31*
