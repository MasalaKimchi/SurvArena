# SurvArena Project State

SurvArena is being developed as a TabArena-like benchmark and toolkit for right-censored tabular survival analysis. This page is the repository-level source of truth for benchmark coverage, paper readiness, AI-agent workflows, and prioritized development tasks.

## Current Positioning

SurvArena has two main surfaces: `SurvivalPredictor` for fitting survival models on user datasets, and a config-driven benchmark runner for shared-split comparisons across datasets, methods, seeds, and modes.

Paper framing: SurvArena is a reproducible benchmarking framework for tabular survival analysis across classical, ensemble, boosting, deep learning, AutoML, and foundation-model approaches.

## Current Benchmark Scope

Ready-to-run datasets: `support`, `metabric`, `aids`, `gbsg2`, `flchain`, and `whas500`.

Large-track/future dataset candidate: `kkbox`, currently configured as a local-only dataset path rather than a ready manuscript dataset.

Maintained benchmark configs currently present in this checkout are `smoke.yaml`, `standard_v1.yaml`, and `manuscript_v1.yaml`. AutoGluon, foundation, and KKBox tracks should remain optional until their configs and evidence bundles are restored or added.

## Method Coverage

Implemented families include classical survival models, AFT models, survival SVM, tree ensembles, boosting models, neural survival models, AutoGluon, and optional foundation-model adapters.

Foundation adapters are experimental and should remain optional until benchmark coverage and runtime reliability are better documented.

## Existing Evidence Artifacts

SurvArena already exports fold results, seed summaries, overall summaries, leaderboards, ranking summaries, pairwise comparisons, confidence intervals, failure summaries, missing-metric summaries, HPO ledgers, compact run ledgers, experiment manifests, and experiment navigator artifacts.

## Paper-Critical Missing Pieces

1. Benchmark coverage matrix showing dataset-method-mode success/failure and artifact paths.
2. Manuscript report generator that creates paper-ready tables from `results/summary/<run_id>`.
3. Preprocessing and split audit report confirming train-only preprocessing and shared split reuse.
4. Runtime and failure-mode summary table.
5. Dataset expansion plan beyond the six standard datasets.
6. Contribution guide for adding new survival datasets and model adapters.
7. Calibration or time-specific risk evaluation report if calibrated survival probabilities are claimed.
8. Stable Notion/GitHub AI-agent loop for weekly triage.

## AI-Agent Operating Model

Use four role-specific agents:

- Code Review Agent: inspect code, tests, docs, API consistency, reproducibility risks, and fragile logic.
- Benchmark Curator Agent: read benchmark artifacts, update coverage, summarize failures, and recommend reruns.
- Paper Evidence Agent: convert benchmark artifacts into manuscript-ready claims, tables, and limitations.
- Implementation Agent: implement one GitHub issue at a time, add tests, update docs, and open a focused PR.

## Weekly Agent Review Template

Each weekly update should include implemented changes, benchmark evidence, paper-blocking gaps, recommended tickets, PRs ready for review, experiments to rerun, and manuscript claims now supported by evidence.

## Recommended Milestone

Milestone: SurvArena v0.1 manuscript benchmark readiness.

Status: not manuscript-ready yet. The framework and export plumbing are close, but the repository does not yet contain a current full evidence bundle that supports paper claims.

Current evidence:

- Six standard datasets are documented and configured: `support`, `metabric`, `aids`, `gbsg2`, `flchain`, and `whas500`.
- `kkbox` is configured and documented as a large/local-only track, but remains outside the ready manuscript suite until credentials, cache preparation, runtime, and dataset statistics are reproducible.
- Present benchmark configs are `smoke.yaml`, `standard_v1.yaml`, `manuscript_v1.yaml`, and `local_feasible_hpo_v1.yaml`. Removed docs references to absent `manuscript_autogluon_v1.yaml`, `smoke_foundation.yaml`, and `smoke_aft.yaml`; those configs are not present in this checkout and should not be treated as completed evidence.
- There are 25 method configs. `manuscript_v1.yaml` covers 23 native methods in no-HPO/default-policy mode; `standard_v1.yaml` covers `coxph`, `coxnet`, `rsf`, and `deepsurv` in paired no-HPO/HPO mode.
- Export code/tests cover fold results, leaderboards, run diagnostics, coverage matrices, runtime/failure summaries, manifests, and navigators. Checked-in run evidence is only a tiny `whas500`/`smoke`/`weibull_aft` no-HPO result with two successful folds.
- Local milestone probe on 2026-05-04 passed environment validation, `smoke.yaml` dry-run, `manuscript_v1.yaml` dry-run, a targeted `standard_v1.yaml` dry-run, and the six-dataset native smoke matrix: 138/138 native dataset-method combinations completed with 276/276 successful folds under `results/local_milestone_probe/`.
- Local feasible paired no-HPO/HPO benchmark on 2026-05-05 ran all six standard datasets x 23 native methods with 3 outer folds x 3 repeats/seeds x 2 modes under `results/local_feasible_hpo_v1_all/`. Final artifacts now cover 138/138 dataset-method combinations, 2,484/2,484 successful fold rows, and plot-ready mode summaries/deltas after fixing and rerunning the `flchain` neural-adapter batch-normalization singleton-batch edge case.
- Optional `tabpfn_survival` is not locally smoke-ready: `support` timed out after 900 seconds and the remaining dataset repeats were skipped as an optional foundation readiness blocker.
- Dataset and method contribution guides exist.

Checklist:

- [x] Core benchmark protocol documented around shared splits, train-side preprocessing, seeded runs, no-HPO/HPO governance, compact artifacts, and manifests.
- [x] Standard six-dataset suite documented and represented in benchmark YAML.
- [x] Main-paper native method portfolio represented in `manuscript_v1.yaml`.
- [x] Compact artifact exporters implemented for fold results, leaderboards, diagnostics, coverage, runtime/failure, manifests, and navigators.
- [x] Dataset and method contribution guides available.
- [x] Align docs/state references with actual benchmark configs, or add the missing AutoGluon/foundation/AFT configs before mentioning them as maintained tracks.
- [x] Regenerate fresh smoke artifacts with the current exporter and verify coverage matrix, runtime/failure summary, manifest, navigator, and per-experiment README are emitted across the six-dataset native smoke matrix.
- [x] Fix runtime/failure summaries so successful no-HPO rows are not marked `missing_metrics` only because `validation_score` is blank while primary/test metrics are present.
- [x] Produce an actual dataset-method-mode coverage matrix from fresh artifacts across all six standard datasets and manuscript methods.
- [x] Produce a method coverage/status table with successes, failures, missing metrics, runtime, memory, and artifact paths.
- [ ] Add or identify a manuscript report generator that reads `results/summary/...` and emits paper-ready tables for datasets, methods, metrics, ranks, CIs, pairwise tests, failures, and runtime.
- [ ] Add a preprocessing/split audit export proving train-only preprocessing and shared split reuse for the final run bundle.
- [ ] Validate one reproducible smoke benchmark run and one standard/manuscript-shaped pilot after exporter/config alignment.
- [ ] Decide whether calibration, net benefit, AutoGluon, foundation models, HPO, or KKBox appear in the manuscript, appendix, or limitations only; require separate evidence bundles for any promoted claim.

Remaining blockers:

- Full local feasible paired no-HPO/HPO artifacts now exist for the six-dataset native suite with no failed fold rows after rerunning the six affected `flchain` neural adapter combinations.
- No checked-in manuscript report artifact or obvious source generator is present for final paper tables.
- Benchmark config references are now aligned across README/docs/project state versus `configs/benchmark/`.
- Current result evidence includes smoke-scale native coverage plus a complete local feasible paired no-HPO/HPO benchmark. It supports exporter/config/runtime assessment and preliminary HPO-vs-default analysis, but still needs manuscript table generation before final claims.
- Optional `tabpfn_survival` timed out locally and should stay out of default smoke/manuscript claims until readiness is improved or the benchmark contract gives it a bounded budget.
- Runtime/failure summaries no longer treat blank no-HPO `validation_score` as a missing metric; remaining missing-metric signals should reflect benchmark/test metrics.
- KKBox, AutoGluon, and foundation paths should remain optional/appendix/exploratory until dedicated configs and evidence are restored or added.

Targeted experiments needed, in order:

1. Use `results/local_feasible_hpo_v1_all/combined_fold_results_success.csv`, `mode_metric_summary.csv`, and `hpo_vs_no_hpo_delta_summary.csv` to generate preliminary HPO-vs-default figures and tables.
2. Run the locked `manuscript_v1.yaml` full no-HPO/default-policy benchmark for main-paper evidence if the local feasible HPO run is treated as a sensitivity/budget study.
3. Add separate appendix-track experiments for AutoGluon, foundation adapters, or KKBox only after their configs and readiness checks are aligned.

Exit criteria:

- Fresh artifact bundle covers all promoted datasets, methods, modes, seeds, and split geometry.
- Dataset coverage matrix, method coverage matrix, runtime/failure summary, and preprocessing/split audit are generated from the final artifacts.
- Manuscript report tables are reproducible from `results/summary/...` without manual spreadsheet work.
- Claims about HPO, calibration, net benefit, AutoGluon, foundation models, or KKBox are either backed by targeted evidence or explicitly scoped as future/appendix/limitations.
