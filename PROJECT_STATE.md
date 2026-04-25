# SurvArena Project State

SurvArena is being developed as a TabArena-like benchmark and toolkit for right-censored tabular survival analysis. This page is the repository-level source of truth for benchmark coverage, paper readiness, AI-agent workflows, and prioritized development tasks.

## Current Positioning

SurvArena has two main surfaces: `SurvivalPredictor` for fitting survival models on user datasets, and a config-driven benchmark runner for shared-split comparisons across datasets, methods, seeds, and modes.

Paper framing: SurvArena is a reproducible benchmarking framework for tabular survival analysis across classical, ensemble, boosting, deep learning, AutoML, and foundation-model approaches.

## Current Benchmark Scope

Ready-to-run datasets: `support`, `metabric`, `aids`, `gbsg2`, `flchain`, and `whas500`.

Large-track/future dataset candidate: `kkbox`, currently present as a local-loader placeholder rather than a ready manuscript dataset.

Maintained benchmark configs include `smoke.yaml`, `standard_v1.yaml`, `manuscript_v1.yaml`, `manuscript_autogluon_v1.yaml`, and `smoke_foundation.yaml`.

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

Exit criteria: dataset coverage matrix, method coverage matrix, experiment registry, manuscript report generator, leakage/preprocessing audit export, runtime/failure summary, reproducible smoke benchmark validation, reproducible standard benchmark validation, paper-ready dataset table, paper-ready method taxonomy table, and contribution guides.
