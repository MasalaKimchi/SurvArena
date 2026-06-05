# Documentation Index

Last reviewed against the repository docs: 2026-06-05.

Use this page as the routing map when you know what you want to do but not
which document owns it.

## First-Time Users

| Task | Read |
| --- | --- |
| Install SurvArena and verify imports | [`environment.md`](environment.md) |
| Run a small user-dataset pilot | [`datasets.md`](datasets.md#user-dataset-pilot) |
| Understand benchmark execution before fitting models | [`benchmarking_workflow.md`](benchmarking_workflow.md) |
| Understand the retained benchmark contract | [`protocol.md`](protocol.md) |
| Plan runtime for no-HPO and HPO runs | [`training_strategy.md`](training_strategy.md) |

## Reference Docs

| Topic | Read |
| --- | --- |
| Built-in datasets and user-data requirements | [`datasets.md`](datasets.md) |
| Model IDs, families, and backend packages | [`methods.md`](methods.md) |
| Optional TCGA/UCSC Xena cancer cohorts | [`cancer_survival_datasets.md`](cancer_survival_datasets.md) |
| Optional foundation-model adapters | [`foundation_models.md`](foundation_models.md) |
| Manuscript evidence bundle and local machine notes | [`manuscript_evidence.md`](manuscript_evidence.md) |
| AutoGluon-style UX comparison | [`autogluon_comparison.md`](autogluon_comparison.md) |

## Contributors

| Contribution | Read |
| --- | --- |
| Add a method adapter | [`contributing_method_adapters.md`](contributing_method_adapters.md) |
| Add a built-in or local-only dataset | [`contributing_datasets.md`](contributing_datasets.md) |

## Canonical Surfaces

- Main maintained benchmark config:
  [`configs/benchmark/manuscript_v1.yaml`](../configs/benchmark/manuscript_v1.yaml)
- Matching HPO-only manuscript config:
  [`configs/benchmark/manuscript_hpo_v1.yaml`](../configs/benchmark/manuscript_hpo_v1.yaml)
- Main CLI entry points:
  `survarena pilot`, `survarena fit`, `survarena compare`, and
  `survarena benchmark ...`
- Batch-worker entry point:
  `python -m survarena.run_benchmark --config ...`

Prefer `--config` in examples and scripts. `--benchmark-config` is still
accepted as a compatibility alias.
