# Training Strategy and Runtime Budget

This note explains how SurvArena trains and evaluates models under the benchmark
YAML profiles. It focuses on the distinction between outer folds, inner folds,
no-HPO evaluation, and HPO evaluation.

## Core Structure

SurvArena uses repeated nested cross-validation for benchmark profiles:

- The outer loop estimates generalization performance.
- The inner loop is used only when a model configuration must be selected.
- The selected configuration is refit on the full outer-training split before
  evaluating the held-out outer-test split.
- Split definitions are shared across methods and persisted under
  `data/splits/<task_id>/`.

For right-censored survival tasks, outer folds are stratified by event status.
Preprocessing is fit only on training-side data, then applied to validation or
test rows.

## YAML Profiles

| Config | Profile | Datasets | Outer folds | Outer repeats | Inner folds | Seeds used by default | Comparison modes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `configs/benchmark/smoke.yaml` | `smoke` | 6 | 2 | 1 | 2 | 1 | `no_hpo` |
| `configs/benchmark/smoke_aft.yaml` | `smoke` | 6 | 2 | 1 | 2 | 1 | `no_hpo`, `hpo` |
| `configs/benchmark/standard_v1.yaml` | `standard` | 6 | 5 | 3 | 3 | 3 repeats from `[11, 22, 33, 44, 55]` | `no_hpo`, `hpo` |
| `configs/benchmark/manuscript_v1.yaml` | `manuscript` | 6 | 5 | 3 | 3 | 3 repeats from `[11, 22, 33, 44, 55]` | `no_hpo` |

Although standard and manuscript configs list five seeds, the repeated outer
loop uses three repeats by default. That gives 15 outer splits per dataset:

```text
5 outer folds x 3 repeats = 15 outer evaluations per dataset
6 datasets x 15 outer evaluations = 90 outer evaluations per method
```

Use `--limit-seeds` for exploratory runs. With `--limit-seeds 1`, repeated
nested CV is reduced to one repeat.

## No-HPO Mode

No-HPO mode evaluates the method's configured default parameters. In benchmark
runs, no-HPO no longer performs inner CV because there is no hyperparameter
selection to make.

Per outer split:

```text
fit default parameters on the outer-training split
predict on the outer-test split
compute metrics
```

Fit count:

```text
1 final fit per outer split
```

Examples for one method across all six standard datasets:

| Config shape | Outer evaluations | Approximate no-HPO fits |
| --- | ---: | ---: |
| Smoke | `6 x 2 x 1 = 12` | 12 |
| Standard or manuscript | `6 x 5 x 3 = 90` | 90 |

The repeated no-HPO runs are still meaningful: each run uses a different
outer-train/outer-test split, so the benchmark estimates split-to-split
variation for the fixed default policy.

## HPO Mode

HPO mode uses the same outer splits as no-HPO mode, then searches
hyperparameters on inner folds inside each outer-training split. Native methods
use Optuna when the method YAML defines a `search_space`.

The standard HPO budget is:

```yaml
hpo:
  enabled: true
  max_trials: 30
  timeout_seconds: 1800
  sampler: tpe
  pruner: median
  n_startup_trials: 10
```

Optuna receives both `n_trials` and `timeout`. Search stops when either 30
trials complete or 1800 seconds elapse. The timeout is checked between trials,
so a long in-progress trial can run beyond the nominal timeout.

Per outer split with `inner_folds: 3` and `max_trials: 30`:

```text
1 default validation x 3 inner folds = 3 inner fits
30 HPO trials x 3 inner folds = 90 inner fits
1 final refit on the outer-training split
```

Approximate HPO fit count for one method across all six standard datasets:

```text
90 outer evaluations x (3 default inner fits + 30 trials x 3 inner folds + 1 final refit)
= 8,460 fits
```

If a method has no `search_space`, HPO is marked disabled for that method and
the benchmark falls back to the default parameters.

If an Optuna trial samples parameters that fail to train or produce a non-finite
selection metric, that candidate is recorded as invalid and the selector keeps a
valid incumbent. When no sampled candidate improves on the default validation
score, the HPO-mode final fit uses the method defaults.

## Dual-Mode Runs

A no-HPO plus HPO comparison uses paired outer splits:

```yaml
comparison_modes: [no_hpo, hpo]
```

For one method on the standard profile:

```text
no-HPO: 90 default final fits
HPO:    8,460 HPO-mode fits
total:  approximately 8,550 fits
```

The important statistical unit is the paired outer result. The no-HPO and HPO
records share the same dataset, split, seed, and method ID, so downstream
exports can compare the default policy against the tuned policy directly.

## Runtime Estimates

Wall-clock time depends heavily on dataset size, model family, CPU/GPU
availability, and whether a trial reaches the timeout. The following estimates
are practical planning ranges for one method across all six standard datasets
with 5 outer folds x 3 repeats.

| Model family | No-HPO only | No-HPO + HPO, 30 trials / 1800s |
| --- | ---: | ---: |
| Fast classical and tabular boosting (`coxph`, `coxnet`, AFT, XGBoost, CatBoost) | minutes | 10-60 minutes |
| Medium tree or neural survival models (`rsf`, `extra_survival_trees`, `mtlr`, `pmf`, `logistic_hazard`, `cox_time`) | 5-20 minutes | 1-6 hours |
| Slow boosting or deep models (`gradient_boosting_survival`, `componentwise_gradient_boosting`, `deepsurv`, `deepsurv_moco`) | 30-90 minutes | 16-40 hours |

The strict configured upper bound for HPO search alone is:

```text
90 HPO outer evaluations x 1800 seconds = 45 hours
```

That bound excludes final refits, metric computation, no-HPO runs, retries, and
timeout overrun from an in-progress trial. Most fast methods finish by trial
count well before the timeout; slow deep methods can approach the timeout.

## Practical Run Tiers

Use these tiers to avoid spending manuscript-level compute before the design is
locked:

| Tier | Suggested shape | Purpose |
| --- | --- | --- |
| Debug | `smoke.yaml`, one method, one dataset | Verify plumbing and artifacts |
| Pilot | standard/manuscript shape with `--limit-seeds 1` | Estimate runtime and catch failures |
| Budget study | `comparison_modes: [no_hpo, hpo]`, `max_trials: 10-15`, `timeout_seconds: 600-900` | Learn whether HPO changes results enough to justify final cost |
| Final | manuscript or standard shape, locked methods/datasets, full repeats | Manuscript-grade reporting |

The shipped manuscript config is the main-paper native default/no-HPO benchmark.
Dual-mode no-HPO/HPO comparisons are represented by `standard_v1.yaml` or by a
new, intentionally added benchmark config and should be treated as sensitivity
or budget-analysis evidence unless explicitly promoted. AutoGluon is separated
into an appendix track via `manuscript_autogluon_v1.yaml`; foundation adapters
remain exploratory under `smoke_foundation.yaml`.

Keep final benchmark YAML immutable once a run starts, and use `--resume` for
restartable execution.
