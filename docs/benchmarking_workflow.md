# Benchmarking Workflow

SurvArena benchmark runs compare complete survival-modeling pipelines under a
shared protocol. A run starts from YAML configuration, resolves datasets and
method adapters, creates reusable split definitions, and writes compact
experiment artifacts for downstream reporting.

Last reviewed against the CLI and benchmark configs: 2026-05-18.

```mermaid
flowchart TD
    A["Benchmark YAML<br/>datasets, methods, metrics, budgets"] --> B["Resolve configs<br/>dataset metadata + method search spaces"]
    B --> C["Load dataset<br/>time/event labels + tabular features"]
    C --> D["Create or reuse shared splits<br/>outer folds, repeats, seeds"]
    D --> E{"Comparison mode"}
    E --> F["No-HPO track<br/>fit configured defaults"]
    E --> G["HPO track<br/>inner-fold search within budget"]
    G --> H["Select best params<br/>by primary validation metric"]
    F --> I["Refit on outer-train data"]
    H --> I
    I --> J["Evaluate outer-test fold<br/>risk, survival curves, timing, memory"]
    J --> K["Compute metrics<br/>Uno C, Harrell C, IBS, AUC, calibration, net benefit"]
    K --> L["Aggregate results<br/>folds, seeds, ranks, pairwise tests, CIs"]
    L --> M["Write compact artifacts<br/>fold results, leaderboard, diagnostics, manifest"]
```

## What Is Compared

Each method is evaluated as a full pipeline: model-specific preprocessing,
training, optional tuning, refit, prediction, and metric computation. The same
split definitions are reused across methods so differences reflect model
behavior rather than different train/test partitions.

The standard and manuscript configs use right-censored survival targets and the
same built-in dataset suite by default:

- `support`
- `metabric`
- `aids`
- `gbsg2`
- `flchain`
- `whas500`

## Run Geometry

The maintained benchmark profile is manuscript-grade; smaller runs are created
by selecting one dataset/method and limiting seeds from the same config:

| Profile | Purpose | Typical geometry | Intended use |
| --- | --- | --- | --- |
| `manuscript` | Full reporting run | Repeated nested CV with the full native and foundation portfolio | Paper-grade result tables and statistical summaries |

The manuscript config emits the `no_hpo` track and fits configured defaults
directly on each outer-training split.

Use the benchmark subcommands for a staged run:

```bash
survarena benchmark plan --config configs/benchmark/manuscript_v1.yaml
survarena benchmark doctor --config configs/benchmark/manuscript_v1.yaml --check-imports
survarena benchmark run --config configs/benchmark/manuscript_v1.yaml --dataset whas500 --method coxph --limit-seeds 1
survarena benchmark report results/manuscript_elo
```

`python -m survarena.run_benchmark` remains the thin module entry point for
batch workers and scripts that do not need the broader `survarena benchmark`
command group.

## Output Flow

Benchmark results are written under a generated results directory or the
directory supplied with `--output-dir`.

Core artifacts include:

- fold-level metric tables
- leaderboards
- run diagnostics
- experiment manifests with config and environment metadata

For the full protocol and artifact contract, see [`protocol.md`](protocol.md).

## Dataset-by-Dataset, Model-by-Model Manuscript Runs

When manuscript-grade runs are too expensive to execute as one monolithic job,
run each dataset/method pair independently with resume support:

```bash
scripts/run_manuscript_by_dataset_model.sh
```

Defaults:

- config: `configs/benchmark/manuscript_v1.yaml`
- output root: `results/manuscript_dataset_model`
- retries per run: `1`

Useful overrides:

```bash
MAX_RETRIES=2 LIMIT_SEEDS=2 OUTPUT_ROOT=results/manuscript_partial \
  scripts/run_manuscript_by_dataset_model.sh
```

The script writes one output directory per dataset and method plus:

- per-run logs in `results/.../<dataset>/<method>.log`
- aggregate status table in `results/.../run_status.csv`

This pattern keeps long-running methods from blocking faster methods and makes
retry/resume straightforward without re-running completed pairs.
