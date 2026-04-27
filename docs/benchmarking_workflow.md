# Benchmarking Workflow

SurvArena benchmark runs compare complete survival-modeling pipelines under a
shared protocol. A run starts from YAML configuration, resolves datasets and
method adapters, creates reusable split definitions, and writes compact
experiment artifacts for downstream reporting.

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
    L --> M["Write compact artifacts<br/>leaderboards, summaries, manifests, run ledger"]
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

The maintained benchmark profiles differ mainly by runtime and statistical
strength:

| Profile | Purpose | Typical geometry | Intended use |
| --- | --- | --- | --- |
| `smoke` | Fast wiring check | Small folds and one seed | CI, local validation, adapter sanity checks |
| `standard` | Balanced benchmark | Repeated nested CV with native core models | Routine comparisons under realistic runtime |
| `manuscript` | Full reporting run | Repeated nested CV with the full native portfolio | Paper-grade result tables and statistical summaries |

`comparison_modes` controls whether a config emits `no_hpo`, `hpo`, or both
tracks. No-HPO fits configured defaults directly. HPO uses the configured inner
folds and budget, then refits the selected configuration before outer-test
evaluation.

## Output Flow

Benchmark results are written under
`results/summary/<benchmark_id>_<model_name>_<timestamp>/`. The
human entry point is the generated experiment `README.md`; the machine entry
point is `experiment_navigator.json`.

Core artifacts include:

- fold-level metric tables
- seed and overall summaries
- leaderboards
- manuscript comparison reports, with detailed ranking and significance files
  depending on the configured artifact layout
- failure and missing-metric summaries when available
- compact per-run ledgers
- experiment manifests with config and environment metadata

For the full protocol and artifact contract, see [`protocol.md`](protocol.md).
