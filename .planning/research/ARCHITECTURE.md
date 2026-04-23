# Architecture Research

**Domain:** Survival model benchmark system (Python, brownfield extension)
**Researched:** 2026-04-23
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Control Layer (CLI/API + Config Contracts)                                │
├────────────────────────────────────────────────────────────────────────────┤
│  run_benchmark.py  cli.py  api/compare.py  configs/benchmark/*.yaml       │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────────────┐
│ Orchestration Layer (Deterministic Execution Engine)                       │
├────────────────────────────────────────────────────────────────────────────┤
│ benchmark/runner.py  benchmark/tuning.py  data/splitters.py               │
│  - split materialization/reuse     - seeded runs/noise tracks             │
│  - no-HPO and HPO modes            - retry + resume semantics             │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────────────┐
│ Method & Evaluation Layer                                                   │
├────────────────────────────────────────────────────────────────────────────┤
│ methods/* adapters    evaluation/metrics.py    evaluation/statistics.py   │
│  - fit/predict API    - survival metrics         - pairwise + ELO + tests │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────────────┐
│ Artifact Layer (Compact, Reproducible Outputs)                            │
├────────────────────────────────────────────────────────────────────────────┤
│ logging/export.py   logging/manifest.py   logging/tracker.py              │
│  - fold/seed/leaderboard exports   - run ledger + compact index            │
│  - manuscript summaries             - hash/versioned manifest contract      │
└────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| Control surface (`cli.py`, `run_benchmark.py`, `api/compare.py`) | Accept benchmark intents, enforce config schema, expose repeatable entrypoints | Thin adapters around one canonical runner contract |
| Experiment orchestrator (`benchmark/runner.py`) | Deterministic run graph over dataset x method x split x robustness track x mode | Nested iteration with explicit seed propagation, resume, retry, and manifest hashing |
| Tuning engine (`benchmark/tuning.py`) | Inner-CV model selection and HPO/no-HPO mode handling | Method-aware search space + fixed budgets + seed-controlled samplers |
| Dataset/split subsystem (`data/loaders.py`, `data/splitters.py`, `data/robustness.py`) | Data loading, split persistence, robustness perturbations | Immutable split definitions keyed by task/config hash, then read-only in execution |
| Method adapter registry (`methods/registry.py`, `methods/*`) | Uniform fit/predict interface across heterogeneous libraries | Registry lookup + adapter classes implementing shared base behavior |
| Metric engine (`evaluation/metrics.py`) | Fold-level survival metric computation | Pure functions over predictions and targets with explicit horizons/thresholds |
| Statistical inference engine (`evaluation/statistics.py`) | Cross-dataset ranking, pairwise significance, ELO updates, CIs | Deterministic DataFrame transforms with explicit metric direction and correction choices |
| Artifact/export subsystem (`logging/export.py`) | Produce bounded, schema-versioned outputs for downstream use | One write path with schema versions, run ledgers, and compact representations |

## Recommended Project Structure

```
survarena/
├── api/                     # stable user-facing Python and CLI APIs
├── benchmark/               # orchestration, tuning, benchmark execution graph
├── data/                    # loaders, split generation/reuse, robustness tracks
├── methods/                 # model adapters grouped by family
├── evaluation/              # metrics + statistical comparison engines
├── logging/                 # manifests, run ledgers, export contracts
└── utils/                   # seeds, timing, env helpers

configs/
├── benchmark/               # benchmark-level profiles/protocol contracts
└── methods/                 # per-method defaults/search spaces

results/
└── summary/exp_<timestamp>/ # immutable experiment collections
```

### Structure Rationale

- **`benchmark/` is the system spine:** Orchestration remains centralized so reproducibility logic is implemented once, not duplicated per entrypoint.
- **`evaluation/` is split by role:** Metric computation and cross-method statistics evolve independently while sharing fold-level contracts.
- **`logging/` owns storage contracts:** Serialization details are isolated from runner logic so artifact schema changes remain controlled.
- **`configs/` mirrors runtime layers:** Benchmark protocol decisions and method hyperparameter policies stay externally auditable.

## Architectural Patterns

### Pattern 1: Deterministic Run Graph with Immutable Split Contract

**What:** Materialize/reuse splits once, then execute a deterministic Cartesian run graph over benchmark dimensions.
**When to use:** Always, because reproducibility is a first-class requirement.
**Trade-offs:** Slightly higher up-front bookkeeping but dramatically better rerun consistency and failure recovery.

**Example:**
```python
splits = load_or_create_splits(...)
for dataset_id in datasets:
    for method_id in methods:
        for split in splits:
            record = evaluate_split(..., seed=split.seed, split=split)
```

### Pattern 2: Two-Stage Evaluation Pipeline (Metric Stage -> Comparison Stage)

**What:** Compute per-run/per-fold metrics first; compute pairwise, significance, rank, and ELO only from normalized aggregated tables.
**When to use:** Any benchmark where both local performance and global ranking are required.
**Trade-offs:** Adds an explicit intermediate table boundary, but prevents "on-the-fly" ranking bugs and simplifies auditability.

**Example:**
```python
fold = export_fold_results(...)
seed = export_seed_summary(..., fold)
leaderboard = export_leaderboard(..., seed)
export_manuscript_comparison(..., leaderboard, fold_results=fold)
```

### Pattern 3: Ledger + Compact Ledger Dual-Write

**What:** Write a complete run ledger for maximal traceability and a compact ledger that hoists repeated manifest fields into shared metadata.
**When to use:** Long benchmark runs with repeated config metadata across thousands of records.
**Trade-offs:** Slight exporter complexity for large storage savings and faster downstream scans.

**Example:**
```python
shared_manifest = infer_shared_manifest(run_records)
compact_records = strip_shared_manifest(run_records, shared_manifest)
write_jsonl_gz("..._run_records_compact.jsonl.gz", compact_records)
```

## Data Flow

### Request Flow

```
Benchmark Config (YAML)
    ↓
Control surface (CLI/API)
    ↓
Runner builds execution graph
    ↓
Dataset load + split fetch/materialize
    ↓
Per-run loop:
  preprocess -> tune/select -> fit -> infer -> metrics
    ↓
Fold results table
    ↓
Seed summary / leaderboard
    ↓
Pairwise + significance + ELO + CI
    ↓
Comprehensive experiment artifact set
```

### State Management

```
Config + split manifests (immutable inputs)
    ↓
Run records (append-only during execution)
    ↓
Deterministic post-processing tables
    ↓
Schema-versioned exports and indexes
```

### Key Data Flows

1. **Reproducibility flow:** config hash + split indices hash + method config hash are attached to each run record, enabling deterministic replay and stale-input detection.
2. **Inference flow:** fold-level outcomes are transformed into seed- and dataset-level summaries before any cross-method inference step.
3. **Storage flow:** detailed run payloads are persisted as append-only records, then summarized into compact leaderboard/manuscript artifacts.

## Build Order Implications

1. **Artifact contract first:** Freeze run ledger schema and comprehensive output schema before adding more methods; otherwise every new method churns storage format.
2. **Split contract second:** Ensure split generation/reuse and hash validation are complete before pairwise/ELO, because ranking quality depends on aligned comparisons.
3. **Metric parity third:** Confirm both no-HPO and HPO modes emit the same metric columns and statuses, enabling fair pairwise joins.
4. **Comparison layer fourth:** Add pairwise significance and ELO atop stable leaderboard tables, not directly in runner loops.
5. **Compaction last:** Once output schema is stable, add minimal-redundancy single-file packing to avoid repeated migration work.

## Integration Strategy: Full Pairwise Matchup and ELO

### Boundary Placement

- Keep pairwise and ELO inside `evaluation/statistics.py`; do not place ranking math in `runner.py`.
- `logging/export.py` remains the integration point that calls ranking/statistical routines and writes all comparison artifacts.

### Canonical Matchup Grain

- Compute matchups at **dataset-level aggregated performance** (leaderboard rows), with optional significance checks from fold-level paired observations.
- Join key for significance should be: `benchmark_id + dataset_id + split_id + seed + method_id`.

### Full Pairwise Coverage

- For each benchmark and dataset, generate all ordered method pairs (`A vs B` and `B vs A`) for explicit win/loss/tie accounting.
- Keep directionality explicit in output schema (`method_id`, `opponent_method_id`) to avoid ambiguity.

### ELO Strategy

- Initialize ratings uniformly (e.g., 1500), process deterministic matchup order sorted by `dataset_id`, and use fixed `k_factor`.
- Persist `initial_rating`, `k_factor`, and `elo_matches` with each final rating row for auditability.
- Treat ties as 0.5 outcomes; enforce metric-direction aware win resolution.

### Statistical Guardrails

- Pairwise significance should remain independent of ELO updates (ELO for ranking signal, significance for inferential claims).
- Apply multiple-comparison correction at export time (Holm default) and report significant wins/losses separately.

## Strategy: Single Comprehensive Results File with Minimal Redundancy

### Recommended Artifact Pattern

- Keep one canonical file per experiment collection:  
  **`<benchmark_id>_comprehensive_results.parquet`** (columnar, compressed, schema-versioned).
- Preserve JSONL run ledger only as optional debug/audit supplement during transition.

### Table Design Inside One File

- Use a **long-format fact table** with row grain = one executed run (`dataset`, `method`, `split`, `seed`, `track`, `mode`).
- Store heavy nested structures (e.g., full HPO trials) in sidecar columns as compact JSON blobs only when needed.
- Hoist repeated metadata to file-level attributes (or companion metadata row group) rather than repeating per row.

### Minimal Redundancy Tactics

1. **Shared metadata dictionary:** `benchmark_config`, `method_defaults`, and static manifest fields stored once.
2. **Normalized identifiers:** integer surrogate IDs for dataset/method/track/mode columns.
3. **Columnar compression:** use ZSTD in Parquet for high compression with fast scan performance.
4. **Derived outputs not duplicated:** rank summary, pairwise, ELO, CI are reproducible derived views from canonical table and need not be independently materialized for every workflow.

### Compatibility Transition Plan

- Phase 1: dual-write current CSV/JSON outputs and canonical Parquet comprehensive file.
- Phase 2: downstream readers shift to canonical file + computed views.
- Phase 3: keep only essential human-readable summaries plus canonical file.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 0-1k runs | Single-process execution, in-memory DataFrames, one experiment directory per run |
| 1k-100k runs | Batch writes to canonical Parquet, chunked post-processing, explicit retry/resume checkpoints |
| 100k+ runs | Partitioned Parquet datasets, out-of-core aggregation, optional distributed execution backend |

### Scaling Priorities

1. **First bottleneck:** repeated serialization of wide CSV/JSON artifacts; fix with canonical compressed columnar output.
2. **Second bottleneck:** O(m^2) pairwise comparisons as method count grows; fix with cached matchup matrices and incremental updates.

## Anti-Patterns

### Anti-Pattern 1: Ranking Directly from Raw Fold Stream

**What people do:** Update ELO/pairwise live inside training loop.
**Why it's wrong:** Ordering artifacts and partial failures bias rankings.
**Do this instead:** Rank from stabilized post-aggregation tables with deterministic ordering.

### Anti-Pattern 2: Multi-File Artifact Explosion as Source of Truth

**What people do:** Treat many CSV/JSON derivatives as independent truth sources.
**Why it's wrong:** Drift and redundancy make audits and reproducibility brittle.
**Do this instead:** Maintain one canonical comprehensive artifact and generate derivative views on demand.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| PyArrow/Parquet | Columnar canonical artifact writer/reader | Enables compact single-file storage and efficient slicing |
| SciPy statistics | Paired non-parametric significance tests | Keep test settings explicit and versioned in artifact metadata |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `benchmark/runner.py` ↔ `evaluation/metrics.py` | direct function calls with run-level payloads | Metrics remain pure; runner handles orchestration concerns |
| `logging/export.py` ↔ `evaluation/statistics.py` | table-in/table-out DataFrame APIs | Stable contracts allow statistical evolution without runner changes |
| `benchmark/runner.py` ↔ `logging/manifest.py` | manifest object per run | Guarantees hash and schema metadata propagation |

## Sources

- SurvArena project context: `/Users/justin/Documents/SurvArena/.planning/PROJECT.md`
- Protocol and output contract: `/Users/justin/Documents/SurvArena/docs/protocol.md`
- Execution orchestration: `/Users/justin/Documents/SurvArena/survarena/benchmark/runner.py`
- Statistical comparison implementation: `/Users/justin/Documents/SurvArena/survarena/evaluation/statistics.py`
- Export and compact ledger patterns: `/Users/justin/Documents/SurvArena/survarena/logging/export.py`
- Apache Arrow Python Parquet docs: [https://arrow.apache.org/docs/python/parquet.html](https://arrow.apache.org/docs/python/parquet.html)
- Apache Arrow Dataset docs: [https://arrow.apache.org/docs/python/dataset.html](https://arrow.apache.org/docs/python/dataset.html)
- JSON Lines format notes: [https://jsonlines.org/](https://jsonlines.org/)
- SciPy Wilcoxon reference: [https://scipy.github.io/devdocs/reference/generated/scipy.stats.wilcoxon.html](https://scipy.github.io/devdocs/reference/generated/scipy.stats.wilcoxon.html)

---
*Architecture research for: Survival benchmark modernization*
*Researched: 2026-04-23*
