# Architecture

**Analysis Date:** 2026-04-23

## Pattern Overview

**Overall:** Layered, config-driven Python toolkit with adapter registry dispatch and filesystem-backed experiment artifacts.

**Key Characteristics:**
- Keep entry points thin: parse flags and delegate (`survarena/cli.py`, `survarena/run_benchmark.py`).
- Centralize orchestration in service modules (`survarena/api/predictor.py`, `survarena/api/compare.py`, `survarena/benchmark/runner.py`).
- Resolve model backends indirectly by `method_id` via `survarena/methods/registry.py`.

## Layers

**Interface Layer:**
- Purpose: expose CLI and importable API contracts.
- Location: `survarena/cli.py`, `survarena/run_benchmark.py`, `survarena/__init__.py`, `survarena/api/__init__.py`
- Contains: argument parsing, command dispatch, public symbol exports.
- Depends on: `survarena.api`, `survarena.benchmark.runner`, `survarena.config`.
- Used by: local CLI runs and Python consumers.

**Application Layer:**
- Purpose: execute fit/compare/benchmark workflows.
- Location: `survarena/api/predictor.py`, `survarena/api/compare.py`, `survarena/benchmark/runner.py`, `survarena/benchmark/tuning.py`
- Contains: validation planning, fold evaluation loops, HPO orchestration, retry/resume logic.
- Depends on: data, methods, evaluation, logging layers.
- Used by: interface layer only.

**Data Layer:**
- Purpose: load datasets, normalize schema, preprocess features, and persist reusable splits.
- Location: `survarena/data/`
- Contains: dataset loaders (`loaders.py`), user ingestion (`user_dataset.py`), split manifesting (`splitters.py`), perturbation tracks (`robustness.py`), tabular preprocessing (`preprocess.py`).
- Depends on: `numpy`, `pandas`, `scikit-learn`, YAML.
- Used by: predictor and benchmark application services.

**Model Adapter Layer:**
- Purpose: provide unified `fit/predict_risk/predict_survival` behavior over heterogeneous libraries.
- Location: `survarena/methods/`
- Contains: base contract (`base.py`), adapter families (`classical/`, `tree/`, `boosting/`, `deep/`, `foundation/`, `automl/`), and registry map (`registry.py`).
- Depends on: selected model libraries plus `survarena/methods/preprocessing.py`.
- Used by: `survarena/benchmark/runner.py` and `survarena/api/predictor.py`.

**Evaluation and Artifact Layer:**
- Purpose: compute metrics/statistics and write benchmark outputs.
- Location: `survarena/evaluation/` and `survarena/logging/`
- Contains: metric bundles (`metrics.py`), statistical summaries (`statistics.py`), manifest/ledger/export writers (`manifest.py`, `tracker.py`, `export.py`).
- Depends on: SciPy/Pandas/NumPy and filesystem I/O.
- Used by: benchmark and compare workflows.

## Data Flow

**Predictor Flow (`survarena fit`):**

1. `survarena/cli.py` parses `fit` options and instantiates `SurvivalPredictor` from `survarena/api/predictor.py`.
2. `survarena/data/user_dataset.py` loads and validates labeled tabular data into `SurvivalDataset` from `survarena/data/schema.py`.
3. Portfolio is selected via `survarena/automl/presets.py` and validated against `survarena/methods/registry.py`.
4. Validation folds are built by `survarena/automl/validation.py`, and tuning is executed by `survarena/benchmark/tuning.py`.
5. Best models are refit and evaluated in `survarena/api/predictor.py`; artifacts are written to `results/predictor/`.

**Benchmark Flow (`python -m survarena.run_benchmark` and compare API):**

1. Benchmark YAML is loaded by `survarena/run_benchmark.py` through `survarena/config.py`.
2. Datasets come from `survarena/data/loaders.py` (built-in) or `survarena/data/user_dataset.py` (user data).
3. Split manifests are created/reused by `survarena/data/splitters.py` under `data/splits/<task_id>/`.
4. Optional robustness tracks are applied via `survarena/data/robustness.py`.
5. Split execution occurs in `survarena/benchmark/runner.py`; each split creates a run payload and status.
6. Summaries and manuscript artifacts are exported by `survarena/logging/export.py` into `results/summary/exp_*/`.

**State Management:**
- Runtime state is in-memory in orchestrator objects (notably `SurvivalPredictor` in `survarena/api/predictor.py`) and persisted to files for reproducibility.

## Key Abstractions

**Method Contract:**
- Purpose: enforce shared adapter API.
- Examples: `survarena/methods/base.py`, implementations in `survarena/methods/classical/`, `survarena/methods/deep/`, `survarena/methods/foundation/`.
- Pattern: abstract base class + polymorphic adapters.

**Registry Lookup:**
- Purpose: map `method_id` to adapter class.
- Examples: `_REGISTRY_TARGETS` and `get_method_class()` in `survarena/methods/registry.py`.
- Pattern: lazy import with memoization.

**Run Manifest:**
- Purpose: capture hash-linked run metadata and status.
- Examples: `RunManifest` in `survarena/logging/manifest.py` and payload serialization in `survarena/logging/tracker.py`.
- Pattern: append-only JSON/JSONL export contract.

## Entry Points

**CLI Entry Point:**
- Location: `survarena/cli.py` (registered in `pyproject.toml`).
- Triggers: `survarena fit`, `survarena compare`, `survarena foundation-check`.
- Responsibilities: parse arguments, call API, print JSON outputs.

**Benchmark Entry Point:**
- Location: `survarena/run_benchmark.py`.
- Triggers: `python -m survarena.run_benchmark`.
- Responsibilities: load benchmark config, run benchmark, support dry-run/resume/retries.

**Library Entry Point:**
- Location: `survarena/__init__.py`.
- Triggers: `from survarena import SurvivalPredictor, compare_survival_models`.
- Responsibilities: stable import surface for external callers.

## Error Handling

**Strategy:** fail fast for invalid config/data, but continue benchmark loops when a single split-method run fails.

**Patterns:**
- Input and config validation with explicit `ValueError`/`RuntimeError` in `survarena/api/compare.py`, `survarena/data/user_dataset.py`, and `survarena/api/predictor.py`.
- Exception capture and failure payload emission in `survarena/benchmark/runner.py` (`status = "failed"` records).
- Split-integrity and stratification checks before split reuse in `survarena/data/splitters.py`.

## Cross-Cutting Concerns

**Logging:** Structured run and artifact export in `survarena/logging/tracker.py` and `survarena/logging/export.py`.
**Validation:** Data/split/metric validation in `survarena/data/user_dataset.py`, `survarena/data/splitters.py`, and `survarena/evaluation/metrics.py`.
**Authentication:** Optional foundation readiness/auth checks in `survarena/methods/foundation/readiness.py`.

---

*Architecture analysis: 2026-04-23*
# Architecture

**Analysis Date:** 2026-04-23

## Pattern Overview

**Overall:** Layered Python package with config-driven orchestration and registry-resolved model adapters.

**Key Characteristics:**
- Keep entry points thin: parse arguments and delegate (`survarena/cli.py`, `survarena/run_benchmark.py`, `survarena/__init__.py`).
- Route all training/evaluation orchestration through service modules (`survarena/api/predictor.py`, `survarena/api/compare.py`, `survarena/benchmark/runner.py`).
- Add new model families via adapter + registry + config triad (`survarena/methods/base.py`, `survarena/methods/registry.py`, `configs/methods/*.yaml`).

## Layers

**Entry/Interface Layer:**
- Purpose: Parse commands and expose public API.
- Location: `survarena/cli.py`, `survarena/run_benchmark.py`, `survarena/__init__.py`, `survarena/api/__init__.py`
- Contains: CLI argument parsing, public API exports.
- Depends on: `survarena.api`, `survarena.benchmark.runner`, `survarena.config`.
- Used by: End users running `survarena` CLI or importing package APIs.

**Application Service Layer:**
- Purpose: Coordinate predictor and benchmark use cases.
- Location: `survarena/api/predictor.py`, `survarena/api/compare.py`, `survarena/benchmark/runner.py`, `survarena/benchmark/tuning.py`
- Contains: Workflow orchestration, split-evaluate loops, model selection/HPO, retry/resume behavior.
- Depends on: data, methods, evaluation, logging, and config layers.
- Used by: CLI entry points and direct Python API calls.

**Data Layer:**
- Purpose: Load datasets, validate schema, preprocess features, and create/reuse splits.
- Location: `survarena/data/`
- Contains: loaders (`loaders.py`), user data ingestion (`user_dataset.py`), split persistence (`splitters.py`), preprocessing (`preprocess.py`), robustness perturbations (`robustness.py`), typed schema (`schema.py`).
- Depends on: NumPy/Pandas/Scikit-learn and YAML.
- Used by: predictor, compare, and benchmark runner.

**Model Adapter Layer:**
- Purpose: Provide a uniform fit/predict interface over heterogeneous survival backends.
- Location: `survarena/methods/`
- Contains: base interface (`base.py`), dynamic registry (`registry.py`), classical/tree/boosting/deep/foundation/autogluon adapters.
- Depends on: external ML libraries plus preprocessing utilities (`survarena/methods/preprocessing.py`).
- Used by: `survarena/api/predictor.py` and `survarena/benchmark/runner.py`.

**Evaluation & Reporting Layer:**
- Purpose: Compute metrics/statistics and write benchmark artifacts.
- Location: `survarena/evaluation/`, `survarena/logging/`
- Contains: metric computation (`metrics.py`), statistical analysis (`statistics.py`), run manifests and exports (`manifest.py`, `tracker.py`, `export.py`).
- Depends on: NumPy/Pandas/SciPy/Torchsurv and filesystem I/O.
- Used by: compare and benchmark flows.

## Data Flow

**Predictor Flow (`survarena fit` / `SurvivalPredictor.fit`):**

1. Parse user flags and construct `SurvivalPredictor` in `survarena/cli.py`.
2. Load/validate tabular survival data through `survarena/data/user_dataset.py` and typed schema checks in `survarena/data/schema.py`.
3. Resolve portfolio from data-aware presets in `survarena/automl/presets.py`.
4. Build holdout or bagging validation plan in `survarena/automl/validation.py`.
5. Run inner-loop parameter selection in `survarena/benchmark/tuning.py` using method adapters resolved by `survarena/methods/registry.py`.
6. Refit selected adapters, compute metrics via `survarena/evaluation/metrics.py`, and persist predictor artifacts from `survarena/api/predictor.py` to `results/predictor/`.

**Benchmark/Compare Flow (`run_benchmark` / `compare_survival_models`):**

1. Load benchmark YAML in `survarena/run_benchmark.py` or construct compare config in `survarena/api/compare.py`.
2. Load built-in datasets from `survarena/data/loaders.py` or user datasets from `survarena/data/user_dataset.py`.
3. Create or reuse deterministic split manifests in `survarena/data/splitters.py` (`data/splits/<task_id>/`).
4. Apply robustness perturbations when configured through `survarena/data/robustness.py`.
5. Evaluate each method/split in `survarena/benchmark/runner.py` with per-run manifest payloads from `survarena/logging/manifest.py`.
6. Export fold tables, leaderboards, significance tests, run ledgers, and navigator outputs in `survarena/logging/export.py` to `results/summary/exp_*/`.

**State Management:**
- Runtime state is held in-memory within workflow objects (notably `SurvivalPredictor` in `survarena/api/predictor.py`) and persisted to filesystem artifacts for reproducibility.

## Key Abstractions

**BaseSurvivalMethod contract:**
- Purpose: Standard adapter interface across all model families.
- Examples: `survarena/methods/base.py`, implemented in `survarena/methods/classical/`, `survarena/methods/tree/`, `survarena/methods/boosting/`, `survarena/methods/deep/`, `survarena/methods/foundation/`, `survarena/methods/automl/`.
- Pattern: Adapter + polymorphic dispatch through registry.

**Method registry:**
- Purpose: Resolve `method_id` to concrete adapter class.
- Examples: `survarena/methods/registry.py`.
- Pattern: Lazy dynamic import map with memoized resolution.

**Validation planning abstractions:**
- Purpose: Decouple split construction from model fitting for holdout and bagged OOF modes.
- Examples: `ValidationPlan` and `ResampledFold` in `survarena/automl/validation.py`.
- Pattern: Precompute fold caches, then pass normalized fold payloads into tuning/selection.

**Run manifest + ledger payloads:**
- Purpose: Attach schema, environment snapshot, metrics, and failure metadata to every run.
- Examples: `survarena/logging/manifest.py`, `survarena/logging/export.py`, `survarena/logging/tracker.py`.
- Pattern: Structured event payloads serialized as JSON/JSONL.GZ.

## Entry Points

**CLI entry point (`survarena`):**
- Location: `survarena/cli.py` (registered in `pyproject.toml`).
- Triggers: `survarena fit`, `survarena compare`, `survarena foundation-check`.
- Responsibilities: Parse args, call API services, print JSON summaries.

**Benchmark module entry:**
- Location: `survarena/run_benchmark.py`.
- Triggers: `python -m survarena.run_benchmark`.
- Responsibilities: Load benchmark YAML, execute `run_benchmark`, support dry-run/resume/retry.

**Python API entry:**
- Location: `survarena/__init__.py` re-exporting from `survarena/api/`.
- Triggers: `from survarena import SurvivalPredictor, compare_survival_models`.
- Responsibilities: Stable import surface for library consumers.

## Error Handling

**Strategy:** Validate early, fail clearly for configuration/data issues, and downgrade per-run model failures into structured failure records during benchmark loops.

**Patterns:**
- Defensive input/config checks with `ValueError` and `RuntimeError` in `survarena/api/compare.py`, `survarena/data/user_dataset.py`, and `survarena/benchmark/runner.py`.
- Failure capture with traceback and status payloads in `survarena/benchmark/runner.py` instead of aborting whole benchmark.
- Split integrity and stratification validation before reuse in `survarena/data/splitters.py`.

## Cross-Cutting Concerns

**Logging:** Structured JSON/CSV artifact writing in `survarena/logging/export.py` and `survarena/logging/tracker.py`.
**Validation:** Dataset/split/model portfolio validation in `survarena/data/user_dataset.py`, `survarena/data/splitters.py`, and `survarena/automl/presets.py`.
**Authentication:** Optional Hugging Face token checks for TabPFN readiness in `survarena/methods/foundation/readiness.py`.

---

*Architecture analysis: 2026-04-23*
