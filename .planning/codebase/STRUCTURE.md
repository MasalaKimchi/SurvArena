# Codebase Structure

**Analysis Date:** 2026-04-23

## Directory Layout

```text
SurvArena/
├── survarena/                # Runtime package: API, benchmark engine, methods, data, evaluation, logging
├── configs/                  # YAML contracts for datasets, methods, and benchmark profiles
├── tests/                    # Pytest suite for CLI/API/data/method/statistics behavior
├── scripts/                  # Setup, environment validation, and benchmark helper scripts
├── docs/                     # User-facing protocol and environment docs
├── data/                     # Local placeholders and split-cache root
├── results/                  # Generated benchmark and predictor outputs
├── examples/                 # Notebook and sample output artifacts
├── pyproject.toml            # Build metadata, dependencies, tool settings, CLI script registration
└── requirements.txt          # Editable install shortcut for contributor setup
```

## Directory Purposes

**`survarena/`:**
- Purpose: all executable product logic.
- Contains: layered modules (`api/`, `benchmark/`, `data/`, `methods/`, `evaluation/`, `logging/`, `automl/`, `utils/`).
- Key files: `survarena/cli.py`, `survarena/api/predictor.py`, `survarena/api/compare.py`, `survarena/benchmark/runner.py`, `survarena/methods/registry.py`.

**`configs/`:**
- Purpose: declarative runtime behavior.
- Contains: `configs/datasets/*.yaml`, `configs/methods/*.yaml`, `configs/benchmark/*.yaml`.
- Key files: `configs/benchmark/standard_v1.yaml`, `configs/benchmark/smoke_all_models_no_hpo.yaml`, `configs/benchmark/cloud_comprehensive_all_models_hpo.yaml`.

**`tests/`:**
- Purpose: regression protection and workflow validation.
- Contains: focused test modules and shared test bootstrap.
- Key files: `tests/conftest.py`, `tests/test_predictor_registry.py`, `tests/test_compare_api.py`, `tests/test_metrics_and_tuning.py`, `tests/test_robustness_tracks.py`.

**`scripts/`:**
- Purpose: operational entry points around environment and benchmark execution.
- Contains: shell/python helpers.
- Key files: `scripts/setup_env.sh`, `scripts/check_environment.py`, `scripts/run_cloud_comprehensive.sh`, `scripts/validate_benchmark_protocol.sh`.

## Key File Locations

**Entry Points:**
- `survarena/cli.py`: user CLI command router.
- `survarena/run_benchmark.py`: benchmark module entry.
- `survarena/__init__.py`: package API exports.

**Configuration:**
- `pyproject.toml`: dependencies, scripts, pytest, ruff.
- `configs/benchmark/*.yaml`: benchmark profile and method/dataset matrix.
- `configs/methods/*.yaml`: per-method defaults and search spaces.
- `configs/datasets/*.yaml`: benchmark dataset metadata contracts.

**Core Logic:**
- `survarena/api/predictor.py`: fit, evaluate, and persist predictor artifacts.
- `survarena/api/compare.py`: user-dataset comparison workflow.
- `survarena/benchmark/runner.py`: split execution and retry/resume mechanics.
- `survarena/benchmark/tuning.py`: inner-CV and optional Optuna HPO.
- `survarena/data/splitters.py`: split manifest creation/reuse and validation.
- `survarena/logging/export.py`: fold/summary/manuscript export pipeline.

**Testing:**
- `tests/`: all runtime behavior tests, organized by product surface.

## Naming Conventions

**Files:**
- Use snake_case module names (`survarena/data/user_dataset.py`, `survarena/evaluation/statistics.py`).
- Test files follow `test_<feature>.py` naming (`tests/test_cli.py`, `tests/test_hpo_config.py`).

**Directories:**
- Segment by capability under `survarena/` (`api`, `benchmark`, `data`, `methods`, `evaluation`, `logging`, `automl`, `utils`).

## Where to Add New Code

**New Feature:**
- Primary code: orchestrator changes in `survarena/api/` or `survarena/benchmark/`; reusable domain logic in `survarena/data/`, `survarena/evaluation/`, or `survarena/logging/`.
- Tests: add or extend `tests/test_<feature>.py`; place shared fixtures in `tests/conftest.py`.

**New Component/Module:**
- Implementation: place in nearest domain package under `survarena/`.
- For new methods: add adapter in `survarena/methods/<family>/`, register in `survarena/methods/registry.py`, and add config in `configs/methods/<method_id>.yaml`.
- For new benchmark profiles: add YAML in `configs/benchmark/` and use `survarena/run_benchmark.py`.

**Utilities:**
- Shared helpers: `survarena/utils/` unless the helper is domain-specific (then keep it local, for example `survarena/logging/`).

## Special Directories

**`data/splits/`:**
- Purpose: split JSON files and split manifest files.
- Generated: Yes (via `survarena/data/splitters.py`).
- Committed: No, except placeholder `.gitkeep`.

**`results/`:**
- Purpose: generated predictor and benchmark outputs.
- Generated: Yes (`survarena/api/predictor.py`, `survarena/logging/export.py`).
- Committed: Generally no.

**`.planning/codebase/`:**
- Purpose: codebase mapping artifacts consumed by planning/execution workflows.
- Generated: Yes.
- Committed: project-dependent; currently present in working tree.

---

*Structure analysis: 2026-04-23*
# Codebase Structure

**Analysis Date:** 2026-04-23

## Directory Layout

```text
SurvArena/
├── survarena/               # Core Python package (API, orchestration, methods, data, evaluation, logging)
├── configs/                 # YAML configuration for benchmarks, methods, and datasets
├── tests/                   # Pytest test suite
├── scripts/                 # Environment setup and benchmark helper scripts
├── docs/                    # Project and protocol documentation
├── data/                    # Local dataset/split cache directories
├── results/                 # Generated benchmark and predictor outputs
├── examples/                # Example notebook and sample output artifacts
├── pyproject.toml           # Build metadata, dependencies, tool config, CLI script
└── requirements.txt         # Editable install shortcut for contributor environment
```

## Directory Purposes

**`survarena/`:**
- Purpose: Runtime package containing all executable logic.
- Contains: `api/`, `benchmark/`, `data/`, `methods/`, `evaluation/`, `logging/`, `automl/`, `utils/`.
- Key files: `survarena/cli.py`, `survarena/run_benchmark.py`, `survarena/api/predictor.py`, `survarena/api/compare.py`, `survarena/benchmark/runner.py`, `survarena/methods/registry.py`.

**`configs/`:**
- Purpose: Declarative runtime configuration.
- Contains: Dataset metadata, method defaults/search spaces, benchmark profiles.
- Key files: `configs/benchmark/standard_v1.yaml`, `configs/benchmark/cloud_comprehensive_all_models_hpo.yaml`, `configs/methods/*.yaml`, `configs/datasets/*.yaml`.

**`tests/`:**
- Purpose: Regression and behavior tests for CLI, API, data, methods, and statistics.
- Contains: `pytest` modules and shared fixtures.
- Key files: `tests/test_cli.py`, `tests/test_compare_api.py`, `tests/test_metrics_and_tuning.py`, `tests/conftest.py`.

**`scripts/`:**
- Purpose: Operational helpers for setup and benchmark validation/run.
- Contains: Shell scripts for environment setup and benchmark protocol checks.
- Key files: `scripts/setup_env.sh`, `scripts/run_cloud_comprehensive.sh`, `scripts/validate_benchmark_protocol.sh`.

## Key File Locations

**Entry Points:**
- `survarena/cli.py`: Primary CLI (`fit`, `compare`, `foundation-check`).
- `survarena/run_benchmark.py`: Benchmark module entry point.
- `survarena/__init__.py`: Public API export surface.

**Configuration:**
- `pyproject.toml`: package metadata, dependencies, tool config, and console script mapping.
- `configs/benchmark/*.yaml`: benchmark profiles and datasets/method lists.
- `configs/methods/*.yaml`: per-method default params and search spaces.
- `configs/datasets/*.yaml`: built-in dataset metadata contracts.

**Core Logic:**
- `survarena/api/predictor.py`: AutoML-like predictor workflow and artifact persistence.
- `survarena/api/compare.py`: User-dataset benchmark-style comparison workflow.
- `survarena/benchmark/runner.py`: Split-level benchmark execution engine.
- `survarena/benchmark/tuning.py`: Inner-CV selection and Optuna-backed HPO orchestration.
- `survarena/data/splitters.py`: deterministic split creation, validation, and manifest reuse.
- `survarena/logging/export.py`: benchmark artifact export pipeline.

**Testing:**
- `tests/`: test modules matching product surfaces (`cli`, `compare`, `statistics`, `robustness`, `hpo`, etc.).

## Naming Conventions

**Files:**
- Snake_case module names across package and tests (examples: `survarena/data/user_dataset.py`, `tests/test_predictor_registry.py`).
- One primary responsibility per module (workflow modules in `survarena/api/` and `survarena/benchmark/`; utility modules in `survarena/utils/`).

**Directories:**
- Domain-oriented package segmentation by capability (`api`, `benchmark`, `data`, `methods`, `evaluation`, `logging`, `automl`, `utils`) under `survarena/`.

## Where to Add New Code

**New Feature:**
- Primary code: add domain logic under `survarena/` by layer (user-facing orchestration in `survarena/api/`, benchmark orchestration in `survarena/benchmark/`, reusable data logic in `survarena/data/`).
- Tests: add corresponding `tests/test_<feature>.py` and reuse fixtures in `tests/conftest.py`.

**New Component/Module:**
- Implementation: place in the closest domain package under `survarena/`.
- Implementation: if it introduces a new model adapter, add class in `survarena/methods/<family>/`, register in `survarena/methods/registry.py`, and add method config in `configs/methods/<method_id>.yaml`.
- Implementation: if it introduces a benchmark profile, add YAML in `configs/benchmark/` and keep CLI wiring in `survarena/run_benchmark.py` unchanged unless new arguments are required.

**Utilities:**
- Shared helpers: `survarena/utils/` for cross-cutting helpers; prefer domain-local helpers first (for example, `survarena/logging/` for export-specific utilities).

## Special Directories

**`data/splits/`:**
- Purpose: persisted split definitions and split manifests for reproducibility.
- Generated: Yes.
- Committed: No (cache/output style directory).

**`results/`:**
- Purpose: predictor artifacts and benchmark summary outputs.
- Generated: Yes.
- Committed: Generally no (runtime outputs).

**`data/raw/`, `data/processed/`, `data/splits/`:**
- Purpose: local data staging and cached split manifests/indexes used by benchmark runs.
- Generated: Yes.
- Committed: No for local datasets/splits beyond placeholders (`.gitkeep` only).

**`.planning/codebase/`:**
- Purpose: generated codebase mapping documents for planner/executor context.
- Generated: Yes.
- Committed: Project-dependent; currently present in repository workspace.

---

*Structure analysis: 2026-04-23*
