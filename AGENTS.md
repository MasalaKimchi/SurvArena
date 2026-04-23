<!-- GSD:project-start source:PROJECT.md -->
## Project

**SurvArena Benchmark Modernization**

SurvArena is a Python benchmark toolkit for comparing survival analysis methods across datasets with reproducible, config-driven runs. This project evolves the existing codebase into a TabArena-style, manuscript-grade benchmarking system focused on practitioners choosing robust models under realistic constraints. It emphasizes comparable no-HPO vs HPO evaluations, strong statistical reporting, and compact, non-redundant result storage.

**Core Value:** A practitioner can trust one benchmark run to produce fair, statistically robust, and compactly stored model comparisons across diverse survival datasets.

### Constraints

- **Ecosystem**: Python-only package coverage (exclusive) — non-Python model ecosystems are out of scope.
- **Runtime Budget**: Wall-clock time is the primary operational constraint — phase plans must prioritize efficient benchmark design.
- **Benchmark Scope**: Medium balanced dataset suite — broad enough for external validity without unbounded runtime.
- **Quality Gate**: All touched-code lint/type/test checks must pass — maintain maintainability while evolving benchmark logic.
- **Storage Contract**: Results must be compact and non-redundant, with one comprehensive artifact per experiment collection.
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python (requires `>=3.10`) - all application logic in `survarena/`, tests in `tests/`, and tooling in `scripts/`.
- YAML - experiment and model configuration in `configs/benchmark/`, `configs/methods/`, and `configs/datasets/`.
- Markdown - project and protocol documentation in `README.md` and `docs/protocol.md`.
- Shell - operational scripts in `scripts/setup_env.sh`, `scripts/run_cloud_comprehensive.sh`, and `scripts/validate_benchmark_protocol.sh`.
## Runtime
- CPython (preferred 3.11, supported 3.10/3.11/3.12) documented in `README.md` and `requirements.txt`.
- `pip` with editable install from `pyproject.toml`.
- Lockfile: missing (`requirements.txt` installs `-e ".[dev]"`; no resolved lock artifact detected).
## Frameworks
- Tabular ML stack: `numpy`, `pandas`, `scipy`, and `scikit-learn` in `survarena/data/preprocess.py`, `survarena/data/splitters.py`, and `survarena/automl/validation.py`.
- Survival modeling stack: `scikit-survival`, `lifelines`, `pycox`, `torch`, and `torchsurv` in `survarena/methods/` and `survarena/evaluation/metrics.py`.
- `pytest` configured in `pyproject.toml` (`[tool.pytest.ini_options]`) with test modules under `tests/`.
- `setuptools.build_meta` package build backend in `pyproject.toml`.
- `ruff` linting configured in `pyproject.toml` (`line-length = 120`, `target-version = "py310"`).
## Key Dependencies
- `numpy==1.26.4` and `pandas==2.2.2` - dataframe and array core in `survarena/api/predictor.py` and `survarena/benchmark/runner.py`.
- `scikit-learn==1.6.1` - split and preprocessing primitives in `survarena/data/preprocess.py` and `survarena/data/splitters.py`.
- `torch==2.2.2` and `torchsurv==0.1.5` - deep model and metric runtime in `survarena/methods/deep/` and `survarena/evaluation/metrics.py`.
- `PyYAML==6.0.2` - YAML configuration loading in `survarena/config.py` and `survarena/data/loaders.py`.
- `autogluon.tabular==1.5.0` - AutoML backend in `survarena/automl/autogluon_backend.py`.
- `xgboost==3.2.0` and `catboost==1.2.10` - boosting adapters in `survarena/methods/boosting/tabular_boosting.py`.
- `psutil==5.9.8` - runtime telemetry in `survarena/logging/tracker.py` and `survarena/utils/env.py`.
- `matplotlib==3.9.2` and `seaborn==0.13.2` - plotting/report support from `survarena/api/predictor.py` and docs/examples.
## Configuration
- Packaging, dependency pins, script entry points, pytest, and ruff are defined in `pyproject.toml`.
- Runtime benchmark and method controls are defined in `configs/benchmark/*.yaml` and `configs/methods/*.yaml`.
- Dataset metadata contracts are defined in `configs/datasets/*.yaml`.
- Build metadata uses `setuptools` via `pyproject.toml`.
- Console script entry point is `survarena = "survarena.cli:main"` in `pyproject.toml`.
## Platform Requirements
- Repo-local virtual environment workflow via `scripts/setup_env.sh` and verification via `scripts/check_environment.py`.
- Writable local filesystem for split cache and outputs in `data/splits/`, `results/predictor/`, and `results/summary/`.
- CLI/batch execution model using `survarena/cli.py` and `survarena/run_benchmark.py`.
- No service container, web server, or orchestrator contract is defined in repository code.
# Technology Stack
## Languages
- Python 3.10+ - Core package, CLI entrypoints, benchmark pipeline, method adapters, and exports in `survarena/` (`pyproject.toml`, `survarena/cli.py`, `survarena/run_benchmark.py`).
- YAML - Config-driven benchmark, method, and dataset definitions in `configs/benchmark/`, `configs/methods/`, and `configs/datasets/`.
- Markdown - User/dev protocol and environment documentation in `README.md` and `docs/`.
- Shell (Bash) - Environment/bootstrap and benchmark execution wrappers in `scripts/setup_env.sh`, `scripts/run_cloud_comprehensive.sh`, and `scripts/validate_benchmark_protocol.sh`.
## Runtime
- CPython runtime with 3.11 preferred and 3.10/3.12 supported (`README.md`, `docs/environment.md`, `scripts/setup_env.sh`).
- Package metadata requires Python >=3.10 (`pyproject.toml`).
- `pip` (editable install workflow via `python -m pip install -e ...`) (`requirements.txt`, `scripts/setup_env.sh`).
- Build backend: `setuptools.build_meta` with `setuptools>=68` and `wheel` (`pyproject.toml`).
- Lockfile: missing (no lockfile detected at repository root).
## Frameworks
- NumPy 1.26.4 + pandas 2.2.2 + SciPy 1.13.1 - Data frame, arrays, and numeric routines (`pyproject.toml`, `survarena/data/*.py`).
- scikit-learn 1.6.1 - Preprocessing/model support utilities (`pyproject.toml`, `survarena/methods/preprocessing.py`).
- scikit-survival 0.24.1 + lifelines 0.30.0 - Classical survival estimators and utilities (`pyproject.toml`, `survarena/methods/classical/*.py`).
- pycox 0.3.0 + torchtuples 0.2.2 + torch 2.2.2 + torchsurv 0.1.5 - Deep survival models and metric tooling (`pyproject.toml`, `survarena/methods/deep/*.py`, `survarena/evaluation/metrics.py`).
- AutoGluon Tabular 1.5.0 - AutoML-backed survival adapter and Mitra foundation path (`pyproject.toml`, `survarena/automl/autogluon_backend.py`, `survarena/methods/foundation/mitra_survival.py`).
- pytest >=8.3,<9 - Test runner configured under `[tool.pytest.ini_options]` in `pyproject.toml`, tests in `tests/`.
- Ruff >=0.15,<0.16 - Linting/style enforcement configured in `[tool.ruff]` in `pyproject.toml`.
- `python -m compileall survarena` and dry-run benchmark checks used as smoke-validation commands (`README.md`, `docs/environment.md`).
## Key Dependencies
- `scikit-survival==0.24.1` - Built-in dataset loading and model ecosystem bridge (`survarena/data/loaders.py`, `survarena/methods/classical/*.py`).
- `pycox==0.3.0` - Deep survival adapters and built-in dataset access (`survarena/methods/deep/pycox_models.py`, `survarena/data/loaders.py`).
- `torchsurv==0.1.5` - Metric computation backbone for protocol outputs (`docs/protocol.md`, `survarena/evaluation/metrics.py`).
- `autogluon.tabular==1.5.0` - AutoGluon survival backend and optional foundation integration (`survarena/automl/autogluon_backend.py`, `survarena/methods/automl/autogluon_survival.py`).
- `PyYAML==6.0.2` - YAML config parsing (`survarena/config.py`, `survarena/data/loaders.py`).
- `psutil==5.9.8` - Process/hardware telemetry for manifests (`survarena/logging/tracker.py`, `survarena/utils/env.py`).
- `pyarrow==20.0.0` - Parquet input support (`survarena/data/io.py`).
- `xgboost==3.2.0` and `catboost==1.2.10` - Gradient-boosting survival adapters (`survarena/methods/boosting/tabular_boosting.py`).
## Configuration
- Project dependency and tool configuration is centralized in `pyproject.toml`.
- Benchmark behavior is configured through YAML files in `configs/benchmark/*.yaml`, plus dataset/method metadata in `configs/datasets/*.yaml` and `configs/methods/*.yaml`.
- Script-level environment controls include `PYTHON_BIN`, `VENV_DIR`, `INSTALL_EXTRAS`, `PYTHONUNBUFFERED`, `DATASET`, and `METHOD` (`scripts/setup_env.sh`, `scripts/run_cloud_comprehensive.sh`).
- Foundation-model authentication uses `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` when gated TabPFN weights are needed (`survarena/methods/foundation/readiness.py`, `README.md`).
- Packaging/build metadata and entry points are in `pyproject.toml` (`survarena = "survarena.cli:main"`).
- No container build descriptors detected (no `Dockerfile*` / `docker-compose*.yml` in repository root).
## Platform Requirements
- Local Python virtual environment (`.venv`) with editable install path expected (`scripts/setup_env.sh`, `requirements.txt`).
- Writable local filesystem for split cache and experiment artifacts in `data/splits/` and `results/` (`docs/environment.md`, `survarena/logging/export.py`).
- CPU-based tabular/deep ML dependencies installed in the active environment (`pyproject.toml`, `README.md`).
- CLI/batch execution target, not a long-running web service (`survarena/cli.py`, `survarena/run_benchmark.py`).
- “Cloud” runs are remote worker CLI jobs launched with the same Python module entrypoint (`scripts/run_cloud_comprehensive.sh`).
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Use snake_case module names in `survarena/` (for example `survarena/data/user_dataset.py`, `survarena/benchmark/tuning.py`).
- Use `test_<subject>.py` naming in `tests/` (for example `tests/test_compare_api.py`, `tests/test_statistics_strong.py`).
- Use snake_case function names (`compare_survival_models` in `survarena/api/compare.py`, `load_or_create_splits` in `survarena/data/splitters.py`).
- Prefix internal helpers with `_` for non-public scope (`_resolve_compare_methods` in `survarena/api/compare.py`).
- Use descriptive snake_case variables (`benchmark_cfg_hash`, `resolved_thresholds`, `method_cfg_cache`) in `survarena/api/compare.py` and `survarena/benchmark/runner.py`.
- Use `PascalCase` for dataclasses and core classes (`SurvivalPredictor`, `PredictorModelResult`, `SplitDefinition`, `RunManifest`).
## Code Style
- Style follows `ruff` settings in `pyproject.toml` (`line-length = 120`, `target-version = "py310"`).
- Use `from __future__ import annotations` consistently in runtime and test modules (`survarena/api/predictor.py`, `tests/test_metrics_and_tuning.py`).
- `ruff` is the configured linter in `pyproject.toml`.
- Code uses strict explicit exceptions and type hints in public function signatures (`survarena/cli.py`, `survarena/data/user_dataset.py`).
## Import Organization
- Not used; imports are package-qualified (`from survarena.benchmark.runner import run_benchmark`).
## Error Handling
- Validate arguments early and raise `ValueError`/`TypeError` with explicit messages (`survarena/api/predictor.py`, `survarena/data/splitters.py`).
- Catch broad exceptions at split-run boundary and downgrade to structured failed records (`survarena/benchmark/runner.py`).
## Logging
- Use `write_json` and `write_jsonl_gz` from `survarena/logging/tracker.py`.
- Store run status/metrics in `run_payload` objects from `survarena/benchmark/runner.py`.
## Comments
- Add short rationale comments only where behavior is non-obvious (for example scikit-survival target field note in `survarena/data/loaders.py`).
- Not applicable (Python codebase).
- Python docstrings are sparse; readability comes primarily from typed signatures and clear naming.
## Function Design
## Module Design
## Naming Patterns
- Use `snake_case.py` file names across source and tests (examples: `survarena/api/predictor.py`, `survarena/evaluation/statistics.py`, `tests/test_predictor_registry.py`).
- Prefix tests with `test_` and keep one responsibility area per file (examples: `tests/test_cli.py`, `tests/test_hpo_config.py`).
- Use `snake_case` for functions and methods, including private helpers prefixed with `_` (examples: `survarena/cli.py` `_parse_csv_list`, `survarena/benchmark/runner.py` `_autogluon_metadata`).
- Use verb-first names for behavior (`read_yaml`, `run_benchmark`, `compute_primary_metric_score`).
- Use `snake_case` for locals/attributes and add trailing underscore for fitted/runtime state on classes (examples in `survarena/api/predictor.py`: `dataset_`, `leaderboard_`, `best_model_`).
- Keep constants in `UPPER_SNAKE_CASE` at module scope (examples: `survarena/evaluation/statistics.py` `MAXIMIZE_METRICS`, `MINIMIZE_METRICS`).
- Use `PascalCase` for classes/dataclasses (examples: `survarena/data/schema.py` `SurvivalDataset`, `survarena/api/predictor.py` `PredictorModelResult`).
- Use builtin generics and PEP 604 unions (`dict[str, Any]`, `Path | None`) throughout `survarena/` and `tests/`.
## Code Style
- Use Ruff as the canonical style tool via `pyproject.toml` `[tool.ruff]`.
- Keep line length at 120 and Python target at 3.10+ (`pyproject.toml` sets `line-length = 120`, `target-version = "py310"`).
- Prefer explicit UTF-8 file I/O (`survarena/config.py`, `survarena/logging/tracker.py`, tests writing manifest files).
- Run `ruff check survarena tests scripts` as documented in `README.md`.
- Keep lint suppressions rare and scoped; existing suppressions are targeted (examples: `survarena/benchmark/runner.py` `# noqa: BLE001`, `# type: ignore[arg-type]`).
## Import Organization
- Not detected. Use absolute package imports rooted at `survarena` (example: `from survarena.data.schema import SurvivalDataset` in `survarena/api/predictor.py`).
## Error Handling
- Validate inputs early and raise `ValueError` with specific messages (`survarena/api/compare.py`, `survarena/automl/validation.py`, `survarena/data/user_dataset.py`).
- Raise `RuntimeError` for invalid method lifecycle calls (predict before fit) across model adapters (`survarena/methods/tree/rsf.py`, `survarena/methods/deep/deepsurv.py`).
- In benchmark execution, convert runtime failures into structured payload rows instead of crashing whole runs (`survarena/benchmark/runner.py` `evaluate_split`).
## Logging
- Use `print()` for CLI and progress output (`survarena/cli.py`, `survarena/run_benchmark.py`, `survarena/benchmark/runner.py`).
- Persist machine-readable JSON/JSONL/GZ artifacts via helpers in `survarena/logging/tracker.py`.
- Prefer explicit run manifests/metrics payloads over ad-hoc logs (`survarena/benchmark/runner.py`, `survarena/logging/manifest.py`).
## Comments
- Add concise comments only when algorithmic intent is not obvious (example: conservative Nemenyi approximation note in `survarena/evaluation/statistics.py`).
- Keep most code self-explanatory via naming and type hints; comment density is intentionally low across `survarena/`.
- Not applicable for this Python codebase.
- Python docstrings are minimal; rely on typed signatures and tests for behavioral specification (examples in `survarena/` and `tests/`).
## Function Design
## Module Design
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Keep entry points thin: parse flags and delegate (`survarena/cli.py`, `survarena/run_benchmark.py`).
- Centralize orchestration in service modules (`survarena/api/predictor.py`, `survarena/api/compare.py`, `survarena/benchmark/runner.py`).
- Resolve model backends indirectly by `method_id` via `survarena/methods/registry.py`.
## Layers
- Purpose: expose CLI and importable API contracts.
- Location: `survarena/cli.py`, `survarena/run_benchmark.py`, `survarena/__init__.py`, `survarena/api/__init__.py`
- Contains: argument parsing, command dispatch, public symbol exports.
- Depends on: `survarena.api`, `survarena.benchmark.runner`, `survarena.config`.
- Used by: local CLI runs and Python consumers.
- Purpose: execute fit/compare/benchmark workflows.
- Location: `survarena/api/predictor.py`, `survarena/api/compare.py`, `survarena/benchmark/runner.py`, `survarena/benchmark/tuning.py`
- Contains: validation planning, fold evaluation loops, HPO orchestration, retry/resume logic.
- Depends on: data, methods, evaluation, logging layers.
- Used by: interface layer only.
- Purpose: load datasets, normalize schema, preprocess features, and persist reusable splits.
- Location: `survarena/data/`
- Contains: dataset loaders (`loaders.py`), user ingestion (`user_dataset.py`), split manifesting (`splitters.py`), perturbation tracks (`robustness.py`), tabular preprocessing (`preprocess.py`).
- Depends on: `numpy`, `pandas`, `scikit-learn`, YAML.
- Used by: predictor and benchmark application services.
- Purpose: provide unified `fit/predict_risk/predict_survival` behavior over heterogeneous libraries.
- Location: `survarena/methods/`
- Contains: base contract (`base.py`), adapter families (`classical/`, `tree/`, `boosting/`, `deep/`, `foundation/`, `automl/`), and registry map (`registry.py`).
- Depends on: selected model libraries plus `survarena/methods/preprocessing.py`.
- Used by: `survarena/benchmark/runner.py` and `survarena/api/predictor.py`.
- Purpose: compute metrics/statistics and write benchmark outputs.
- Location: `survarena/evaluation/` and `survarena/logging/`
- Contains: metric bundles (`metrics.py`), statistical summaries (`statistics.py`), manifest/ledger/export writers (`manifest.py`, `tracker.py`, `export.py`).
- Depends on: SciPy/Pandas/NumPy and filesystem I/O.
- Used by: benchmark and compare workflows.
## Data Flow
- Runtime state is in-memory in orchestrator objects (notably `SurvivalPredictor` in `survarena/api/predictor.py`) and persisted to files for reproducibility.
## Key Abstractions
- Purpose: enforce shared adapter API.
- Examples: `survarena/methods/base.py`, implementations in `survarena/methods/classical/`, `survarena/methods/deep/`, `survarena/methods/foundation/`.
- Pattern: abstract base class + polymorphic adapters.
- Purpose: map `method_id` to adapter class.
- Examples: `_REGISTRY_TARGETS` and `get_method_class()` in `survarena/methods/registry.py`.
- Pattern: lazy import with memoization.
- Purpose: capture hash-linked run metadata and status.
- Examples: `RunManifest` in `survarena/logging/manifest.py` and payload serialization in `survarena/logging/tracker.py`.
- Pattern: append-only JSON/JSONL export contract.
## Entry Points
- Location: `survarena/cli.py` (registered in `pyproject.toml`).
- Triggers: `survarena fit`, `survarena compare`, `survarena foundation-check`.
- Responsibilities: parse arguments, call API, print JSON outputs.
- Location: `survarena/run_benchmark.py`.
- Triggers: `python -m survarena.run_benchmark`.
- Responsibilities: load benchmark config, run benchmark, support dry-run/resume/retries.
- Location: `survarena/__init__.py`.
- Triggers: `from survarena import SurvivalPredictor, compare_survival_models`.
- Responsibilities: stable import surface for external callers.
## Error Handling
- Input and config validation with explicit `ValueError`/`RuntimeError` in `survarena/api/compare.py`, `survarena/data/user_dataset.py`, and `survarena/api/predictor.py`.
- Exception capture and failure payload emission in `survarena/benchmark/runner.py` (`status = "failed"` records).
- Split-integrity and stratification checks before split reuse in `survarena/data/splitters.py`.
## Cross-Cutting Concerns
## Pattern Overview
- Keep entry points thin: parse arguments and delegate (`survarena/cli.py`, `survarena/run_benchmark.py`, `survarena/__init__.py`).
- Route all training/evaluation orchestration through service modules (`survarena/api/predictor.py`, `survarena/api/compare.py`, `survarena/benchmark/runner.py`).
- Add new model families via adapter + registry + config triad (`survarena/methods/base.py`, `survarena/methods/registry.py`, `configs/methods/*.yaml`).
## Layers
- Purpose: Parse commands and expose public API.
- Location: `survarena/cli.py`, `survarena/run_benchmark.py`, `survarena/__init__.py`, `survarena/api/__init__.py`
- Contains: CLI argument parsing, public API exports.
- Depends on: `survarena.api`, `survarena.benchmark.runner`, `survarena.config`.
- Used by: End users running `survarena` CLI or importing package APIs.
- Purpose: Coordinate predictor and benchmark use cases.
- Location: `survarena/api/predictor.py`, `survarena/api/compare.py`, `survarena/benchmark/runner.py`, `survarena/benchmark/tuning.py`
- Contains: Workflow orchestration, split-evaluate loops, model selection/HPO, retry/resume behavior.
- Depends on: data, methods, evaluation, logging, and config layers.
- Used by: CLI entry points and direct Python API calls.
- Purpose: Load datasets, validate schema, preprocess features, and create/reuse splits.
- Location: `survarena/data/`
- Contains: loaders (`loaders.py`), user data ingestion (`user_dataset.py`), split persistence (`splitters.py`), preprocessing (`preprocess.py`), robustness perturbations (`robustness.py`), typed schema (`schema.py`).
- Depends on: NumPy/Pandas/Scikit-learn and YAML.
- Used by: predictor, compare, and benchmark runner.
- Purpose: Provide a uniform fit/predict interface over heterogeneous survival backends.
- Location: `survarena/methods/`
- Contains: base interface (`base.py`), dynamic registry (`registry.py`), classical/tree/boosting/deep/foundation/autogluon adapters.
- Depends on: external ML libraries plus preprocessing utilities (`survarena/methods/preprocessing.py`).
- Used by: `survarena/api/predictor.py` and `survarena/benchmark/runner.py`.
- Purpose: Compute metrics/statistics and write benchmark artifacts.
- Location: `survarena/evaluation/`, `survarena/logging/`
- Contains: metric computation (`metrics.py`), statistical analysis (`statistics.py`), run manifests and exports (`manifest.py`, `tracker.py`, `export.py`).
- Depends on: NumPy/Pandas/SciPy/Torchsurv and filesystem I/O.
- Used by: compare and benchmark flows.
## Data Flow
- Runtime state is held in-memory within workflow objects (notably `SurvivalPredictor` in `survarena/api/predictor.py`) and persisted to filesystem artifacts for reproducibility.
## Key Abstractions
- Purpose: Standard adapter interface across all model families.
- Examples: `survarena/methods/base.py`, implemented in `survarena/methods/classical/`, `survarena/methods/tree/`, `survarena/methods/boosting/`, `survarena/methods/deep/`, `survarena/methods/foundation/`, `survarena/methods/automl/`.
- Pattern: Adapter + polymorphic dispatch through registry.
- Purpose: Resolve `method_id` to concrete adapter class.
- Examples: `survarena/methods/registry.py`.
- Pattern: Lazy dynamic import map with memoized resolution.
- Purpose: Decouple split construction from model fitting for holdout and bagged OOF modes.
- Examples: `ValidationPlan` and `ResampledFold` in `survarena/automl/validation.py`.
- Pattern: Precompute fold caches, then pass normalized fold payloads into tuning/selection.
- Purpose: Attach schema, environment snapshot, metrics, and failure metadata to every run.
- Examples: `survarena/logging/manifest.py`, `survarena/logging/export.py`, `survarena/logging/tracker.py`.
- Pattern: Structured event payloads serialized as JSON/JSONL.GZ.
## Entry Points
- Location: `survarena/cli.py` (registered in `pyproject.toml`).
- Triggers: `survarena fit`, `survarena compare`, `survarena foundation-check`.
- Responsibilities: Parse args, call API services, print JSON summaries.
- Location: `survarena/run_benchmark.py`.
- Triggers: `python -m survarena.run_benchmark`.
- Responsibilities: Load benchmark YAML, execute `run_benchmark`, support dry-run/resume/retry.
- Location: `survarena/__init__.py` re-exporting from `survarena/api/`.
- Triggers: `from survarena import SurvivalPredictor, compare_survival_models`.
- Responsibilities: Stable import surface for library consumers.
## Error Handling
- Defensive input/config checks with `ValueError` and `RuntimeError` in `survarena/api/compare.py`, `survarena/data/user_dataset.py`, and `survarena/benchmark/runner.py`.
- Failure capture with traceback and status payloads in `survarena/benchmark/runner.py` instead of aborting whole benchmark.
- Split integrity and stratification validation before reuse in `survarena/data/splitters.py`.
## Cross-Cutting Concerns
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, or `.github/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
