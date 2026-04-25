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
# Technology Stack
## Languages
- Python 3.10+ - Core package, CLI entrypoints, benchmark pipeline, method adapters, and exports in `survarena/` (`pyproject.toml`, `survarena/cli.py`, `survarena/run_benchmark.py`).
- YAML - Config-driven benchmark, method, and dataset definitions in `configs/benchmark/`, `configs/methods/`, and `configs/datasets/`.
- Markdown - User/dev protocol and environment documentation in `README.md` and `docs/`.
- Shell (Bash) - Environment/bootstrap and benchmark validation wrappers in `scripts/setup_env.sh` and `scripts/validate_benchmark_protocol.sh`.
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
- Script-level environment controls include `PYTHON_BIN`, `VENV_DIR`, `INSTALL_EXTRAS`, and `PYTHONUNBUFFERED` (`scripts/setup_env.sh`, benchmark CLI commands).
- Foundation-model authentication uses `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` when gated TabPFN weights are needed (`survarena/methods/foundation/readiness.py`, `README.md`).
- Packaging/build metadata and entry points are in `pyproject.toml` (`survarena = "survarena.cli:main"`).
- No container build descriptors detected (no `Dockerfile*` / `docker-compose*.yml` in repository root).
## Platform Requirements
- Local Python virtual environment (`.venv`) with editable install path expected (`scripts/setup_env.sh`, `requirements.txt`).
- Writable local filesystem for split cache and experiment artifacts in `data/splits/` and `results/` (`docs/environment.md`, `survarena/logging/export.py`).
- CPU-based tabular/deep ML dependencies installed in the active environment (`pyproject.toml`, `README.md`).
- CLI/batch execution target, not a long-running web service (`survarena/cli.py`, `survarena/run_benchmark.py`).
- “Cloud” runs are remote worker CLI jobs launched with `python -m survarena.run_benchmark`.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

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
- Python docstrings are minimal; rely on typed signatures and tests for behavioral specification (examples in `survarena/` and `tests/`).
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

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
