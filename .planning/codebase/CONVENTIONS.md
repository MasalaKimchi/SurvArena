# Coding Conventions

**Analysis Date:** 2026-04-23

## Naming Patterns

**Files:**
- Use snake_case module names in `survarena/` (for example `survarena/data/user_dataset.py`, `survarena/benchmark/tuning.py`).
- Use `test_<subject>.py` naming in `tests/` (for example `tests/test_compare_api.py`, `tests/test_statistics.py`).

**Functions:**
- Use snake_case function names (`compare_survival_models` in `survarena/api/compare.py`, `load_or_create_splits` in `survarena/data/splitters.py`).
- Prefix internal helpers with `_` for non-public scope (`_resolve_compare_methods` in `survarena/api/compare.py`).

**Variables:**
- Use descriptive snake_case variables (`benchmark_cfg_hash`, `resolved_thresholds`, `method_cfg_cache`) in `survarena/api/compare.py` and `survarena/benchmark/runner.py`.

**Types:**
- Use `PascalCase` for dataclasses and core classes (`SurvivalPredictor`, `PredictorModelResult`, `SplitDefinition`, `RunManifest`).

## Code Style

**Formatting:**
- Style follows `ruff` settings in `pyproject.toml` (`line-length = 120`, `target-version = "py310"`).
- Use `from __future__ import annotations` consistently in runtime and test modules (`survarena/api/predictor.py`, `tests/test_metrics_and_tuning.py`).

**Linting:**
- `ruff` is the configured linter in `pyproject.toml`.
- Code uses strict explicit exceptions and type hints in public function signatures (`survarena/cli.py`, `survarena/data/user_dataset.py`).

## Import Organization

**Order:**
1. Future imports (`from __future__ import annotations`)
2. Standard library imports (`argparse`, `pathlib`, `typing`)
3. Third-party imports (`numpy`, `pandas`, `sklearn`, `torch`)
4. First-party imports (`from survarena...`)

**Path Aliases:**
- Not used; imports are package-qualified (`from survarena.benchmark.runner import run_benchmark`).

## Error Handling

**Patterns:**
- Validate arguments early and raise `ValueError`/`TypeError` with explicit messages (`survarena/api/predictor.py`, `survarena/data/splitters.py`).
- Catch broad exceptions at split-run boundary and downgrade to structured failed records (`survarena/benchmark/runner.py`).

## Logging

**Framework:** structured JSON/CSV exports via internal logging modules, not Python `logging`.

**Patterns:**
- Use `write_json` and `write_jsonl_gz` from `survarena/logging/tracker.py`.
- Store run status/metrics in `run_payload` objects from `survarena/benchmark/runner.py`.

## Comments

**When to Comment:**
- Add short rationale comments only where behavior is non-obvious (for example scikit-survival target field note in `survarena/data/loaders.py`).

**JSDoc/TSDoc:**
- Not applicable (Python codebase).
- Python docstrings are sparse; readability comes primarily from typed signatures and clear naming.

## Function Design

**Size:** orchestration functions are large but organized into helper calls (`SurvivalPredictor.fit` in `survarena/api/predictor.py`, `run_benchmark` in `survarena/benchmark/runner.py`).

**Parameters:** keyword-only arguments are preferred for public workflow APIs (`SurvivalPredictor.__init__`, `compare_survival_models`).

**Return Values:** return plain dict payloads and typed dataclasses for structured contracts (`fit_summary()`, `SplitDefinition`, `FoundationRuntimeStatus`).

## Module Design

**Exports:** explicit re-export via `__init__.py` modules (`survarena/__init__.py`, `survarena/api/__init__.py`).

**Barrel Files:** minimal barrels are used for package entry points; most modules are imported directly by concrete path.

---

*Convention analysis: 2026-04-23*
# Coding Conventions

**Analysis Date:** 2026-04-23

## Naming Patterns

**Files:**
- Use `snake_case.py` file names across source and tests (examples: `survarena/api/predictor.py`, `survarena/evaluation/statistics.py`, `tests/test_predictor_registry.py`).
- Prefix tests with `test_` and keep one responsibility area per file (examples: `tests/test_cli.py`, `tests/test_hpo_config.py`).

**Functions:**
- Use `snake_case` for functions and methods, including private helpers prefixed with `_` (examples: `survarena/cli.py` `_parse_csv_list`, `survarena/benchmark/runner.py` `_autogluon_metadata`).
- Use verb-first names for behavior (`read_yaml`, `run_benchmark`, `compute_primary_metric_score`).

**Variables:**
- Use `snake_case` for locals/attributes and add trailing underscore for fitted/runtime state on classes (examples in `survarena/api/predictor.py`: `dataset_`, `leaderboard_`, `best_model_`).
- Keep constants in `UPPER_SNAKE_CASE` at module scope (examples: `survarena/evaluation/statistics.py` `MAXIMIZE_METRICS`, `MINIMIZE_METRICS`).

**Types:**
- Use `PascalCase` for classes/dataclasses (examples: `survarena/data/schema.py` `SurvivalDataset`, `survarena/api/predictor.py` `PredictorModelResult`).
- Use builtin generics and PEP 604 unions (`dict[str, Any]`, `Path | None`) throughout `survarena/` and `tests/`.

## Code Style

**Formatting:**
- Use Ruff as the canonical style tool via `pyproject.toml` `[tool.ruff]`.
- Keep line length at 120 and Python target at 3.10+ (`pyproject.toml` sets `line-length = 120`, `target-version = "py310"`).
- Prefer explicit UTF-8 file I/O (`survarena/config.py`, `survarena/logging/tracker.py`, tests writing manifest files).

**Linting:**
- Run `ruff check survarena tests scripts` as documented in `README.md`.
- Keep lint suppressions rare and scoped; existing suppressions are targeted (examples: `survarena/benchmark/runner.py` `# noqa: BLE001`, `# type: ignore[arg-type]`).

## Import Organization

**Order:**
1. Standard library imports first.
2. Third-party imports second.
3. Local `survarena.*` imports last.

**Path Aliases:**
- Not detected. Use absolute package imports rooted at `survarena` (example: `from survarena.data.schema import SurvivalDataset` in `survarena/api/predictor.py`).

## Error Handling

**Patterns:**
- Validate inputs early and raise `ValueError` with specific messages (`survarena/api/compare.py`, `survarena/automl/validation.py`, `survarena/data/user_dataset.py`).
- Raise `RuntimeError` for invalid method lifecycle calls (predict before fit) across model adapters (`survarena/methods/tree/rsf.py`, `survarena/methods/deep/deepsurv.py`).
- In benchmark execution, convert runtime failures into structured payload rows instead of crashing whole runs (`survarena/benchmark/runner.py` `evaluate_split`).

## Logging

**Framework:** print + structured artifact writers

**Patterns:**
- Use `print()` for CLI and progress output (`survarena/cli.py`, `survarena/run_benchmark.py`, `survarena/benchmark/runner.py`).
- Persist machine-readable JSON/JSONL/GZ artifacts via helpers in `survarena/logging/tracker.py`.
- Prefer explicit run manifests/metrics payloads over ad-hoc logs (`survarena/benchmark/runner.py`, `survarena/logging/manifest.py`).

## Comments

**When to Comment:**
- Add concise comments only when algorithmic intent is not obvious (example: conservative Nemenyi approximation note in `survarena/evaluation/statistics.py`).
- Keep most code self-explanatory via naming and type hints; comment density is intentionally low across `survarena/`.

**JSDoc/TSDoc:**
- Not applicable for this Python codebase.
- Python docstrings are minimal; rely on typed signatures and tests for behavioral specification (examples in `survarena/` and `tests/`).

## Function Design

**Size:** No strict limit enforced; modules mix short utilities and orchestration functions. Keep orchestration in entry modules (`survarena/benchmark/runner.py`, `survarena/api/predictor.py`) and pure helpers in focused modules (`survarena/config.py`, `survarena/logging/tracker.py`).

**Parameters:** Prefer keyword-only arguments for complex APIs (`survarena/benchmark/runner.py` `evaluate_split`, `run_benchmark`; `survarena/api/compare.py` and predictor APIs).

**Return Values:** Return typed dictionaries/dataframes for serializable outputs and dataclasses for structured domain records (`survarena/evaluation/metrics.py`, `survarena/data/schema.py`, `survarena/logging/manifest.py`).

## Module Design

**Exports:** Keep package exports minimal and explicit in `__init__.py` files; place major entry points in `survarena/cli.py` and `survarena/run_benchmark.py`.

**Barrel Files:** Lightweight barrels are used only to expose package-level symbols (`survarena/api/__init__.py`, `survarena/methods/*/__init__.py`); core behavior remains in concrete modules.

---

*Convention analysis: 2026-04-23*
