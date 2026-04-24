# Testing Patterns

**Analysis Date:** 2026-04-23

## Test Framework

**Runner:**
- `pytest` (configured in `pyproject.toml` with `testpaths = ["tests"]`)
- Config: `pyproject.toml`

**Assertion Library:**
- Native `pytest` assertions plus `numpy.testing` and `pandas.testing` helpers in `tests/test_predictor_registry.py` and `tests/test_io_config.py`.

**Run Commands:**
```bash
pytest                                   # Run all tests
pytest -k predictor                      # Run subset by expression
pytest tests/test_compare_api.py         # Run a specific module
```

## Test File Organization

**Location:**
- Separate `tests/` directory (not co-located with implementation files).

**Naming:**
- `test_<area>.py` naming pattern (`tests/test_cli.py`, `tests/test_hpo_config.py`, `tests/test_robustness_tracks.py`).

**Structure:**
```text
tests/
├── conftest.py
├── test_cli.py
├── test_compare_api.py
├── test_predictor_registry.py
└── test_statistics.py
```

## Test Structure

**Suite Organization:**
```python
def test_compute_primary_metric_score_dispatches_to_harrell(monkeypatch) -> None:
    called: dict[str, np.ndarray] = {}
    def fake_harrell(**kwargs) -> float:
        called.update(kwargs)
        return 0.73
    monkeypatch.setattr("survarena.evaluation.metrics.compute_harrell_c_index", fake_harrell)
    score = compute_primary_metric_score(...)
    assert score == 0.73
```

**Patterns:**
- Setup pattern: mostly inline setup in each test function, with minimal shared bootstrap in `tests/conftest.py`.
- Teardown pattern: rely on pytest fixture scope and temporary directories rather than explicit teardown code.
- Assertion pattern: direct, specific assertions on output schema, values, and artifact existence.

## Mocking

**Framework:** `pytest` monkeypatching and lightweight fake classes.

**Patterns:**
```python
monkeypatch.setattr("survarena.evaluation.metrics.compute_uno_c_index", fake_uno)
with pytest.raises(ValueError, match="Unsupported primary metric"):
    compute_primary_metric_score(...)
```

**What to Mock:**
- External/slow runtime dependencies (AutoGluon backends, heavy model fit calls) as seen in `tests/test_cli.py` and `tests/test_autogluon_backend.py`.

**What NOT to Mock:**
- Pure transformation and stats helpers where deterministic real execution is cheap (`tests/test_statistics.py`, `tests/test_metrics_and_tuning.py`).

## Fixtures and Factories

**Test Data:**
```python
frame = pd.DataFrame({"time": [1.0, 2.0], "event": [1, 0], "age": [50, 60]})
dataset = load_user_dataset(frame, time_col="time", event_col="event")
```

**Location:**
- Simple inline data factories in each test module; shared path insertion fixture logic only in `tests/conftest.py`.

## Coverage

**Requirements:** None enforced in repository config (no coverage gate detected in `pyproject.toml`).

**View Coverage:**
```bash
pytest --cov=survarena --cov-report=term-missing
```

## Test Types

**Unit Tests:**
- Core focus. Examples: metric math and tuning helpers in `tests/test_metrics_and_tuning.py`, split behavior in `tests/test_robustness_tracks.py`.

**Integration Tests:**
- CLI/API wiring and artifact generation, for example `tests/test_compare_api.py`, `tests/test_cli.py`, `tests/test_predictor_edge_cases.py`.

**E2E Tests:**
- Not used as a dedicated browser/system framework; workflows are covered through Python-level integration tests.

## Common Patterns

**Async Testing:**
```python
Not applicable: current runtime APIs in `survarena/` are synchronous.
```

**Error Testing:**
```python
with pytest.raises(ValueError, match="num_bag_sets > 1 requires num_bag_folds >= 2"):
    predictor.fit(..., num_bag_folds=0, num_bag_sets=2)
```

---

*Testing analysis: 2026-04-23*
# Testing Patterns

**Analysis Date:** 2026-04-23

## Test Framework

**Runner:**
- `pytest` (declared in `pyproject.toml` optional `dev` dependencies).
- Config: `pyproject.toml` under `[tool.pytest.ini_options]` with `testpaths = ["tests"]`.

**Assertion Library:**
- Native `pytest` assertions with Python `assert`, plus `pytest.raises`.
- `numpy.testing` helpers for array assertions (`tests/test_validation.py`, `tests/test_metrics_and_tuning.py`).

**Run Commands:**
```bash
pytest                           # Run all tests
pytest -k predictor              # Focused subset (no dedicated watch mode configured)
pytest --cov=survarena           # Coverage (plugin-dependent; not configured in repo)
```

## Test File Organization

**Location:**
- Tests are centralized in `tests/` rather than colocated with implementation.
- Shared bootstrap fixture lives in `tests/conftest.py` (repo root insertion for imports).

**Naming:**
- Use `test_*.py` naming (examples: `tests/test_cli.py`, `tests/test_predictor_edge_cases.py`, `tests/test_robustness_tracks.py`).

**Structure:**
```text
tests/
├── conftest.py
├── test_cli.py
├── test_compare_api.py
├── test_predictor_registry.py
└── ... (feature-focused test modules)
```

## Test Structure

**Suite Organization:**
```python
def test_compare_survival_models_writes_benchmark_style_outputs(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame({...})
    monkeypatch.setattr("survarena.api.compare.evaluate_split", fake_evaluate_split)
    summary = compare_survival_models(frame, time_col="time", event_col="event", models=["coxph"], output_dir=tmp_path)
    assert summary["benchmark_id"] == "user_compare_fixed"
```

**Patterns:**
- Setup pattern: build inline toy `DataFrame`/`ndarray` fixtures in each test for readability (`tests/test_compare_api.py`, `tests/test_predictor_edge_cases.py`).
- Teardown pattern: rely on pytest fixture lifecycle (`tmp_path`, `monkeypatch`) instead of manual teardown.
- Assertion pattern: validate both outputs and side effects (artifact files, serialized manifest content, captured stdout).

## Mocking

**Framework:** `pytest` `monkeypatch` fixture (and occasional `pytest.MonkeyPatch()` manual context in `tests/test_presets.py`).

**Patterns:**
```python
monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
monkeypatch.setitem(__import__("sys").modules, "autogluon.tabular", fake_module)
```

**What to Mock:**
- External/optional dependencies and module imports (`tests/test_autogluon_backend.py`).
- Expensive training/search internals to isolate API behavior (`tests/test_predictor_registry.py`, `tests/test_predictor_edge_cases.py`).
- Split builders and benchmark runners for deterministic file-output tests (`tests/test_compare_api.py`).

**What NOT to Mock:**
- Dataframe/array transformation invariants and numeric behavior checks (use real in-memory data in `tests/test_validation.py`, `tests/test_metrics_and_tuning.py`).
- Serialization path existence checks where filesystem behavior is core (`tests/test_predictor_edge_cases.py`).

## Fixtures and Factories

**Test Data:**
```python
def _dataset(frame: pd.DataFrame, *, time: list[float], event: list[int]) -> SurvivalDataset:
    return SurvivalDataset(
        metadata=DatasetMetadata(dataset_id="toy", name="toy", source="unit_test"),
        X=frame.reset_index(drop=True),
        time=np.asarray(time, dtype=float),
        event=np.asarray(event, dtype=int),
    )
```

**Location:**
- Module-level helper factories are preferred inside each test module (`tests/test_validation.py`, `tests/test_robustness_tracks.py`).
- Global fixture usage is minimal; `tests/conftest.py` currently handles import path setup only.

## Coverage

**Requirements:** None enforced by repo config (no coverage threshold config detected in `pyproject.toml` or dedicated coverage config files).

**View Coverage:**
```bash
pytest --cov=survarena --cov-report=term-missing
```

## Test Types

**Unit Tests:**
- Primary test type; validates pure functions, metric computation, config parsing, validation logic (`tests/test_metrics_and_tuning.py`, `tests/test_hpo_config.py`, `tests/test_validation.py`).

**Integration Tests:**
- Lightweight file-system and pipeline integrations that validate generated artifacts and orchestration behavior (`tests/test_compare_api.py`, `tests/test_statistics.py`, `tests/test_cli.py`).

**E2E Tests:**
- Not used as a separate framework (no browser/system test harness detected).

## Common Patterns

**Async Testing:**
```python
# Not used in current suite; no async tests or pytest-asyncio markers detected.
def test_example_sync_only() -> None:
    assert True
```

**Error Testing:**
```python
with pytest.raises(ValueError, match="Unsupported primary metric"):
    compute_primary_metric_score(
        primary_metric="ibs",
        train_time=np.asarray([1.0, 2.0]),
        train_event=np.asarray([1, 0]),
        eval_time=np.asarray([1.5, 2.5]),
        eval_event=np.asarray([1, 0]),
        eval_risk_scores=np.asarray([0.2, 0.4]),
    )
```

---

*Testing analysis: 2026-04-23*
