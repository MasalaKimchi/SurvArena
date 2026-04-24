# Technology Stack

**Analysis Date:** 2026-04-23

## Languages

**Primary:**
- Python (requires `>=3.10`) - all application logic in `survarena/`, tests in `tests/`, and tooling in `scripts/`.

**Secondary:**
- YAML - experiment and model configuration in `configs/benchmark/`, `configs/methods/`, and `configs/datasets/`.
- Markdown - project and protocol documentation in `README.md` and `docs/protocol.md`.
- Shell - operational scripts in `scripts/setup_env.sh`, `scripts/run_cloud_comprehensive.sh`, and `scripts/validate_benchmark_protocol.sh`.

## Runtime

**Environment:**
- CPython (preferred 3.11, supported 3.10/3.11/3.12) documented in `README.md` and `requirements.txt`.

**Package Manager:**
- `pip` with editable install from `pyproject.toml`.
- Lockfile: missing (`requirements.txt` installs `-e ".[dev]"`; no resolved lock artifact detected).

## Frameworks

**Core:**
- Tabular ML stack: `numpy`, `pandas`, `scipy`, and `scikit-learn` in `survarena/data/preprocess.py`, `survarena/data/splitters.py`, and `survarena/automl/validation.py`.
- Survival modeling stack: `scikit-survival`, `lifelines`, `pycox`, `torch`, and `torchsurv` in `survarena/methods/` and `survarena/evaluation/metrics.py`.

**Testing:**
- `pytest` configured in `pyproject.toml` (`[tool.pytest.ini_options]`) with test modules under `tests/`.

**Build/Dev:**
- `setuptools.build_meta` package build backend in `pyproject.toml`.
- `ruff` linting configured in `pyproject.toml` (`line-length = 120`, `target-version = "py310"`).

## Key Dependencies

**Critical:**
- `numpy==1.26.4` and `pandas==2.2.2` - dataframe and array core in `survarena/api/predictor.py` and `survarena/benchmark/runner.py`.
- `scikit-learn==1.6.1` - split and preprocessing primitives in `survarena/data/preprocess.py` and `survarena/data/splitters.py`.
- `torch==2.2.2` and `torchsurv==0.1.5` - deep model and metric runtime in `survarena/methods/deep/` and `survarena/evaluation/metrics.py`.
- `PyYAML==6.0.2` - YAML configuration loading in `survarena/config.py` and `survarena/data/loaders.py`.

**Infrastructure:**
- `autogluon.tabular==1.5.0` - AutoML backend in `survarena/automl/autogluon_backend.py`.
- `xgboost==3.2.0` and `catboost==1.2.10` - boosting adapters in `survarena/methods/boosting/tabular_boosting.py`.
- `psutil==5.9.8` - runtime telemetry in `survarena/logging/tracker.py` and `survarena/utils/env.py`.
- `matplotlib==3.9.2` and `seaborn==0.13.2` - plotting/report support from `survarena/api/predictor.py` and docs/examples.

## Configuration

**Environment:**
- Packaging, dependency pins, script entry points, pytest, and ruff are defined in `pyproject.toml`.
- Runtime benchmark and method controls are defined in `configs/benchmark/*.yaml` and `configs/methods/*.yaml`.
- Dataset metadata contracts are defined in `configs/datasets/*.yaml`.

**Build:**
- Build metadata uses `setuptools` via `pyproject.toml`.
- Console script entry point is `survarena = "survarena.cli:main"` in `pyproject.toml`.

## Platform Requirements

**Development:**
- Repo-local virtual environment workflow via `scripts/setup_env.sh` and verification via `scripts/check_environment.py`.
- Writable local filesystem for split cache and outputs in `data/splits/`, `results/predictor/`, and `results/summary/`.

**Production:**
- CLI/batch execution model using `survarena/cli.py` and `survarena/run_benchmark.py`.
- No service container, web server, or orchestrator contract is defined in repository code.

---

*Stack analysis: 2026-04-23*
# Technology Stack

**Analysis Date:** 2026-04-23

## Languages

**Primary:**
- Python 3.10+ - Core package, CLI entrypoints, benchmark pipeline, method adapters, and exports in `survarena/` (`pyproject.toml`, `survarena/cli.py`, `survarena/run_benchmark.py`).

**Secondary:**
- YAML - Config-driven benchmark, method, and dataset definitions in `configs/benchmark/`, `configs/methods/`, and `configs/datasets/`.
- Markdown - User/dev protocol and environment documentation in `README.md` and `docs/`.
- Shell (Bash) - Environment/bootstrap and benchmark execution wrappers in `scripts/setup_env.sh`, `scripts/run_cloud_comprehensive.sh`, and `scripts/validate_benchmark_protocol.sh`.

## Runtime

**Environment:**
- CPython runtime with 3.11 preferred and 3.10/3.12 supported (`README.md`, `docs/environment.md`, `scripts/setup_env.sh`).
- Package metadata requires Python >=3.10 (`pyproject.toml`).

**Package Manager:**
- `pip` (editable install workflow via `python -m pip install -e ...`) (`requirements.txt`, `scripts/setup_env.sh`).
- Build backend: `setuptools.build_meta` with `setuptools>=68` and `wheel` (`pyproject.toml`).
- Lockfile: missing (no lockfile detected at repository root).

## Frameworks

**Core:**
- NumPy 1.26.4 + pandas 2.2.2 + SciPy 1.13.1 - Data frame, arrays, and numeric routines (`pyproject.toml`, `survarena/data/*.py`).
- scikit-learn 1.6.1 - Preprocessing/model support utilities (`pyproject.toml`, `survarena/methods/preprocessing.py`).
- scikit-survival 0.24.1 + lifelines 0.30.0 - Classical survival estimators and utilities (`pyproject.toml`, `survarena/methods/classical/*.py`).
- pycox 0.3.0 + torchtuples 0.2.2 + torch 2.2.2 + torchsurv 0.1.5 - Deep survival models and metric tooling (`pyproject.toml`, `survarena/methods/deep/*.py`, `survarena/evaluation/metrics.py`).
- AutoGluon Tabular 1.5.0 - AutoML-backed survival adapter and Mitra foundation path (`pyproject.toml`, `survarena/automl/autogluon_backend.py`, `survarena/methods/foundation/mitra_survival.py`).

**Testing:**
- pytest >=8.3,<9 - Test runner configured under `[tool.pytest.ini_options]` in `pyproject.toml`, tests in `tests/`.

**Build/Dev:**
- Ruff >=0.15,<0.16 - Linting/style enforcement configured in `[tool.ruff]` in `pyproject.toml`.
- `python -m compileall survarena` and dry-run benchmark checks used as smoke-validation commands (`README.md`, `docs/environment.md`).

## Key Dependencies

**Critical:**
- `scikit-survival==0.24.1` - Built-in dataset loading and model ecosystem bridge (`survarena/data/loaders.py`, `survarena/methods/classical/*.py`).
- `pycox==0.3.0` - Deep survival adapters and built-in dataset access (`survarena/methods/deep/pycox_models.py`, `survarena/data/loaders.py`).
- `torchsurv==0.1.5` - Metric computation backbone for protocol outputs (`docs/protocol.md`, `survarena/evaluation/metrics.py`).
- `autogluon.tabular==1.5.0` - AutoGluon survival backend and optional foundation integration (`survarena/automl/autogluon_backend.py`, `survarena/methods/automl/autogluon_survival.py`).

**Infrastructure:**
- `PyYAML==6.0.2` - YAML config parsing (`survarena/config.py`, `survarena/data/loaders.py`).
- `psutil==5.9.8` - Process/hardware telemetry for manifests (`survarena/logging/tracker.py`, `survarena/utils/env.py`).
- `pyarrow==20.0.0` - Parquet input support (`survarena/data/io.py`).
- `xgboost==3.2.0` and `catboost==1.2.10` - Gradient-boosting survival adapters (`survarena/methods/boosting/tabular_boosting.py`).

## Configuration

**Environment:**
- Project dependency and tool configuration is centralized in `pyproject.toml`.
- Benchmark behavior is configured through YAML files in `configs/benchmark/*.yaml`, plus dataset/method metadata in `configs/datasets/*.yaml` and `configs/methods/*.yaml`.
- Script-level environment controls include `PYTHON_BIN`, `VENV_DIR`, `INSTALL_EXTRAS`, `PYTHONUNBUFFERED`, `DATASET`, and `METHOD` (`scripts/setup_env.sh`, `scripts/run_cloud_comprehensive.sh`).
- Foundation-model authentication uses `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` when gated TabPFN weights are needed (`survarena/methods/foundation/readiness.py`, `README.md`).

**Build:**
- Packaging/build metadata and entry points are in `pyproject.toml` (`survarena = "survarena.cli:main"`).
- No container build descriptors detected (no `Dockerfile*` / `docker-compose*.yml` in repository root).

## Platform Requirements

**Development:**
- Local Python virtual environment (`.venv`) with editable install path expected (`scripts/setup_env.sh`, `requirements.txt`).
- Writable local filesystem for split cache and experiment artifacts in `data/splits/` and `results/` (`docs/environment.md`, `survarena/logging/export.py`).
- CPU-based tabular/deep ML dependencies installed in the active environment (`pyproject.toml`, `README.md`).

**Production:**
- CLI/batch execution target, not a long-running web service (`survarena/cli.py`, `survarena/run_benchmark.py`).
- “Cloud” runs are remote worker CLI jobs launched with the same Python module entrypoint (`scripts/run_cloud_comprehensive.sh`).

---

*Stack analysis: 2026-04-23*
