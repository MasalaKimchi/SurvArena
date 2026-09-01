# Environment

Last reviewed against `pyproject.toml`, `uv.lock`, and setup scripts: 2026-09-01.

## Supported Python

- 3.11 preferred
- 3.10 and 3.12 supported

## Locked uv Setup

```bash
curl -LsSf https://astral.sh/uv/0.12.8/install.sh | sh
uv sync --locked
uv run --no-sync python scripts/check_environment.py
```

The repository requires uv 0.12.8, and `.python-version` selects CPython 3.11
by default. `uv sync --locked` refuses a stale `uv.lock` instead of resolving a
new environment. After that explicit sync, use `uv run --no-sync` so commands
cannot silently update either the environment or lock.

The default sync:

- creates the repo-local `.venv`
- installs SurvArena's pinned runtime dependencies
- installs the default `dev` dependency group containing Ruff, mypy, and pytest

Optional foundation extras remain explicit:

```bash
uv sync --locked --extra foundation
uv sync --locked --extra foundation-tabarena
uv sync --locked --extra foundation-tabpfn
uv sync --locked --extra foundation-mitra
uv run --no-sync python scripts/check_environment.py --include-foundation
uv run --no-sync survarena foundation-check
```

Optional tracking support uses the same lock:

```bash
uv sync --locked --extra tracking
```

Always run benchmark commands through the explicitly synced environment:

```bash
uv run --no-sync python -c "import sys; print(sys.executable)"
uv run --no-sync survarena benchmark run --config configs/benchmark/manuscript_v1.yaml --dry-run
```

## Pip Compatibility Setup

The setup script and manual pip install remain available for contributors who
cannot use uv. These paths resolve independently and are not equivalent to the
committed lock.

```bash
PYTHON_BIN=python3.11 ./scripts/setup_env.sh
source .venv/bin/activate
```

Manual equivalent:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/check_environment.py
```

Foundation extras can also be appended to that compatibility install:

```bash
python -m pip install -e ".[dev,foundation-tabpfn,foundation-tabarena]"
python -m pip install -e ".[foundation]"
```

## Reproducibility Notes

`pyproject.toml` is the source of truth for dependency constraints and optional
extras. `uv.lock` is the generated, cross-platform resolution used by developer
machines and CI. Confirm that metadata and lock still agree before committing:

```bash
uv lock --check
```

For an intentional single-package update, first edit that package's intended
constraint in `pyproject.toml`—especially when it is exactly pinned—then update
and review the resolution:

```bash
uv lock --upgrade-package NAME
git diff -- uv.lock
uv lock --check
```

`requirements.txt` is only a pip compatibility wrapper and is not a lockfile.
An environment freeze can help diagnose a compatibility-path run:

```bash
python -m pip freeze --all > results/<bundle>/environment-freeze.txt
python -VV > results/<bundle>/python-version.txt
```

Neither the cross-platform developer lock nor a local freeze is the canonical
manuscript-release environment. Citable benchmark evidence still requires the
separately governed Phase 6 Linux/amd64 environment and release recipe.

Build the maintained Markdown documentation with the isolated docs group and
strict warnings:

```bash
uv sync --locked --only-group docs --python 3.11
uv run --no-sync sphinx-build -n -W --keep-going -b html docs docs/_build/html
```

## What the Check Covers

- virtual environment detection
- core imports such as `numpy`, `pandas`, `torch`, `torchsurv`, `autogluon.tabular`, `lifelines`, `sksurv`, `xgboost`, and `catboost`
- optional foundation imports such as `tabpfn` and `autogluon.tabular`
- runtime readiness messages for wired foundation adapters
- smoke tests for Harrell C-index, Uno C-index, integrated Brier score, and time-dependent AUC

## Smoke Checks

```bash
uv run --no-sync python -m compileall survarena
uv run --no-sync survarena benchmark run --config configs/benchmark/manuscript_v1.yaml --dry-run
uv run --no-sync survarena benchmark plan --config configs/benchmark/manuscript_v1.yaml
uv run --no-sync survarena benchmark doctor --config configs/benchmark/manuscript_v1.yaml --check-imports
```

Use `uv run --no-sync python scripts/check_environment.py --include-foundation`
and `uv run --no-sync survarena foundation-check` for optional foundation
dependency readiness.

End-to-end protocol spot-check (dry run plus a tiny fit and artifact checks;
see `docs/protocol.md`):

```bash
uv run --no-sync ./scripts/validate_benchmark_protocol.sh
```

Optional environment overrides: `BENCHMARK_CONFIG`, `WORK_DIR`, `PYTHON_BIN`.

## Output Locations

- splits: `data/splits/<task_id>/`
- predictor artifacts: `results/predictor/<dataset_name>/`
- benchmark runs: generated result folders or the explicit `--output-dir`

Benchmark experiment folders contain core CSV outputs plus the experiment
manifest.

Treat timestamped benchmark output folders as local run artifacts unless you are
intentionally publishing curated results.

To preview removable local caches and generated artifacts:

```bash
./scripts/clean_local_artifacts.sh
```
