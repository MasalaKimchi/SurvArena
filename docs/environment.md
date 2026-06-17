# Environment

Last reviewed against `pyproject.toml` and setup scripts: 2026-06-05.

## Supported Python

- 3.11 preferred
- 3.10 and 3.12 supported

## Quick Setup

```bash
PYTHON_BIN=python3.11 ./scripts/setup_env.sh
source .venv/bin/activate
```

The setup script:

- creates `.venv`
- installs SurvArena in editable mode with developer tooling and the default
  manuscript TabPFN and TabICL foundation dependencies
- runs `scripts/check_environment.py`

Useful overrides:

- `PYTHON_BIN=python3.11 ./scripts/setup_env.sh`
- `PYTHON_BIN=python3.10 ./scripts/setup_env.sh`
- `PYTHON_BIN=python3.12 ./scripts/setup_env.sh`
- `INSTALL_EXTRAS=dev ./scripts/setup_env.sh` for a core-only contributor environment
- `INSTALL_EXTRAS=dev,foundation ./scripts/setup_env.sh`
- `INSTALL_EXTRAS=dev,foundation-tabarena ./scripts/setup_env.sh`
- `INSTALL_EXTRAS=dev,foundation-tabpfn ./scripts/setup_env.sh`
- `INSTALL_EXTRAS=dev,foundation-mitra ./scripts/setup_env.sh`
- `VENV_DIR=.venv311 PYTHON_BIN=python3.11 ./scripts/setup_env.sh`

## Manual Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/check_environment.py
```

Optional foundation extras:

```bash
python -m pip install -e ".[dev,foundation-tabpfn,foundation-tabarena]"
python -m pip install -e ".[foundation]"
python scripts/check_environment.py --include-foundation
survarena foundation-check
```

Install only one foundation backend when isolating dependency issues:

```bash
python -m pip install -e ".[foundation-tabarena]"
python -m pip install -e ".[foundation-tabpfn]"
python -m pip install -e ".[foundation-mitra]"
```

Always run benchmark commands with the activated repo-local environment:

```bash
source .venv/bin/activate
python -c "import sys, tabicl; print(sys.executable); print(tabicl.__file__)"
survarena benchmark run --config configs/benchmark/manuscript_v1.yaml --dry-run
```

Optional tracking extras:

```bash
python -m pip install -e ".[tracking]"
```

## Reproducibility Notes

`pyproject.toml` is the source of truth for supported dependency pins and
optional extras. `requirements.txt` is only a convenience wrapper around the
default editable contributor install.

Do not treat `requirements.txt` as a lockfile. Before publishing or archiving a
benchmark bundle, write an environment freeze next to the retained artifacts:

```bash
python -m pip freeze --all > results/<bundle>/environment-freeze.txt
python -VV > results/<bundle>/python-version.txt
```

A committed constraints or lock file is useful once the project chooses one
exact platform and extras combination for manuscript reproduction.

## What the Check Covers

- virtual environment detection
- core imports such as `numpy`, `pandas`, `torch`, `torchsurv`, `autogluon.tabular`, `lifelines`, `sksurv`, `xgboost`, and `catboost`
- optional foundation imports such as `tabpfn` and `autogluon.tabular`
- runtime readiness messages for wired foundation adapters
- smoke tests for Harrell C-index, Uno C-index, integrated Brier score, and time-dependent AUC

## Smoke Checks

```bash
python -m compileall survarena
survarena benchmark run --config configs/benchmark/manuscript_v1.yaml --dry-run
survarena benchmark plan --config configs/benchmark/manuscript_v1.yaml
survarena benchmark doctor --config configs/benchmark/manuscript_v1.yaml --check-imports
```

Use `python scripts/check_environment.py --include-foundation` and
`survarena foundation-check` for optional foundation dependency readiness.

End-to-end protocol spot-check (dry run plus a tiny fit and artifact checks;
see `docs/protocol.md`):

```bash
./scripts/validate_benchmark_protocol.sh
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
