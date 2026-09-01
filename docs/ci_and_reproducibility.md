# CI and Reproducibility

**Status date:** 2026-09-01

**Target:** `survbench-1.0-rc1`

SurvArena uses the same committed `uv.lock` for contributor checks and GitHub
Actions. This developer/CI lock is intentionally distinct from the canonical
Linux/amd64 release environment governed by Phase 6.

## Pull-request CI

Workflow: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)

Pushes and pull requests targeting `main`, plus manual dispatches, run these
read-only jobs:

| Job | Python | Contract |
|---|---:|---|
| `quality` | 3.11 | Reject a stale lock, then run Ruff with GitHub annotations |
| `type` | 3.11 | Run mypy over the declared incremental kernel scope |
| `import-smoke` | 3.10, 3.11, 3.12 | Import `survarena` with only the six locked lightweight dependencies |
| `full-tests` | 3.11, 3.12 | Install the locked runtime and test group, run full pytest, then compile the package |
| `docs` | 3.11 | Build every tracked Markdown guide with nitpicky, fail-on-warning Sphinx |

All jobs have explicit timeouts, cancel superseded runs, and keep
`permissions: contents: read`. Checkout credentials are not persisted.
Official actions are pinned to audited commit SHAs rather than floating tags.
Foundation extras remain outside baseline pull-request CI because their gated
weights, authentication, runtime, and hardware needs require capability-specific
validation.

## Manual benchmark smoke

Workflow:
[`.github/workflows/benchmark-smoke.yml`](../.github/workflows/benchmark-smoke.yml)

This dispatch-only workflow runs WHAS500/CoxPH with one seed from the same lock.
It requires successful fold rows with finite Uno C, checks the compact artifact
set, rejects the redundant JSON leaderboard, and uploads the temporary smoke
directory for diagnosis. It does not create citable evidence or merge results
into the repository.

## Local Equivalents

Use the exact uv version required by `pyproject.toml`. Every install is locked,
and every subsequent command disables implicit synchronization.

Lock, Ruff, and scoped mypy:

```bash
uv lock --check
uv sync --locked --only-group quality --python 3.11
uv run --no-sync ruff check --output-format=github survarena tests scripts
uv run --no-sync python -m mypy \
  survarena/core \
  survarena/benchmark/resume.py \
  survarena/data/splitters.py \
  scripts/audit_manuscript_publishability.py
```

Lazy imports across the supported Python range:

```bash
for version in 3.10 3.11 3.12; do
  UV_PROJECT_ENVIRONMENT=".venv-import-${version}" \
    uv sync --locked --only-group import-smoke --python "${version}"
  UV_PROJECT_ENVIRONMENT=".venv-import-${version}" \
    uv run --no-sync python -c "import survarena"
done
```

Full tests and compile checks run on the preferred and maximum supported
versions. The lightweight import matrix separately retains the Python 3.10
minimum-version guard.

```bash
for version in 3.11 3.12; do
  UV_PROJECT_ENVIRONMENT=".venv-test-${version}" \
    uv sync --locked --no-default-groups --group test --python "${version}"
  UV_PROJECT_ENVIRONMENT=".venv-test-${version}" \
    uv run --no-sync python -m pytest -q
  UV_PROJECT_ENVIRONMENT=".venv-test-${version}" \
    uv run --no-sync python -m compileall -q survarena
done
```

Strict documentation:

```bash
UV_PROJECT_ENVIRONMENT=.venv-docs \
  uv sync --locked --only-group docs --python 3.11
UV_PROJECT_ENVIRONMENT=.venv-docs \
  uv run --no-sync sphinx-build -n -W --keep-going -b html docs docs/_build/html
```

## Lock and Cache Contract

`astral-sh/setup-uv` installs uv 0.12.8 and selects Python explicitly in every
job. Its cache key includes `uv.lock`; stable dependency-profile suffixes keep
unrelated environments separate:

| Profile | Jobs |
|---|---|
| `quality` | Lock/Ruff and scoped mypy |
| `docs` | Strict Sphinx |
| `import-smoke` | All lightweight import lanes |
| `full-tests` | Both full-test lanes |
| `benchmark-smoke` | Manual semantic benchmark |

`uv lock --check` rejects stale project metadata. `uv sync --locked` refuses
to re-resolve it, and `uv run --no-sync` prevents command execution from
changing the installed environment. CI caches package artifacts managed by uv;
it does not cache project virtual environments.

## Reproducibility Boundary

- `pyproject.toml` remains the source of dependency constraints and extras.
- `uv.lock` is the generated cross-platform developer/CI resolution.
- `requirements.txt` and `scripts/setup_env.sh` remain pip compatibility
  paths and are not lock-equivalent.
- Developer-machine and pull-request outputs are diagnostic only.
- Citable evidence requires the separately governed Phase 6 Linux/amd64 image,
  release lock, recipe, and independent reproduction.
- Historical matrices that predate behavior-changing fixes remain invalidated.

This repository change validates the workflow definitions and local equivalents;
it does not claim that a hosted GitHub Actions run occurred.

## Phase 6 Release Gate

Before `survbench-1.0-rc1` can be cited, the release process must build a
non-editable wheel, produce the canonical platform-specific lock and
digest-pinned image, run semantic sentinels inside it, reproduce
recipe/split/result identities within declared tolerances, and regenerate all
reports from one canonical result collection.
