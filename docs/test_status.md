# Test and Verification Status

**Observed locally:** 2026-08-31

**Scope:** Phase 1 of `v2.0 Verified Benchmark Kernel`

**Environment:** project `.venv`, CPython 3.12, full declared scientific stack

This page distinguishes checks actually executed in the current workspace from automation that is only configured and benchmark evidence that remains invalidated.

## Locally Executed

| Gate | Command | Observed result |
|---|---|---|
| Lint | `.venv/bin/ruff check survarena tests scripts` | Passed |
| Incremental static types | `.venv/bin/python -m mypy survarena/core survarena/benchmark/resume.py survarena/data/splitters.py scripts/audit_manuscript_publishability.py` | No issues in 11 source files |
| Full suite | `.venv/bin/python -m pytest -q` | 256 passed, 6 skipped |
| Byte compilation | `.venv/bin/python -m compileall -q survarena` | Passed |
| Strict evidence audit | `.venv/bin/python scripts/audit_manuscript_publishability.py --strict` | Executed without traceback; expected `publishable=false`, exit 2 |

The six skips are optional foundation/integration cases with explicit availability guards. They are not broad skips added to make Phase 1 green.

The mypy gate is intentionally incremental, not whole-package coverage. Its exact scope is repeated in `pyproject.toml`, the README, and `.github/workflows/ci.yml`; `follow_imports = "skip"` prevents those targets from silently expanding into the untyped legacy dependency graph. Expansion belongs to later phases as typed contracts become authoritative.

## Semantic Benchmark Smoke

The Phase 1 smoke uses one WHAS500 seed with CoxPH in the no-HPO arm. Passing requires more than a zero process exit: at least one successful row, a finite Uno C value, dataset/method/arm identity columns, and the compact manifest/navigator/fold/leaderboard/diagnostics artifacts. The final observed output directory is temporary and is not retained as benchmark evidence.

## Configured but Not Independently Observed

- [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) defines Ruff, incremental mypy, light import, full pytest, and compileall jobs.
- [`.github/workflows/benchmark-smoke.yml`](../.github/workflows/benchmark-smoke.yml) defines the clean Linux WHAS500/CoxPH smoke and semantic assertions.
- No successful hosted GitHub Actions execution or clean Docker build is claimed as of this status date.

## Not Established by Phase 1

- Statistical validity of pairwise wins, ranks, Friedman/post-hoc tests, or cross-dataset uncertainty.
- Group-disjoint inner HPO/early-stopping validation and killable timeout/resource isolation.
- Live-runner use of typed protocol/capability contracts and the canonical SQLite collection.
- A frozen representative suite/roster, balanced HPO budgets, or leakage-free ensemble selection.
- Citable benchmark scores or manuscript conclusions. Historical result matrices require regeneration after Phases 2–5 stabilize behavior.

## Canonical Local Commands

```bash
.venv/bin/ruff check survarena tests scripts
.venv/bin/python -m mypy survarena/core survarena/benchmark/resume.py survarena/data/splitters.py scripts/audit_manuscript_publishability.py
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q survarena
.venv/bin/python scripts/audit_manuscript_publishability.py --strict
```

An audit exit code of 2 is currently an expected scientific/release blocker verdict. Any traceback, import error, or rendering crash is a verification failure.
