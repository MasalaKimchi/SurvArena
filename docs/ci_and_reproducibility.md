# CI and Reproducibility

**Status date:** 2026-09-01

**Target:** `survbench-1.0-rc1`

SurvArena separates fast development assurance, semantic benchmark smoke, and citable release reproduction. The first two are defined now; a locked, independently reproduced release environment is Phase 6 work.

## Defined Workflows

### Pull-request CI

Workflow: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)

| Job | Python | Contract |
|---|---:|---|
| `lint` | 3.11 | `ruff check survarena tests scripts` |
| `type` | 3.11 | Mypy 2.3.1 over the declared incremental kernel scope, without recursively claiming imported legacy modules |
| `import-smoke` | 3.10, 3.11, 3.12 | Lazy `import survarena` using a light dependency subset |
| `test` | 3.11, 3.12 | Editable `[dev]` install, full `pytest -q`, and `compileall` |

Foundation extras remain outside baseline CI. Their gated weights, authentication, runtime, and hardware needs require separate capability-specific jobs before release.

### Manual benchmark smoke

Workflow: [`.github/workflows/benchmark-smoke.yml`](../.github/workflows/benchmark-smoke.yml)

This manual workflow runs WHAS500/CoxPH with one seed and no external dataset download. It then asserts:

- exactly one fold-results artifact is discovered;
- required dataset, method, split, seed, arm, status, and Uno C columns exist;
- at least one row succeeded and its Uno C is finite;
- dataset/method/arm are WHAS500, CoxPH, and `no_hpo`;
- the compact fold, leaderboard, diagnostics, manifest, navigator, and README artifacts exist;
- a redundant JSON leaderboard was not emitted.

The workflow uploads the temporary smoke directory for diagnosis. It does not create citable evidence or merge results into the repository.

## Local Equivalence

The README repeats the exact Ruff, mypy, pytest, and compile commands. `scripts/validate_benchmark_protocol.sh` remains a local convenience wrapper for a focused benchmark, while the workflow contains the release-facing semantic assertions.

## Current Reproducibility Boundary

- Direct dependencies are pinned in `pyproject.toml`, but there is no fully resolved hashed transitive lock.
- `Dockerfile` and `.dockerignore` are scaffolding; no successful canonical image build is claimed here.
- Hosted CI definitions have been syntax-checked locally, and their committed source snapshot passed equivalent local gates plus the five-fold semantic smoke; no completed remote run is claimed.
- Developer-machine numbers are diagnostic only. Citable results must come from the future digest-pinned Linux/amd64 release environment.
- Historical benchmark matrices predate behavior-changing fixes and must not be mixed into the release collection.

## Phase 6 Release Gate

Before `survbench-1.0-rc1` can be cited, CI must build a non-editable wheel, resolve a hashed lock, build a digest-pinned OCI image, run semantic sentinel checks inside it, reproduce recipe/split/result identities twice within declared tolerances, regenerate all reports from one canonical collection, and independently reproduce at least one complete dataset-method matrix.

Until then, environment snapshots may help diagnosis but are not a substitute for the release lock and image digest:

```bash
python -m pip freeze --all > environment-freeze.txt
python -VV > python-version.txt
```
