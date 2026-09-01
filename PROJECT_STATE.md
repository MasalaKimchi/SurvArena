# SurvArena Project State

**Status date:** 2026-09-01

**Software milestone:** `v2.0 Verified Benchmark Kernel`

**First citable protocol target:** `survbench-1.0-rc1`

**Roadmap:** [`.planning/ROADMAP.md`](.planning/ROADMAP.md)

SurvArena is a Python benchmark toolkit for single-event, right-censored tabular survival analysis. The v2 objective is one fair, statistically defensible, provenance-complete benchmark run whose compact canonical result collection can regenerate every comparison and report.

## Current Verified Behavior

The 2026-09-01 local development environment has executed the full package test suite, Ruff, an explicit incremental mypy scope, and package byte-compilation. A clean archive of the committed snapshot passed the same development gates and a five-fold WHAS500/CoxPH semantic smoke. The strict manuscript audit now executes without the optional `tabulate` dependency and returns an actionable blocker verdict rather than crashing. Split-cache mismatches remain fail-closed and report field differences, the manifest path, and the explicit `--regenerate-splits` recovery flag.

Phase 1 establishes code-health and semantic-smoke evidence only. It does not validate the manuscript's statistical conclusions or make historical result matrices citable.

## Evidence Validity

Retained result roots may demonstrate structural coverage, but all clinical/genomics no-HPO, HPO, foundation, Elo, table, and figure artifacts generated before the current behavior-changing fixes are **invalidated as current release evidence**. They must not support leaderboard or manuscript claims for `survbench-1.0-rc1`.

The invalidating changes include training/refit semantics, arm parity, group-aware splitting, split fingerprints, metric eligibility, calibration handling, ranking/significance logic, and failure filtering. Regeneration is deliberately deferred until the scientific comparison kernel, execution boundaries, typed protocol, and canonical result store are verified; regenerating earlier would produce another expensive transitional matrix.

Run the release audit with:

```bash
python scripts/audit_manuscript_publishability.py --strict
```

Exit code 2 is currently expected and means named release-evidence blockers remain. A traceback is a code failure.

## Remaining v2 Work

1. **Scientific comparison kernel:** exact matched-cell pairing, dataset-level inference, complete common support, metric reference/edge tests, and prediction eligibility.
2. **Execution and data safety:** killable process trees, separated budgets/RNG domains, group-safe inner validation, run-scoped telemetry, and adapter capability conformance.
3. **Authoritative protocol and results:** make strict typed recipes and one conflict-rejecting SQLite collection the live runner path.
4. **Representative protocol pilot:** freeze 12–18 curated tasks and a balanced 12–16 method roster, then validate default/tuned/ensemble semantics on sentinels.
5. **Release reproduction:** hashed dependency lock, digest-pinned Linux/amd64 image, clean wheel/container gates, regenerated evidence, and independent reproduction.

## What Is Not Yet Proven

- Hosted CI has been defined but no successful remote run is claimed here.
- The current Dockerfile and dependency resolution are not a locked canonical environment.
- Pairwise win rates and cross-dataset inference still require Phase 2 corrections.
- Typed `ProtocolSpec`/`ResultStore` infrastructure is not yet the authoritative execution path.
- No retained leaderboard, Elo rating, statistical table, or manuscript figure is current citable evidence.

See [the verification status](docs/test_status.md) for exact executed commands and [CI/reproducibility](docs/ci_and_reproducibility.md) for configured-versus-observed automation.
