---
status: complete
completed: 2026-06-14
---

# Manuscript Publishability Audit Summary

## Outcome

Added a reproducible publication-readiness gate for local manuscript artifacts:

- `scripts/audit_manuscript_publishability.py`
- `docs/manuscript_publishability.md`

The audit currently reports `publishable=false`.

## Findings

- Clinical no-HPO retained evidence is row-complete for the historical matrix: 2,835 / 2,835 rows.
- Clinical HPO retained evidence is missing: 0 / 2,835 rows.
- Genomics no-HPO retained evidence is partial: 1,260 / 2,025 rows.
- Clinical discrete-hazard foundation evidence is partial: 246 / 420 rows and 16 / 28 complete dataset-method pairs.
- Canonical discrete-hazard foundation evidence still needs either reruns under canonical IDs or a locked provenance bridge from alias-run artifacts.

## Documentation Updates

- Linked the audit from `README.md`.
- Added the audit to `docs/index.md`.
- Added publication-readiness guidance to `docs/manuscript_run_status.md`.
- Updated `PROJECT_STATE.md` so the top-level status does not imply publication readiness.

## Verification

- `python scripts/audit_manuscript_publishability.py`
- `python scripts/audit_manuscript_publishability.py --strict` exits 2 as expected while evidence is incomplete.
- `python -m ruff check survarena tests scripts configs`
- `python -m pytest -q`
- `python -m compileall survarena scripts`
- `./scripts/validate_benchmark_protocol.sh`
- `git diff --check`
- `code-review-graph` incremental update and minimal review
