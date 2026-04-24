---
phase: 01-deterministic-execution-foundation
status: all_fixed
fix_scope: critical_warning
findings_in_scope: 2
fixed: 2
skipped: 0
iteration: 1
fixed_at: 2026-04-24T12:00:00Z
---

# Phase 01: Code review fix report

**Scope:** `critical_warning` (WR-01, WR-02)  
**Code commit:** `7be3070` — `fix(01): validate method IDs before YAML load; binary event fingerprint`

## WR-01 — Unknown `method_id` before YAML load

**Change:** After `registered_method_ids()`, build `unknown_methods` and `raise ValueError` before constructing `method_cfg_cache`. Removed redundant per-iteration registry check.  
**Tests:** `test_unknown_method_rejected_before_read_yaml` — asserts `read_yaml` is not called when the method is unknown.

## WR-02 — Event fingerprint collisions

**Change:** `_event_fingerprint` now requires `np.isin(..., [0, 1, False, True]).all()` and encodes with `uint8` instead of lossy `int8`.  
**Tests:** `test_event_fingerprint_rejects_non_binary_labels` — `ValueError` for `[0, 1, 2]`.

## Verification

- `python -m pytest` — 127 passed, 6 skipped (after this commit).

---

_Artifact: `/gsd-code-review-fix` single pass (no `--auto`)._
