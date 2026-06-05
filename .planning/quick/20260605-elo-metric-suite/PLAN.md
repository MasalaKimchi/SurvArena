---
status: in_progress
created: 2026-06-05
---

# Quick Task: Elo metric suite

## Goal

Update manuscript Elo generation so the default path builds a suite of comparable metric-specific Elo artifacts rather than only Uno C, and remove raw `calibration_slope_50` from rankable/report metric defaults in favor of calibration absolute-error metrics.

## Scope

- Keep single-metric Elo generation available via `--metric`.
- Add multi-metric/default behavior and an index file.
- Exclude raw calibration slope/intercept values from comparable Elo discovery.
- Preserve raw calibration diagnostics in fold outputs where already produced.
- Add focused tests and run lint/test checks.
