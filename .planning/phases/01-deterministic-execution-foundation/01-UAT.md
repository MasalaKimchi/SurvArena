---
status: testing
phase: 01-deterministic-execution-foundation
source: 01-01-SUMMARY.md, 01-02-SUMMARY.md
started: 2026-04-23T20:00:00Z
updated: 2026-04-24T15:00:00Z
---

## Current Test
<!-- OVERWRITE each test - shows where we are -->

number: 3
name: Interrupted run can resume without redoing good completes
expected: |
  Stopping a benchmark mid-collection and re-invoking the same command skips work that already finished successfully (per integrity-valid success in the run ledger), while still allowing retries for failures or incomplete cells.
awaiting: user response

## Tests

### 1. Non-canonical benchmark profile fails before execution
expected: Invalid profile name fails at entry with a clear error; no model/dataset execution starts.
result: pass

### 2. Split manifest mismatch is a hard fail unless you opt in to regeneration
expected: If on-disk split manifest does not match the current split definition, the run stops with an explicit error. Adding the intended operator flag to regenerate splits (e.g. --regenerate-splits) allows a clean re-generation path instead of silent drift.
result: pass

### 3. Interrupted run can resume without redoing good completes
expected: Stopping a benchmark mid-collection and re-invoking the same command skips work that already finished successfully (per integrity-valid success in the run ledger), while still allowing retries for failures or incomplete cells.
result: [pending]

### 4. Run ledger exposes per-attempt status, retry count, and failure payload
expected: Exported run records include top-level or normalized fields for status, retry_attempt, and structured failure information on each attempt so you can audit retries and failures without losing history.
result: [pending]

## Summary

total: 4
passed: 2
issues: 0
pending: 2
skipped: 0

## Gaps

[none yet]
