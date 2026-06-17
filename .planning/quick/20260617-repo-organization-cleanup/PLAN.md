---
status: in_progress
created: 2026-06-17
task: repo-organization-cleanup
---

# Repo Organization Cleanup

Address repository organization/ease-of-use improvements from items 3-7:

- Split oversized orchestration where behavior can be preserved safely.
- Add reproducibility/install guidance without changing dependency semantics.
- Reduce public-facing clutter from planning artifacts through documentation and ignore rules.
- Add local artifact cleanup ergonomics.
- Separate CLI parsing from command handlers where reviewable.
- Audit `results/` and consolidate/remove outputs that are not directly used for Elo comparison.

Validation:

- Run targeted tests for CLI/API/benchmark paths touched.
- Run lint on changed source/scripts/tests.
- Use code-review-graph for architecture and impact review.
