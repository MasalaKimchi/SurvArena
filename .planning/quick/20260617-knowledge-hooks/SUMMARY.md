---
status: complete
---

# Quick Task 20260617: Knowledge Hooks

## Summary

Installed local Git hook automation to refresh code-review-graph knowledge and generated markdown wiki pages on commit and push events.

## Changes

- Added `scripts/update_code_knowledge.sh` as the shared hook runner.
- Added tracked hook wrappers in `scripts/git-hooks/post-commit` and `scripts/git-hooks/pre-push`.
- Configured this checkout with `core.hooksPath=scripts/git-hooks`.
- Generated the initial `.code-review-graph/wiki/` markdown pages.

## Validation

- `scripts/update_code_knowledge.sh manual`
- `bash -n scripts/update_code_knowledge.sh scripts/git-hooks/post-commit scripts/git-hooks/pre-push`
- `scripts/update_code_knowledge.sh pre-push </dev/null`
- `code-review-graph status --repo /Users/justin/Documents/SurvArena`
