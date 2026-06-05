---
status: complete
completed: 2026-06-05
---

# Docs Usability Pass Summary

Improved the documentation from a first-time user perspective:

- shortened `README.md` into a task-oriented front door
- added `docs/index.md` as a documentation routing map
- added `docs/methods.md` for method IDs and adapter families
- moved manuscript evidence and local machine notes into
  `docs/manuscript_evidence.md`
- updated environment and benchmark docs for current defaults and canonical
  `--config` examples
- kept detailed protocol/runtime/reference content in dedicated docs

Verification:

- scanned for stale `--benchmark-config` examples; remaining mentions are
  compatibility notes only
- ran `python -m compileall survarena`
