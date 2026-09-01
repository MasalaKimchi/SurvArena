---
phase: 01
slug: truthful-green-baseline
status: verified
threats_open: 0
asvs_level: 1
created: 2026-09-01
register_authored_at_plan_time: true
---

# Phase 1 — Security

> Phase-local verification of the evidence-integrity, provenance, execution-isolation, release-integrity, and supply-chain controls declared in the three Phase 1 plans.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|---|---|---|
| Benchmark execution → evidence artifacts | A successful fit is converted into predictions, metrics, manifests, and optional serialized models. | Model outputs, run identity, failure metadata |
| Historical cache → current protocol | Previously persisted split and result identities may be reused only when compatible. | Split indices, manifest fingerprints, HPO arm identity |
| Local verification → configured CI | Documented commands and automated workflow definitions must make the same bounded assurance claim. | Source, dependency specifications, test results |
| Transitional results → release claims | Historical matrices must not be promoted after behavior-changing implementation work. | Leaderboards, reports, manuscript evidence |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation and evidence | Status |
|---|---|---|---|---|---|
| T-01-01 | Evidence integrity | Artifact export | Mitigate | Predictions are written before optional model serialization; serialization failure is structured in the artifact manifest without erasing successful scientific evidence. Regression tests assert retained predictions, failure status, and run-level success. | closed |
| T-01-02 | Reproducibility | Parallel scheduler | Mitigate | Production execution retains `ProcessPoolExecutor`; tests replace that boundary with a synchronous recording fake and assert the process scheduler is selected. | closed |
| T-01-03 | Provenance | Resume/run identity | Mitigate | Completion keys and run IDs include `hpo_mode`; legacy arm-less rows complete `no_hpo` only. Tests cover both comparison arms and arm-qualified artifact identity. | closed |
| T-01-04 | Assurance | Static analysis gate | Mitigate | The exact 11-file/module mypy scope is declared in `pyproject.toml`, CI, README, and status docs; `follow_imports = "skip"` prevents an accidental whole-package claim. | closed |
| T-02-01 | Release integrity | Strict publication audit | Mitigate | The audit uses an internal renderer and subprocess tests assert a written report, blocker verdict, expected exit 2, and no traceback. | closed |
| T-02-02 | Provenance | Split cache compatibility | Mitigate | Reuse failure reports the manifest path and bounded missing/unexpected/changed fields; replacement requires explicit `--regenerate-splits`. Rejection is tested as non-mutating. | closed |
| T-02-03 | Log integrity | Markdown audit tables | Mitigate | Scalar rendering escapes pipes, backslashes, and newlines consistently; exact fixtures cover malformed-table inputs and missing values. | closed |
| T-02-04 | Compatibility | Legacy split manifests | Mitigate | Missing and added identity keys remain observable incompatibilities; explicit regeneration creates a new reusable manifest rather than normalizing a legacy manifest in place. | closed |
| T-03-01 | Assurance | CI and documentation | Mitigate | CI, README, and verification docs repeat the same Ruff, bounded mypy, pytest, and compileall commands and distinguish executed local evidence from configured hosted automation. | closed |
| T-03-02 | Scientific evidence | Semantic smoke workflow | Mitigate | The WHAS500/CoxPH smoke asserts successful rows, finite Uno C, dataset/method/arm identities, the compact artifact set, and absence of a redundant JSON leaderboard. A clean committed snapshot passed the same contract. | closed |
| T-03-03 | Publication integrity | Retained benchmark matrices | Mitigate | `PROJECT_STATE.md`, `docs/test_status.md`, and the strict audit explicitly invalidate pre-fix matrices and block publication claims until regeneration. | closed |
| T-03-04 | Supply chain | Development CI dependencies | Mitigate | Workflows install the declared development extra or the exact mypy version/ranged Ruff dependency from `pyproject.toml`. Full transitive locking, digest pinning, and independent reproduction remain explicitly assigned to Phase 6. | closed |

All plan-time threats have a verified phase-local mitigation. “Closed” does not imply release-wide security or reproducibility completion: the broader timeout/process-tree, exact fingerprint, strict protocol, conflict-safe store, action pinning, and environment-lock controls remain milestone requirements in Phases 3, 4, and 6.

---

## Accepted Risks Log

No accepted risks. Deferred milestone requirements are tracked work, not accepted release risks.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|---|---:|---:|---:|---|
| 2026-09-01 | 12 | 12 | 0 | Codex (inline `gsd-security-auditor` fallback) |

---

## Sign-Off

- [x] All threats have a disposition.
- [x] Accepted risks are documented (none).
- [x] `threats_open: 0` confirmed.
- [x] `status: verified` set in frontmatter.

**Approval:** verified 2026-09-01
