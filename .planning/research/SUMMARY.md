# Project Research Summary

**Project:** SurvArena
**Domain:** Practitioner-focused survival benchmarking platform (Python-first)
**Researched:** 2026-04-23
**Confidence:** HIGH

## Executive Summary

SurvArena should be built as a reproducible benchmark system for practitioner model selection, not as a dashboard product. The strongest pattern across stack, features, architecture, and pitfalls research is consistent: first establish deterministic execution contracts (splits, seeds, budgets, schemas), then layer statistical inference and ranking on top of stable fold-level results. Experts in this space prioritize validity and traceability over presentation, because benchmark trust collapses quickly when leakage, budget asymmetry, or artifact drift appears.

The recommended approach is a Python 3.12 baseline with a pinned scientific stack (`numpy`/`pandas`/`scipy`/`scikit-learn`/`scikit-survival`) and Optuna for budgeted HPO. Architecturally, keep clear boundaries: control surface -> deterministic runner -> method/metrics/statistics engines -> canonical artifact/export layer. Product scope for v1 should center on no-HPO vs HPO parity, multi-metric pairwise comparisons with correction, and a compact canonical artifact that supports manuscript-grade outputs.

The dominant risks are methodological, not implementation-only: leakage, unfair tuning budgets, multiplicity errors in pairwise claims, and overinterpretation of ELO. Mitigation is to enforce contracts in code and CI (split hashing, realized-budget tracking, correction metadata gates, ELO coverage/sensitivity checks, immutable provenance-rich artifacts). If these controls are implemented early, downstream roadmap and requirements can move quickly with low rework risk.

## Key Findings

### Recommended Stack

The stack research supports a modern but conservative baseline: Python 3.12 target, fully pinned package versions, and deterministic local execution as default. The strongest technical recommendation is to treat reproducibility settings as first-class configuration (sampler seeds, fixed split manifests, explicit budget caps, and schema versioning) rather than optional run metadata.

**Core technologies:**
- `Python 3.12.x` (3.13 secondary) for package compatibility and stable runtime behavior.
- `numpy 2.4.4` + `pandas 3.0.2` + `scipy 1.17.1` for data plumbing and bootstrap/statistical primitives.
- `scikit-learn 1.8.0` + `scikit-survival 0.27.0` for estimator/CV conventions and survival-native metrics.
- `optuna 4.8.0` for budgeted HPO with seedable, deterministic workflows.
- `pyarrow 24.0.0` + `joblib 1.5.3` for compact canonical artifacts and controlled local parallelism.

### Expected Features

Feature research is explicit that v1 must deliver trust-building benchmark mechanics before hosted UX. Table stakes include reproducible split governance, no-HPO/HPO parity, multi-metric evaluation, significance-aware pairwise comparison, and CLI/API parity. Differentiation is strongest where SurvArena combines TabArena-style global ranking with manuscript-grade statistical reporting and missing-metric/failure transparency.

**Must have (table stakes):**
- Dataset-balanced benchmark profiles (`smoke`, `research`, `manuscript`) with fixed runtime/split contracts.
- Reproducible split manifests and hash checks across reruns.
- No-HPO and HPO parity for each method under shared budget policies.
- Pairwise significance-aware comparison and corrected inference outputs.
- Canonical compact experiment artifact with complete lineage.

**Should have (competitive):**
- Pairwise battle matrix + ELO leaderboard with bootstrap confidence intervals.
- Manuscript-grade export bundle (critical difference, multiple-comparison summaries, rank summaries).
- Failure/missing-metric transparency integrated with quality metrics.

**Defer (v2+):**
- Hosted collaboration UI and scheduled benchmark service.
- Cross-language execution expansion (non-Python method ecosystems).
- Always-on auto-ensemble expansion across all methods.

### Architecture Approach

Architecture research strongly recommends a layered brownfield extension, with one deterministic orchestration spine and strict separation between metric computation and statistical inference. The system should produce fold-level normalized tables first, then derive pairwise/ELO/inference outputs in a second stage, and finally export to a canonical schema-versioned artifact.

**Major components:**
1. Control layer (`cli.py`, `run_benchmark.py`, `api/compare.py`) - validates contracts and exposes stable entrypoints.
2. Orchestration layer (`benchmark/runner.py`, `benchmark/tuning.py`) - executes deterministic run graph with split/seed/budget enforcement.
3. Method + evaluation layer (`methods/*`, `evaluation/metrics.py`, `evaluation/statistics.py`) - computes metrics and pairwise/ELO/correction logic.
4. Artifact layer (`logging/export.py`, manifests/trackers) - writes canonical outputs, indexes, and reproducibility metadata.

### Critical Pitfalls

1. **Dataset leakage through split/preprocessing violations** - enforce split-first, fold-local transforms, hash-validated split manifests, and leakage audits.
2. **Unfair HPO budgets** - codify one budget policy and log requested vs realized usage for every method/track.
3. **Multiple-comparison errors** - require predeclared correction policy and adjusted p-values/effect sizes in exports.
4. **ELO misuse** - treat ELO as secondary, require matchup coverage thresholds and sensitivity diagnostics.
5. **Reproducibility/artifact drift** - enforce immutable provenance-rich canonical artifacts and generated-only publication tables.

## Implications for Roadmap

Based on combined research, suggested phase structure:

### Phase 1: Protocol Contracts and Data Hygiene
**Rationale:** All later ranking/inference quality depends on leak-free aligned evaluation units.
**Delivers:** Split governance, manifest hashing, fold-local preprocessing enforcement, leakage audit checks.
**Addresses:** Reproducibility split governance, baseline profile integrity.
**Avoids:** Leakage pitfall and early invalid benchmark claims.

### Phase 2: Fair Execution and HPO Policy
**Rationale:** Fairness between methods/modes must be fixed before producing comparative claims.
**Delivers:** Shared budget policy, no-HPO/HPO parity enforcement, realized-budget logging and invalid-run handling.
**Uses:** Optuna deterministic settings and runner-level policy injection.
**Avoids:** Asymmetric-budget ranking distortions.

### Phase 3: Metrics and Inference Foundation
**Rationale:** Multi-metric fold outputs and corrected pairwise statistics are the decision core.
**Delivers:** Normalized fold/seed/leaderboard tables, multiplicity-corrected pairwise outputs, effect-size-aware summaries.
**Implements:** Two-stage evaluation pattern (metrics stage -> comparison stage).
**Avoids:** False-positive statistical narratives.

### Phase 4: Ranking Robustness Layer
**Rationale:** ELO adds interpretability only after pairwise/statistical base is stable.
**Delivers:** Full pairwise matchup matrix, deterministic ELO updates, bootstrap CI, coverage and sensitivity diagnostics.
**Addresses:** Practitioner global ranking needs and uncertainty communication.
**Avoids:** Schedule-dependent and overconfident leaderboard claims.

### Phase 5: Canonical Artifact and Provenance Hardening
**Rationale:** Requirements and roadmap both need a single source of truth to prevent output drift.
**Delivers:** Canonical comprehensive artifact (Parquet), schema versioning, content hashes, reproducibility replay scripts.
**Uses:** PyArrow/Parquet compaction and manifest contracts.
**Avoids:** Artifact redundancy and unreproducible manuscript tables.

### Phase 6: Productization and Extensions (v1.x+)
**Rationale:** Interface/scale enhancements should follow benchmark validity and storage contract stability.
**Delivers:** Decision-policy presets, robustness track expansion, artifact introspection tools; later hosted UI/service.
**Addresses:** Adoption and workflow acceleration after scientific core is stable.
**Avoids:** Premature UX scope that destabilizes core contracts.

### Phase Ordering Rationale

- Contract-first ordering reflects hard dependencies: split integrity -> budget parity -> statistics -> ranking -> artifact hardening.
- Architecture-aligned grouping reduces churn by isolating runner, statistics, and export responsibilities.
- Risk-heavy claims (pairwise significance and ELO) are delayed until prerequisite data quality gates are in place.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2:** Method-family-specific HPO budget normalization may need targeted policy calibration research.
- **Phase 4:** ELO robustness protocol design (coverage thresholds, K-factor policy, ordering sensitivity) needs additional validation.
- **Phase 6:** Hosted service architecture and operations patterns should be researched separately from core benchmark phases.

Phases with standard patterns (likely skip research-phase):
- **Phase 1:** Split governance and leakage controls are well-established in sklearn and benchmark practice.
- **Phase 3:** Corrected pairwise statistical workflows have mature references and existing implementation anchors.
- **Phase 5:** Canonical artifact/versioning/provenance patterns are mature in modern data engineering.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Mostly official package/docs evidence with explicit pinning and compatibility constraints. |
| Features | HIGH | Strong convergence between project protocol needs and benchmark literature patterns. |
| Architecture | HIGH | Directly grounded in current SurvArena module boundaries and proven benchmark design patterns. |
| Pitfalls | MEDIUM-HIGH | Risks are well-supported; exact mitigation thresholds (e.g., ELO gates) still require project-specific tuning. |

**Overall confidence:** HIGH

### Gaps to Address

- **ELO policy thresholds:** define publishable coverage/sensitivity gates during phase planning before requirements freeze.
- **Budget normalization edge cases:** validate per-method search-space harmonization criteria for fairness audits.
- **Canonical artifact migration path:** decide exact dual-write retirement criteria and downstream reader transition milestones.

## Sources

### Primary (HIGH confidence)
- Official package metadata/docs from `STACK.md` (PyPI + official docs for `scikit-survival`, `optuna`, `xgboost`, `catboost`, `scipy`, `statsmodels`, `pandas`).
- Internal code/protocol anchors from `ARCHITECTURE.md` (`docs/protocol.md`, `survarena/benchmark/runner.py`, `survarena/evaluation/statistics.py`, `survarena/logging/export.py`).
- Methodological references from `PITFALLS.md` (`scikit-learn` pitfalls guidance, Demsar/JMLR, Berrar, AMLB).

### Secondary (MEDIUM confidence)
- Comparative benchmark framing from `FEATURES.md` and TabArena public materials.
- ELO robustness literature as directional guidance for ranking diagnostics.

### Tertiary (LOW confidence)
- Ecosystem usage snapshots and directional repository signals cited in `STACK.md`; useful for trend context but not binding design evidence.

---
*Research completed: 2026-04-23*
*Ready for roadmap: yes*
