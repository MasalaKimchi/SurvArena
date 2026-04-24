# Codebase Concerns

**Analysis Date:** 2026-04-23

## Tech Debt

**Large orchestration modules:**
- Issue: workflow-heavy functions mix validation, control flow, model lifecycle, and artifact persistence in single modules.
- Files: `survarena/api/predictor.py`, `survarena/benchmark/runner.py`
- Impact: harder reasoning, higher regression risk when adding new benchmark/predictor behavior.
- Fix approach: extract sub-workflows into dedicated service helpers (selection, refit, export, failure handling) with narrower unit tests.

**Dynamic config dictionaries without schema enforcement:**
- Issue: benchmark and method configs are read as loosely-typed dicts with distributed key checks.
- Files: `survarena/config.py`, `survarena/api/compare.py`, `survarena/benchmark/runner.py`, `survarena/benchmark/tuning.py`
- Impact: silent config drift risk and higher chance of runtime key errors in new profiles.
- Fix approach: introduce typed config validation models before execution.

## Known Bugs

**Large benchmark includes unsupported placeholder dataset:**
- Symptoms: comprehensive cloud benchmark can fail when attempting `kkbox` load.
- Files: `configs/benchmark/cloud_comprehensive_all_models_hpo.yaml`, `survarena/data/loaders.py`
- Trigger: running benchmark profiles that include `kkbox`.
- Workaround: remove `kkbox` from the dataset list or implement a real loader before running that profile.

## Security Considerations

**Untrusted pickle deserialization path:**
- Risk: loading arbitrary pickle files can execute malicious code.
- Files: `survarena/api/predictor.py`
- Current mitigation: caller-controlled local path only; no remote download path in code.
- Recommendations: document trusted-file-only requirement and optionally add a safer serialization mode for shared artifacts.

## Performance Bottlenecks

**Repeated preprocessing and model fitting across folds/tracks:**
- Problem: each split and robustness track repeats preprocessor fit and model fit from scratch.
- Files: `survarena/benchmark/runner.py`, `survarena/benchmark/tuning.py`, `survarena/data/robustness.py`
- Cause: correctness-first design prioritizes isolation over caching.
- Improvement path: cache reusable fold transforms and add optional parallel execution at dataset/method/split granularity.

## Fragile Areas

**Method registry and method config synchronization:**
- Files: `survarena/methods/registry.py`, `configs/methods/*.yaml`, `configs/benchmark/*.yaml`
- Why fragile: adding/removing method ids requires coordinated updates across registry, method YAML, and benchmark lists.
- Safe modification: add method adapter -> add registry entry -> add method config -> add targeted tests in `tests/test_method_adapters.py` and `tests/test_cli.py`.
- Test coverage: good baseline exists, but broad combinatorial matrix coverage remains partial.

## Scaling Limits

**Disk and runtime growth in comprehensive benchmarks:**
- Current capacity: per-run artifact strategy writes many CSV/JSON/JSONL files under `results/summary/exp_*/`.
- Limit: long cloud profiles with many methods/datasets/tracks can create very large output directories and long wall-clock runtime.
- Scaling path: add optional artifact pruning/compression tiers and checkpointed resumable execution by dataset/method chunks.

## Dependencies at Risk

**Optional heavy backends and environment variance:**
- Risk: foundation and AutoML dependencies are large and environment-sensitive.
- Impact: benchmark runs may fail due to missing optional packages or auth/runtime readiness.
- Migration plan: keep readiness checks in `survarena/methods/foundation/readiness.py` and harden preflight checks in `scripts/check_environment.py`.

## Missing Critical Features

**No automated CI workflow in repository:**
- Problem: `.github/workflows/` is empty, so tests/lint are not automatically enforced on branch changes.
- Blocks: consistent pre-merge quality gates and reproducible validation in shared collaboration.

## Test Coverage Gaps

**System-level benchmark execution path gaps:**
- What's not tested: full long-running benchmark execution matrix (multi-dataset + all robustness tracks + retries) as a single end-to-end run.
- Files: `survarena/benchmark/runner.py`, `survarena/logging/export.py`, `configs/benchmark/cloud_comprehensive_all_models_hpo.yaml`
- Risk: integration regressions may appear only in large profile runs.
- Priority: High

---

*Concerns audit: 2026-04-23*
# Codebase Concerns

**Analysis Date:** 2026-04-23

## Tech Debt

**Broad exception swallowing across runtime paths:**
- Issue: Multiple runtime-critical code paths catch `Exception` and either return defaults or continue, which hides root causes and weakens observability during failures.
- Files: `survarena/benchmark/runner.py`, `survarena/automl/autogluon_backend.py`, `survarena/methods/foundation/readiness.py`, `survarena/logging/tracker.py`, `survarena/utils/env.py`
- Impact: Real regressions can be recorded as partial success with sparse diagnostics, and remediation gets delayed because stack traces are not always surfaced where failures originate.
- Fix approach: Use narrower exception types, preserve causal error data in structured payloads, and fail-fast for non-recoverable setup/runtime errors.

**Duplicated benchmark orchestration logic:**
- Issue: Benchmark orchestration behavior is split across two similar flows (`compare_survival_models` and `run_benchmark`) with overlapping export and evaluation concerns.
- Files: `survarena/api/compare.py`, `survarena/benchmark/runner.py`, `survarena/run_benchmark.py`
- Impact: Feature drift risk is high (different defaults and capabilities by entry point), and maintenance requires synchronized edits in multiple orchestration paths.
- Fix approach: Extract a shared orchestration service that both CLI and API entry points call with explicit capability flags.

**Hardcoded optimization direction utility:**
- Issue: `_metric_direction_for_optimization()` always returns `"maximize"` regardless of metric input.
- Files: `survarena/benchmark/tuning.py`
- Impact: Future expansion to minimization metrics can silently produce incorrect HPO behavior if the helper is reused without correction.
- Fix approach: Encode explicit metric-direction mapping (matching `survarena/evaluation/statistics.py`) and validate unsupported metrics early.

## Known Bugs

**Benchmark command can report completion even when all runs fail:**
- Symptoms: The benchmark pipeline writes outputs and prints completion even when `evaluate_split` returns failed records for all combinations.
- Files: `survarena/benchmark/runner.py`, `survarena/run_benchmark.py`, `survarena/logging/export.py`
- Trigger: Any run where method setup/training consistently fails (for example missing optional dependencies or runtime incompatibility).
- Workaround: Inspect `*_failure_summary.csv` and `*_fold_results.csv` status columns manually before trusting benchmark completion.

## Security Considerations

**Unsafe deserialization boundary for predictor artifacts:**
- Risk: `pickle.load` executes code during deserialization and trusts file provenance.
- Files: `survarena/api/predictor.py`, `tests/test_predictor_edge_cases.py`
- Current mitigation: Type check after load (`isinstance`) and serialization version check in manifest.
- Recommendations: Treat predictor artifacts as trusted-only, add signed/hashed artifact verification before load, and provide a safer serialization format for untrusted environments.

**Environment token discovery without centralized secret policy enforcement:**
- Risk: Foundation auth readiness relies on ambient environment or local token cache checks rather than explicit secret source policy.
- Files: `survarena/methods/foundation/readiness.py`
- Current mitigation: Runtime readiness warnings and explicit guidance in error messages.
- Recommendations: Add configurable secret source policy and explicit "strict mode" to block accidental implicit credential usage.

## Performance Bottlenecks

**Quadratic pairwise statistics expansion:**
- Problem: Pairwise win-rate/significance/Elo computations iterate over method pairs and dataset groups with nested loops.
- Files: `survarena/evaluation/statistics.py`, `survarena/logging/export.py`
- Cause: O(methods^2) comparisons per benchmark-dataset group plus repeated dataframe merges and per-pair tests.
- Improvement path: Vectorize pair generation, cache merged frames, and gate expensive statistics behind profile flags for large method portfolios.

**Large in-memory accumulation during benchmark runs:**
- Problem: Full run payloads and records are accumulated in memory until all datasets/methods/splits finish.
- Files: `survarena/benchmark/runner.py`, `survarena/logging/export.py`
- Cause: `all_records`, `run_records`, and `hpo_trial_rows` are append-only lists flushed only at the end.
- Improvement path: Stream per-run writes (append mode) and periodically checkpoint summary artifacts to keep memory bounded.

**Bootstrap CI cost scales quickly with benchmark size:**
- Problem: Bootstrap confidence intervals use per-method repeated resampling with default `n_bootstrap=1000`.
- Files: `survarena/evaluation/statistics.py`, `survarena/logging/export.py`
- Cause: Loop-heavy mean-resample computation for every benchmark/method group.
- Improvement path: Make bootstrap iterations profile-aware, expose configurable defaults, and optionally parallelize grouped bootstrap work.

## Fragile Areas

**Strict split stratification tolerance for all datasets:**
- Files: `survarena/data/splitters.py`
- Why fragile: A fixed tolerance (`0.03`) can fail valid small or heavily imbalanced datasets due to sampling variance.
- Safe modification: Make tolerance adaptive to sample size/event prevalence and log diagnostic deltas before raising.
- Test coverage: Gaps; no direct tests exercise split manifest reuse/integrity and stratification validation edge cases.

**Foundation/runtime readiness path relies on optional dependency probing:**
- Files: `survarena/methods/foundation/readiness.py`, `survarena/methods/foundation/tabpfn_survival.py`, `survarena/methods/foundation/mitra_survival.py`
- Why fragile: Optional dependency checks and auth probing can vary by environment and produce state-dependent behavior.
- Safe modification: Keep error rewriting deterministic and add integration tests per missing-dependency/auth scenario.
- Test coverage: Partial; readiness catalog behavior is tested, but full end-to-end gated-model runtime paths remain environment-sensitive.

## Scaling Limits

**Single-process benchmark execution path:**
- Current capacity: One process iterating datasets × methods × splits × robustness tracks serially.
- Limit: Wall-clock grows linearly with run matrix and can become impractical for comprehensive portfolios.
- Scaling path: Introduce task-level parallel workers with deterministic seeding and synchronized artifact writes.

**Artifact volume growth under robustness and HPO:**
- Current capacity: Results exported as many CSV/JSON/JSONL files per run and per benchmark.
- Limit: Disk footprint and post-processing latency increase rapidly with high trial counts and robustness tracks.
- Scaling path: Add artifact retention policies, compression for large tabular outputs, and optional reduced-output profiles.

## Dependencies at Risk

**Pinned heavy ML stack with tight cross-library coupling:**
- Risk: Strict pins across `torch`, `torchsurv`, `autogluon.tabular`, `tabpfn`, `catboost`, and `xgboost` increase environment brittleness and upgrade friction.
- Impact: Installation/runtime breakage risk rises across OS/Python combinations and optional-feature matrices.
- Migration plan: Maintain a tested compatibility matrix and split optional stacks into profile-specific lockfiles/extras with CI coverage.

## Missing Critical Features

**No trust boundary for model artifact loading:**
- Problem: Artifact loading assumes local trust and does not enforce signature/hash provenance checks.
- Blocks: Secure deployment patterns where artifacts may move across systems or users.

**No benchmark failure policy gate:**
- Problem: End-to-end benchmark runs do not enforce a minimum success threshold before considering a run valid.
- Blocks: Reliable CI gating and automated publication workflows for benchmark outputs.

## Test Coverage Gaps

**Benchmark orchestration and retry/resume semantics:**
- What's not tested: Direct behavior of retry loops, resume keying, and failure-threshold policy in the real benchmark runner.
- Files: `survarena/benchmark/runner.py`, `survarena/run_benchmark.py`
- Risk: Silent regressions in long-running benchmark workflows and inconsistent resume outcomes.
- Priority: High

**Split persistence and integrity validation edges:**
- What's not tested: Manifest compatibility checks, split-file corruption handling, and adaptive behavior under class imbalance.
- Files: `survarena/data/splitters.py`
- Risk: Invalid split reuse or false-negative validation failures can invalidate experimental results.
- Priority: High

**Security behavior around predictor deserialization:**
- What's not tested: Rejection/tamper scenarios for malformed or untrusted predictor artifacts.
- Files: `survarena/api/predictor.py`, `tests/test_predictor_edge_cases.py`
- Risk: Unsafe artifact ingestion pathways remain unverified under adversarial inputs.
- Priority: High

---

*Concerns audit: 2026-04-23*
