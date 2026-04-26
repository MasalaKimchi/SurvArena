# Contributing: Method Adapters

SurvArena exposes a uniform adapter interface for heterogeneous survival modeling backends. Adding a new method is
intentionally config-driven and test-gated so benchmark runs remain comparable.

This guide documents the minimal contract for adding a new method adapter that can be used in:

- `SurvivalPredictor` (AutoML-style workflow)
- the benchmark runner (config-driven comparisons)

## Quick Checklist

Before opening a PR, confirm:

- [ ] adapter class implements the SurvArena method interface (`fit` + prediction surface)
- [ ] adapter handles right-censored labels (`time`, `event`) consistently with existing methods
- [ ] optional dependencies are declared and handled gracefully (clear error if missing)
- [ ] a method config exists in `configs/methods/` with a stable `method_id`
- [ ] at least one test covers registration + a minimal fit/predict smoke on a tiny synthetic dataset
- [ ] docs and config notes indicate whether the method is manuscript-scope, optional, or experimental

## Where Things Live

- Adapter base interface: `survarena/methods/base.py`
- Registry (method id -> adapter): `survarena/methods/registry.py`
- Family implementations:
  - `survarena/methods/classical/`
  - `survarena/methods/tree/`
  - `survarena/methods/boosting/`
  - `survarena/methods/deep/`
  - `survarena/methods/automl/`
  - `survarena/methods/foundation/` (optional / experimental)
- Method metadata/config: `configs/methods/*.yaml`

## Adapter Contract (Minimum)

SurvArena adapters are expected to:

1. Validate inputs early and raise `ValueError` with actionable messages when user/config inputs are invalid.
2. Fit only on training data and retain any fitted state on `self` (use trailing underscore naming such as `model_`).
3. Expose prediction outputs in the expected format for SurvArena evaluation. Not all methods can produce all output
   types; if an output type is not supported, the adapter must either:
   - omit it in a structured way that the evaluator can treat as "missing", or
   - raise a clear exception that the benchmark runner can capture as a fold-level failure row.

Use existing adapters in the closest family as a template (for example, `survarena/methods/tree/rsf.py` for tree
ensembles, or `survarena/methods/classical/coxph.py` for classical models).

## Adding a New Method (Step-by-Step)

1) **Pick a stable `method_id`**

The `method_id` is the identifier used in benchmark configs and CLI flags. Keep it:

- short, lowercase, snake_case
- stable across releases (avoid renaming once published)

2) **Implement the adapter**

Add a new module under the appropriate family folder (for example: `survarena/methods/boosting/my_new_method.py`).

Common expectations:

- explicit handling of `random_state` / seed if the backend supports it
- deterministic behavior when possible
- consistent preprocessing integration with SurvArena's preprocessing utilities

3) **Register the method**

Update the method registry so the new `method_id` resolves to your adapter. Prefer the existing registry pattern (lazy
imports) and keep changes minimal.

4) **Add `configs/methods/<method_id>.yaml`**

Include:

- the `method_id`
- a human-readable model name
- a family label (classical/tree/boosting/deep/automl/foundation)
- any backend-specific parameters
- optional dependency notes (and whether the method is experimental)

If a method requires non-default optional dependencies, ensure SurvArena fails fast with a clear message when those
dependencies are missing.

5) **Add tests**

Add a focused test file under `tests/` that checks:

- registry resolution for the new `method_id`
- a tiny synthetic fit/predict round-trip (keep runtime low)

Avoid integration tests that require large datasets or long fits unless absolutely necessary.

## Optional Dependencies and Failure Modes

SurvArena supports a "best-effort" benchmark policy: one failing method should not crash an entire benchmark run.

If your method depends on an optional package:

- raise a clear error at import-time or construction-time with instructions to install the right extra, or
- implement a readiness check similar to other optional adapters.

If the method cannot produce a required evaluation output for the current benchmark config, prefer producing a missing
metric rather than crashing (unless the limitation should be treated as an explicit failure).

## Benchmark Inclusion

Method inclusion is controlled by benchmark configs under `configs/benchmark/`.

When adding a method:

- keep it out of manuscript configs by default unless it is stable and well-understood
- add it to smoke configs only if runtime is small and dependencies are available in CI/dev environments

