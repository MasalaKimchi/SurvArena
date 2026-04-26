# Contributing: Datasets

SurvArena benchmarks are config-driven. Adding a dataset means:

1) defining a stable dataset id and metadata contract
2) implementing (or selecting) a loader
3) ensuring splits, preprocessing, and evaluation assumptions remain comparable

This guide covers built-in datasets under `configs/datasets/` as well as local-only datasets that cannot be redistributed.

## Quick Checklist

Before opening a PR, confirm:

- [ ] dataset has a stable id (lowercase, snake_case)
- [ ] `time_col` and `event_col` are clearly defined and validated
- [ ] event coding is explicit (what value means "event occurred")
- [ ] row/feature counts and censoring notes are recorded (even if approximate at config time)
- [ ] license/redistribution status is documented
- [ ] smoke/dry-run behavior is stable even when optional local data are unavailable
- [ ] a minimal test covers loader validation / missing-data behavior when applicable

## Where Things Live

- Dataset configs: `configs/datasets/*.yaml`
- Built-in loaders: `survarena/data/loaders.py`
- User dataset ingestion: `survarena/data/user_dataset.py`
- Split logic + persistence: `survarena/data/splitters.py`

## Required Metadata (Dataset Config)

Each dataset config should provide, at minimum:

- `dataset_id`: stable id used in benchmark configs
- `source`: where the dataset comes from (package name, URL, or "local")
- `time_col`: duration column name
- `event_col`: event indicator column name
- `event_value` (or equivalent): which value indicates an observed event (not censored)
- feature typing hints when available (numeric/categorical/etc.)
- notes covering:
  - censoring assumptions
  - known quirks (missingness, time units, label encoding)
  - whether the dataset is manuscript-scope, optional track, or local-only

If any of these are unknown at first, add a TODO and exclude from manuscript configs until resolved.

## Loader Expectations

Loaders should:

- return a consistent schema (features + `time_col` + `event_col`)
- validate presence of required columns
- validate event coding where possible
- raise `ValueError` with actionable messages when inputs/files are missing or malformed

Prefer using existing upstream loaders (for example from `scikit-survival` or `pycox`) when feasible, and wrap them with
SurvArena schema validation.

## Local-Only / Non-Redistributable Datasets

If a dataset cannot be redistributed (license, terms, privacy), treat it as **local-only**:

- configs may exist, but loaders must check for required local files and fail gracefully
- docs must state the expected file layout and how to obtain the data
- smoke configs and CI should not assume the dataset is present

This is the intended status for placeholder large-track datasets until a reproducible download path exists.

## Adding a Dataset (Step-by-Step)

1) **Add a config**

Create `configs/datasets/<dataset_id>.yaml` describing the metadata above.

2) **Implement or extend a loader**

If the dataset is available via an upstream Python package, add a loader entry using that package.

If it is local-only, implement a loader that:

- checks file existence
- validates columns/types
- provides a clear error describing the required local files and schema

3) **Add to benchmark configs (optional)**

Only include a new dataset in benchmark configs once:

- loader and schema are stable
- runtime cost is understood
- censoring/time unit semantics are documented

Start by adding it to a smoke config for a single method + seed, then broaden.

## Minimum Tests

Prefer small, fast tests. Suitable patterns:

- loader validation: missing required columns triggers `ValueError`
- missing local files: dataset is skipped or produces a clear error without crashing dry-run planning
- schema checks: returned dataset includes `time_col` + `event_col` and non-empty features

