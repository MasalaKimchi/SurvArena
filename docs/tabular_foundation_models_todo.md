# To-Do: Tabular Foundation Models for SurvArena

This roadmap outlines the work needed to support tabular foundation-model style
survival learning in SurvArena, in a way that feels comparable to modern
AutoGluon model portfolios.

## Current status

- Experimental `tabpfn` support is now available as `tabpfn_survival`.
- The current implementation uses TabPFN embeddings with a Cox survival head.
- This is a first integration point, not the final architecture.
- SurvArena now exposes a dedicated `foundation` preset plus a foundation-model
  catalog so users can explicitly inspect and request this portfolio family.

## Phase 1: Abstractions

- Define a `TabularEmbeddingBackbone` interface for pretrained tabular encoders.
- Define a `SurvivalHead` interface for Cox, discrete-time, and hazard-based
  prediction heads.
- Add a `FoundationSurvivalMethod` base class under `survarena/methods/foundation/`.
- Separate backbone freezing, partial fine-tuning, and full fine-tuning modes.
- Standardize how embeddings, risk scores, and survival curves are produced.

## Phase 2: Data and preprocessing

- Add feature-metadata objects that preserve numeric, categorical, ordinal,
  datetime, and text semantics.
- Add a preprocessing path that can emit both classic tabular matrices and
  backbone-specific encoded inputs.
- Preserve column order, feature typing, and missingness masks for pretrained
  backbones.
- Add dataset diagnostics that flag unsupported feature patterns early.

## Phase 3: Model adapters

- Implement adapters for candidate pretrained tabular backbones.
- Support frozen-embedding inference for cheap comparisons.
- Support fine-tuned backbone training for stronger final models.
- Add a fallback survival head that can train on extracted embeddings only.
- Define a common prediction contract so foundation models match
  `predict_risk(...)` and `predict_survival(...)`.

## Phase 4: Search and portfolio logic

- Extend presets so portfolio selection can include foundation models when the
  dataset size and feature profile make sense.
- Add heuristics for when to skip expensive backbones on very small datasets.
- Add tuning knobs for freeze strategy, head type, embedding dimension, and
  fine-tuning depth.
- Add resource-aware scheduling so expensive models run late in the search.

## Phase 5: Evaluation and artifacts

- Log backbone name, checkpoint identifier, fine-tuning mode, and parameter
  counts in the run manifest.
- Export embedding diagnostics and calibration summaries.
- Track inference latency and memory separately for backbone and survival head.
- Compare frozen vs fine-tuned variants in leaderboards.

## Phase 6: User-facing UX

- Expose a simple user flag such as `presets="best"` plus
  `enable_foundation_models=True`.
- Add a dedicated `foundation` preset for users who explicitly want those
  models considered.
- Explain when foundation models were skipped and why.
- Show backbone-specific results clearly in the leaderboard.

## Phase 7: Practical first implementation

- Start with one well-scoped foundation-model adapter rather than many.
- First support inference on extracted embeddings plus a Cox-style head.
- Then add selective fine-tuning and compare against classical/deep baselines.
- Keep the first release optional and clearly experimental.

## Key Principle

Foundation models should be integrated as **optional portfolio members** inside
the same high-level `SurvivalPredictor` workflow, not as a separate product that
forces users to learn a new interface.
