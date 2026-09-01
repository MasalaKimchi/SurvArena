# SurvArena → The Living Survival Benchmark: Evaluation & Revamp Plan

*Target: become the single representative benchmark for right-censored tabular
survival analysis — the TabArena of survival — as a living, public, community-
extensible platform.*

Decisions locked with the maintainer:

- **Scope:** single-event right-censored, done definitively (no competing risks,
  time-varying covariates, or multi-state for v1). A tight, defensible scope is a
  feature, not a limitation — TabArena deliberately covered only
  classification + regression.
- **Rebuild style:** strangler-fig refactor. Preserve the strong evaluation /
  metrics / statistics core; aggressively restructure everything around it.
- **End goal:** a living public benchmark — public Elo leaderboard, a
  model-submission API, versioned dataset/protocol releases with a DOI, and a
  maintainer governance model.

---

## 0. Executive summary

SurvArena today is a rigorously engineered, honestly documented *manuscript
toolkit*. To become *the* survival benchmark it must cross three chasms that
TabArena crossed and SurvArena has not:

1. **From a repo that produces a paper → to a living platform.** TabArena is a
   continuously maintained system with a public leaderboard, curated frozen
   datasets, a model-submission protocol, and a maintainer team. SurvArena
   produces local CSVs and a one-off Elo bundle. This is the largest gap.
2. **From no-HPO evidence → to tuned-and-ensembled evidence.** TabArena's
   central scientific finding — that deep learning and foundation models only
   catch up to GBDTs *when extensively tuned and ensembled* — is only visible
   under a tuned + ensembled protocol. SurvArena's maintained arm is `no_hpo`,
   which structurally cannot surface that finding, and it has no ensemble
   entrant at all.
3. **From 17 narrow datasets → to a curated, versioned, domain-diverse suite.**
   TabArena curated 51 datasets from 1,053 with explicit exclusion policies
   spanning many domains and sizes. SurvArena has 7 tiny clinical datasets plus
   10 TCGA cohorts — two domains, almost no scale diversity, and download-on-the-
   fly provenance with no checksums or version pinning.

What we keep (it is genuinely strong): the IPCW metric implementations, the
nested-CV protocol, the leakage-free preprocessing discipline, the true
iterative Elo, the correct Wilcoxon / Nemenyi statistics, the resume/budget
engineering, and the audit culture (the `H*/M*/D*` annotations).

What we rebuild: the orchestration monolith, the artifact layer (CSV sprawl →
queryable results store), the model-adapter contract (positional 6-arg fit →
capability-typed interface — which also *structurally fixes* the fairness bug),
the config system (YAML sprawl → typed, versioned protocol), packaging, and CI.

What we add: a tuned + ensembled headline arm, an ensemble-selection entrant, a
proper-scoring-rule headline metric with D-calibration, a curated + versioned +
checksummed dataset suite, and the living-benchmark platform (leaderboard,
submissions, releases, governance).

**Net:** roughly a two-quarter program. The evaluation science is ~70% there;
the platform, data curation, and tuned/ensembled protocol are the real build.

---

## 0.5 Compute model — laptop-only reality (governs everything below)

**Hard constraint:** the maintainer has **one MacBook, no cloud budget, no HPC**.
Large experiments and large data cannot be processed locally. This does not sink
the living-benchmark goal, but it dictates *who computes what*. Three principles
govern every section below:

1. **The laptop is the orchestrator + curator, never the cluster.** The Mac
   designs the protocol, curates datasets, runs only the small core tier, drives
   CI, and validates submissions. It never produces the full at-scale matrix.
2. **Free CI is the batch worker — and the canonical environment.** GitHub
   Actions gives free linux/amd64 runners on public repos (≈2-core / 7 GB, 6 h
   per job). Sharded by (dataset × method × fold), CPU-friendly cells fit inside
   a job; a scheduled (cron) workflow grinds a few shards per run and commits
   result rows back to the store — a slow, free, reproducible batch cluster. It
   doubles as the **canonical env** (pinned amd64 container), so the Mac's arm64
   nondeterminism never touches citable numbers. *The Mac is dev-only; CI
   produces the official results.*
3. **The community brings the scale.** In a living benchmark, submitters run
   *their* model against the frozen suite on *their own* compute and submit a
   validated shard. You maintain the reference core + protocol + harness; the
   board grows on contributed compute.

The durable point: **the lasting contribution is the protocol, the
curated+versioned datasets, the harness, and the submission system — not a
heroic one-shot matrix you personally computed.** TabArena's ~25M runs came from
clusters; you will never match that solo, and you don't need to. You ship the
*standard* and a credible reference core; the numbers accrete.

Consequences baked into the plan:

- Datasets are **tiered by cost** (§5.2): Tier 0 laptop-core (you run it), Tier 1
  CI-grindable (free CI populates it over weeks), Tier 2 community-frontier
  (specified + checksummed, populated by contributors with compute). Big-data
  *preprocessing* also happens in CI, never on the Mac.
- Models default to **CPU-bounded** settings in Tier 0; heavy deep/foundation
  models are optional locally and run on CI/community for the full set. The
  **ensemble entrant stays cheap** (post-hoc greedy selection over *cached*
  validation predictions — no extra training), so it remains a headline result
  at zero marginal compute.
- Every run unit is **small, resumable, idempotent** — which SurvArena already
  supports (resume + split caching + process isolation). This is precisely what
  makes laptop + free-CI execution viable.
- Ambition is scoped honestly: **v1.0 ships with Tier 0 fully populated and Tier
  1 partially populated via CI.** The leaderboard is real and citable at that
  scope and grows over time. "Representative" is reached incrementally.

---

## 1. Scorecard: TabArena pillars vs SurvArena today

| Pillar (TabArena standard) | SurvArena today | Gap | Priority |
| --- | --- | --- | --- |
| **Living platform** — public leaderboard, maintainers, continuous updates | Local CSVs + one Elo bundle; manuscript-scoped | **Severe** | P0 |
| **Model-submission protocol** — standardized way for the community to add models | None; adapters are internal-only, edited in-tree | **Severe** | P0 |
| **Versioned, frozen datasets** — checksummed, curated, reproducible | Download-on-the-fly from pycox/sksurv; no checksums/versions | **High** | P0 |
| **Dataset diversity** — 51 datasets, many domains, many scales | 17 datasets, 2 domains (clinical + TCGA), all small | **High** | P1 |
| **Tuned + ensembled comparison** — the headline protocol | HPO exists but maintained arm is `no_hpo`; no ensemble entrant | **High** | P1 |
| **Standardized, well-tuned model implementations** — vetted search spaces | 33 real adapters (great breadth) but uneven HPO spaces | Medium | P1 |
| **One task-appropriate headline metric + Elo aggregation** | Multiple metric-specific Elo ladders; no declared primary | Medium | P1 |
| **Rigorous evaluation protocol** — nested CV, repetition, clustering | Strong: 5×3 nested CV, shared splits, dataset-clustered CIs | **Small (lead)** | maintain |
| **Correct metric/stat implementations** | Strong, but 2 residual bugs (holdout fairness, Elo CI) | Small | P0 (quick) |
| **Reproducibility infra** — containers, lockfiles, CI, 25M runs | Resume/budgets/caching good; no CI, no lockfile, no container; ~2.8k rows | Medium | P0/P1 |
| **Packaging health** | `datetime.UTC` breaks the advertised Py3.10 support; eager heavy imports | Small | P0 (quick) |
| **Scope** — clearly bounded | Single-event only, clearly bounded and documented | **None** | keep |

Reading: SurvArena *leads* TabArena on evaluation-science rigor and model breadth
for its scope, and *trails badly* on everything that makes a benchmark "living"
and "representative" — platform, curated/versioned data, and tuned+ensembled
evidence.

---

## 2. What's missing — deep gap analysis

### 2.1 Living-benchmark infrastructure (the defining gap)

A benchmark becomes "the" benchmark when other people can trust it, cite a
frozen version of it, and add their model to it without your involvement.
SurvArena has none of the machinery for that:

- **No public leaderboard.** Results live as per-run CSVs under `results/…`.
  There is no hosted, queryable, always-current ladder.
- **No results store.** Artifacts are scattered CSVs keyed by directory
  convention. There is no immutable, schema-versioned run-record database you can
  query ("give me Uno's C for method X across all datasets at protocol v1"),
  which is the substrate a leaderboard and a submission system both need.
- **No model-submission path.** Adding a model means editing `registry.py` and
  the config tree in-repo. There is no external plugin mechanism, no "run my
  model against the frozen suite and validate the outputs" runner, no shard
  merge-back.
- **No versioned releases.** `manuscript_v1.yaml` versions a *paper*, not a
  *benchmark protocol*. There is no `SurvBench-v1.0` = {frozen dataset list +
  checksums + protocol + seeds} that a model is evaluated *against* and that a
  citation can point to. No DOI, no changelog contract.
- **No governance.** No maintainer roster, no "how a dataset/model gets added or
  retired," no deprecation policy. `PROJECT_STATE.md` is close in spirit but is
  internal narrative, not public contract.

### 2.2 Datasets — curation, diversity, versioning

- **Too few, too narrow.** 7 clinical (SUPPORT, METABRIC, NWTCO, AIDS, GBSG2,
  FLCHAIN, WHAS500) + 10 TCGA (5 cohorts × {full, 1000-gene}). Only two domains.
  Real survival analysis spans **churn, reliability/predictive-maintenance,
  credit/lending, recidivism, customer lifetime, and large-registry
  epidemiology** — none represented.
- **No scale diversity.** Every clinical set is hundreds to ~2k rows. TabArena
  deliberately spans small→large; the interesting foundation-model-vs-GBDT
  crossover *is a function of dataset size*, so a size-diverse suite is
  scientifically necessary. SurvArena retired its one large set (KKBox).
- **Reproducibility risk in acquisition.** Clinical sets are pulled live from
  `pycox`/`sksurv`; a version bump upstream can silently change data. There are
  **no checksums, no frozen snapshots, no dataset version IDs**.
- **No survival-specific curation policy.** TabArena documents exclusion rules
  (leakage, duplicates, redundant features, ambiguous targets, bad splits).
  Survival needs *additional* rules: censoring-rate sanity bounds, adequacy of
  follow-up, event-definition clarity, informative-censoring red flags,
  competing-risks contamination (deaths from other causes mislabeled), and
  group/cluster structure disclosure.
- **Licensing under-documented** for the clinical suite (the contributor
  checklist mandates it; the dataset docs omit it).

### 2.3 Models — tuned+ensembled, missing families, brittle contract

- **The maintained evidence is `no_hpo`.** For a representative benchmark this is
  backwards: the headline must be *tuned + ensembled* (with a no-HPO arm as a
  secondary "default-hyperparameters" view). Without it you cannot reproduce
  TabArena's core finding for survival.
- **No ensemble entrant.** TabArena treats post-hoc ensembling (greedy
  Caruana-style selection over the model portfolio) as a first-class competitor,
  and it usually *wins*. SurvArena has AutoGluon-backed foundation adapters but
  no "ensemble of the survival portfolio" as an evaluated method. This is both a
  missing baseline and a missing scientific result.
- **Uneven HPO search spaces.** `hpo_config.py` exists but per-model search-space
  quality/coverage is inconsistent; TabArena invests heavily in vetted spaces.
- **Missing model families (still within single-event scope):**
  - Flexible parametric / Royston–Parmar spline models (`flexsurv`-style).
  - Modern neural survival: a transformer (SurvTRACE-style), a neural-ODE
    (SODEN-style), Nnet-survival / N-MTLR variants beyond what's present.
  - Trivial floors that every benchmark needs: **Kaplan–Meier marginal** (a
    covariate-free baseline) and a median/`always-censored` sanity floor.
  - A survival-native AutoML entrant *and* the ensemble entrant above.
- **Brittle adapter contract.** The positional 6-arg `fit(X, t, e, Xv, tv, ev)`
  with "methods that don't early-stop just drop the val args" is exactly what
  produces the fairness bug (§2.6). It should be a typed capability interface.

### 2.4 Evaluation — headline metric, proper scoring, protocol

- **No declared primary metric.** SurvArena exports many metric-specific Elo
  ladders but never says "*this* is the number." A representative benchmark must
  take a defensible stance. Recommendation (detailed in §4): a **proper scoring
  rule (IPCW Integrated Brier Score) as the headline** — the survival analog of
  TabArena's log-loss/RMSE, since it rewards the full predicted distribution —
  with **Uno's C as co-primary discrimination**.
- **Missing distributional calibration.** The suite has slope/intercept
  calibration but not **D-calibration** (Haider et al.), the standard test that a
  model's predicted survival *distribution* is calibrated. For a benchmark that
  claims to measure "how good is the survival curve," this is a required metric.
- **Missing right-censored log-likelihood** as a reported proper score
  (complements IBS; some models optimize it directly).
- **No runtime↔quality Pareto reporting.** TabArena foregrounds time budgets.
  SurvArena records fit/inference time but does not present the accuracy-vs-cost
  frontier that practitioners actually choose on.

### 2.5 Reproducibility & platform engineering

- **No CI.** `.github/workflows/` is empty. For a public benchmark, CI must run
  lint + tests + a smoke benchmark + leaderboard regeneration on every change.
- **No dependency lockfile / container.** `docs/environment.md` explicitly says
  don't treat `requirements.txt` as a lockfile. A living benchmark needs a
  pinned, hashed lock (uv/pip-tools) and a Docker image so a submitted model runs
  in a byte-reproducible environment.
- **Scale.** ~2,835 clinical + ~1,619 genomics rows vs TabArena's ~25M runs.
  You don't need 25M, but you need enough seeds×folds×datasets×models×(tuned)
  that Elo intervals are tight — and you need the sharding/orchestration to
  produce it repeatedly and cheaply.

### 2.6 Correctness carryovers (from the prior audit — fix first)

- **Cross-method fairness asymmetry.** The ~15% validation holdout is carved
  method-blind; early-stopping models exploit ~100% of train while
  non-early-stopping models (CoxPH, RSF, AFT, pooled foundation) silently train
  on 85%. Directional handicap against simpler models. *(runner.py:243–312)*
- **Elo bootstrap CI bug.** Duplicate-unit re-grouping inflates CI widths
  ~1.4–1.6×, and it resamples fold-level rather than dataset-clustered. Ratings
  are unaffected; only the CIs are wrong. *(evaluation/_ratings.py:141–172)*
- **Packaging.** `from datetime import UTC` (tuning.py:4) requires Python 3.11+
  but `pyproject.toml` says `>=3.10` and the README advertises 3.10 — a 3.10
  install passes the gate then crashes at import.

---

## 3. Target architecture (bold, strangler-fig)

The core idea: **freeze a small, trusted "kernel" (data spec, splits, metrics,
statistics, protocol) behind clean interfaces, then replace the sprawling
orchestration/artifact/config/model-contract layers around it.** Nothing in the
kernel gets rewritten from scratch; everything else does.

### 3.1 Package restructure

Split the flat `survarena/` package into four layers with a strict dependency
direction (`platform → bench → models → core`, never upward):

```text
survarena/
  core/            # THE PRESERVED KERNEL — no heavy ML deps, importable anywhere
    data/          #   DatasetSpec, loaders-as-registry, schema, feature roles
    splits/        #   nested-CV geometry, group-disjoint, cache fingerprinting
    metrics/       #   IPCW C, IBS, AUC, D-calibration, proper scores (torch-free where possible)
    stats/         #   Elo (fixed), Wilcoxon, Nemenyi, dataset-clustered bootstrap
    protocol/      #   ProtocolSpec (versioned): geometry, seeds, metric set, eligibility
    results/       #   RunResult schema + immutable results store (parquet/DuckDB)
  models/          # adapters behind a capability-typed interface, via entry-points
    base.py        #   SurvivalModel Protocol + ModelCapabilities
    classical/ tree/ boosting/ deep/ foundation/ ensemble/
  bench/           # orchestration: plan → run → resume → aggregate → export
    runner.py      #   thin; delegates fit/eval to core+models
    orchestrator/  #   process pool, sharding, budgets, checkpointing
  platform/        # the "living" layer
    leaderboard/   #   build a static site from the results store
    submit/        #   validate + run an external model against a frozen release
    release/       #   freeze/version a benchmark snapshot, checksums, DOI/changelog
  cli.py           # typer-based CLI over the above
```

Two immediate wins fall out of this: (1) `import survarena.core.metrics` no
longer drags in torch/autogluon (fixes the eager-import chain and the
`datetime.UTC` crash surfacing everywhere), and (2) external contributors depend
only on `survarena.core` + `survarena.models.base` to ship a model.

### 3.2 Capability-typed model interface (this fixes the fairness bug structurally)

Replace the positional 6-arg `fit` with an explicit contract. The runner then
uses declared capabilities — not method name guesses — to decide how to spend the
training data, so no model is silently handicapped.

```python
# survarena/models/base.py
@dataclass(frozen=True)
class ModelCapabilities:
    uses_validation: bool          # consumes a val fold at fit time
    supports_early_stopping: bool  # needs val to pick #iterations/epochs
    refit_on_full_train: bool      # after selection, refit on train∪val
    native_categoricals: bool
    supports_gpu: bool
    deterministic_given_seed: bool

class SurvivalModel(Protocol):
    capabilities: ClassVar[ModelCapabilities]
    def fit(self, train: SurvivalDataset, val: SurvivalDataset | None) -> None: ...
    def predict_survival(self, X, times) -> np.ndarray: ...  # S(t|x) on a grid
    def predict_risk(self, X) -> np.ndarray: ...             # scalar risk, higher = riskier
```

Runner policy driven by capabilities (pseudocode):

```python
if model.capabilities.uses_validation:
    sub, val = stratified_holdout(train, frac=0.15, seed=split.seed)
    model.fit(sub, val)
    if model.capabilities.refit_on_full_train and not needs_val_at_predict:
        model.fit(train, None)          # give the data back — no 15% penalty
else:
    model.fit(train, None)              # non-early-stoppers train on 100%
```

This makes the train/val policy an explicit, audited property of each model
instead of an accident of which positional args an adapter happens to ignore.

### 3.3 Results store replaces CSV sprawl

Define one immutable, schema-versioned record and persist it to a partitioned
Parquet dataset queried through DuckDB:

```python
@dataclass(frozen=True)
class RunResult:
    schema_version: int
    protocol_id: str            # e.g. "survbench-v1.0"
    dataset_id: str; dataset_version: str; dataset_sha256: str
    method_id: str; method_version: str; hpo_mode: str
    split_id: str; seed: int
    status: Literal["success","failed"]; ineligible_reason: str
    metrics: dict[str, float]   # uno_c, ibs, d_calibration_pvalue, ...
    fit_seconds: float; peak_rss_mb: float
    env_fingerprint: str        # lockfile hash + git sha + hardware class
```

Stored as `results/<protocol_id>/dataset=<id>/method=<id>/part-*.parquet`.
Benefits: one query language for leaderboard + reports + audits; trivial shard
merge (concat partitions); dedup by primary key; the env fingerprint makes every
number traceable to an exact environment. The existing resume/eligibility logic
ports onto this near-directly (it already keys on the same tuple).

### 3.4 Typed, versioned protocol (retire YAML sprawl)

Collapse the many benchmark YAMLs into **one validated, versioned ProtocolSpec**
(pydantic). A protocol version freezes the geometry, seeds, metric set, and
eligibility rules; datasets and methods are referenced by versioned IDs.

```python
class ProtocolSpec(BaseModel):
    protocol_id: str                    # "survbench-v1.0"
    outer_folds: int = 5; repeats: int = 3
    inner_folds: int = 3
    arms: list[Literal["tuned","default"]] = ["tuned","default"]
    hpo_budget_trials: int = 50
    primary_metric: str = "ibs"         # proper scoring rule (see §4)
    secondary_metrics: list[str] = ["uno_c","d_calibration","ibll","auc_t"]
    datasets: list[DatasetRef]          # each pinned to a version + sha256
    methods: list[MethodRef]
```

`manuscript_v1.yaml` becomes *a saved ProtocolSpec instance*, not a bespoke
schema. Retired configs are deleted (they already shouldn't be cited).

### 3.5 Data flow, end to end

```text
DatasetRegistry(version, sha256)  ─┐
                                   ├─► split cache (group-disjoint, fingerprinted)
ProtocolSpec(survbench-v1.0)  ─────┘        │
                                            ▼
                         per (dataset,method,split,seed,arm) unit
                                            │  fit via capability policy
                                            ▼
                         core.metrics ► RunResult ► Parquet/DuckDB store
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    ▼                        ▼                       ▼
             stats: Elo/CD/Wilcoxon   leaderboard site        release freeze + DOI
```

---

## 4. Evaluation protocol v2 — "single-event, done right"

### 4.1 Headline metric decision

TabArena picks exactly one metric per task type and aggregates by Elo. Survival's
literature is fragmented (Harrell's C vs Uno's C vs IBS vs time-AUC), so the
benchmark must *take a position and defend it*:

- **Primary (the Elo headline): IPCW Integrated Brier Score (IBS).** It is a
  strictly proper scoring rule over the predicted survival *distribution* — the
  survival analog of log-loss/RMSE — so it rewards both calibration and
  discrimination and cannot be gamed by rank-only tricks. Already implemented and
  audited in `core.metrics`.
- **Co-primary discrimination: Uno's C (IPCW).** Less censoring-biased than
  Harrell's; what clinicians read. Reported alongside, with its own Elo ladder.
- **Secondary suite:** integrated right-censored log-likelihood (proper score,
  complements IBS), **D-calibration** (distributional calibration — *new, must
  add*), time-dependent AUC at standardized horizons, and absolute-error
  calibration slope/intercept (already present).
- **Efficiency axis:** fit/inference seconds and peak memory → a runtime↔IBS
  Pareto front in the leaderboard.

Keep the per-metric Elo ladders SurvArena already produces, but *label one as
official* and drive the default leaderboard sort from it.

### 4.2 Protocol geometry

- Nested CV, **5 outer folds × 3 repeats** (keep), inner 3-fold for HPO;
  group-disjoint whenever `group_col` is declared (keep — this is a strength).
- **Dataset-specific repetition** (TabArena-style): allow small datasets more
  repeats to tighten intervals; encode per-dataset in the DatasetSpec.
- **Two arms, tuned is headline:** `tuned` (per-model vetted search space, fixed
  trial budget — reproducible, not wall-clock-dependent, which SurvArena already
  does well) and `default` (documented default hyperparameters, the current
  no-HPO view) as secondary.
- **Ensembling as a first-class entrant** (see §6.4), evaluated like any model.

### 4.3 Aggregation & statistics

- **Elo** (already true iterative Elo) as the primary aggregator — it sidesteps
  the incomplete-block problem that biases Friedman/Nemenyi when not every model
  runs on every dataset.
- **Fix the Elo bootstrap CI** (replicate value arrays instead of re-grouping;
  cluster by dataset) so intervals are valid.
- Keep dataset-clustered bootstrap for metric means; keep Wilcoxon + Nemenyi CD
  as secondary views with an explicit "complete-block subset only" guard.
- Add a **normalized-score** view (per-dataset min-max or vs-best normalization)
  as an interpretable companion to Elo, as TabArena reports.

---

## 5. Dataset suite v1 — curation policy + target list

### 5.1 Survival-specific curation policy (write this down, publicly)

Adopt TabArena's exclusions (leakage, duplicates, redundant/constant features,
ambiguous targets, bad splits) **plus** survival-specific gates:

1. Event indicator unambiguous; a single, clearly-defined event of interest.
2. Censoring rate within sane bounds (roughly 5–95%); extreme ends flagged.
3. Adequate follow-up (enough events after the last usable horizon for IPCW).
4. No obvious informative-censoring or competing-risks contamination (or it is
   disclosed and the dataset is quarantined to a "hard" tier).
5. Group/cluster structure disclosed via `group_col` (prevents subject leakage).
6. License permits redistribution *or* a reproducible fetch+checksum script is
   provided.

### 5.2 Target suite: broaden from 2 domains to ~6, span scales

Grow from 17 to ~30–40 curated tasks. Keep all current sets; add domain and
scale diversity (candidates — final list gated by the policy above):

| Domain | Candidates | Why |
| --- | --- | --- |
| Clinical (have) | SUPPORT, METABRIC, GBSG2, FLCHAIN, WHAS500, NWTCO, AIDS; add Rotterdam, colon, veteran, PBC, DLBCL | Core, but currently the *whole* suite |
| Genomics (have) | 5 TCGA cohorts × {full,1000} | Keep; high-dimensional regime |
| **Churn** | KKBox (revive — large), Telco churn | Large-n, business-relevant, size diversity |
| **Reliability / PdM** | NASA C-MAPSS (turbofan RUL), Backblaze drives | Non-clinical censoring, very large n |
| **Credit / lending** | Lending Club / Freddie Mac loan survival | Huge n, economic domain |
| **Recidivism / social** | Rossi recidivism, Broward | Classic survival, policy-relevant |
| **Registry epidemiology** | SEER subset (gated access), MIMIC-derived mortality | Very large n; the "does size flip the ranking" question |

Source aggregators to mine under the policy: **SurvSet** (70+ curated survival
datasets), **SurvBoard** (multi-omics), pycox/sksurv (current).

**Cost tiers (laptop-only reality, per §0.5).** The suite is partitioned by who
can afford to compute it, and only Tier 0 is maintainer-run:

- **Tier 0 — laptop-core (~8–12 sets):** all current small clinical sets + one or
  two mid-size (e.g. FLCHAIN, a churn set). Fully runnable on the Mac in
  hours; this is the smoke/iteration set and the first citable reference board.
- **Tier 1 — CI-grindable (~10–15 sets):** medium data, CPU models. *Not* run on
  the Mac — a scheduled Actions workflow grinds shards over weeks and commits
  results. Big-data preprocessing also lives here (download+clean+checksum in CI,
  publish a processed snapshot artifact), so the laptop never touches large data.
- **Tier 2 — community-frontier (open-ended):** large-registry / very-high-dim /
  heavy-model cells (SEER, KKBox at full size, foundation models on big data).
  Fully *specified and checksummed* but populated by contributors who bring
  compute. Ships as a documented open frontier, not a maintainer deliverable.

### 5.3 Versioning & provenance

- Each dataset → a `DatasetSpec` with `version`, `sha256`, `source_url`,
  `license`, `n`, `p`, `censoring_rate`, `domain`, `event_definition`,
  `group_col`, and a deterministic `fetch()` that verifies the checksum.
- Ship a **frozen snapshot manifest** per benchmark release; a dataset can only
  change under a new version, which forces a new protocol release.

---

## 6. Model portfolio v1 — standardized, tuned, ensembled

### 6.1 Keep the breadth (it's a lead over peers)

The 33 real adapters already exceed TabArena's model count and cover classical,
penalized, AFT, trees, boosting, deep/pycox, and foundation models — retain them,
ported onto the capability interface.

### 6.2 Add the missing pieces (within single-event scope)

- **Baseline floors:** Kaplan–Meier marginal (covariate-free) + a trivial
  reference, so every leaderboard has an honest floor.
- **Flexible parametric:** Royston–Parmar spline model.
- **Modern neural:** a transformer survival model (SurvTRACE-style) and a
  neural-ODE (SODEN-style) to represent the current DL frontier.
- **A survival AutoML entrant** and the ensemble entrant (§6.4).

### 6.3 Standardized, vetted HPO spaces

One reviewed search space per model, checked into `models/<family>/spaces.py`,
with fixed trial budgets (reproducible, machine-speed-independent — keep the
current n_trials approach). This is high-leverage, unglamorous work; it's what
makes "tuned" credible.

### 6.4 Ensembling as a first-class competitor (new headline result)

Implement greedy ensemble selection (Caruana) over the portfolio's *validation
survival predictions*, averaging survival functions (or hazards) rather than
labels. Evaluate it exactly like any other method. TabArena's ensemble usually
wins; establishing whether that holds for survival is a genuine scientific
contribution and a reason to cite *this* benchmark.

---

## 7. The living-benchmark platform

### 7.1 Public leaderboard

Generate a **static site** from the DuckDB results store (no server to babysit):
official IBS Elo ladder + per-metric ladders, per-dataset drill-down, the
runtime↔quality Pareto, coverage/eligibility tables, and CIs. Host on GitHub
Pages / a HuggingFace Space. Regenerated by CI on every merged results shard.

### 7.2 Model-submission protocol

The mechanism that makes it "living":

1. A contributor implements the `SurvivalModel` interface in a small package and
   declares it via a Python entry-point (no core edits).
2. `survarena submit --model my_pkg:MyModel --release survbench-v1.0` runs the
   **frozen** protocol on the **frozen, checksummed** datasets inside the pinned
   container, producing a results shard + env fingerprint.
3. CI validates outputs (schema, coverage, no-leakage assertions,
   determinism-under-seed check), then merges the shard and regenerates the
   leaderboard. Human maintainer approves.

### 7.3 Versioned releases, DOI, governance

- **`survbench-v1.0`** = frozen {dataset manifest + checksums + ProtocolSpec +
  seeds}. Models are evaluated *against a release*. Data/protocol changes → new
  release; a Zenodo DOI per release makes it citable.
- **Governance doc:** maintainer roster, how datasets/models are added/retired,
  deprecation policy, and a public changelog. Promote `PROJECT_STATE.md`'s
  culture into a public `GOVERNANCE.md` + `MAINTAINERS.md`.

### 7.4 CI/CD (fill the empty `.github/workflows/`)

- `lint-test`: ruff + pytest matrix (3.11, 3.12) on every PR.
- `smoke-bench`: one dataset × a few methods end-to-end, asserting metric ranges.
- `leaderboard`: regenerate + publish on merges to results.
- `release`: freeze manifest, checksum, tag, push DOI metadata.

---

## 8. Phased roadmap (strangler-fig sequencing)

Each phase ships something usable and keeps tests green; the old runner keeps
working until the new path supersedes it.

### Phase 0 — Stop the bleeding (≈1–2 weeks)
Fix what's cheap and blocking. **Exit:** clean install on 3.11/3.12, CI green,
known bugs closed, no evidence-behavior surprises.
- Fix packaging: `requires-python >=3.11` (or use `timezone.utc`); make
  `survarena/__init__` import lazily so core is importable without torch.
- Fix the Elo bootstrap CI (value-array replication + dataset clustering).
- Fix the holdout fairness gap *tactically* now (refit-on-full-train for
  non-early-stoppers) pending the capability interface in Phase 1.
- Stand up CI (lint + test + smoke). Add a hashed lockfile (uv) + Dockerfile.

### Phase 1 — Kernel + contracts (≈3–4 weeks)
Carve out `core/`, land the capability interface and results store. **Exit:** all
runs flow through the capability policy and write `RunResult` to Parquet; the
fairness fix is now structural; a DuckDB query reproduces the current leaderboard.
- Extract `core/{data,splits,metrics,stats,protocol,results}` behind interfaces.
- Implement `ModelCapabilities` + `SurvivalModel`; port all 33 adapters.
- Implement the Parquet/DuckDB results store; port resume/eligibility onto it.
- Replace YAML sprawl with `ProtocolSpec`; `manuscript_v1` → a saved instance.

### Phase 2 — Protocol v2 + tuned/ensembled (≈4–6 weeks)
Make the evidence representative. **Exit:** `survbench-v1.0` protocol runs the
tuned arm + ensemble entrant; IBS is the declared headline; D-calibration + IBLL
reported; regenerate the (now trustworthy) evidence matrices.
- Vetted per-model HPO spaces; tuned arm as headline, default arm secondary.
- Implement the greedy ensemble entrant.
- Add D-calibration + integrated right-censored log-likelihood metrics.
- Add the normalized-score aggregate; ship runtime↔quality Pareto.

### Phase 3 — Dataset & model expansion (open-ended, CI/community-paced)
Earn "representative" *incrementally* — this phase is bounded by compute you don't
have, so it is scoped by tier, not by a deadline. **Exit:** ≥30 datasets
*specified + versioned + checksummed* across ≥6 domains; **Tier 0 fully
populated locally, Tier 1 accumulating via scheduled CI, Tier 2 left as a
documented frontier**; new model families landed; curation policy published.
- Write + publish the survival curation policy; build the `DatasetRegistry` with
  checksums/versions and deterministic fetchers (fetch/preprocess for big sets
  runs *in CI*, never on the Mac).
- Onboard churn / reliability / credit / recidivism / registry datasets via
  SurvSet/SurvBoard + custom loaders, assigned to tiers by cost (KKBox/SEER →
  Tier 2, community-populated).
- Add KM floor + Royston–Parmar (both cheap, Tier 0); land transformer +
  neural-ODE models but run them only on CI/community tiers.

### Phase 4 — Go live (≈4–6 weeks)
Become "the" benchmark. **Exit:** public leaderboard live; a stranger can submit
a model via the documented protocol and land on the board; `survbench-v1.0`
tagged with a DOI; governance published.
- Static leaderboard site from the results store; publish via CI.
- Submission runner + validation CI + shard merge-back.
- Freeze + checksum + DOI `survbench-v1.0`; publish `GOVERNANCE.md` /
  `MAINTAINERS.md` / `CONTRIBUTING` for models and datasets.
- Delete the strangled legacy paths once the new pipeline owns all evidence.

---

## 9. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| **Laptop-only, no budget** → can't self-produce at-scale reference numbers | Ship the *standard*, not a heroic matrix (§0.5): Tier 0 run locally, Tier 1 ground out by free scheduled CI, Tier 2 populated by community submissions; leaderboard is honest about coverage and accretes over time |
| Free CI limits (6 h/job, 2-core, ephemeral) | Shard to sub-hour units; resume + idempotent writes commit partial progress back to the store; schedule cron runs; keep heavy models off CI (community tier) |
| arm64 (Mac) vs amd64 (CI) nondeterminism | Canonical numbers are produced only in the pinned amd64 container on CI; the Mac is dev-only and never the source of citable results |
| Tuned + expanded suite explodes compute cost | Fixed trial budgets (already the design); shard by (dataset×method); cache splits; cheap `default` arm for fast iteration; prioritize a "core" dataset tier for every-PR smoke |
| Refactor regresses the trusted evaluation core | Strangler-fig: kernel stays behind interfaces with golden-value tests; new path must reproduce current numbers before old path is deleted |
| Dataset licensing / access (SEER, KKBox, MIMIC) | Curation policy requires redistributable-or-fetch+checksum; gate restricted sets behind an access script; never vendor non-redistributable data |
| Submission runner is a security/repro surface | Run in the pinned container, no network at eval time, deterministic-seed check, schema + leakage assertions before merge; human approval |
| Scope creep back into competing risks | Explicit v1 non-goals doc; competing risks is a *future* protocol (v2), not a v1 slippage |
| Foundation adapters unstable / heavy | Keep behind extras + readiness checks (already present); bounded-CPU settings; mark ineligible cleanly rather than failing the board |

---

## 10. First two weeks (concrete)

1. Land Phase 0 packaging + import fixes; turn on CI (ruff + pytest + smoke), and
   make the repo public so GitHub Actions becomes your free amd64 batch worker.
2. Fix the Elo bootstrap CI and the holdout fairness gap; add regression tests
   that pin both behaviors.
3. Add a uv lockfile + Dockerfile; verify a clean containerized `benchmark run`.
4. Draft `core/`'s public interfaces (`SurvivalModel`, `ModelCapabilities`,
   `RunResult`, `ProtocolSpec`, `DatasetSpec`) as stubs + a one-page ADR each —
   this de-risks Phase 1 before any code moves.
5. Write the v1 non-goals + survival dataset-curation policy (public docs).
6. Pick the "core" dataset tier (≈6–8 sets spanning sizes) for the every-PR smoke
   benchmark and the first tuned+ensembled pilot.

---

## Appendix — what to preserve verbatim

These are audited and correct; port them unchanged behind the new interfaces:
IPCW Uno's C / Harrell's-on-full-set, IPCW IBS, link-scale calibration, true
iterative Elo (point estimates), symmetric paired Wilcoxon with pre-mirror
Holm/BH, per-k Nemenyi CD with one-rank-per-method-per-dataset, dataset-clustered
metric bootstrap, group-disjoint splitting with content fingerprinting, the
process-pool determinism model, resume integrity checks, and eligibility
filtering. The audit culture (`H*/M*/D*` annotations) should survive the
refactor — carry the annotations onto the new code.

