# Stack Research

**Domain:** Practitioner-focused survival analysis benchmark platform (Python-only v1)
**Researched:** 2026-04-23
**Confidence:** HIGH (with explicit LOW-confidence notes where applicable)

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| Python | 3.12.x (target), 3.13.x (secondary) | Runtime baseline | Most core stack packages now require `>=3.11`; using 3.12 as default is the safest balance of maturity and performance while keeping future 3.13 compatibility open. | HIGH |
| NumPy | 2.4.4 | Numeric backbone | Required foundation for all survival/statistics packages; modern versions align with current SciPy/sklearn ecosystem in 2026. | HIGH |
| pandas | 3.0.2 | Dataset IO + tabular transformations | Stable and widely adopted for benchmark data plumbing; direct Parquet support supports compact artifact contracts. | HIGH |
| SciPy | 1.17.1 | Core statistical primitives | Includes robust bootstrap CI utilities (`scipy.stats.bootstrap`) needed for manuscript-grade uncertainty reporting. | HIGH |
| scikit-learn | 1.8.0 | Unified ML API + CV utilities | Still the de facto API layer for estimator interfaces, CV, splits, and metric workflows; minimizes custom orchestration code. | HIGH |
| scikit-survival | 0.27.0 | Primary survival modeling/evaluation package | Built on sklearn conventions and exposes survival-native metrics (IPCW C-index, IBS, dynamic AUC) plus scorer wrappers for tuning loops. | HIGH |
| Optuna | 4.8.0 | Budgeted HPO engine | Best fit for strict wall-clock budgets: TPE + pruning + seedable studies; deterministic behavior is straightforward in sequential mode. | HIGH |

### Supporting Libraries

| Library | Version | Purpose | When to Use | Confidence |
|---------|---------|---------|-------------|------------|
| lifelines | 0.30.3 | Classical survival baselines + diagnostics | Include for practitioner-trusted Cox/AFT baselines and interpretability-first comparisons. | HIGH |
| xgboost | 3.2.0 | Tree boosting survival models (`survival:cox`, `survival:aft`) | Use for high-performing nonlinear/tabular tracks where training speed and strong baselines matter. | HIGH |
| catboost | 1.2.10 | Categorical-friendly survival objectives (`Cox`, `SurvivalAft`) | Use when datasets have substantial categorical signal or minimal preprocessing tolerance. | HIGH |
| statsmodels | 0.14.6 | Multiple-testing correction + classical inference utilities | Use for pairwise significance pipelines with FWER/FDR control (`multipletests`) in manuscript outputs. | HIGH |
| scikit-posthocs | 0.12.0 | Post-hoc pairwise procedures | Use when benchmark protocol requires nonparametric post-hoc families beyond base SciPy/statsmodels primitives. | MEDIUM |
| pyarrow | 24.0.0 | Columnar artifact format + compression | Use as primary storage engine for compact, non-redundant experiment artifacts (Parquet + compression). | HIGH |
| joblib | 1.5.3 | Local parallel execution backend | Use for deterministic, low-overhead parallelism on a single host before introducing cluster complexity. | HIGH |
| openskill | 6.2.0 | ELO/skill-style ranking utilities | Use only if you want a maintained off-the-shelf skill rating implementation instead of custom deterministic Elo code. | MEDIUM |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| ruff (0.15.11) | Lint + format speed gate | Keep rule set strict on touched benchmark paths; fast feedback helps maintain quality in large experiment runs. |
| pytest (9.0.3) | Regression and protocol tests | Add contract tests for reproducibility (seed stability, split reuse, summary schema invariants). |
| mypy (1.20.2) | Static typing on core pipeline | Prioritize `survarena/benchmark`, `survarena/evaluation`, and export/statistics modules for safer refactors. |

## Installation

```bash
# Core runtime
uv pip install \
  "numpy==2.4.4" "pandas==3.0.2" "scipy==1.17.1" \
  "scikit-learn==1.8.0" "scikit-survival==0.27.0" \
  "optuna==4.8.0" "pyarrow==24.0.0" "joblib==1.5.3"

# Benchmark model support
uv pip install \
  "lifelines==0.30.3" "xgboost==3.2.0" "catboost==1.2.10" \
  "statsmodels==0.14.6" "scikit-posthocs==0.12.0"

# Optional ranking helper
uv pip install "openskill==6.2.0"

# Dev quality gates
uv pip install -U "ruff==0.15.11" "pytest==9.0.3" "mypy==1.20.2"
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Optuna | Ray Tune (`ray==2.55.1`) | Use Ray Tune only when you truly need distributed HPO across many workers/nodes and can accept additional orchestration overhead and weaker determinism guarantees. |
| scikit-survival + lifelines | pycox / torch-native survival stacks | Use deep stacks only for a clearly separate deep-model track where extra wall-clock cost is intentional and budgeted. |
| pyarrow Parquet artifacts | CSV-per-metric artifact sprawl | Use CSV only for tiny human-readable extracts; keep canonical benchmark outputs in Parquet/JSON manifests. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Ray Tune as default HPO backend | Distributed tuning adds overhead and reproducibility complexity; Ray docs explicitly note exact reproducibility is difficult in distributed settings. | Default to Optuna seeded sequential/local studies; promote Ray only for explicit scale-out needs. |
| Unbounded/exhaustive grid search | Wall-clock explodes combinatorially and conflicts with medium-breadth benchmark constraints. | Budgeted TPE + pruning with fixed trial/time caps per model/dataset. |
| Deep-learning survival methods in mandatory v1 baseline suite | Python-only v1 with wall-clock constraints favors strong classical + tree baselines first; deep models can dominate runtime and maintenance budget. | Keep deep models optional/flagged and run in separate tracks. |
| Pickle-only or wide CSV-only canonical outputs | Hard to query, deduplicate, and version robustly for manuscript workflows. | Parquet + lightweight JSON manifest + deterministic export schema. |
| Single-source p-value reporting without multiplicity correction | Inflates false positives in large pairwise benchmark grids. | Use `statsmodels.stats.multitest.multipletests` with declared correction policy. |

## Stack Patterns by Variant

**If strict reproducibility is more important than throughput:**
- Use Optuna in sequential mode with fixed sampler/pruner seeds and fixed study names/storage.
- Keep local process-level parallelism conservative (`joblib`) and avoid distributed schedulers.
- Freeze split manifests and write canonical Parquet summaries only once per run.

**If throughput pressure dominates and modest nondeterminism is acceptable:**
- Introduce Ray Tune only for the heaviest search spaces and largest datasets.
- Keep a deterministic "audit rerun" mode in Optuna local mode for verification.
- Separate exploratory and publication tracks so manuscript artifacts remain reproducible.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `scikit-survival==0.27.0` | `python>=3.11`, `scikit-learn==1.8.0` | Maintain Python 3.12 baseline to reduce cross-package friction. |
| `numpy==2.4.4` | `scipy==1.17.1`, `pandas==3.0.2` | Use pinned core triad to avoid silent ABI/behavior drift. |
| `optuna==4.8.0` | `python>=3.9` | Reproducibility still depends on seeded sampler and deterministic objective function. |
| `xgboost==3.2.0` | `python>=3.10` | Supports survival objectives needed for benchmark model diversity. |
| `catboost==1.2.10` | Python 3.x | Includes survival objectives (`Cox`, `SurvivalAft`) for categorical-heavy datasets. |

## Sources

- [PyPI: scikit-survival JSON](https://pypi.org/pypi/scikit-survival/json) — version (`0.27.0`) and Python requirement (`>=3.11`). (HIGH)
- [PyPI: lifelines JSON](https://pypi.org/pypi/lifelines/json) — version (`0.30.3`) and Python requirement (`>=3.11`). (HIGH)
- [PyPI: optuna JSON](https://pypi.org/pypi/optuna/json) — version (`4.8.0`). (HIGH)
- [Optuna FAQ](https://optuna.readthedocs.io/en/stable/faq.html) — reproducibility guidance (seeded sampler, sequential preference for strict reproducibility). (HIGH)
- [Context7: Optuna docs lookup](https://optuna.readthedocs.io/en/stable/faq.html) — deterministic sampler and RDB-storage patterns. (HIGH)
- [PyPI: ray JSON](https://pypi.org/pypi/ray/json) and [Ray Tune FAQ](https://docs.ray.io/en/latest/tune/faq.html) — scale-out option and reproducibility caveats in distributed tuning. (HIGH)
- [Context7: Ray docs lookup](https://github.com/ray-project/ray/blob/master/doc/source/tune/faq.md) — seed placement for driver/trainable and reproducibility caveats. (HIGH)
- [PyPI: xgboost JSON](https://pypi.org/pypi/xgboost/json) + [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html) — survival objectives (`survival:cox`, `survival:aft`). (HIGH)
- [PyPI: catboost JSON](https://pypi.org/pypi/catboost/json) + [CatBoost loss functions](https://catboost.ai/docs/en/concepts/loss-functions-regression) — `Cox` and `SurvivalAft` availability. (HIGH)
- [SciPy bootstrap docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) — bootstrap CI support for manuscript-grade uncertainty quantification. (HIGH)
- [statsmodels multipletests docs](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html) — multiple-testing correction methods for pairwise benchmark analysis. (HIGH)
- [pandas to_parquet docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html) — compact Parquet artifact strategy and compression options. (HIGH)
- [Web-search ecosystem snapshots](https://github.com/nliulab/Survival-Benchmark) — directional signals for package usage in survival benchmark practice. (LOW)

---
*Stack research for: practitioner-focused survival benchmarking (Python-only v1)*
*Researched: 2026-04-23*
