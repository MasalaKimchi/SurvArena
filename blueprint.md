# SurvArena: An AutoML-Ready Benchmarking Framework for Tabular Survival Analysis

## 1. Objective

SurvArena is a reproducible, extensible, and fair framework for **tabular survival analysis**.
Its purpose is twofold:

1. provide an AutoML-style entrypoint for users who want to run strong survival models on
   their own data with minimal configuration
2. provide a rigorous benchmark engine for comparing classical, machine-learning, and
   deep-learning survival models under a standardized protocol

In practice, SurvArena should feel as easy to start with as AutoGluon while preserving the
benchmarking discipline needed for credible research comparisons. Users should be able to
bring a CSV, Parquet file, or DataFrame, specify survival labels, and let the framework
handle preprocessing, model selection, tuning, evaluation, and artifact export.

The benchmark is designed to answer the following questions:

1. Which methods perform best on **discrimination**?
2. Which methods perform best on **calibration / overall prediction quality**?
3. Which methods are most **stable across random seeds and data splits**?
4. Which methods are most **compute-efficient**?
5. Which methods generalize best across datasets with different censoring rates, feature types, and sample sizes?

The framework should be implemented as a normal GitHub repository first, with all experiment
results logged to disk as structured files, so that a future public leaderboard can be built
on top of the same outputs.

---

## 2. Scope

### Included
- A high-level `SurvivalPredictor` interface for user-owned datasets
- Right-censored tabular survival datasets
- Classical models, tree-based models, boosting models, deep survival models, and future tabular foundation-model integrations
- Repeated evaluation across multiple seeds and data splits
- Per-dataset, per-seed, and aggregate performance summaries
- Logging of metrics, runtime, memory, configuration, and software environment
- Preset-driven execution modes such as `fast`, `medium`, and `best`

### Not included initially
- Competing risks
- Multi-state survival
- Time-varying covariates
- Longitudinal sequence models
- Multi-modal imaging + tabular models

These can be added later as separate tracks.

---

## 3. Benchmark Philosophy

The key principle is:

> Compare **pipelines**, not single lucky runs.

A method is not just a model architecture. A method includes:
- preprocessing
- hyperparameter search space
- search budget
- early stopping
- random seeds
- train/validation/test protocol
- metric used for model selection

SurvArena should therefore benchmark a **fully specified evaluation procedure** for each method.

At the same time, SurvArena should hide this complexity from most end users. The benchmark
machinery is the backend; the default product surface should be simple enough that a user can
fit a portfolio of survival models without writing dataset loaders, YAML configs, or custom
preprocessing code.

### 3.1 Product Layers

SurvArena should expose two complementary layers:

#### Simple mode
- user passes a DataFrame or file path
- user specifies `time_col` and `event_col`
- framework infers schema and preprocessing
- framework trains and ranks a portfolio of models
- framework returns leaderboard, best model, risk predictions, and survival predictions

#### Research mode
- user selects dataset, method, and benchmark configs explicitly
- framework reuses persisted splits and strict evaluation rules
- framework produces reproducible artifacts for comparisons and reporting

This dual-layer design makes the system accessible to practitioners without sacrificing rigor.

### 3.2 AutoML-Like User Experience

The ideal entrypoint is a high-level API such as:

```python
from survarena import SurvivalPredictor

predictor = SurvivalPredictor(
    label_time="time",
    label_event="event",
    presets="medium",
    eval_metric="harrell_c",
)

predictor.fit(train_data="train.csv")
leaderboard = predictor.leaderboard()
pred_risk = predictor.predict_risk("test.csv")
pred_survival = predictor.predict_survival("test.csv")
```

The equivalent CLI should require only the essentials:

```bash
survarena fit --train train.csv --time-col time --event-col event --presets medium
```

Under this interface, the framework should automatically:

- validate right-censored survival labels
- infer feature types and preprocessing steps
- choose an appropriate model portfolio from available methods
- tune models within a user-facing budget or time limit
- produce a clear leaderboard and export reusable artifacts

Users should not need to understand the internal benchmark protocol to get value quickly.

---

## 4. Recommended Initial Dataset Suite

Use datasets that are established, public, and commonly reused in survival benchmarking.

## 4.1 Core Small / Medium Tabular Datasets

### SUPPORT
- Hospital survival dataset
- Mixed clinical/tabular variables
- Common benchmark for censored survival prediction
- Moderate sample size, heterogeneous clinical features

### METABRIC
- Breast cancer survival dataset
- Common in deep survival literature
- Mix of clinical and genomic/tabular features

### GBSG / GBSG2
- German Breast Cancer Study Group dataset
- Classic survival benchmark
- Moderate dimensionality, widely available in survival toolkits

### FLCHAIN
- Serum free light chain study
- Popular benchmark in survival evaluation papers
- Good for classical and ML methods

### WHAS500
- Worcester Heart Attack Study dataset
- Smaller clinical dataset
- Useful as a low-sample benchmark

## 4.2 Large Tabular Dataset

### KKBox
- Large-scale churn / survival-style dataset
- Very large sample size
- Useful for testing scalability and compute efficiency
- Should be treated as a separate “large-scale” benchmark track because repeated nested CV may be too expensive

## 4.3 Optional Additional Dataset Sources

### SurvSet
A broad repository of survival datasets intended for benchmarking and standardization.

### PyCox datasets
Convenient packaged access to several common survival datasets.

### scikit-survival datasets
Useful for canonical clinical survival datasets and consistent loading APIs.

---

## 5. Dataset Description Table

Maintain a machine-readable metadata file for every dataset with:

- dataset_id
- source
- citation
- n_samples
- n_features
- event_rate
- censoring_rate
- missingness_rate
- feature_types
- recommended_split_strategy
- recommended_primary_metric
- notes

Example:

```yaml
dataset_id: support
name: SUPPORT
task_type: right_censored_survival
source: scikit-survival
n_samples: null
n_features: null
event_col: event
time_col: time
group_col: null
feature_types:
  - numerical
  - categorical
primary_metric: uno_c
split_strategy: stratified_event
notes: Mixed clinical variables; moderate censoring.
