# SurvArena: A GitHub-First Benchmarking Framework for Tabular Survival Analysis

## 1. Objective

SurvArena is a reproducible, extensible, and fair benchmarking framework for **tabular survival analysis**. Its purpose is to compare classical, machine-learning, and deep-learning survival models under a standardized protocol across multiple public datasets, using consistent preprocessing, hyperparameter search budgets, repeated evaluation, and transparent logging.

The benchmark is designed to answer the following questions:

1. Which methods perform best on **discrimination**?
2. Which methods perform best on **calibration / overall prediction quality**?
3. Which methods are most **stable across random seeds and data splits**?
4. Which methods are most **compute-efficient**?
5. Which methods generalize best across datasets with different censoring rates, feature types, and sample sizes?

The framework should be implemented as a normal GitHub repository first, with all experiment results logged to disk as structured files, so that a future public leaderboard can be built on top of the same outputs.

---

## 2. Scope

### Included
- Right-censored tabular survival datasets
- Classical models, tree-based models, boosting models, and deep survival models
- Repeated evaluation across multiple seeds and data splits
- Per-dataset, per-seed, and aggregate performance summaries
- Logging of metrics, runtime, memory, configuration, and software environment

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

### PBC
- Primary biliary cirrhosis dataset
- Classic survival analysis dataset
- Includes missing values, useful for testing preprocessing rigor

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