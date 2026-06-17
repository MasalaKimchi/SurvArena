# Pooled Discrete-Time Hazard Adapter

SurvArena does not treat TabPFN, TabICL, TabM, or RealTabPFN as native survival
foundation models. The foundation adapters evaluate whether tabular foundation
models designed primarily for classification can be adapted to right-censored
time-to-event data under a transparent Python benchmark contract.

## Historical Horizon Adapter

Earlier foundation experiments used censored-aware cumulative horizon
classification. For each training-fold horizon `tau_k`, those adapters trained
one binary classifier for `P(T <= tau_k | X)` using:

- positive rows: observed events with `Y_i <= tau_k`
- negative rows: subjects with `Y_i > tau_k`
- excluded rows: subjects censored before or at `tau_k`

The per-horizon event probabilities are monotonicity-corrected before survival
is reconstructed as `S(tau_k | X) = 1 - P(T <= tau_k | X)`.

## Default Pooled Discrete-Time Hazard Adapter

The maintained foundation adapter uses one binary classifier over stacked
patient-interval rows. With `Y_i = min(T_i, C_i)` and
`delta_i = 1[T_i <= C_i]`, a training-fold-only event-quantile grid is built:

```text
0 = tau_0 < tau_1 < ... < tau_K
```

For interval `k`, subject `i` contributes a row only when the subject is known
to be at risk at interval start:

```text
Y_i > tau_{k-1}
```

The binary label is:

```text
z_ik = 1  if delta_i = 1 and tau_{k-1} < Y_i <= tau_k
z_ik = 0  if Y_i > tau_k
```

Rows censored inside the interval are excluded because event status within the
interval is unknown. The classifier estimates:

```text
h_k(X_i) = P(tau_{k-1} < T_i <= tau_k | T_i > tau_{k-1}, X_i)
```

Input features are the benchmark-preprocessed tabular covariates plus
training-derived interval features:

- interval index
- log interval end
- interval width
- Kaplan-Meier survival at the interval endpoint

Survival is reconstructed by cumulative products:

```text
S(tau_k | X_i) = product_{j=1..k} (1 - h_j(X_i))
```

This construction guarantees non-increasing survival curves.

## Weighting And Metadata

The default `subject_weighting: normalized` assigns each stacked row weight
`1 / m_i`, where `m_i` is the number of rows contributed by subject `i`.
Backbones receive `sample_weight` only when their `fit` signature supports it.
Otherwise SurvArena records that weighting was requested but not applied.

Fold results and run metadata include audit fields for the time grid, stacked
row counts, interval event counts, excluded censored rows, weighting support,
hazard bounds, and fallback status. IPCW is reserved for future work and is
currently represented as `censoring_weighting: none`.
