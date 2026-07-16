# RSBS confirmatory benchmark protocol

## Scientific claim

The exploratory work does **not** support replacing likelihood with KMD. The confirmatory hypothesis is narrower:

> In settings with adequately supported time-varying effects, a matched flexible intensity model trained with validation-selected Reference-Scaled Bregman Survival (RSBS) improves proper distributional scores relative to the same architecture trained by right-censored log score alone, without material loss under proportional-hazards truth.

## Objective

For positive candidate interval hazards `q`, exposures `E`, events `D`, and a fixed training-only reference hazard `a`, use

```text
L_RSBS = L_log + eta L_beta,a, eta >= 0
```

with

```text
L_beta,a = sum_k E_k a_k (q_k/a_k)^beta / beta
                 - D_k (q_k/a_k)^(beta-1)/(beta-1).
```

The beta=1 limit is the log-intensity score; beta=2 gives `E q^2/(2a) - D q/a`. Estimate `a` inside training folds or by cross-fitting and detach it from gradients.

## Design

- Five outer folds repeated five times.
- Shared outer and inner splits for every method.
- Preprocessing fitted only inside each outer fold.
- Equal hyperparameter-trial and runtime budgets.
- Preserve failed trials and runtime records.
- Tune architecture, optimization, time bins, `beta`, `eta`, and the reference model only on inner folds.

## Comparators

CoxPH; penalized Cox; DeepCox; Cox-Time; discrete hazard; PC-Hazard; random survival forest; gradient boosting; matched low-rank PEXP-NLL; matched RSBS.

## Metrics

Primary: right-censored log score and integrated Brier score over predeclared supported follow-up.

Secondary: calibration intercept/slope, RMST error, Harrell C, Uno C, Antolini C, time-dependent Uno C, cumulative/dynamic AUC, and decision-curve net benefit.

Concordance metrics are discrimination estimands and must not be interpreted as calibration metrics.

## Censoring

- Estimate marginal or conditional censoring distributions using training folds only.
- Cross-fit conditional censoring weights.
- Report weight distributions, effective sample size, truncation horizon, and positivity violations.
- Do not score beyond supported follow-up.

## PH characterization

Report global Schoenfeld tests, multiplicity-adjusted feature tests, estimated time-varying coefficients, event support by time interval, and performance differences between matched nonlinear-PH and time-varying architectures. Do not use one PH-test p-value as an automatic model-selection rule.

## Simulations

Vary sample size, event fraction, censoring fraction, covariate dimension, nonlinear strength, reversal amplitude, reversal time, delayed effects, cure fractions, multimodality, and censoring dependence. Include linear PH truth as a negative control.

## Uncertainty

- Paired outer-fold contrasts.
- Dataset-clustered bootstrap for aggregate claims.
- 95% percentile and BCa intervals.
- Holm adjustment across confirmatory secondary endpoints.
- Report effect sizes and failure rates, not winner counts alone.

## KMD role

Use the off-diagonal KMD U-statistic as a held-out specification diagnostic. Build critics on separate folds and calibrate rejection thresholds with a wild multiplier bootstrap. Do not treat finite-sample negative U-statistics as numerical errors.

## Decision rule

No method is declared superior unless the pre-specified primary paired interval excludes zero and the gain is not driven by unsupported follow-up, failed competitors, unequal tuning budgets, or a single dataset.
