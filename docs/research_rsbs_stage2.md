# Stage 2 research findings: beyond Cox loss

This exploratory study tested KMD, censoring-aware concordance estimands, matched PH/non-PH architectures, and a proper reference-scaled intensity objective.

## Main findings

- Under linear PH truth, CoxPH remained best.
- Under nonlinear PH truth, nonlinear PH models closed nearly all of the gap; time variation was unnecessary.
- Under crossing hazards, relaxing PH reduced known-truth survival ISE from about 0.0166 for CoxPH and 0.0126 for nonlinear PH to about 0.0058 for a time-varying piecewise hazard.
- The implemented KMD V/U training penalties were around `1e-6` and did not materially change optimization. KMD is better retained as a held-out residual specification test until its normalization and covariance estimation are redesigned.
- The measurable objective change came from a proper reference-scaled quadratic Bregman component, but its gain over matched likelihood was small and heterogeneous.
- Across nine public datasets, no method was uniformly best. Random survival forests, CoxPH, flexible likelihood, and RSBS each led on different datasets.

## Concordance taxonomy

Harrell and Uno concordance evaluate a static risk ordering. Antolini concordance evaluates subject-specific survival curves at the earlier event time and therefore permits rankings to change over time. A time-dependent Uno estimator adds marginal IPCW to this dynamic comparison. If censoring is independent only conditional on covariates, pairwise conditional censoring weights and cross-fitting are needed.

A calibration-deformation experiment raised oracle survival curves to powers from 0.5 to 2.0. Harrell, Uno, Antolini, and dynamic Uno concordance stayed essentially unchanged, while integrated Brier score and known-truth error were minimized only at the undeformed oracle. Concordance is therefore a discrimination estimand, not a measure of probabilistic correctness.

## Revised method

For candidate intensity `q`, fixed positive reference intensity `a`, exposure `E`, and event count `D`, use the beta-Bregman score

```text
L_beta,a = sum_k E_k a_k (q_k/a_k)^beta / beta
                 - D_k (q_k/a_k)^(beta-1)/(beta-1).
```

The beta=1 limit is the log-intensity score. Beta=2 gives `E q^2/(2a) - D q/a`. The proposed training objective is

```text
L_RSBS = L_log + eta L_beta,a, eta >= 0.
```

Its population excess risk is an at-risk weighted Bregman divergence between the true and candidate intensities. A nonnegative mixture with the strictly proper log score remains strictly proper on identifiable follow-up support. The reference must be estimated from training data or by cross-fitting and detached from model gradients.

## KMD diagnostic

For moment residuals `xi_i`, use the off-diagonal statistic

```text
(||sum_i xi_i||^2 - sum_i ||xi_i||^2) / [2 n (n-1)].
```

It removes the V-statistic self-noise term and may be negative in finite samples. Build critics and evaluate moments on separate folds, then use a wild multiplier bootstrap for a held-out goodness-of-fit test.

## Limitations

The public datasets are mostly small tabular benchmarks, five split seeds give exploratory rather than publication-grade intervals, and some IPCW scores were not estimable beyond censoring support. The study supports a confirmatory protocol, not a universal superiority claim.
