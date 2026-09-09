# MoiraiVaR pipeline audit Q&A

This document is an audit brief for an independent agent. It describes the
intended corrected pipeline and the checks that must pass before using new
results in the paper.

## Q1. What is the forecasting origin?

**A.** At origin `t`, the model observes the 60 returns ending at `t`:

```text
x_t = (r[t-59], ..., r[t])
```

The model must not receive `r[t+1]` or any later return as an input feature.

## Q2. What does horizon `h` mean after the correction?

**A.** Horizon `h` is the next-`h`-day realized RMS volatility:

```text
y[t,h] = sqrt(mean(r[t+1:t+h+1] ** 2))
```

Thus `h=1,3,5,10,21` refer to one-, three-, five-, ten-, and 21-day future
return windows. They are no longer offsets of a 60-day rolling window.

## Q3. Can target construction use future returns?

**A.** Yes, future returns are allowed in a supervised target. They are not
allowed in the input. A target is valid only when its entire future window is
inside the same train, validation, or test split.

## Q4. How are split-boundary leaks prevented?

**A.** `load_and_split_dataset()` removes the final `max(horizon)` origins from
each split. The target of every retained origin therefore ends before that
split's boundary. Validation and test inputs may use historical context from a
previous period, but their labels must remain inside their own period.

## Q5. Which return is used by the VaR loss?

**A.** The dataset supplies `r[t+1]`, the next return after the forecasting
origin. This matches a one-step VaR target and avoids using the origin return
as an implicit future observation.

## Q6. What is the Student-t parameterization?

**A.** Model output is interpreted as a volatility standard deviation. For
`nu > 2`, the standardized Student-t quantile is:

```text
q_alpha = t.ppf(alpha, nu) * sqrt((nu - 2) / nu)
```

Training and `VarBacktester.var_threshold()` must use the same expression.
Values `nu <= 2` are invalid for a finite-variance standard-deviation
parameterization and must not be silently accepted.

## Q7. How is `nu` estimated?

**A.** The training notebook estimates `nu` from training returns using the
moment relation `nu = 4 + 6 / excess_kurtosis`, then clips it to the configured
finite-variance range. An audit must verify that evaluation does not refit `nu`
from pooled test predictions.

## Q8. Does one run train five separate models?

**A.** No. One model outputs the five horizons jointly. The corrected target
changes all five output columns, so every retained configuration must be
trained or regenerated against the same target definition.

## Q9. What must be rerun for a full benchmark?

**A.** If the paper keeps all 26 configurations, regenerate the predictions or
rerun training for every compared family: four GARCH models, twelve Transformer
configurations, six Moirai configurations, and four hybrid configurations. At
minimum, retrain the primary MoiraiVaR model and its MSE-only baseline, then
re-evaluate all reusable baselines on common target and test timestamps.

## Q10. What is the minimum reviewer-facing experiment?

**A.** For one primary backbone, run `lambda` values `{0, 0.05, 0.1, 0.2,
0.5, 1.0}`; tune a scalar-rescaling baseline on validation; and report MSE,
MAE, QLIKE, quantile loss, violation rate, and independence diagnostics. Do not
claim that the VaR objective adds value unless it beats the rescaling baseline.

## Q11. What remains outside this code change?

**A.** The Transformer, GARCH, and hybrid dataset builders now implement the
same causal target/export schema. Their existing CSV artifacts remain legacy:
every compared family must be rerun and merged on verified common keys before
scientific comparison. The statistical analysis pipeline also needs a verified
train-only source for per-market `nu` rather than fitting it from the merged
prediction file.

## Q12. What evidence should the auditor request?

**A.** Request: (1) sample counts per split after horizon filtering; (2) a check
that no target index crosses a split boundary; (3) the exact `nu` source and
values; (4) training/evaluation quantile formulas; (5) common timestamp counts
across model families; (6) the lambda-versus-rescaling table; and (7) raw
violation rates and quantile losses behind every composite rank.

## Audit verdict rule

The corrected Moirai pipeline is internally aligned only if Q1--Q6 pass. A
paper-level benchmark claim is supported only if Q7--Q12 also pass, or if the
claim is narrowed to the subset that was actually regenerated and audited.
