# MoiraiVaR: horizon-wise accuracy–risk analysis

## Executive conclusion

The strongest evidence for MoiraiVaR is not that it dominates every model on
tail-risk calibration. The evidence is that a **validation-selected
MoiraiVaR–Moirai 2 configuration** improves point volatility forecasting at
short and medium horizons while remaining Pareto-efficient on the
accuracy–risk plane.

The primary configuration used below is **MoiraiVaR–Moirai 2 (λ=0.05)**. The
value of λ was selected from validation performance and then held fixed for
test evaluation. This avoids choosing a different λ after observing each test
horizon.

The defensible paper claim is therefore:

> VaR-aware adaptation is an effective improvement to point volatility
> forecasting, with the clearest gains at horizons 1–5 days and a persistent
> accuracy–risk trade-off at longer horizons. It is not a universal
> replacement for econometric models when tail calibration is the only
> objective.

## Evaluation scope and naming

Results come from Modal volume run `20260911_020709` after the active
rolling-60 target contract was applied. The analysis contains 9 equity indices,
5 horizons (`h ∈ {1, 3, 5, 10, 21}`), and 41 configurations. The validation
sanity gate retained 36 configurations for test ranking.

Naming follows the paper convention:

| Internal identifier | Paper name |
|---|---|
| `Moirai_VAR / moirai2 / lambda_0.05` | MoiraiVaR–Moirai 2 (λ=0.05) |
| `Moirai / moirai2 / baseline` | Moirai 2 baseline |
| `Moirai_VAR / moirai_moe / lambda_*` | MoiraiVaR–Moirai-MoE (λ=*) |
| `GARCH / statistical / FI-GARCH` | FI-GARCH |
| `GARCH / statistical / GARCH` | GARCH |
| `GARCH / statistical / GJR-GARCH` | GJR-GARCH |

The target used in this corrected run is the active project contract:

```text
input:      r[t-59 : t+1]
target(h):  std(r[t+h-59 : t+h+1], ddof=0)
VaR return: r[t+1]
```

The manuscript and the older audit prompt must be reconciled before submission:
the paper should describe this rolling-60 endpoint target explicitly rather
than the former future-window standard-deviation definition.

## 1. Horizon-wise point-forecast accuracy

![MoiraiVaR horizon accuracy](figures/moiraivar/moiraivar_horizon_accuracy.png)

The left panel shows mean squared error averaged over the 9 markets. The right
panel compares the validation-selected MoiraiVaR–Moirai 2 configuration with
the best non-MoiraiVaR configuration at each horizon. The comparison includes
the Moirai baseline and the econometric families, so it is conservative.

| Horizon | Primary configuration | Test MSE | Improvement over best non-MoiraiVaR | Paired wins vs Moirai 2 baseline |
|---:|---|---:|---:|---:|
| 1 | MoiraiVaR–Moirai 2 (λ=0.05) | 0.010860 | 13.6% | 8/9 markets |
| 3 | MoiraiVaR–Moirai 2 (λ=0.05) | 0.013334 | 7.3% | 6/9 markets |
| 5 | MoiraiVaR–Moirai 2 (λ=0.05) | 0.016446 | 7.2% | 7/9 markets |
| 10 | MoiraiVaR–Moirai 2 (λ=0.05) | 0.026121 | 3.6% | 4/9 markets |
| 21 | MoiraiVaR–Moirai 2 (λ=0.05) | 0.051652 | 0.6% | 4/9 markets |

The strongest results are at horizons 1 and 5. An exploratory paired Wilcoxon
test against the Moirai 2 baseline gives unadjusted one-sided `p=0.0098` at
`h=1` and `p=0.0371` at `h=5` (`n=9` markets). These should be reported as
supporting evidence, not as a substitute for a larger market or rolling-origin
inference design.

At `h=10` and `h=21`, the average MSE advantage is smaller and the paired
market evidence is mixed. The paper should describe these horizons as
competitive or Pareto-efficient rather than claiming a decisive accuracy win.

## 2. Horizon-wise Pareto analysis

![MoiraiVaR horizon Pareto frontier](figures/moiraivar/moiraivar_horizon_pareto.png)

For each horizon, accuracy and risk scores were normalized within that horizon.
Accuracy combines MSE, MAE, and QLIKE. Risk combines Student-t VaR pass rate
and absolute violation-rate error at 1% and 5%. A configuration is Pareto
efficient when no other configuration is at least as good on both axes and
strictly better on one.

| Horizon | Pareto configurations | MoiraiVaR configurations on frontier |
|---:|---:|---:|
| 1 | 6 | 3 |
| 3 | 6 | 2 |
| 5 | 6 | 3 |
| 10 | 4 | 1 |
| 21 | 4 | 2 |

The validation-selected MoiraiVaR–Moirai 2 (λ=0.05) point is on the Pareto
frontier at **all five horizons**. This is the most useful balanced claim:
MoiraiVaR does not need to win the risk axis outright to be useful; it provides
very high forecast accuracy while retaining a non-dominated accuracy–risk
trade-off.

The frontier also exposes the complementary role of the model families:

- MoiraiVaR occupies the high-accuracy side of the frontier.
- FI-GARCH and GJR-GARCH occupy the stronger-risk-calibration side.
- Moirai 2 baseline lies between these extremes.

Thus, the paper should not frame MoiraiVaR as a universal replacement for
GARCH. It should frame the method as a **risk-aware point-forecast adaptation**
that is especially attractive when accurate multi-horizon volatility forecasts
and usable tail-risk outputs are required together.

## 3. Why the architecture is a point-forecast improvement

![MoiraiVaR architecture](figures/moiraivar/moiraivar_architecture.png)

MoiraiVaR preserves the point-forecast pathway:

1. A 60-return context is encoded by the pretrained Moirai, Moirai 2, or
   Moirai-MoE backbone.
2. Mean pooling produces a shared latent representation.
3. The MLP head predicts non-negative volatility forecasts at five horizons.

The VaR-aware improvement is an additional risk branch and training signal,
not a replacement for the point forecast. The one-step predicted volatility is
converted into a standardized Student-t VaR threshold, and the joint loss is

```text
L = L_volatility + λ_VaR · L_pinball(r[t+1], VaR[t+1|t])
```

This gives the head two simultaneous responsibilities:

- preserve accurate point estimates of future volatility;
- produce a scale that remains useful when translated into a lower-tail
  quantile for risk decisions.

That architectural interpretation is stronger than saying that VaR-aware
training simply adds another output. The risk objective changes the learned
point-forecast representation while retaining the original multi-horizon
forecast interface.

## 4. The common λ trade-off

![MoiraiVaR lambda trade-off](figures/moiraivar/moiraivar_lambda_tradeoff.png)

The λ sweep was evaluated across Moirai, Moirai 2, and Moirai-MoE. The first
panel shows the mean normalized accuracy and risk components; the remaining
panels show the mean scores produced by SAW and TOPSIS under both weighting
policies. The final panel averages the four policy-specific ranks, where a
smaller rank is better.

| Decision rule | Common λ selected | Interpretation |
|---|---:|---|
| Mean Pareto frontier | 0.05, 0.10, 0.50 | These λ values are non-dominated on the mean accuracy–risk plane. |
| SAW 50:50 | 0.50 | Best mean composite score when accuracy and risk receive equal block weight. |
| SAW 30:70 | 0.50 | Best mean composite score when risk receives greater weight. |
| TOPSIS 50:50 | 0.05 | Best mean closeness to the ideal solution. |
| TOPSIS 30:70 | 0.05 | Same result under the risk-heavier policy. |
| Consensus mean rank | **0.05** | Best average rank across SAW/TOPSIS and both weight policies. |

The methods disagree because they reward different geometries. SAW favors the
more risk-oriented λ=0.50 operating point, while TOPSIS favors λ=0.05 because
it remains closer to the joint ideal without giving up as much accuracy. The
average rank is 10.58 for λ=0.05 and 11.08 for λ=0.50; the next Pareto option,
λ=0.10, has a weaker consensus rank of 12.75.

### Recommended paper decision

Use **λ=0.05 as the primary common setting**:

- it is the validation-selected global setting for Moirai 2;
- it is on the Pareto frontier at every horizon;
- it wins the common TOPSIS comparison under both weight policies;
- it has the best consensus rank across the four decision rules.

Report **λ=0.50 as the risk-heavier sensitivity operating point**, not as a
universal optimum. This lets the paper show that increasing the VaR-loss
weight moves the system toward stronger risk-side composite scores, but can
reduce point-forecast accuracy.

The common-λ analysis is a diagnostic of the completed test set after the
validation gate. It must not be used to select a different λ after observing
test results. The primary λ remains the validation choice, and all other λ
values are ablation/sensitivity points.

## 5. What should and should not be claimed

### Supported claims

- MoiraiVaR–Moirai 2 (λ=0.05), selected on validation, has the lowest test MSE
  among the evaluated configurations at horizons 1, 3, 5, and 10.
- Its largest relative gains occur at short horizons, especially `h=1` and
  `h=5`.
- It remains Pareto-efficient at every evaluated horizon.
- VaR-aware adaptation improves the point-forecast/risk trade-off without
  changing the basic Moirai backbone or multi-horizon output interface.

### Claims to avoid

- “MoiraiVaR is the best risk-calibrated model at every horizon.” GARCH models
  generally occupy the stronger-risk side of the frontier.
- “λ=0.2 is the optimal global setting.” The test-only `h=21` winner is not a
  valid primary selection; λ=0.05 is the validation-selected global setting
  for Moirai 2.
- “VaR-aware training always improves every market.” The horizon-10 and
  horizon-21 paired market results are mixed.
- “All prediction rows are independent observations.” The paper should report
  market–horizon blocks and acknowledge overlapping rolling targets.

## 6. Reproducibility and remaining analysis items

Source artifacts used for this report:

- [Active target contract](target_contract.md)
- [MoiraiVaR data construction](../experiments/moirai_var_aware/data.py)
- [MoiraiVaR loss](../experiments/moirai_var_aware/losses.py)
- [Validation-only selection gate](../output/benchmark_analysis_20260911_020709/mcdm_frozen/ValidationSelectionGate.csv)
- [Horizon summary CSV](../output/benchmark_analysis_20260911_020709/horizon_analysis/horizon_model_summary.csv)
- [Horizon Pareto CSV](../output/benchmark_analysis_20260911_020709/horizon_analysis/horizon_pareto.csv)
- [λ trade-off summary CSV](../output/benchmark_analysis_20260911_020709/horizon_analysis/lambda_tradeoff_summary.csv)
- [λ trade-off by backbone CSV](../output/benchmark_analysis_20260911_020709/horizon_analysis/lambda_tradeoff_by_backbone.csv)

Before finalizing the conference table, add the scalar-rescaling control to the
horizon-wise Pareto input and include pinball loss plus Christoffersen
independence diagnostics directly in the risk component. The current frontier
is therefore a balanced accuracy–coverage analysis, not the final complete
tail-risk score.
