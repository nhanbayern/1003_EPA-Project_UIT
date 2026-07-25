# View Reviews

**Paper ID**
61

**Paper Title**
Forecasting Accuracy Does Not Guarantee Risk Control: A Unified Evaluation Framework for
Volatility Models

```
Reviewer # 1
```
## Questions

```
1. Summary of the paper
The paper studies whether strong volatility forecasting accuracy also leads to better
financial risk control. It compares ten econometric, deep-learning, hybrid, and foundation-
model approaches across nine stock indices from 2010–2025, using both forecasting
errors and VaR backtesting. The main result is that Autoformer achieves the best
prediction accuracy, while GARCH-LSTM Hybrid and Reformer perform better for VaR risk
calibration. Overall, the paper argues that volatility models should be evaluated not only by
forecast error but also by their ability to support reliable risk management.
2. Strong points
Good motivation and interesting benchmark scope.
The main result is interesting. Table I clearly shows that Autoformer has the best predictive
accuracy, with QLIKE 0.019, MSE 0.048, and MAE 0.163, while GARCH-LSTM Hybrid has
the best pass rate, 0.467, followed by Reformer at 0.422. This supports the paper’s central
claim that forecast accuracy and risk validity can diverge.
3. Weak points and comments
The presentation of the unified model is unclear. it seems that it is a compositition of
multiple existing benchmarks. Also, there is no mention of data source (for verification of
validity) and data processing.
the claim: "The Moirai-based results further support this point. Relative to the original
Moirai-MoE architecture, Moirai-MoEGARCHs reduces QLIKE from 24.273 to 0.040 and
raises
pass rate to approximately zero" is not supported by table I (with pass rate 0.111).
The paper defines a pass when both Kupiec and Christoffersen p-values exceed 0.05, but
does not mention sample size.
5. Overall score
Weak Accept
```

**Reviewer # 2**

## Questions

```
1. Summary of the paper
This paper presents a unified evaluation framework for financial volatility forecasting that
jointly assesses prediction accuracy and risk management performance. The authors
benchmark ten econometric, deep learning, hybrid, and foundation-model approaches on
nine global stock market indices over the 2010–2025 period using both forecasting metrics
(MSE, MAE, and QLIKE) and Value-at-Risk (VaR) backtesting. Experimental results reveal
that the most accurate forecasting model does not necessarily provide the most reliable
risk control: while Autoformer achieves the best prediction accuracy, GARCH-LSTM Hybrid
and Reformer deliver superior VaR backtesting performance. These findings demonstrate
the trade-off between forecasting precision and risk calibration and highlight the necessity
of evaluating volatility models using both predictive and risk-oriented criteria to support
more informed financial decision-making.
2. Strong points
The paper addresses a practically important problem by proposing a unified evaluation
framework that bridges the gap between volatility forecasting accuracy and financial risk
management. Its comprehensive benchmark across multiple model families, nine
international stock market indices, and several forecasting horizons provides a thorough
empirical comparison under a consistent evaluation protocol. The inclusion of both
prediction metrics and VaR backtesting offers valuable insights into the trade-off between
forecasting performance and risk calibration, leading to a clear and practically relevant
conclusion that forecasting accuracy alone is insufficient for selecting models in risk-
sensitive financial applications.
3. Weak points and comments
(1) The novelty of the proposed framework is relatively limited. The paper mainly combines
conventional forecasting metrics (MSE, MAE, QLIKE) with standard VaR backtesting
methods into a unified evaluation protocol. Although useful in practice, the framework is
largely an integration of existing evaluation techniques rather than a fundamentally new
methodology or theoretical contribution.
(2) The proposed Moirai-MoE-GARCHs model is insufficiently validated. While the
architecture is described in detail, the experimental results show that it performs worse
than several existing methods, particularly GARCH-LSTM Hybrid and Reformer, in terms
of risk evaluation. Consequently, the proposed model does not convincingly demonstrate a
performance advantage over current state-of-the-art approaches.
(3)The empirical analysis remains descriptive rather than analytical. The paper reports
performance comparisons but does not investigate why forecasting accuracy fails to
correlate with VaR performance. Additional analyses, such as calibration error, error
```

```
distribution, uncertainty estimation, or regime-specific behavior, would provide stronger
scientific insights.
(4) No statistical significance analysis is provided. Although differences between models
are reported, the paper does not conduct statistical tests (e.g., Diebold–Mariano test,
Model Confidence Set, Friedman/Nemenyi tests, or bootstrap confidence intervals) to
determine whether the observed improvements are statistically meaningful.
(5) The evaluation protocol focuses on only one risk measure. Risk assessment relies
solely on 5% VaR backtesting using Kupiec and Christoffersen tests. Other widely adopted
financial risk metrics, such as Expected Shortfall (ES), Conditional VaR, or additional
regulatory backtests, are not considered, limiting the comprehensiveness of the
evaluation.
(6) The experimental settings lack sufficient implementation details. Important
hyperparameters, optimization settings, computational cost, training time, hardware
configuration, and reproducibility information are only briefly mentioned or omitted, making
it difficult for other researchers to reproduce the reported results.
(7)The discussion of practical implications is somewhat limited. While the paper concludes
that forecasting accuracy does not guarantee better risk management, it does not provide
clear guidance on how practitioners should select models under different application
scenarios or how the proposed framework can influence real-world financial decision-
making.
Overall, the paper presents a useful benchmarking study with practical value, but its
scientific contribution would be significantly strengthened by providing deeper
methodological novelty, more rigorous statistical validation, richer empirical analyses, and
stronger evidence demonstrating the advantages of the proposed approach over existing
methods.
5. Overall score
Weak Reject
```
**Reviewer # 3**

## Questions

```
1. Summary of the paper
The paper proposes a unified evaluation protocol that jointly reports forecasting-error
metrics (MSE, MAE, QLIKE) and VaR backtesting performance (Kupiec, Christoffersen,
pass rate) for ten volatility models — econometric, deep-learning, hybrid, and foundation-
model — across nine stock indices from 2010–2025. It also introduces a new variant,
Moirai-MoE-GARCHs, built on top of Moirai-MoE with an added regime-aware expert-
routing mechanism and a dedicated volatility head. The central empirical finding is that the
```

model with the lowest forecasting error (Autoformer) is not the model with the best VaR
backtest pass rate (GARCH-LSTM Hybrid, Reformer).

**2. Strong points**

1. The core empirical question — whether forecast accuracy and risk-calibration rankings
diverge — is well motivated and practically relevant for model selection in risk-sensitive
deployment.
2. The data pipeline is careful about avoiding look-ahead bias (strictly causal windowing)
and the distribution-aware train/val/test split, with explicit Jarque–Bera and Student-t
diagnostics, is a nice methodological touch that most volatility-forecasting papers skip.
3. The breadth of the benchmark (9 indices across 4 continents, 10 model families, 5
horizons) is a genuine strength for a benchmarking paper.

**3. Weak points and comments**

1. Section V states that Moirai-MoE-GARCHs "raises pass rate to approximately zero," but
Table I reports its pass rate as 0.111, while "approximately zero" (0.000) describes the
original Moirai-MoE. This is not a minor wording slip — it is the sentence used to justify the
paper's own proposed architecture, and as written it materially misstates the paper's
central contribution's performance.
2. Moirai-MoE-GARCHs ranks 9th of 10 on pass rate and is not the best model on any
metric in Table I. The paper's narrative device (highlighting the fact that accuracy and risk
rankings diverge) does not require this model to be introduced at all — the same
conclusion could be demonstrated purely by comparing Autoformer against GARCH-LSTM
Hybrid/Reformer. This raises the question of what the architectural contribution in Section
III-C is actually adding to the paper's argument.
3. No ablation or model-selection criterion (BIC, cross-validated likelihood) is given for why
three latent states were chosen, nor is there any inspection of what the inferred regimes
correspond to economically (e.g., calm vs. crisis periods).
4. Hyperparameters, training budgets, and how the ten models were tuned (equally, or
individually optimized per model family) are not reported. Without this, the ranking in Table
I could partly reflect uneven tuning effort rather than intrinsic model capability.
5. Sample size per dataset-horizon cell is not reported, which matters directly for the
reliability of the Kupiec/Christoffersen p-values that pass rate is built on (both tests are
asymptotic and can be unreliable with few violations in small samples).

**5. Overall score**
Reject


