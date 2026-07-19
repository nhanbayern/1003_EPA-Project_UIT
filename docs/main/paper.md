# Forecasting Accuracy Does Not Guarantee Risk Control: A Unified Evaluation Framework for Volatility Models

**Abstract**—Volatility forecasting is often judged by predictive error, whereas financial risk management is judged by Value-at-Risk (VaR) backtests. This separation makes model selection difficult when the real objective is risk-sensitive deployment. This paper presents a unified evaluation framework that jointly measures forecasting accuracy and risk calibration on nine stock market indices from 2010 to 2025. The benchmark covers ten models from econometric, transformer-based, hybrid, and foundation-model variants, and reports MSE, MAE, QLIKE, violation rate, and VaR backtesting performance over horizons 1, 3, 5, 10, and 21. The results show a clear mismatch between the two objectives. Autoformer achieves the best forecasting accuracy, with the lowest QLIKE (0.019), MSE (0.048), and MAE (0.163), while GARCH-LSTM Hybrid and Reformer achieve the highest overall backtesting pass rates, 0.467 and 0.422, respectively. The main finding is that improving forecasting accuracy does not guarantee better risk management. The main contribution is a standardized evaluation protocol that exposes this trade-off and supports model selection under both predictive and risk criteria.

**Index Terms**—volatility forecasting, risk management, Value-at-Risk backtesting, time-series models, hybrid models

## I. INTRODUCTION

Financial volatility forecasts are rarely used in isolation. In practice, they support risk-sensitive decisions such as capital allocation, loss control, and VaR monitoring. However, model comparison is still dominated by prediction losses such as MSE, MAE, or QLIKE, even though these metrics do not directly test whether a forecast is useful for risk management. This creates a practical gap. A model can produce low average forecasting error but still fail when the target task is tail-risk control. The central problem of this paper is therefore not only which model forecasts volatility more accurately, but also which model turns volatility forecasts into more reliable risk signals.

To address this gap, we build a unified evaluation framework that combines forecasting accuracy with VaR backtesting. The benchmark uses nine stock indices from Europe, Asia, the United States, and Vietnam, and compares econometric, deep learning, hybrid, and foundation-model variants under the same protocol.

Our contributions are threefold:
* We define a unified evaluation framework that jointly reports forecasting accuracy and risk-management performance across multiple horizons and markets.
* We benchmark ten model instances spanning GARCH-family baselines, transformer-based forecasters, a hybrid GARCH-LSTM model, and two Moirai-based variants.
* We provide empirical evidence that the model with the best forecasting accuracy is not the model with the best risk-management performance: Autoformer leads on MSE, MAE, and QLIKE, whereas GARCH-LSTM Hybrid and Reformer lead on backtesting pass rate.

The key message is simple: improving forecasting accuracy does not guarantee better risk management, so volatility models should be evaluated within a unified framework rather than by prediction loss alone.

## II. RELATED WORK

### A. Econometric Volatility Models
Econometric volatility modeling remains the classical foundation of financial risk analysis. GARCH models conditional variance dynamics directly and remains a standard benchmark for volatility forecasting [1]. Extensions such as GJR-GARCH incorporate asymmetric responses to shocks [2], while FI-GARCH captures long-memory behavior in volatility [3]. Evidence from emerging markets also reinforces that mean and variance should be modeled separately: ARIMA is more effective for short-term trend, whereas GARCH is better aligned with volatility clustering and conditional variance dynamics [4]. Related evidence on good and bad volatility further shows that negative-shock volatility is more informative for market risk than positive-shock volatility, which supports asymmetric volatility specifications such as GJR-GARCH [5]. These models are attractive because their structure is closely aligned with volatility clustering, persistence, and tail-sensitive financial behavior.

### B. Deep Learning Time-Series Models
Deep learning approaches aim to learn richer temporal dependencies from sequential data. LSTM introduced gated recurrent memory for long-range dependency modeling [6]. Transformer architectures later replaced recurrence with attention, enabling more scalable sequence modeling [7]. In time-series forecasting, Informer improves efficiency for long sequences [8], Autoformer adds decomposition and auto-correlation mechanisms for series forecasting [9], and Reformer reduces transformer complexity for long-context modeling [10]. A recent analysis of transformer forecasting shows that their strongest gains usually appear in large-data settings with exploitable long-range structure, whereas high noise, limited data, and weak temporal structure reduce the benefit of attention-based models [11]. Hybrid approaches have also emerged. In particular, the GARCH-LSTM hybrid proposed by Zhao et al. combines financial inductive bias with neural representation learning [12].

### C. Risk Backtesting and Research Gap
Risk evaluation is typically studied through VaR backtesting rather than forecast-error minimization. Kupiec's coverage test checks whether the realized violation frequency matches the nominal risk level [13]. Christoffersen's framework further tests whether violations are independent over time [14]. This distinction matters because volatility forecasting quality and risk-management usefulness need not coincide [15].

The literature is therefore split in practice. Forecasting papers often emphasize predictive losses, while risk papers emphasize VaR calibration and backtesting validity. Regime-dependent evidence also shows that predictive accuracy changes materially between stable and turbulent states, so a model that works well in calm markets can degrade sharply when volatility rises [16]. What remains limited is a unified comparison across heterogeneous model families under both views. Our work addresses this gap by evaluating forecasting and risk performance within one standardized protocol.

## III. METHODOLOGY

### A. Task and Data Pipeline
The study uses nine stock market indices. The international indices (DAX 40, Euronext 100, IBEX 35, KOSPI Index, Nikkei 225, SMI, and S&P 500) are sourced from Investing.com, whereas the Vietnamese indices (VN30 and VN-Index) are sourced from vnstock. The benchmark spans the period from 2010 to 2025. The initial raw datasets contain only six basic price and volume columns: `time`, `open`, `high`, `low`, `close`, and `volume`.

Daily log returns are computed from closing prices as:
$$ r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) \quad (1) $$

where $P_{t}$ is the closing price at time $t$. To avoid leakage, each input sample contains only past information, namely the 60-return window:

$$ x_{t}=(r_{t-60},r_{t-59},...,r_{t-1}) \quad (2) $$

For each forecast origin $t$, the model predicts future volatility at horizons $h\in\{1,3,5,10,21\}$. Thus, the forecasting task is a causal mapping from past returns to future volatility, with inputs observed on $[t-60,t-1]$ and targets constructed on $[t,t+h-1]$. At prediction time, the model never observes any return from $t$ onward.

Accordingly, the realized volatility target for horizon $h$ is defined from future returns only:

$$ \sigma_{t,h}=\sqrt{\frac{1}{h}\sum_{i=0}^{h-1}(r_{t+i}-\overline{r}_{t,h})^{2}} \quad (3) $$

where

$$ \overline{r}_{t,h}=\frac{1}{h}\sum_{i=0}^{h-1}r_{t+i} \quad (4) $$

This protocol ensures a strictly causal setup: the model forecasts future volatility from past returns rather than reconstructing a target that overlaps with the input window. The train, validation, and test splits are then applied without mixing future observations into earlier inputs.

The data split is distribution-aware. It searches over candidate cut points $(i,j)$ with minimum segment length $\max(30,0.1n)$, estimates a Student-t degree-of-freedom parameter $\nu$ on each segment from excess kurtosis via $\nu = 4+6/k$ and accepts only splits for which all three $\nu$ values are finite and each stays within 20% of their mean. Among feasible candidates, it selects the split that minimizes the total deviation of the three values from that mean; if no feasible split exists, it falls back to a 60/20/20 partition. The resulting sample sizes for the train, validation, and test splits across all indices are detailed in Table I.

**Table I: Sample Sizes and Splits per Dataset (2010–2025)**
| Index | Source | Train | Validation | Test | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| DAX 40 | Investing.com | 1983 | 1104 | 972 | 4059 |
| Euronext 100 | Investing.com | 2005 | 1115 | 979 | 4099 |
| IBEX 35 | Investing.com | 1815 | 1362 | 923 | 4100 |
| KOSPI Index | Investing.com | 1944 | 1109 | 881 | 3934 |
| Nikkei 225 | Investing.com | 1677 | 1174 | 1062 | 3913 |
| SMI | Investing.com | 1987 | 1135 | 901 | 4023 |
| S&P 500 | Investing.com | 1988 | 1109 | 927 | 4024 |
| VN30 | vnstock | 1971 | 1096 | 925 | 3992 |
| VN-Index | vnstock | 1964 | 1103 | 925 | 3992 |

The return-distribution diagnostics also determine the distributional assumption used in the rest of the paper. Jarque-Bera rejects normality at the 5% level for all nine datasets in both train-validation and test, with p-values equal to zero up to machine precision on train-validation and at most $1.04\times10^{-191}$ on test. In contrast, the fitted Student-t model remains plausible across all splits: the estimated $\nu$ values lie in [2.706, 4.926] for train, [2.434, 3.772] for validation, and [2.385, 5.963] for test, while the corresponding fit p-values lie in [0.242, 0.842], [0.359, 0.970], and [0.507, 0.960], respectively. We therefore explicitly reject the Gaussian return assumption and use Student-t as the fixed distributional assumption throughout training and risk evaluation, rather than selecting the distribution family dynamically for each dataset or horizon.

### B. Compared Models
The benchmark contains the following model instances:
* **Econometric baselines:** GARCH, GJR-GARCH, and FI-GARCH.
* **Deep learning time-series models:** Transformer, Informer, Autoformer, and Reformer.
* **Hybrid and foundation-model variants:** GARCH-LSTM Hybrid [12], the original Moirai-MoE foundation model [17], and our MoiraiMoEGARCHS variant.

This design allows comparison between models that emphasize financial structure, models that emphasize sequence learning, and models that attempt to combine both perspectives.

### C. MoiraiMoEGARCHs Architecture
According to the project design notes, the original Moirai-MoE model from Liu et al. [17] suffers from a semantic mismatch between the predicted returns distribution and the volatility training target in this financial setting. This mismatch causes scale collapse and leads to unrealistically small volatility outputs. Our MoiraiMoEGARCHS variant addresses this issue by introducing a dedicated volatility head and optimizing the loss directly on the volatility target.

As summarized in Fig. 1, returns are first encoded by a transformer backbone. Regime information from a three-state HMM then drives hard routing to three experts: a GLU-FFN expert for nonlinear mapping, a GARCHNN expert for short-term volatility, and a HARGARCH expert for multi-scale volatility. The expert outputs are merged by a mask-based mechanism, and the final volatility head uses a Linear -> GELU -> Linear -> Softplus design to predict positive volatility for multiple horizons. The project notes specify Student-t negative log-likelihood for the volatility objective, matching the reported heavy-tailed, non-Gaussian behavior of returns.

### D. Evaluation Protocol
We evaluate each model from both a forecasting and a risk-management perspective. Let $y_t$ denote the realized volatility target, $\hat{y}_t$ the predicted volatility, $r_t$ the realized return, and $n$ the number of evaluation points.

#### 1) Forecasting Metrics: 
Forecasting accuracy is measured by MSE, MAE, and QLIKE. The mean squared error penalizes large forecast errors more heavily. The mean absolute error is easier to interpret and less sensitive to extreme deviations. We also report the quasi-likelihood loss,

$$ QLIKE=\frac{1}{n}\sum_{t=1}^{n}\left(\frac{y_t}{\hat{y}_t}-\log\left(\frac{y_t}{\hat{y}_t}\right)-1\right) \quad (5) $$

a standard volatility-forecasting metric that is particularly suitable when the target is strictly positive and noisy. For all three metrics, lower values indicate better predictive accuracy.

#### 2) VaR Construction and Risk Metrics: 
To assess whether volatility forecasts remain useful for risk control, the pipeline converts predicted volatility into a one-step-ahead VaR forecast:

$$ \widehat{VaR}_{t}^{\alpha} = \mu + \hat{\sigma}_{t}q_{\alpha} \quad (6) $$

where $\mu$ is the assumed mean return, typically set to zero, $\hat{\sigma}_t$ is the predicted volatility, and $q_{\alpha}$ is the $\alpha$-quantile of the assumed return distribution. Here $q_{\alpha}$ is taken from a Student-t distribution, not a normal distribution, because normality is rejected for every dataset and the fitted Student-t parameters remain stable across train, validation, and test. In this study, risk is evaluated at the 5% tail level.

A VaR violation occurs when the realized return falls below the predicted threshold. The corresponding indicator is

$$ I_t = \mathbf{1}(r_t < \widehat{VaR}_{t}^{\alpha}) \quad (7) $$

The empirical violation rate is then

$$ \text{Violation Rate} = \frac{1}{n}\sum_{t=1}^{n}I_t \quad (8) $$

For a well-calibrated model, this rate should remain close to $\alpha$; a much larger value indicates risk underestimation, whereas a much smaller value indicates overly conservative forecasts.

We further apply the Kupiec unconditional coverage test to verify whether the observed number of violations matches the expected tail probability. Let $x=\sum_{t=1}^{n}I_{t}$ and $\hat{p}=x/n$. The likelihood-ratio statistic is

$$ LR_{uc} = -2 \log\left(\frac{(1-\alpha)^{n-x}\alpha^{x}}{(1-\hat{p})^{n-x}\hat{p}^{x}}\right) \quad (9) $$

with p-value

$$ p_{uc}=1-F_{\chi^{2}(1)}(LR_{uc}) \quad (10) $$

In the evaluation pipeline, these quantities are reported as `kupiec_lr` and `kupiec_p`. A model is not rejected on unconditional coverage when `kupiec_p` exceeds 0.05.

To test whether violations cluster over time, we use the Christoffersen independence test. Let $n_{00}$, $n_{01}$, $n_{10}$, and $n_{11}$ denote the transition counts of the violation indicator sequence. The restricted and unrestricted likelihoods define the statistic

$$ LR_{ind}=-2(\log L_{\text{restricted}}-\log L_{\text{unrestricted}}) \quad (11) $$

with p-value

$$ p_{ind}=1-F_{\chi^{2}(1)}(LR_{ind}) \quad (12) $$

These are reported as `lr_ind` and `lr_ind_p`. When `lr_ind_p` is above 0.05, the model is not rejected on independence.

Finally, we summarize risk reliability through an overall pass rate. For each dataset-horizon pair, a model is treated as passing the backtest when both `kupiec_p` and `lr_ind_p` exceed 0.05. The aggregate pass rate is

$$ \text{Pass Rate} = \frac{\text{number of passed cases}}{\text{total number of evaluated cases}} \quad (13) $$

A higher pass rate indicates that the model remains statistically acceptable more often across datasets and forecast horizons. This metric is important because low forecasting error alone does not guarantee reliable tail-risk control.

## IV. EXPERIMENTS AND RESULTS

Table I reports the aggregate quantitative results. Fig. 2 visualizes the provided result summaries for forecasting error, violation rate, backtest success counts, and overall pass rate.

**TABLE I: AGGREGATE RESULTS OVER ALL DATASETS AND FORECAST HORIZONS.**
*(LOWER IS BETTER FOR QLIKE, MSE, AND MAE. HIGHER IS BETTER FOR PASS RATE.)*

| Model | Pass rate | QLIKE | MSE | MAE |
| :--- | :---: | :---: | :---: | :---: |
| GARCH-LSTM Hybrid | 0.467 | 0.042 | 0.121 | 0.254 |
| Reformer | 0.422 | 0.045 | 0.116 | 0.256 |
| GJR-GARCH | 0.333 | 0.051 | 0.162 | 0.266 |
| FI-GARCH | 0.267 | 0.028 | 0.098 | 0.203 |
| Autoformer | 0.200 | 0.019 | 0.048 | 0.163 |
| GARCH | 0.178 | 0.035 | 0.112 | 0.221 |
| Transformer | 0.133 | 0.054 | 0.133 | 0.298 |
| Informer | 0.133 | 0.063 | 0.158 | 0.325 |
| MoiraiMoEGARCHS | 0.111 | 0.040 | 0.104 | 0.237 |
| Moirai MoE | 0.000 | 24.273 | 1.122 | 0.988 |

Table I shows that GARCH-LSTM Hybrid achieves the highest pass rate, followed by Reformer. Autoformer records the lowest QLIKE, MSE, and MAE. Moirai_MoE records the lowest pass rate and the largest forecasting errors among the compared models.

## V. EMPIRICAL ANALYSIS

The results reveal a consistent trade-off between forecasting accuracy and risk management. Autoformer is the strongest model on predictive error, but its pass rate is only 0.200. In contrast, GARCH-LSTM Hybrid reaches the highest pass rate, 0.467, despite materially higher QLIKE, MSE, and MAE than Autoformer. Reformer shows a similar pattern, with the second-best pass rate and moderate forecasting error.

This trade-off also appears within the econometric family. FI-GARCH provides the strongest accuracy metrics among the econometric baselines, whereas GJR-GARCH attains the highest pass rate in that group. The ranking therefore changes once the evaluation objective moves from average error to risk validity.

The Moirai-based results further support this point. Relative to the original Moirai-MoE model, MoiraiMoEGARCHS reduces QLIKE from 24.273 to 0.040 and raises pass rate to approximately zero. Even so, it remains below the leading hybrid and transformer baselines in overall risk-management performance.

Overall, the benchmark identifies GARCH-LSTM Hybrid and Reformer as the strongest models for risk-sensitive deployment, while Autoformer is the strongest model for volatility point forecasting. The unified framework makes this distinction explicit.

## VI. DISCUSSION

The contrast between Autoformer and the best risk-oriented models helps explain why a unified evaluation framework is necessary. Transformer-based forecasters are effective at extracting temporal patterns and therefore reduce average prediction error. However, tail-risk control depends on calibration, violation behavior, and the temporal structure of extreme events, not only on central prediction accuracy. This is consistent with recent evidence that transformer success in time-series forecasting often reflects strong pattern matching in favorable settings rather than robust recovery of the underlying data-generating dynamics, especially when data are noisy or limited [11].

Econometric and hybrid models retain an advantage here because they are closer to the statistical structure of financial volatility. Volatility clustering, persistence, and asymmetric responses to shocks are directly modeled by the GARCH family and are partially preserved in the GARCH-LSTM hybrid. This makes these models more reliable when the evaluation target is VaR backtesting rather than point-forecast loss alone. The result is also coherent with evidence that variance forecasting should be treated separately from mean forecasting [4] and that downside, or bad, volatility contributes more to market risk than upside volatility [5].

The benchmark also highlights an evaluation mismatch in prior practice. If model selection were based only on MSE, MAE, or QLIKE, Autoformer would be the natural choice. If the target were risk management, GARCH-LSTM Hybrid or Reformer would be preferred instead. This mismatch is exactly why forecasting papers and risk papers should not remain separate when models are intended for financial decision support.

Finally, the foundation-model variants show an important limitation. General sequence modeling ability does not automatically transfer to financial volatility and tail-risk calibration. The improved MoiraiMoEGARCHs design is much more stable than the original Moirai-MoE model, but the remaining gap suggests that calibration and domain alignment are still open problems for foundation-style volatility forecasting. This limitation is plausible in a regime-dependent market environment, where predictive relationships change between stable and stressed periods and purely generic sequence modeling is less reliable without explicit regime awareness [16].

The contrast between Autoformer and GARCH-LSTM Hybrid also points to a concrete next research direction. Since Autoformer is strongest on forecasting accuracy while GARCH-LSTM Hybrid is strongest on risk-oriented performance, a promising line of work is to study architectures that combine Autoformer-style long-range sequence modeling with GARCH-style volatility structure. More broadly, our results motivate further research on hybrid statistical and machine-learning models that explicitly optimize the trade-off between forecasting accuracy and risk-management performance instead of treating these objectives separately.

## VII. CONCLUSION AND FUTURE WORK

### A. Conclusion
This paper presented a unified evaluation framework for volatility forecasting and risk management. By combining forecasting metrics with VaR backtesting over nine stock indices and multiple horizons, the framework shows that accuracy improvements do not necessarily translate into better risk control. Autoformer is the best model for predictive accuracy, while GARCH-LSTM Hybrid and Reformer provide the best overall balance when risk management is included in the evaluation.

The main insight is therefore operational rather than cosmetic: choosing a volatility model only from MSE, MAE, or QLIKE can lead to a different conclusion than choosing a model for risk-sensitive deployment. A unified evaluation framework is required to make that trade-off visible.

### B. Future Work
Future work should extend this benchmark in three directions. First, probabilistic forecasting can replace point volatility prediction with direct distributional or quantile forecasts. Second, economic evaluation can be added through deployment-oriented criteria such as P&L, drawdown, and transaction cost. Third, better calibration remains necessary, especially for foundation-model variants and other models intended to improve tail-risk reliability. An especially important direction is to design Autoformer-GARCH and related hybrid statistical-machine-learning models that can better balance forecasting accuracy with risk-management performance.

## REFERENCES
1. T. Bollerslev, "Generalized autoregressive conditional heteroskedasticity," Journal of Econometrics, vol. 31, no. 3, pp. 307-327, 1986.
2. L. R. Glosten, R. Jagannathan, and D. E. Runkle, "On the relation between the expected value and the volatility of the nominal excess return on stocks," The Journal of Finance, vol. 48, no. 5, pp. 1779-1801, 1993.
3. R. T. Baillie, T. Bollerslev, and H. O. Mikkelsen, "Fractionally integrated generalized autoregressive conditional heteroskedasticity," Journal of Econometrics, vol. 74, no. 1. pp. 3-30, 1996.
4. K. Macharia, E. Atitwa, D. Mugo, and M. Kawira, "Modeling stock price trends and volatility in emerging markets using ARIMA and GARCH approaches." International Journal of Advanced and Applied Sciences, vol. 12, no. 7, pp. 134-143, 2025.
5. D. Umoru, E. Oseremen, E. T. Omoluabi, G. Ohiokha, F. 1. Ohiokha, B. I. Mohammad, I. U. E. Enaberue, B. Oyakhiromhe, A. I. M. Olade, A. P. Anoghene et al., "Volatility risk premium and market risk forecasting: Good vs. bad volatility in emerging and developed markets," Risk Governance and Control: Financial Markets & Institutions, vol. 16. no. 1. pp. 138-151, 2026.
6. S. Hochreiter and J. Schmidhuber, "Long short-term memory." Neural Computation, vol. 9, no. 8. pp. 1735-1780, 1997.
7. A. Vaswani et al., "Attention is all you need," in Advances in Neural Information Processing Systems 30, 2017, pp. 5998-6008.
8. H. Zhou et al., "Informer: Beyond efficient transformer for long sequence time-series forecasting," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 12, pp. 11106-11115, 2021.
9. H. Wu et al., "Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting," in Advances in Neural Information Processing Systems 34, 2021, pp. 22419-22430.
10. N. Kitaev, L. Kaiser, and A. Levskaya, "Reformer: The efficient transformer," in International Conference on Learning Representations, 2020.
11. Y. Chen, N. Céspedes, and P. Barnaghi, "A closer look at transformers for time series forecasting: Understanding why they work and where they struggle," in Proceedings of the 42nd International Conference on Machine Learning, vol. 267, PMLR, 2025, pp. 7763-7780.
12. P. Zhao, H. Zhu, W. S. H. Ng, and D. L. Lee, "From GARCH to neural network for volatility forecast," in Proceedings of the AAAl Conference on Artificial Intelligence, vol. 38, no. 15, 2024, pp. 16998-17006.
13. P. H. Kupiec, "Techniques for verifying the accuracy of risk measurement models," The Journal of Derivatives, vol. 3, no. 2, pp. 73-84. 1995.
14. P. F. Christoffersen, "Evaluating interval forecasts," International Economic Review, vol. 39, no. 4, pp. 841-862, 1998.
15. P. F. Christoffersen and F. X. Diebold, "How relevant is volatility forecasting for financial risk management?" The Review of Economics and Statistics, vol. 82, no. 1, pp. 12-22, 2000.
16. M. Fikri, "The impact of market volatility regimes on gold price prediction accuracy: A VIX-based machine learning approach," Datokarama Journal of Information Technology, vol. 1, no. 2, pp. 29-44, 2025.
17. X. Liu et al., "Moirai-MoE: Empowering time series foundation models with sparse mixture of experts," in Proceedings of the 42nd International Conference on Machine Learning, vol. 267, PMLR, 2025, pp. 38940-38962.