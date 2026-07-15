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