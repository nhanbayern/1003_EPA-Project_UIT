# Methodology

This section details the data processing pipeline, the problem formulation, and the implementation specifics for the ten volatility forecasting models benchmarked in this study. The methodology ensures a unified evaluation framework where models from distinct paradigms—econometric, deep learning, hybrid, and foundation models—are trained and tested under strictly identical conditions.

## 1. Data and Problem Formulation

### 1.1. Data Source and Preprocessing
The study utilizes nine major stock market indices: DAX 40, Euronext 100, IBEX 35, KOSPI Index, Nikkei 225, SMI, S&P 500, VN30, and VN-Index. To maintain consistency, historical daily price and volume data for all nine indices from 2010 to 2025 were sourced exclusively from Investing.com. 

Daily percentage log returns are computed from closing prices as follows:
$$ r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) \times 100 $$
where $P_t$ is the closing price at day $t$. 

### 1.2. Forecasting Task and Causality
The forecasting task is formulated as a multi-horizon prediction problem. At any forecast origin $t$, the models are given a lookback window of 60 days of historical returns:
$$ x_t = (r_{t-60}, r_{t-59}, \dots, r_{t-1}) $$

The objective is to predict the realized volatility at five future horizons: $h \in \{1, 3, 5, 10, 21\}$. To ensure strict causality and prevent data leakage, the target realized volatility at horizon $h$ is defined as the 60-day rolling standard deviation ending exactly at $t+h-1$:
$$ \sigma_{t,h} = \sqrt{\frac{1}{60}\sum_{i=1}^{60}(r_{t+h-i} - \bar{r}_{t,h})^2} $$
where $\bar{r}_{t,h}$ is the sample mean of the returns in that 60-day window. Crucially, at prediction time $t$, the models have no access to any return data from time $t$ onward.

### 1.3. Distribution-Aware Data Splitting
Financial returns consistently exhibit heavy tails, leading to the rejection of the Gaussian distribution (Jarque-Bera p-value $\approx 0$). Consequently, this study adopts the Student-t distribution for risk modeling. The dataset is divided chronologically into training, validation, and test sets. To preserve the statistical properties of the series across splits, a distribution-aware algorithm determines the cut points by ensuring the estimated Student-t degrees of freedom ($\nu$) remain stable across the three subsets.

## 2. Model Implementations and Hyperparameters

The benchmark comprises three families of models: traditional econometric models, end-to-end deep learning architectures (Transformers and Hybrid), and pre-trained time-series foundation models (Moirai). 

### 2.1. Econometric Models (GARCH Family)
Econometric models explicitly capture volatility clustering and persistence by modeling conditional variance dynamics [1]. 
- **Models:** Standard GARCH(1,1) [1], GJR-GARCH(1,1) (incorporating asymmetric responses to negative shocks) [2], and FI-GARCH(1,d,1) (fractionally integrated for long-memory processes) [3].
- **Implementation:** Implemented using the `arch` Python package. To ensure convergence on long financial time series, a rolling history window of the most recent 1,000 observations is used to fit the models dynamically at each test step.
- **Assumptions & Outputs:** The models assume a Student-t distribution for the innovations. They are configured with a zero-mean equation (`mean="Zero"`). The multi-horizon conditional variance forecasts $\hat{\sigma}_{t+h}^2$ are generated recursively, and the final predicted volatility is extracted as $\sqrt{\hat{\sigma}_{t+h-1}^2}$ for each horizon $h$.

### 2.2. Transformer-Based Deep Learning Models
Transformer architectures leverage self-attention mechanisms [4] to capture long-range dependencies in time series without recurrent sequential processing.
- **Models:** Vanilla Transformer [4], Informer (utilizing ProbSparse attention for efficiency) [5], Autoformer (incorporating series decomposition and auto-correlation) [6], and Reformer (using locality-sensitive hashing) [7].
- **Configuration:** Models are configured with a sequence length of 60, a label length of 30, and an output prediction length of 21. The hidden dimension (`d_model`) is set to 128, with 4 encoder layers.
- **Training:** The models directly map the input return sequence to the multi-horizon volatility target. They are trained using a custom Student-t Negative Log-Likelihood (NLL) loss function, which inherently accommodates the heavy tails of financial returns. Optimization is performed using AdamW (Learning Rate = $10^{-3}$, Weight Decay = $10^{-4}$) for 30 epochs with a batch size of 128 and early stopping (patience = 5).

### 2.3. GARCH-LSTM Hybrid
The GARCH-LSTM hybrid [8] fuses the financial inductive bias of econometric models with the nonlinear representation capabilities of neural networks.
- **Architecture:** A custom recurrent cell (`GARCH_LSTM_Cell`) embeds the GARCH conditional variance update equation directly into the LSTM gating mechanism. It takes both the previous return and the previous variance as inputs to update the hidden state and compute the current variance.
- **Training and Forecasting:** The model is trained for 1-step-ahead prediction ($h=1$) using teacher forcing (feeding true historical variances during training). The loss function is the Student-t NLL. During inference, multi-horizon forecasts up to $h=21$ are generated autoregressively by feeding the predicted variance back into the cell, assuming expected future returns to be zero.

### 2.4. Moirai-Based Foundation Models
Moirai models are large-scale, pre-trained time-series foundation models developed by Salesforce [9], utilizing a masked encoder architecture supporting variable patch sizes.
- **Models:** Moirai 1.0, Moirai 2.0, and Moirai-MoE (Mixture of Experts). The `small` parameter variants are utilized.
- **Implementation:** Moirai's universal representation capability is leveraged via a feature extraction paradigm. The 60-day return sequences are padded to 64 days to align with a fixed patch size of 16 (yielding 4 patches per sequence). The pre-trained backbone is frozen, and its pooled representations are fed into a downstream Multi-Layer Perceptron (MLP) regression head.
- **Regression Head & Training:** The MLP consists of two linear layers (hidden dimension 256) with ReLU activation and Dropout (0.2), projecting the representations to the 5 target horizons. The MLP is fine-tuned using Mean Squared Error (MSE) loss, AdamW optimizer (LR = $10^{-3}$), batch size of 32 (train) / 64 (eval), and an early stopping patience of 7 epochs.

## 3. References

[1] T. Bollerslev, "Generalized autoregressive conditional heteroskedasticity," *Journal of Econometrics*, vol. 31, no. 3, pp. 307-327, 1986.
[2] L. R. Glosten, R. Jagannathan, and D. E. Runkle, "On the relation between the expected value and the volatility of the nominal excess return on stocks," *The Journal of Finance*, vol. 48, no. 5, pp. 1779-1801, 1993.
[3] R. T. Baillie, T. Bollerslev, and H. O. Mikkelsen, "Fractionally integrated generalized autoregressive conditional heteroskedasticity," *Journal of Econometrics*, vol. 74, no. 1. pp. 3-30, 1996.
[4] A. Vaswani et al., "Attention is all you need," in *Advances in Neural Information Processing Systems 30*, 2017, pp. 5998-6008.
[5] H. Zhou et al., "Informer: Beyond efficient transformer for long sequence time-series forecasting," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 35, no. 12, pp. 11106-11115, 2021.
[6] H. Wu et al., "Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting," in *Advances in Neural Information Processing Systems 34*, 2021, pp. 22419-22430.
[7] N. Kitaev, L. Kaiser, and A. Levskaya, "Reformer: The efficient transformer," in *International Conference on Learning Representations*, 2020.
[8] P. Zhao, H. Zhu, W. S. H. Ng, and D. L. Lee, "From GARCH to neural network for volatility forecast," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 38, no. 15, 2024, pp. 16998-17006.
[9] X. Liu et al., "Moirai-MoE: Empowering time series foundation models with sparse mixture of experts," in *Proceedings of the 42nd International Conference on Machine Learning*, vol. 267, PMLR, 2025, pp. 38940-38962.
