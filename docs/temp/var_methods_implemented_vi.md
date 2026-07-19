# Các phương pháp VaR đã implement

Tài liệu này mô tả các phương pháp Value-at-Risk (VaR) đang được implement trong pipeline `stats_analysis`. Pipeline chỉ sử dụng file dự báo có sẵn:

```text
output/merged_all_predictions.csv
```

Không train lại mô hình volatility. Các giá trị VaR được tính từ `log_return` và `predict_volatility` trong file prediction.

## 1. Quy ước chung

Với mỗi dòng dữ liệu:

```text
dataset, branch, tier, model, time, horizon, log_return, predict_volatility
```

Pipeline sinh thêm các cột VaR và violation theo từng phương pháp:

```text
normal_var, normal_vio
student_t_var, student_t_vio
fhs_var, fhs_vio
```

Quy ước dấu:

```text
VaR_t = threshold âm của return
Violation xảy ra nếu log_return_t < VaR_t
```

Ví dụ:

```text
log_return_t = -3.0
VaR_t = -2.5

Vì -3.0 < -2.5 nên đây là một VaR violation.
```

Output được chia theo mức tail risk:

```text
output/stats_analysis/var_5pct/
output/stats_analysis/var_1pct/
```

Trong mỗi folder có file row-level:

```text
var_predictions.csv
```

## 2. Normal VaR

Normal VaR giả định standardized return tuân theo phân phối chuẩn.

Công thức:

```text
VaR_t^alpha = mu + predict_volatility_t * z_alpha
```

Trong đó:

```text
mu = mean return, hiện set bằng 0
predict_volatility_t = volatility do model dự báo
z_alpha = alpha-quantile của phân phối chuẩn N(0,1)
alpha = 0.05 hoặc 0.01
```

Ý nghĩa:

```text
Normal VaR dùng volatility forecast của model,
nhưng giả định phần tail của return là Gaussian.
```

Vai trò trong nghiên cứu:

```text
Baseline đơn giản để so sánh với Student-t và FHS.
```

Hạn chế:

```text
Phân phối chuẩn thường đánh giá thấp tail risk nếu dữ liệu heavy-tailed.
```

Paper/tài liệu tham khảo tương ứng:

```text
J.P. Morgan/Reuters (1996), RiskMetrics - Technical Document.
```

Lý do dùng reference này:

```text
RiskMetrics là tài liệu kinh điển cho variance-covariance/parametric VaR,
trong đó VaR được tính bằng volatility/covariance forecast kết hợp với
quantile của phân phối chuẩn.
```

## 3. Student-t VaR

Student-t VaR giả định standardized return tuân theo phân phối Student-t.

Công thức:

```text
VaR_t^alpha = mu + predict_volatility_t * t_alpha,nu
```

Trong đó:

```text
t_alpha,nu = alpha-quantile của Student-t với bậc tự do nu
nu = degree of freedom được estimate theo từng dataset
```

Pipeline estimate `nu` từ `log_return` theo từng `dataset`. Sau đó dùng cùng `nu` cho các model/horizon thuộc dataset đó.

Ý nghĩa:

```text
Student-t VaR vẫn dùng predict_volatility của model,
nhưng cho phép tail dày hơn Normal.
```

Vai trò trong nghiên cứu:

```text
Đây là phương pháp VaR chính hiện tại,
phù hợp với phân tích trong docs/data.md vì dữ liệu bị bác bỏ normality
và Student-t fit hợp lý hơn.
```

Paper tham khảo tương ứng:

```text
Bollerslev, T. (1987).
A Conditionally Heteroskedastic Time Series Model for Speculative Prices
and Rates of Return.
The Review of Economics and Statistics, 69(3), 542-547.
```

Lý do dùng reference này:

```text
Bollerslev (1987) là một reference nền tảng cho mô hình volatility
với conditional Student-t errors. Ý tưởng quan trọng là innovation/return
chuẩn hóa có thể có tail dày hơn Gaussian, phù hợp với VaR tail risk.
```

Reference bổ sung cho việc dùng Student-t trong dữ liệu tài chính:

```text
Blattberg, R. C., & Gonedes, N. J. (1974).
A Comparison of the Stable and Student Distributions as Statistical Models
for Stock Prices.
Journal of Business, 47(2), 244-280.
```

## 4. Filtered Historical Simulation VaR

Filtered Historical Simulation (FHS) là phương pháp bán tham số. Nó không giả định residual theo Normal hay Student-t, mà dùng empirical quantile của residual quá khứ.

Đầu tiên chuẩn hóa return bằng volatility forecast:

```text
residual_t = log_return_t / predict_volatility_t
```

Sau đó tại thời điểm `t`, lấy residual quá khứ trong rolling window:

```text
residual_{t-W}, ..., residual_{t-1}
```

với mặc định:

```text
W = 250
min_history = 250
```

Tính empirical quantile:

```text
q_alpha = Quantile_alpha(residual quá khứ)
```

Cuối cùng scale ngược lại:

```text
VaR_t^alpha = mu + predict_volatility_t * q_alpha
```

Điểm quan trọng:

```text
Pipeline dùng shift(1), nên VaR tại thời điểm t chỉ dùng residual quá khứ.
Không dùng residual của chính t hoặc dữ liệu tương lai.
```

Ý nghĩa:

```text
Nếu model dự báo volatility tốt, residual sau chuẩn hóa sẽ ổn định hơn.
Khi đó empirical quantile của residual sẽ tạo VaR tốt hơn.
```

Vai trò trong nghiên cứu:

```text
FHS đánh giá khả năng risk-control của model mà không ép residual theo một phân phối cố định.
```

Paper tham khảo tương ứng:

```text
Barone-Adesi, G., Giannopoulos, K., & Vosper, L. (1999).
VaR without correlations for nonlinear portfolios.
Journal of Futures Markets, 19(5), 583-602.
```

Lý do dùng reference này:

```text
Đây là reference gốc thường được trích cho Filtered Historical Simulation.
FHS chuẩn hóa return bằng volatility/risk forecast, lấy empirical distribution
của standardized residual, rồi scale lại theo volatility hiện tại để tính VaR.
```

Reference đánh giá/backtest FHS:

```text
Pritsker, M. (2001).
The Hidden Dangers of Historical Simulation.
Federal Reserve Board Finance and Economics Discussion Series.
```

Lý do bổ sung:

```text
Pritsker so sánh historical simulation và các biến thể như FHS,
phù hợp khi thảo luận ưu/nhược điểm của phương pháp simulation-based VaR.
```

## 5. EVT-filtered VaR

EVT-filtered VaR đã từng được thử nghiệm nhưng hiện không chạy trong pipeline mặc định.

Lý do bỏ:

```text
EVT rolling phải fit Generalized Pareto Distribution cho tail residual.
Nếu fit tại từng timestamp trên gần 1 triệu dòng,
thời gian chạy rất lâu và không phù hợp cho full pipeline mặc định.
```

Trạng thái hiện tại:

```text
Không sinh evt_var
Không sinh evt_vio
Không dùng EVT trong run_full_var_analysis.py
```

Nếu cần nghiên cứu EVT sau này, nên implement lại theo hướng:

```text
rolling no-leak + refit_frequency
```

Ví dụ:

```text
fit EVT mỗi 20 hoặc 60 ngày,
dùng lại tham số tail giữa các lần refit.
```

Paper tham khảo tương ứng:

```text
McNeil, A. J., & Frey, R. (2000).
Estimation of tail-related risk measures for heteroscedastic financial time series:
an extreme value approach.
Journal of Empirical Finance, 7(3-4), 271-300.
```

Lý do dùng reference này:

```text
McNeil và Frey (2000) là reference trực tiếp cho pipeline filtered EVT:
ước lượng volatility, chuẩn hóa residual, fit Extreme Value Theory cho tail,
rồi recover conditional VaR/Expected Shortfall.
```

Reference nền tảng EVT/POT:

```text
Balkema, A. A., & de Haan, L. (1974).
Residual Life Time at Great Age.
Annals of Probability, 2(5), 792-804.

Pickands, J. (1975).
Statistical Inference Using Extreme Order Statistics.
Annals of Statistics, 3(1), 119-131.
```

Lý do bổ sung:

```text
Hai paper này là nền tảng lý thuyết cho Peaks-over-Threshold và
Generalized Pareto Distribution, tức phần phân phối tail thường dùng trong EVT VaR.
```

## 6. Backtesting

Sau khi có VaR, pipeline kiểm tra violation:

```text
I_t = 1 nếu log_return_t < VaR_t
I_t = 0 nếu ngược lại
```

Các metric backtest hiện có:

```text
violation_rate
Kupiec test
Christoffersen independence test
backtest_pass
```

Các bảng aggregate vẫn giữ convention cũ để không phá báo cáo hiện tại:

```text
stats_by_dataset_horizon.csv
stats_by_model.csv
backtest_pass_cases.csv
stats_run_summary.csv
```

File `var_predictions.csv` là nơi xem chi tiết từng dòng VaR/violation theo từng phương pháp.

Paper tham khảo cho backtesting:

```text
Kupiec, P. H. (1995).
Techniques for Verifying the Accuracy of Risk Measurement Models.
Journal of Derivatives, 3(2), 73-84.

Christoffersen, P. F. (1998).
Evaluating Interval Forecasts.
International Economic Review, 39(4), 841-862.
```

Lý do dùng reference này:

```text
Kupiec test kiểm tra unconditional coverage, tức tỷ lệ violation có khớp
mức alpha kỳ vọng hay không. Christoffersen test bổ sung kiểm tra independence
của chuỗi violation, tức violation có bị clustering theo thời gian hay không.
```

## 7. Lệnh chạy

Chạy full VaR 5% và 1%:

```powershell
python stats_analysis\run_full_var_analysis.py
```

Output:

```text
output/stats_analysis/var_5pct/
output/stats_analysis/var_1pct/
```

Mỗi folder chứa:

```text
stats_by_dataset_horizon.csv
stats_by_model.csv
backtest_pass_cases.csv
stats_run_summary.csv
var_predictions.csv
```
