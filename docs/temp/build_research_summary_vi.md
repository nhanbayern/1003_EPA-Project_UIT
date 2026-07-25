# Build research summary

Tài liệu này mô tả script tổng hợp kết quả nghiên cứu từ các output VaR/forecast/statistical tests đã chạy.

Script chính:

```text
stats_analysis/build_research_summary.py
```

Lệnh chạy:

```powershell
python stats_analysis\build_research_summary.py
```

Output:

```text
output/stats_analysis/research_summary/
```

## 1. Mục tiêu

Các file raw hiện tại đã đủ dữ liệu nhưng quá chi tiết:

```text
output/stats_analysis/var_5pct/
output/stats_analysis/var_1pct/
output/stats_analysis/statistical_tests/
```

Script `build_research_summary.py` gom các kết quả này thành các bảng ngắn hơn để dùng cho phân tích và viết paper.

Script không train model, không tính lại prediction, và không xuất các bảng trung gian lớn. Các aggregate cần thiết từ `var_predictions.csv` được tính trong memory rồi chỉ xuất bảng summary cuối cùng.

## 2. Input

Script đọc các nhóm file sau:

```text
output/stats_analysis/var_5pct/stats_by_model.csv
output/stats_analysis/var_1pct/stats_by_model.csv
output/stats_analysis/var_5pct/var_predictions.csv
output/stats_analysis/var_1pct/var_predictions.csv
output/stats_analysis/statistical_tests/friedman_results.csv
output/stats_analysis/statistical_tests/nemenyi_pairwise_results.csv
output/stats_analysis/statistical_tests/dm_var_pairwise_results.csv
```

Yêu cầu trước khi chạy:

```powershell
python stats_analysis\run_full_var_analysis.py
python stats_analysis\run_statistical_tests.py
```

## 3. Output tables

### `forecast_ranking.csv`

Mục đích:

```text
Xếp hạng model theo forecast accuracy.
```

Metric chính:

```text
mse
mae
qlike
```

Rank:

```text
mse_rank
mae_rank
qlike_rank
avg_forecast_rank
avg_forecast_rank_rank
```

`mse_rank`, `mae_rank`, và `qlike_rank` được tính từ các metric trung bình ở cấp model trong `stats_by_model.csv`.

`avg_forecast_rank` là trung bình đều của ba rank metric:

```text
avg_forecast_rank = mean(mse_rank, mae_rank, qlike_rank)
```

`avg_forecast_rank_rank` là thứ hạng cuối cùng của `avg_forecast_rank`. Rank càng nhỏ càng tốt.

Lưu ý: `avg_forecast_rank` này chỉ là trung bình rank theo metric, không phải average rank theo từng dataset-horizon.

### `var_5pct_ranking.csv`

Mục đích:

```text
Xếp hạng model theo VaR 5% legacy backtest.
```

Metric chính:

```text
pass_rate
violation_rate
kupiec_lr
lr_ind
```

Rank:

```text
pass_rate_rank
violation_rate_rank
kupiec_lr_rank
lr_ind_rank
```

Với `pass_rate`, rank càng nhỏ nghĩa là pass rate càng cao.

### `var_1pct_ranking.csv`

Giống `var_5pct_ranking.csv`, nhưng dùng VaR 1%.

Mục đích:

```text
Kiểm tra robustness ở tail nghiêm ngặt hơn.
```

### `var_method_comparison.csv`

Mục đích:

```text
So sánh Normal VaR, Student-t VaR, và FHS VaR theo từng model.
```

Nguồn:

```text
var_predictions.csv
```

Các method:

```text
normal
student_t
fhs
```

Metric:

```text
violation_rate
abs_violation_error
quantile_loss
```

Trong đó:

```text
abs_violation_error = |violation_rate - alpha|
```

Ý nghĩa:

```text
abs_violation_error nhỏ hơn nghĩa là tỷ lệ vi phạm gần mức VaR kỳ vọng hơn.
quantile_loss nhỏ hơn nghĩa là VaR forecast tốt hơn theo quantile objective.
```

### `accuracy_risk_tradeoff.csv`

Mục đích:

```text
Kiểm tra model forecast tốt có đồng thời kiểm soát rủi ro tốt hay không.
```

Các cột chính:

```text
qlike_rank
mse_rank
mae_rank
pass_rate_rank
risk_accuracy_gap
```

Trong đó:

```text
risk_accuracy_gap = pass_rate_rank - qlike_rank
```

Diễn giải:

```text
risk_accuracy_gap > 0: model forecast rank tốt hơn risk rank.
risk_accuracy_gap < 0: model risk rank tốt hơn forecast rank.
```

### `friedman_summary.csv`

Mục đích:

```text
Tóm tắt Friedman test cho forecast metrics và VaR metrics.
```

Nguồn:

```text
output/stats_analysis/statistical_tests/friedman_results.csv
```

Cột quan trọng:

```text
test_family
var_case
var_method
metric
blocks
models
friedman_stat
friedman_p
significant_0.05
```

Diễn giải:

```text
significant_0.05 = True nghĩa là có khác biệt thống kê giữa các model
trên metric tương ứng.
```

### `nemenyi_significant_pairs.csv`

Mục đích:

```text
Chỉ giữ các cặp model khác biệt có ý nghĩa theo Nemenyi post-hoc.
```

Nguồn:

```text
output/stats_analysis/statistical_tests/nemenyi_pairwise_results.csv
```

Điều kiện lọc:

```text
nemenyi_p < 0.05
```

File này dùng để viết nhận định kiểu:

```text
Model A tốt hơn/kém hơn Model B với khác biệt có ý nghĩa thống kê.
```

### `dm_significant_pairs_summary.csv`

Mục đích:

```text
Tóm tắt Diebold-Mariano pairwise test theo dataset, horizon, và VaR method.
```

Nguồn:

```text
output/stats_analysis/statistical_tests/dm_var_pairwise_results.csv
```

Điều kiện lọc:

```text
dm_p < 0.05
```

Cột chính:

```text
significant_pairs
model_a_wins
model_b_wins
avg_abs_dm_stat
min_dm_p
```

Ý nghĩa:

```text
significant_pairs cho biết có bao nhiêu cặp model khác biệt có ý nghĩa
theo quantile loss VaR trong từng dataset/horizon/method.
```

## 4. Cách dùng trong paper

Luồng phân tích khuyến nghị:

```text
1. Dùng forecast_ranking.csv để xác định model forecast tốt nhất.
2. Dùng var_5pct_ranking.csv và var_1pct_ranking.csv để xác định model kiểm soát VaR tốt nhất.
3. Dùng var_method_comparison.csv để so sánh Normal vs Student-t vs FHS.
4. Dùng accuracy_risk_tradeoff.csv để phân tích trade-off giữa accuracy và risk-control.
5. Dùng friedman_summary.csv để kiểm tra khác biệt tổng thể.
6. Dùng nemenyi_significant_pairs.csv để xác định cặp model nào khác biệt có ý nghĩa.
7. Dùng dm_significant_pairs_summary.csv để kiểm tra pairwise predictive loss theo thời gian.
```

## 5. Lưu ý diễn giải

Các metric forecast:

```text
mse, mae, qlike càng nhỏ càng tốt.
```

Các metric VaR:

```text
abs_violation_error càng nhỏ càng tốt.
quantile_loss càng nhỏ càng tốt.
kupiec_lr càng nhỏ càng tốt.
lr_ind càng nhỏ càng tốt.
pass_rate càng cao càng tốt.
```

FHS có warmup:

```text
FHS cần 250 residual quá khứ, nên các dòng đầu mỗi group có fhs_var = NaN.
Các summary chỉ dùng dòng FHS hợp lệ.
```

Scale:

```text
log_return và predict_volatility đều đang cùng scale, tức đã nhân 100.
```
