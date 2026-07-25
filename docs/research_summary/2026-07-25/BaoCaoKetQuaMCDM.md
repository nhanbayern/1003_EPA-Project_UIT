# Báo cáo Nghiên cứu: Kết quả MCDM và Kiểm định Dominance

**Ngày thực hiện:** 25/07/2026  
**Chủ đề:** Phân tích kết quả xếp hạng MCDM sau khi bổ sung dynamic-tracking metrics, forecast sanity gate và kiểm định dominance cho `GARCH-Autoformer`.

---

## Tóm tắt

Tài liệu này tổng hợp kết quả từ lần chạy MCDM mới nhất:

```text
output/mcdm_results/MCDM20260725170414/
```

Script liên quan:

```text
stats_analysis/run_mcdm_evaluation.py
```

Hai cấu hình được chạy:

```text
5,5: Accuracy 50%, Risk 50%
3,7: Accuracy 30%, Risk 70%
```

Các CSV vẫn giữ chi tiết theo `branch/tier/model`, còn các ảnh phân tích và dominance test được tổng hợp theo `display_group`, tức trung bình qua các tier của cùng một model family.

## 1. Các mô hình bị loại bởi forecast sanity gate

Pipeline loại các mô hình không đạt điều kiện forecast dynamics tối thiểu:

```text
volatility_std_ratio_error <= 0.9
tracking_correlation > 0
```

Danh sách bị loại giống nhau trong cả hai cấu hình `5,5` và `3,7`.

| Model | Std-ratio error | Tracking correlation | Lý do loại |
|---|---:|---:|---|
| Reformer (Tier 3) | 1.000000 | -0.002470 | Dự báo gần như phẳng và tương quan tracking không dương |
| Informer (Tier 1) | 0.648742 | -0.004056 | Tương quan tracking không dương |
| Autoformer (Tier 3) | 0.919195 | 0.006766 | Std-ratio error vượt ngưỡng 0.9 |
| Vanilla (Tier 3) | 1.000000 | -0.000124 | Dự báo gần như phẳng và tương quan tracking không dương |
| Informer (Tier 3) | 1.000000 | -0.001735 | Dự báo gần như phẳng và tương quan tracking không dương |

Nhận xét khoa học:

```text
Các mô hình bị loại chủ yếu thuộc nhóm Transformer có tier lớn. Một số mô hình tạo forecast gần như constant trong từng dataset-horizon. Dù các mô hình này có thể đạt VaR calibration tốt ở một số case, chúng không thể hiện khả năng tracking volatility dynamics. Vì vậy việc loại khỏi MCDM ranking là hợp lý để tránh chọn mô hình có VaR pass rate tốt nhưng forecast không có giá trị động học.
```

## 2. Kết quả cấu hình 5,5

Cấu hình:

```text
Accuracy block = 50%
Risk block = 50%
```

Top kết quả combined MCDM:

| Rank | Model | Accuracy score | Risk score | SAW rank | TOPSIS rank | SAW score | TOPSIS score |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | MoiraiVaR - Moirai 2 (lambda=0.2) | 0.496282 | 0.180438 | 1 | 3 | 0.676720 | 0.589468 |
| 1 | GARCH-Autoformer (Tier 2) | 0.223168 | 0.443182 | 3 | 1 | 0.666350 | 0.669217 |
| 3 | Moirai-MoE | 0.491507 | 0.178509 | 2 | 4 | 0.670016 | 0.586912 |
| 4 | GARCH-Autoformer (Tier 1) | 0.253919 | 0.371223 | 6 | 2 | 0.625142 | 0.640853 |
| 5 | MoiraiVaR - Moirai-MoE (lambda=0.2) | 0.490744 | 0.163560 | 4 | 5 | 0.654304 | 0.572443 |
| 6 | Moirai 2 | 0.500000 | 0.150040 | 5 | 6 | 0.650040 | 0.556233 |

Nhận xét:

```text
Khi accuracy và risk được cân bằng 50/50, nhóm Moirai và MoiraiVaR giữ vị trí rất cao nhờ accuracy score vượt trội. Tuy nhiên, GARCH-Autoformer vẫn cạnh tranh trực tiếp ở top đầu nhờ risk score cao hơn đáng kể. Điều này cho thấy GARCH-Autoformer là mô hình cân bằng hơn giữa tracking/risk, trong khi Moirai-family nổi bật ở forecast accuracy.
```

Ở cấu hình 5,5, `GARCH-Autoformer (Tier 2)` đạt TOPSIS rank 1. Điều này cho thấy khi xét khoảng cách đến nghiệm lý tưởng đa tiêu chí, mô hình này gần ideal solution nhất, dù SAW rank không phải cao nhất.

## 3. Kết quả cấu hình 3,7

Cấu hình:

```text
Accuracy block = 30%
Risk block = 70%
```

Top kết quả combined MCDM:

| Rank | Model | Accuracy score | Risk score | SAW rank | TOPSIS rank | SAW score | TOPSIS score |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | GARCH-Autoformer (Tier 2) | 0.133901 | 0.620455 | 1 | 1 | 0.754355 | 0.740389 |
| 2 | Reformer (Tier 2) | 0.026559 | 0.561948 | 3 | 2 | 0.588507 | 0.694979 |
| 3 | GARCH-Autoformer (Tier 1) | 0.152351 | 0.519713 | 2 | 5 | 0.672064 | 0.673411 |
| 4 | Autoformer (Tier 2) | 0.033116 | 0.537603 | 5 | 3 | 0.570720 | 0.678577 |
| 4 | Wavelet-Autoformer (Tier 2) | 0.040504 | 0.532792 | 4 | 4 | 0.573296 | 0.675658 |
| 6 | Autoformer (Tier 1) | 0.039716 | 0.494650 | 8 | 6 | 0.534366 | 0.657734 |

Nhận xét:

```text
Khi tăng trọng số risk lên 70%, GARCH-Autoformer trở thành mô hình dẫn đầu rõ ràng. Tier 2 đứng hạng 1 ở cả SAW và TOPSIS, trong khi Tier 1 vẫn nằm trong nhóm top 3 combined. Điều này cho thấy ưu thế của GARCH-Autoformer không chỉ đến từ một phương pháp ranking riêng lẻ mà xuất hiện nhất quán trên cả SAW và TOPSIS.
```

So với cấu hình 5,5, nhóm Moirai-family giảm thứ hạng vì risk score thấp hơn. Điều này phản ánh trade-off chính của nghiên cứu:

```text
Forecast accuracy tốt không tự động đảm bảo risk calibration tốt.
```

## 4. Kiểm định dominance của GARCH-Autoformer

Dominance test so sánh `GARCH-Autoformer` với các model family còn lại theo:

```text
SAW Composite Score
TOPSIS Closeness Coefficient
```

H1 được định nghĩa:

```text
H1 = GARCH-Autoformer có score cao hơn model còn lại
```

Kiểm định chính:

```text
Exact binomial test, one-sided
H0: P(GARCH-Autoformer thắng một model còn lại) = 0.5
H1: P(GARCH-Autoformer thắng một model còn lại) > 0.5
```

Z-test được báo thêm như xấp xỉ:

```text
One-proportion z-test
H0: p = 0.5
H1: p > 0.5
```

### Kết quả dominance ở cấu hình 5,5

| Metric | H1 / Total | H1 rate | Exact binomial p-value | z statistic | z-test p-value | Mean relative difference |
|---|---:|---:|---:|---:|---:|---:|
| SAW Composite Score | 11 / 15 | 73.33% | 0.059235 | 1.807392 | 0.035351 | +51.62% |
| TOPSIS Closeness Coefficient | 15 / 15 | 100.00% | 0.000031 | 3.872983 | 0.000054 | +43.52% |

Nhận xét:

```text
Theo TOPSIS, GARCH-Autoformer thắng toàn bộ 15 model family còn lại và exact binomial test bác bỏ H0 rất mạnh. Theo SAW, GARCH-Autoformer thắng 11/15 model, tương đương 73.33%. Exact binomial p-value = 0.059235, sát ngưỡng 5% nhưng chưa đủ để bác bỏ H0 nếu dùng exact test nghiêm ngặt. Z-test cho p-value = 0.035351, nhưng đây chỉ là xấp xỉ và kém ưu tiên hơn exact binomial test khi n nhỏ.
```

Diễn giải khoa học:

```text
Ở cấu hình cân bằng 5,5, bằng chứng thống kê cho ưu thế của GARCH-Autoformer mạnh hơn theo TOPSIS so với SAW. Điều này phù hợp với bản chất TOPSIS: mô hình có cấu hình cân bằng giữa accuracy và risk sẽ gần điểm lý tưởng hơn.
```

### Kết quả dominance ở cấu hình 3,7

| Metric | H1 / Total | H1 rate | Exact binomial p-value | z statistic | z-test p-value | Mean relative difference |
|---|---:|---:|---:|---:|---:|---:|
| SAW Composite Score | 15 / 15 | 100.00% | 0.000031 | 3.872983 | 0.000054 | +96.70% |
| TOPSIS Closeness Coefficient | 15 / 15 | 100.00% | 0.000031 | 3.872983 | 0.000054 | +102.18% |

Nhận xét:

```text
Ở cấu hình risk-sensitive 3,7, GARCH-Autoformer thắng toàn bộ 15 model family còn lại theo cả SAW và TOPSIS. Exact binomial p-value = 0.000031 cho cả hai metric, bác bỏ mạnh H0 rằng xác suất thắng chỉ là 50%. Điều này cung cấp bằng chứng thống kê rõ ràng cho ưu thế của GARCH-Autoformer trong bối cảnh ưu tiên quản trị rủi ro.
```

## 5. Nhận xét tổng hợp

Kết quả cho thấy ranking phụ thuộc mạnh vào mục tiêu triển khai:

```text
Nếu mục tiêu là cân bằng accuracy-risk, MoiraiVaR/Moirai-family và GARCH-Autoformer cùng nằm ở nhóm đầu.
Nếu mục tiêu là risk-sensitive deployment, GARCH-Autoformer vượt trội rõ ràng.
```

Các mô hình bị loại bởi forecast sanity gate cho thấy cần kiểm tra dynamic tracking trước khi tin vào VaR backtesting:

```text
Một forecast gần như đường phẳng có thể tạo violation rate nhìn ổn ở một số case, nhưng không nên được xem là mô hình dự báo volatility hợp lệ.
```

Do đó pipeline hiện tại có ba tầng đánh giá hợp lý hơn:

```text
1. Forecast sanity gate: loại mô hình không tracking volatility dynamics.
2. MCDM ranking: xếp hạng theo accuracy, dynamic tracking và risk calibration.
3. Dominance test: kiểm định tỷ lệ model bị GARCH-Autoformer vượt qua theo SAW/TOPSIS.
```

Kết luận chính:

```text
GARCH-Autoformer là mô hình có bằng chứng mạnh nhất khi ưu tiên risk calibration. Ở cấu hình 3,7, mô hình này thắng 100% các model family còn lại theo cả SAW và TOPSIS, với exact binomial p-value = 0.000031.
```
