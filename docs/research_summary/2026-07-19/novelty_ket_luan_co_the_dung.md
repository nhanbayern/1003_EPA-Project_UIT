# Novelty và kết luận có thể dùng

Tài liệu này tổng hợp các hướng novelty có thể viết từ kết quả hiện tại. Mục tiêu là định vị đóng góp một cách đúng mức: nghiên cứu hiện tại mạnh ở hướng empirical benchmark và evaluation framework, chưa nên claim là một phương pháp VaR hoàn toàn mới.

## 1. Novelty chính nên dùng

Novelty phù hợp nhất:

> Nghiên cứu này đánh giá liệu lợi thế point forecasting của foundation time-series models trong dự báo realized volatility có chuyển hóa sang tail-risk forecasting hay không, thông qua một framework hai tầng kết hợp forecast accuracy và VaR-based distributional evaluation.

Diễn đạt tiếng Anh:

> This study provides a systematic empirical evaluation of whether the point-forecasting advantage of foundation time-series models in realized volatility forecasting transfers to VaR-based tail-risk forecasting across markets, horizons, and distributional assumptions.

Điểm mới nằm ở framework đánh giá và bằng chứng thực nghiệm, không nằm ở việc đề xuất một kiến trúc model hoàn toàn mới.

## 2. Kết luận trung tâm

Kết luận nên nhấn mạnh:

> Volatility forecast accuracy và distributional risk forecasting quality là hai mục tiêu liên quan nhưng không thay thế cho nhau.

Bằng chứng:

- Moirai-family đứng đầu forecast accuracy theo MSE, MAE, QLIKE.
- Moirai-family cũng rất mạnh về quantile loss tổng thể.
- Tuy nhiên, VaR calibration/backtesting không luôn do Moirai dẫn đầu, đặc biệt ở VaR 1%.
- Ranking thay đổi khi chuyển từ Normal sang Student-t hoặc FHS.

Câu có thể dùng:

> Moirai-family models significantly outperform competing baselines in realized volatility point forecasting, but their advantage only partially transfers to VaR forecasting. This indicates that accurate volatility scale prediction is not sufficient for calibrated tail-risk forecasting.

## 3. Novelty 1: Framework đánh giá hai tầng

Đóng góp:

1. Tầng point forecast:
   - MSE
   - MAE
   - QLIKE
   - average forecast rank

2. Tầng distributional risk:
   - VaR 5%
   - VaR 1%
   - Normal VaR
   - Student-t VaR
   - FHS VaR
   - quantile loss
   - Kupiec test
   - Christoffersen independence test

Câu có thể dùng:

> We propose a two-layer evaluation protocol that separates realized volatility point forecasting from VaR-based distributional risk evaluation.

Mức độ novelty: khá tốt cho empirical/applied ML paper; không phải theoretical novelty.

## 4. Novelty 2: Kiểm tra forecast-to-risk transfer

Đóng góp:

Nghiên cứu không chỉ hỏi model nào dự báo volatility tốt nhất, mà hỏi:

> Model dự báo volatility tốt nhất có tạo VaR tốt nhất không?

Bằng chứng thống kê:

- Aggregate Spearman giữa forecast rank và average quantile loss:
  - VaR 5%: `rho = 0.568`, `p = 0.011`
  - VaR 1%: `rho = 0.768`, `p = 0.00012`

- Theo phương pháp VaR:
  - Normal: quan hệ rất mạnh.
  - FHS: quan hệ có ý nghĩa.
  - Student-t: quan hệ yếu hoặc không ổn định.

Câu có thể dùng:

> Forecast accuracy is statistically associated with VaR quantile loss in aggregate, but the relationship is not robust across distributional mappings, especially under Student-t VaR.

Kết luận:

> Lợi thế forecast của Moirai có chuyển hóa một phần sang VaR quantile loss, nhưng không đủ để đảm bảo calibration tốt trong mọi tail setting.

## 5. Novelty 3: Foundation models cho volatility forecasting trên nhiều thị trường

Đóng góp:

Moirai, Moirai2 và Moirai-MoE được benchmark trên:

- 9 thị trường.
- 5 horizons.
- Nhiều nhóm baseline: GARCH-family và Transformer-family.

Bằng chứng:

Forecast Friedman test:

| Metric | Friedman p |
|---|---:|
| MSE | 1.797e-98 |
| MAE | 7.459e-104 |
| QLIKE | 3.919e-119 |

Nemenyi post-hoc:

| Model | MSE significant vs non-Moirai | MAE significant vs non-Moirai | QLIKE significant vs non-Moirai |
|---|---:|---:|---:|
| Moirai2 | 15/16 | 15/16 | 14/16 |
| Moirai-MoE | 15/16 | 14/16 | 14/16 |
| Moirai | 14/16 | 13/16 | 13/16 |

Câu có thể dùng:

> The Moirai-family exhibits statistically significant superiority over most non-Moirai baselines in realized volatility forecasting.

## 6. Novelty 4: Robustness qua nhiều giả định VaR

Đóng góp:

Thay vì chỉ dùng một VaR mapping, nghiên cứu kiểm tra ba giả định:

- Normal: Gaussian tail.
- Student-t: parametric heavy-tail.
- FHS: empirical residual tail.

Bằng chứng:

| VaR case | Method | Violation rate | Abs violation error | Quantile loss |
|---|---|---:|---:|---:|
| 5% | Normal | 0.0590 | 0.0164 | 0.1413 |
| 5% | Student-t | 0.0320 | 0.0221 | 0.1472 |
| 5% | FHS | 0.0527 | 0.0040 | 0.1320 |
| 1% | Normal | 0.0274 | 0.0175 | 0.0529 |
| 1% | Student-t | 0.0060 | 0.0058 | 0.0491 |
| 1% | FHS | 0.0148 | 0.0048 | 0.0458 |

Câu có thể dùng:

> Model ranking is sensitive to the distributional assumption used to map volatility forecasts into VaR thresholds.

Kết luận:

> FHS cho calibration trung bình gần alpha nhất, trong khi Normal thường underestimate risk và Student-t thường bảo thủ hơn.

## 7. Novelty 5: Moirai mạnh tổng thể về quantile loss nhưng không thống trị calibration

Kết quả tổng thể quantile loss qua cả 6 setting:

| Overall rank | Model | Avg quantile-loss rank | Avg quantile loss |
|---:|---|---:|---:|
| 1 | Moirai | 4.83 | 0.091671 |
| 2 | Moirai-MoE | 5.67 | 0.091962 |
| 3 | GJR-GARCH | 6.83 | 0.092065 |
| 4 | Moirai2 | 7.00 | 0.092238 |

Kết luận:

> Xét tổng thể quantile loss qua cả VaR 5%, VaR 1% và ba phương pháp VaR, Moirai-family vẫn là nhóm mạnh nhất. Tuy nhiên, lợi thế này không đồng nhất theo tail level và không đồng nghĩa với pass rate cao nhất.

Câu có thể dùng:

> Moirai-family models achieve the best overall quantile-loss performance, yet they do not dominate VaR calibration metrics such as violation error and joint backtesting pass rate.

## 8. Novelty 6: Calibration không thể suy ra từ forecast accuracy

Calibration trong VaR nghĩa là violation rate phải gần alpha và violations không bị clustering.

Pipeline kiểm định:

```text
kupiec_lr -> kupiec_p -> kupiec_pass
lr_ind -> lr_ind_p -> independence_pass
kupiec_pass AND independence_pass -> backtest_pass
mean(backtest_pass) -> pass_rate
```

Bằng chứng:

- Moirai2 forecast rank #1 nhưng VaR 1% pass rate chỉ `0.133`.
- Reformer Tier 3 Large dẫn đầu VaR 1% pass rate với `0.378`.
- GARCH-family cạnh tranh mạnh hơn ở VaR 1% absolute violation error.

Câu có thể dùng:

> Forecast accuracy measures the quality of the volatility path, whereas VaR calibration evaluates the correctness of tail event probabilities. These objectives are empirically non-equivalent in our results.

## 9. Kiểm định thống kê dùng để bảo vệ novelty

| Mục tiêu | Kiểm định / chỉ số | Vai trò |
|---|---|---|
| Có khác biệt tổng thể giữa model không? | Friedman test | Kiểm tra khác biệt rank qua dataset-horizon blocks |
| Cặp model nào khác biệt? | Nemenyi post-hoc | Chứng minh Moirai khác biệt với đa số baseline |
| Quantile loss khác biệt theo thời gian không? | Diebold-Mariano test | Kiểm tra pairwise loss series |
| VaR có đúng tỷ lệ violation không? | Kupiec test | Kiểm tra unconditional coverage |
| Violation có độc lập không? | Christoffersen independence test | Kiểm tra clustering của violations |
| Forecast accuracy có đi cùng quantile loss không? | Spearman/Kendall rank correlation | Kiểm tra forecast-to-risk transfer |

## 10. Mức độ novelty

Đánh giá thẳng:

> Novelty hiện tại là vừa đến khá mạnh ở hướng empirical evaluation, chưa phải novelty rất mạnh về modeling hoặc theory.

Mạnh ở:

- Benchmark foundation time-series models cho realized volatility.
- Đánh giá đồng thời volatility forecast và VaR/tail-risk forecast.
- Kiểm tra robustness qua Normal, Student-t, FHS.
- Có hệ kiểm định thống kê đầy đủ.

Chưa mạnh ở:

- Chưa đề xuất architecture mới.
- Chưa train model trực tiếp bằng tail-risk objective.
- Chưa có calibration layer mới.
- VaR methods và tests đều là phương pháp chuẩn.

Vì vậy không nên claim:

> We propose a novel VaR forecasting model.

Nên claim:

> We propose a systematic two-layer evaluation framework and provide empirical evidence that volatility point-forecast superiority only partially transfers to VaR-based tail-risk forecasting.

## 11. Cách làm novelty mạnh hơn

Có thể mở rộng bằng experiment `experiments/moirai_var_aware/`:

```text
loss = volatility_loss + lambda_var * VaR_1% quantile_loss
```

Hướng này tạo thêm đóng góp modeling:

> Tail-aware Moirai head for joint volatility and VaR optimization.

Nếu chạy thêm thí nghiệm này, novelty có thể nâng từ empirical evaluation lên methodological extension.

Các thí nghiệm nên chạy:

| Config | lambda_var | Mục tiêu |
|---|---:|---|
| Baseline | 0.0 | MSE-only |
| Mild VaR-aware | 0.1 | Giữ forecast accuracy, thêm tail signal |
| Balanced | 0.2 | Trade-off chính |
| Strong VaR-aware | 0.5 | Ưu tiên VaR 1% |

Nếu loss mới cải thiện VaR 1% quantile loss hoặc pass rate mà không làm MSE/MAE/QLIKE suy giảm quá mạnh, đây sẽ là đóng góp mạnh hơn.

## 12. Kết luận dùng trong paper

Phiên bản tiếng Việt:

> Điểm mới của nghiên cứu là không dừng ở việc benchmark độ chính xác dự báo realized volatility, mà kiểm tra liệu lợi thế của foundation time-series models có chuyển hóa sang dự báo rủi ro tail hay không. Kết quả cho thấy Moirai-family vượt trội có ý nghĩa thống kê về MSE, MAE và QLIKE, đồng thời đạt quantile loss tổng thể tốt nhất. Tuy nhiên, lợi thế này không chuyển hóa hoàn toàn sang VaR calibration, đặc biệt ở VaR 1% và dưới các giả định phân phối khác nhau. Do đó, volatility forecast accuracy và VaR calibration là hai mục tiêu liên quan nhưng không thay thế cho nhau.

Phiên bản tiếng Anh:

> The novelty of this study lies in examining whether the point-forecasting advantage of foundation time-series models in realized volatility forecasting transfers to VaR-based tail-risk forecasting. The results show that Moirai-family models significantly outperform competing baselines in MSE, MAE and QLIKE, and achieve the best overall quantile-loss performance. However, this advantage only partially transfers to VaR calibration, especially at the 1% tail and under different distributional mappings. These findings demonstrate that volatility forecast accuracy and VaR calibration are related but non-equivalent objectives.
