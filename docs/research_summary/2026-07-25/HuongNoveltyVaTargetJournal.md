# Hướng Novelty và Target Journal từ kết quả 25/07/2026

Tài liệu này chỉ dựa trên các kết quả trong thư mục:

```text
docs/research_summary/2026-07-25/
```

Không dùng narrative cũ trong `docs/main/paper.md`.

## 1. Kết luận nhanh

Kết quả hiện tại có thể phát triển thành bài báo theo hướng **risk-sensitive model selection for volatility forecasting**. Điểm có giá trị nhất không phải là "một model mới thắng tất cả", mà là:

```text
Accuracy tốt không đủ để chọn volatility model.
Cần có lớp đánh giá risk calibration và dynamic tracking trước khi xếp hạng.
Khi ưu tiên risk, GARCH-Autoformer nổi lên như ứng viên mạnh nhất.
Khi cân bằng accuracy-risk, Moirai/MoiraiVaR và GARCH-Autoformer cùng nằm trong nhóm đầu.
```

Rank public hiện tại:

| Trạng thái | Mức có thể hướng |
|---|---|
| Chỉ public 3 docs ngày 25/07 như technical report | Workshop / preprint nội bộ / repo note |
| Viết lại thành paper ngắn, có ablation sanity gate | Q2-Q1 ứng dụng, tùy journal |
| Có thêm block-level robustness, ablation weight, ablation threshold, statistical test theo dataset-horizon | Q1 khá cạnh tranh |
| Thêm model/architecture novelty thật sự, có theory hoặc training objective mới rõ ràng | Q1 mạnh hơn, có thể nhắm finance/AI journal tốt |

## 2. Các hướng novelty có thể dẫn đến

### Novelty A: Risk-sensitive MCDM framework cho volatility model selection

Nội dung novelty:

```text
Đề xuất framework chọn model volatility bằng MCDM, kết hợp:
- forecast accuracy: MSE, MAE, QLIKE
- dynamic tracking: volatility_std_ratio_error, tracking_correlation_error
- risk calibration: VaR 1%, VaR 5%, pass rate, absolute violation error
- hai kịch bản preference: 5,5 và 3,7
```

Claim nên dùng:

```text
Một volatility model không nên được chọn chỉ bằng forecast error.
MCDM giúp chuyển bài toán từ "model nào dự báo sát nhất" sang "model nào phù hợp mục tiêu risk deployment nhất".
```

Claim không nên dùng quá mạnh:

```text
Framework này là tối ưu hoặc universal.
```

Cần bổ sung để public tốt:

```text
1. Sensitivity với trọng số MCDM: 50/50, 40/60, 30/70, 20/80.
2. Rank stability: Spearman/Kendall giữa các weight scenario.
3. Block-level ranking theo dataset-horizon, không chỉ aggregate model score.
4. Mô tả rõ vì sao SAW và TOPSIS được chọn.
```

Target journal:

| Độ hoàn thiện | Target |
|---|---|
| Hiện tại, viết gọn thành application paper | Forecasting, Journal of Risk and Financial Management, Data in Brief / MethodsX nếu tập trung pipeline |
| Có robustness và ablation weight đầy đủ | Finance Research Letters, Financial Innovation |
| Có framework decision-support rất chặt, nhấn mạnh MCDM/model selection | Decision Support Systems, Expert Systems with Applications |

Rank khả thi:

```text
B/B+ nếu chỉ là MCDM framework ứng dụng.
A- nếu có robustness tốt và viết thành decision-support paper.
```

### Novelty B: Forecast sanity gate để loại volatility forecast suy biến

Nội dung novelty:

```text
Trước khi đưa model vào MCDM, dùng gate:
- volatility_std_ratio_error <= 0.9
- tracking_correlation > 0

Mục tiêu là loại model có forecast gần như đường phẳng hoặc không đồng biến với true volatility.
```

Điểm mạnh từ kết quả hiện tại:

```text
Một số Transformer tier lớn bị loại vì forecast phẳng hoặc tracking correlation không dương.
Điều này làm ranking đáng tin hơn so với chỉ dùng MSE/MAE/QLIKE/VaR pass rate.
```

Claim nên dùng:

```text
Sanity gate là điều kiện tối thiểu để model được xem là volatility forecaster hợp lệ trong bài toán risk-sensitive selection.
```

Claim không nên dùng:

```text
Model qua gate là model tốt.
Ngưỡng 0.9 là ngưỡng tối ưu.
```

Cần bổ sung:

```text
1. Ablation threshold: 0.8, 0.85, 0.9, 0.95.
2. Báo cáo model nào bị loại/được giữ theo từng threshold.
3. Plot true vs pred cho các model bị loại và top model.
4. Kiểm tra threshold theo từng dataset-horizon thay vì chỉ aggregate model.
```

Target journal:

| Độ hoàn thiện | Target |
|---|---|
| Chỉ thêm như diagnostic trong framework | Finance Research Letters, Forecasting |
| Viết thành methodological diagnostic cho time-series forecast degeneration | Expert Systems with Applications, Applied Soft Computing |
| Có theory/robustness rộng hơn trên nhiều domain time series | International Journal of Forecasting khó hơn nhưng có thể cân nhắc |

Rank khả thi:

```text
B nếu chỉ là heuristic gate.
A- nếu chứng minh gate ổn định, có ablation và case visualization.
```

### Novelty C: Risk-sensitive dominance của GARCH-Autoformer

Nội dung novelty:

```text
Trong scenario 3,7, GARCH-Autoformer Tier 2 đứng rank 1 theo cả SAW và TOPSIS.
Khi aggregate theo model family, GARCH-Autoformer thắng 15/15 model family còn lại theo SAW và TOPSIS.
```

Claim nên dùng:

```text
GARCH-Autoformer là ứng viên mạnh nhất khi objective ưu tiên risk calibration.
Kết quả cho thấy hybrid hóa GARCH inductive bias với Autoformer có lợi thế trong setting risk-sensitive.
```

Claim cần tránh:

```text
GARCH-Autoformer thống kê vượt trội tuyệt đối trên mọi điều kiện.
GARCH-Autoformer là SOTA chung cho volatility forecasting.
```

Lý do cần thận trọng:

```text
Dominance test hiện tại so sánh score aggregate giữa model family.
Đây là bằng chứng hỗ trợ, nhưng chưa mạnh bằng block-level test theo dataset-horizon.
```

Cần bổ sung:

```text
1. Block-level SAW/TOPSIS cho từng dataset-horizon.
2. Friedman/Nemenyi trên rank block-level.
3. Wilcoxon signed-rank hoặc sign test giữa GARCH-Autoformer và từng model family trên 45 block.
4. Report theo market group: Vietnam, Asia, Europe, US.
```

Target journal:

| Độ hoàn thiện | Target |
|---|---|
| Claim thực nghiệm gọn, paper ngắn | Finance Research Letters |
| Có block-level statistical evidence và financial interpretation | International Review of Financial Analysis, Financial Innovation |
| Có architecture/loss mới của GARCH-Autoformer, không chỉ evaluation | Expert Systems with Applications, Applied Soft Computing |

Rank khả thi:

```text
B+ với kết quả hiện tại nếu viết đúng mức.
A- nếu có block-level significance và ablation.
A nếu biến GARCH-Autoformer thành architecture contribution rõ ràng.
```

### Novelty D: Accuracy-risk trade-off map cho các model family

Nội dung novelty:

```text
Kết quả 5,5 cho thấy Moirai/MoiraiVaR có accuracy score rất mạnh.
Kết quả 3,7 cho thấy GARCH-Autoformer vượt lên khi risk weight tăng.
Do đó ranking model phụ thuộc vào deployment preference.
```

Claim nên dùng:

```text
Không có một ranking duy nhất nếu objective thay đổi.
Moirai-family phù hợp hơn khi cần accuracy, GARCH-Autoformer phù hợp hơn khi cần risk calibration.
```

Cần bổ sung:

```text
1. Pareto frontier accuracy-risk.
2. Trade-off curve khi risk weight chạy từ 0 đến 1.
3. Stability region: model nào rank 1 trong khoảng weight nào.
4. Case study: nếu nhà quản trị rủi ro ưu tiên VaR 70%, ranking đổi như thế nào.
```

Target journal:

| Độ hoàn thiện | Target |
|---|---|
| Short empirical note | Finance Research Letters |
| Decision preference/model selection paper | Decision Support Systems |
| Forecasting evaluation paper có Pareto/stability analysis | International Journal of Forecasting, nếu novelty method đủ mạnh |

Rank khả thi:

```text
B nếu là empirical trade-off.
A- nếu có preference-stability methodology rõ.
```

### Novelty E: Stationarity-aware interpretation of volatility forecasting results

Nội dung novelty:

```text
ADF/KPSS cho thấy full sample chỉ 15/45 chuỗi dataset-horizon stationary.
Val/test có nhiều non-stationary hơn train.
Vì vậy ranking phải được đọc trong điều kiện volatility có regime/local non-stationarity.
```

Claim nên dùng:

```text
Stationarity diagnostic giải thích vì sao chỉ dùng forecast error hoặc calibration aggregate có thể gây lệch.
```

Claim không nên dùng:

```text
Stationarity analysis chứng minh model nào tốt hơn.
```

Cần bổ sung:

```text
1. Gắn performance với stationarity class: stationary/mixed/non_stationary.
2. Kiểm tra model nào bền hơn trong non-stationary blocks.
3. Thêm regime split hoặc volatility state split.
4. Nếu đủ, dùng rolling stationarity/regime diagnostics thay cho split 70/15/15 suy ra.
```

Target journal:

| Độ hoàn thiện | Target |
|---|---|
| Bổ trợ cho framework chính | Nên là subsection, không nên là paper riêng |
| Nếu có regime-aware evaluation đầy đủ | Journal of Forecasting, Quantitative Finance, Financial Innovation |
| Nếu có market/regime financial interpretation mạnh | International Review of Financial Analysis |

Rank khả thi:

```text
B- nếu dùng riêng.
B+/A- nếu gắn với regime-aware risk-sensitive evaluation.
```

### Novelty F: MoiraiVaR như một hướng foundation-model risk alignment

Nội dung novelty:

```text
MoiraiVaR với lambda=0.2 đạt top cao trong scenario 5,5 nhờ accuracy score mạnh.
Nhưng khi risk weight tăng, GARCH-Autoformer vượt lên.
Đây là một kết quả hay: foundation model alignment có thể cải thiện forecast, nhưng risk calibration vẫn là thách thức.
```

Claim nên dùng:

```text
Risk-aware fine-tuning/alignment cho foundation time-series model là hướng tiềm năng, nhưng cần calibration tốt hơn để thắng trong risk-sensitive deployment.
```

Cần bổ sung:

```text
1. Ablation lambda: 0.0, 0.1, 0.2, 0.5.
2. Head-only vs full fine-tune.
3. Calibration plots cho VaR 1% và 5%.
4. So sánh Moirai, Moirai 2, Moirai-MoE, MoiraiVaR trên từng horizon.
```

Target journal:

| Độ hoàn thiện | Target |
|---|---|
| Nếu chỉ là empirical result | Finance Research Letters, Forecasting |
| Nếu có loss/alignment method rõ | Expert Systems with Applications, Applied Soft Computing |
| Nếu có foundation-model finance story mạnh | Financial Innovation |

Rank khả thi:

```text
B hiện tại.
A- nếu có ablation lambda và fine-tuning mode đầy đủ.
```

## 3. Nên chọn hướng nào để viết paper?

Thứ tự ưu tiên để public có xác suất cao:

| Ưu tiên | Hướng | Lý do |
|---:|---|---|
| 1 | Novelty A + B + D | Framework rõ, gắn trực tiếp với kết quả hiện có, ít phụ thuộc vào claim model mới |
| 2 | Novelty C + E | Mạnh nếu bổ sung block-level test và regime/stationarity analysis |
| 3 | Novelty F | Hay nhưng cần thêm experiment MoiraiVaR ablation |

Hướng paper nên đặt title theo kiểu:

```text
Risk-Sensitive Multi-Criteria Selection of Volatility Forecasting Models
with Dynamic Tracking Diagnostics
```

Thông điệp chính:

```text
Chúng tôi không chỉ hỏi model nào có forecast error thấp.
Chúng tôi hỏi model nào đủ khả năng tracking volatility và tạo risk signal đúng hơn dưới mức ưu tiên risk khác nhau.
```

## 4. Journal target theo mức tham vọng

### Mức B: dễ public hơn, ít đòi hỏi novelty lý thuyết

| Journal | Rank tham khảo | Fit |
|---|---|---|
| Forecasting | SJR 2025 khoảng 0.693 theo SCImago | Hợp với evaluation/forecasting application |
| Journal of Risk and Financial Management | Q2/Q3 tùy category/năm, cần check lại trước submit | Hợp với VaR/risk-management application |
| MethodsX | Methods note | Hợp nếu đóng góp là reproducible evaluation pipeline |

### Mức B+/A-: target thực tế nếu bổ sung robustness

| Journal | Rank tham khảo | Fit |
|---|---|---|
| Finance Research Letters | JCR Q1, IF 2026 7.1 theo Journal Metrics; SJR 2025 1.711 theo Resurchify | Phù hợp paper ngắn về finance/risk/volatility |
| Financial Innovation | Q1 theo các nguồn journal ranking 2026 | Phù hợp nếu nhấn mạnh AI/foundation model/risk calibration |
| Journal of Forecasting | Journal forecasting chuyên ngành; cần viết method/evaluation chặt hơn | Phù hợp nếu tập trung forecast evaluation và robustness |

### Mức A-/A: khó hơn, cần novelty và validation mạnh

| Journal | Rank tham khảo | Fit |
|---|---|---|
| Expert Systems with Applications | JCR Q1, IF 2026 9.4 theo Journal Metrics; SCImago Q1 | Phù hợp nếu framework được viết như AI decision/evaluation system |
| Decision Support Systems | Q1 theo SCImago; các source journal metric ghi IF 2026 quanh 7.5 | Phù hợp nếu đóng góp là decision-support framework cho model selection |
| Applied Soft Computing | JCR Q1, IF 2026 7.8 theo Journal Metrics | Phù hợp nếu có soft-computing/ML method, ablation và benchmark mạnh |
| International Review of Financial Analysis | JCR Q1, IF 2026 10.2 theo Journal Metrics | Khó hơn; cần financial insight mạnh, không chỉ ML ranking |
| International Journal of Forecasting | SJR 2025 2.272 theo SCImago | Rất khó; cần đóng góp forecasting methodology/evaluation có tính tổng quát |

## 5. Bản đồ novelty -> journal

| Novelty | Nếu chỉ dùng kết quả hiện tại | Nếu bổ sung tối thiểu | Target tốt nhất nên hướng |
|---|---|---|---|
| A. Risk-sensitive MCDM framework | B | B+/A- | Decision Support Systems, Expert Systems with Applications |
| B. Forecast sanity gate | B | A- | Expert Systems with Applications, Applied Soft Computing |
| C. GARCH-Autoformer risk dominance | B+ | A- | Finance Research Letters, International Review of Financial Analysis |
| D. Accuracy-risk trade-off map | B | A- | Decision Support Systems, International Journal of Forecasting |
| E. Stationarity-aware interpretation | B- | B+/A- | Journal of Forecasting, Financial Innovation |
| F. MoiraiVaR risk alignment | B | A- | Financial Innovation, Expert Systems with Applications |

## 6. Việc cần làm tiếp để nâng rank

Checklist ngắn:

```text
1. Viết lại một narrative duy nhất quanh risk-sensitive MCDM.
2. Thêm threshold ablation cho sanity gate.
3. Thêm weight sensitivity từ risk weight 0 đến 1.
4. Thêm block-level SAW/TOPSIS theo 45 dataset-horizon blocks.
5. Chạy Wilcoxon/sign test/Friedman-Nemenyi trên block-level rank.
6. Thêm plot Pareto accuracy-risk.
7. Thêm analysis performance theo stationarity class.
8. Nếu theo hướng MoiraiVaR, thêm lambda ablation và head/full fine-tune.
```

Nếu chỉ làm 1 việc để tăng rank nhanh nhất:

```text
Làm block-level robustness + weight sensitivity.
```

Nếu làm 2 việc:

```text
Thêm threshold ablation cho sanity gate.
```

Nếu làm 3 việc:

```text
Thêm Pareto frontier và stability region cho model ranking.
```

## 7. Nguồn rank/journal tham khảo

Đã check ngày 25/07/2026:

```text
- Expert Systems with Applications: SCImago Q1; Journal Metrics ghi JCR Q1, IF 2026 9.4.
- Finance Research Letters: Journal Metrics ghi JCR Q1, IF 2026 7.1; Resurchify ghi SJR 2025 1.711.
- International Review of Financial Analysis: Journal Metrics ghi JCR Q1, IF 2026 10.2.
- International Journal of Forecasting: SCImago ghi SJR 2025 2.272.
- Applied Soft Computing: Journal Metrics ghi JCR Q1, IF 2026 7.8.
- Decision Support Systems: SCImago Q1; các source journal metric ghi IF 2026 quanh 7.5.
- Forecasting: SCImago ghi SJR 2025 0.693.
```

Cần check lại trên website journal/Clarivate/Scopus trước khi submit vì quartile và IF thay đổi theo năm.

