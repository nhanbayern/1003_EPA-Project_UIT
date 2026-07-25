# Diễn giải kết quả nghiên cứu

Tài liệu này diễn giải các bảng tổng hợp trong `output/stats_analysis/research_summary/`, được xây dựng từ hai bộ kết quả VaR 5% và VaR 1%. Trọng tâm của phân tích không phải là tìm một phương pháp VaR tốt nhất tuyệt đối, mà là kiểm tra liệu mô hình dự báo volatility chính xác theo point forecast có đồng thời dự báo tốt phân phối rủi ro ở phần đuôi hay không. Vì vậy, ba phương pháp VaR gồm Normal, Student-t và Filtered Historical Simulation (FHS) được sử dụng như ba góc nhìn robustness cho cùng một tập volatility forecasts.

## 1. Tổng quan

Kết quả thực nghiệm cho thấy sự tách biệt rõ giữa hai mục tiêu đánh giá. Nhóm Moirai đạt độ chính xác dự báo volatility tốt nhất theo các loss trung bình MSE, MAE và QLIKE. Tuy nhiên, khi các volatility forecasts này được chuyển thành VaR bằng các giả định phân phối khác nhau, các mô hình đứng đầu về backtesting và quantile loss không còn trùng ổn định với nhóm có forecast loss thấp nhất. Điều này củng cố luận điểm trung tâm của nghiên cứu: mô hình dự báo point volatility tốt chưa chắc là mô hình dự báo tốt conditional return distribution, đặc biệt ở tail.

## 2. Độ chính xác dự báo volatility

Bảng `forecast_ranking.csv` xếp hạng mô hình theo các metric dự báo trung bình ở cấp model. Vì MSE, MAE và QLIKE không phụ thuộc mức alpha của VaR, ranking forecast ở VaR 5% và VaR 1% là giống nhau. Ngoài các rank riêng `mse_rank`, `mae_rank` và `qlike_rank`, bảng hiện có thêm `avg_forecast_rank`, được tính bằng trung bình đều của ba rank metric này. Đây là average rank theo metric ở cấp aggregate, không phải average rank theo từng dataset-horizon.

| Rank tổng hợp | Model | MSE | MAE | QLIKE | Avg forecast rank |
|---:|---|---:|---:|---:|---:|
| 1 | Moirai2 | 0.022751 | 0.090622 | 0.009121 | 1.000 |
| 2 | Moirai-MoE | 0.024757 | 0.097559 | 0.009127 | 2.000 |
| 3 | Moirai | 0.042343 | 0.132846 | 0.014733 | 3.000 |
| 4 | FI-GARCH | 0.116380 | 0.236267 | 0.030465 | 4.000 |
| 5 | GARCH | 0.146167 | 0.266114 | 0.040636 | 5.000 |

Moirai2 đứng đầu đồng thời ở cả MSE, MAE và QLIKE, nên cũng đứng đầu theo `avg_forecast_rank`. Moirai-MoE xếp thứ hai với khoảng cách rất nhỏ so với Moirai2, đặc biệt ở QLIKE. Moirai gốc đứng thứ ba, trong khi FI-GARCH và GARCH là hai mô hình econometric tốt nhất trong nhóm còn lại. Kết quả này cho thấy các mô hình foundation-style Moirai có lợi thế rõ rệt trong bài toán point volatility forecasting của bộ dữ liệu hiện tại.

Một điểm cần diễn giải thận trọng là GARCH-LSTM-Hybrid có forecast error rất lớn, với MSE bằng 1,281,840.73, MAE bằng 68.90 và QLIKE bằng 0.35686. Do đó, GARCH-LSTM-Hybrid không phải là mô hình tốt về point forecast trong bộ kết quả này, dù nó lại có kết quả VaR backtesting cạnh tranh ở một số thiết lập. Đây là bằng chứng trực tiếp cho thấy forecast accuracy và distributional risk forecasting là hai tiêu chí khác nhau.

## 3. VaR 5% backtesting

Bảng `var_5pct_ranking.csv` cho thấy GARCH-LSTM-Hybrid đạt pass rate cao nhất ở VaR 5%, dù mô hình này đứng cuối theo QLIKE. Các mô hình Autoformer và Reformer cũng nằm trong nhóm có backtesting tốt ở mức tail 5%.

| Rank | Model | Pass rate | Violation rate | QLIKE |
|---:|---|---:|---:|---:|
| 1 | GARCH-LSTM-Hybrid | 0.2667 | 0.0340 | 0.3569 |
| 2 | Autoformer Tier 2 | 0.2444 | 0.0362 | 0.0980 |
| 3 | Autoformer Tier 1 | 0.2222 | 0.0344 | 0.0929 |
| 3 | Reformer Tier 3 | 0.2222 | 0.0409 | 0.1001 |
| 5 | Informer Tier 2 | 0.1778 | 0.0353 | 0.0850 |
| 5 | Reformer Tier 2 | 0.1778 | 0.0379 | 0.1003 |

Các model đứng đầu VaR 5% thường có violation rate thấp hơn mức kỳ vọng 5%. GARCH-LSTM-Hybrid có violation rate 3.40%, Autoformer Tier 2 là 3.62%, Autoformer Tier 1 là 3.44% và Reformer Tier 3 là 4.09%. Điều này cho thấy nhiều mô hình có xu hướng hơi bảo thủ ở mức tail 5%. Tuy nhiên, pass rate không chỉ phản ánh khoảng cách giữa violation rate và alpha, mà còn phụ thuộc vào Kupiec test và Christoffersen independence test. Vì vậy, một violation rate gần 5% chưa đủ để kết luận mô hình có chất lượng VaR tốt nếu chuỗi violation không độc lập hoặc coverage không vượt qua kiểm định.

## 4. VaR 1% backtesting

Ở VaR 1%, ranking risk-control thay đổi đáng kể. Reformer Tier 3 đạt pass rate cao nhất, trong khi GARCH-LSTM-Hybrid giảm xuống nhóm đồng hạng thứ năm. Điều này cho thấy mô hình tốt ở tail 5% không nhất thiết ổn định khi chuyển sang tail nghiêm ngặt hơn.

| Rank | Model | Pass rate | Violation rate | QLIKE |
|---:|---|---:|---:|---:|
| 1 | Reformer Tier 3 | 0.3778 | 0.0090 | 0.1001 |
| 2 | Autoformer Tier 1 | 0.3556 | 0.0061 | 0.0929 |
| 2 | Informer Tier 1 | 0.3556 | 0.0068 | 0.0972 |
| 4 | Reformer Tier 2 | 0.3333 | 0.0083 | 0.1003 |
| 5 | Autoformer Tier 2 | 0.3111 | 0.0072 | 0.0980 |
| 5 | GARCH-LSTM-Hybrid | 0.3111 | 0.0061 | 0.3569 |

Nhóm Transformer variants chiếm ưu thế rõ hơn ở VaR 1%. Reformer Tier 3 có violation rate 0.90%, gần mức kỳ vọng 1%, và đạt pass rate cao nhất. Trong khi đó, GARCH-LSTM-Hybrid đứng đầu ở VaR 5% nhưng không còn giữ vị trí dẫn đầu ở VaR 1%. Kết quả này nhấn mạnh rằng risk-control là một thuộc tính phụ thuộc alpha; không nên suy luận hiệu quả VaR 1% từ kết quả VaR 5%.

## 5. Ba phương pháp VaR như kiểm tra robustness phân phối

Bảng `var_method_comparison.csv` được sử dụng để kiểm tra độ nhạy của kết luận khi cùng một volatility forecast được chuyển thành tail quantile dưới các giả định phân phối khác nhau. Normal VaR đại diện cho giả định Gaussian tail, Student-t VaR cho phép heavy tail tham số hóa, còn FHS sử dụng phân phối residual thực nghiệm. Do đó, phần này không nhằm chọn một phương pháp VaR tốt nhất tuyệt đối, mà nhằm kiểm tra liệu ưu thế về point forecast có bền vững khi đánh giá ở tầng phân phối rủi ro hay không.

| VaR case | Method | Violation rate | Abs violation error | Quantile loss |
|---|---|---:|---:|---:|
| 5% | Normal | 0.0598 | 0.0172 | 0.4355 |
| 5% | Student-t | 0.0326 | 0.0217 | 0.5273 |
| 5% | FHS | 0.0527 | 0.0040 | 0.4425 |
| 1% | Normal | 0.0280 | 0.0182 | 0.1363 |
| 1% | Student-t | 0.0061 | 0.0058 | 0.1830 |
| 1% | FHS | 0.0148 | 0.0048 | 0.1470 |

Ở VaR 5%, Normal có violation rate trung bình 5.98%, cao hơn mức kỳ vọng và vì vậy có xu hướng underestimate risk. Student-t có violation rate 3.26%, thấp hơn 5%, thể hiện khuynh hướng bảo thủ hơn. FHS đạt violation rate 5.27% và abs violation error 0.0040, gần mức mục tiêu nhất. Ở VaR 1%, Normal tạo quá nhiều violation với tỷ lệ 2.80%, trong khi Student-t bảo thủ hơn với 0.61%. FHS tiếp tục nằm gần alpha hơn hai phương pháp còn lại theo abs violation error.

Điểm quan trọng là cùng một tập volatility forecasts tạo ra kết quả khác nhau khi đi qua ba cách mô hình hóa tail. Normal, Student-t và FHS không chỉ là ba kỹ thuật VaR, mà là ba phép kiểm tra xem thông tin volatility của mô hình có đủ để mô tả phân phối tổn thất ở tail hay không. Việc ranking thay đổi giữa calibration error, quantile loss và backtesting cho thấy point forecast tốt không tự động chuyển hóa thành distributional forecast tốt.

## 6. Ranking mô hình dưới các giả định phân phối khác nhau

Khi xếp hạng theo quantile loss trong từng phương pháp, nhóm mô hình dẫn đầu thay đổi theo cả alpha và giả định phân phối. Ở VaR 5%, nhóm Moirai đứng đầu khi dùng Normal và FHS, trong khi Student-t ưu tiên các biến thể Vanilla và Reformer. Ở VaR 1%, nhóm GARCH-family trở nên cạnh tranh hơn với Normal và FHS, còn Student-t tiếp tục làm nổi bật một số Transformer variants.

| VaR case | Method | Top models theo quantile loss |
|---|---|---|
| 5% | Normal | Moirai, Moirai-MoE, Moirai2 |
| 5% | Student-t | Vanilla Tier 3, Reformer Tier 3, Vanilla Tier 2 |
| 5% | FHS | Moirai, Moirai-MoE, Moirai2 |
| 1% | Normal | FI-GARCH, GJR-GARCH, GARCH |
| 1% | Student-t | Vanilla Tier 3, Informer Tier 1, Reformer Tier 3 |
| 1% | FHS | GJR-GARCH, FI-GARCH, GARCH |

Kết quả này cho thấy ưu thế forecast accuracy của Moirai không chuyển hóa đồng nhất sang mọi bài toán tail distribution. Moirai chuyển hóa tốt sang Normal/FHS VaR ở mức 5%, nhưng lợi thế đó không giữ nguyên ở tail 1%. Khi đánh giá extreme tail, các mô hình GARCH-family và một số Transformer variants có thể tạo quantile loss cạnh tranh hơn tùy giả định phân phối. Đây chính là bằng chứng bổ sung cho kết luận rằng mô hình dự báo volatility chính xác chưa chắc là mô hình dự báo phân phối rủi ro tốt.

## 7. Kiểm định Friedman và Nemenyi

Friedman test xác nhận có khác biệt có ý nghĩa thống kê giữa các mô hình về forecast accuracy. Với cả hai VaR case, p-value của MSE là 1.62e-92, MAE là 2.18e-97 và QLIKE là 6.26e-109. Các giá trị này đều nhỏ hơn 0.05, cho thấy ranking forecast không chỉ phản ánh dao động ngẫu nhiên giữa các dataset-horizon cases.

Đối với các VaR metrics, phần lớn kiểm định Friedman cũng có ý nghĩa thống kê. Ngoại lệ đáng chú ý là Kupiec LR của FHS VaR 5%, với p-value 0.0583, không vượt ngưỡng ý nghĩa 5%. Điều này phù hợp với việc FHS làm violation rate trung bình gần mức 5% cho nhiều mô hình, từ đó thu hẹp khác biệt coverage giữa các model.

Nemenyi post-hoc test củng cố kết quả Friedman. Với forecast metrics, có 81 cặp khác biệt có ý nghĩa cho MSE, 81 cặp cho MAE và 85 cặp cho QLIKE ở mỗi VaR case. Đối với VaR metrics, số cặp significant phụ thuộc mạnh vào metric và giả định phân phối VaR: Normal và Student-t thường phân biệt mô hình rõ hơn, trong khi FHS tạo ít khác biệt hơn ở một số tiêu chí coverage do đã dùng phân phối residual thực nghiệm để điều chỉnh tail.

## 8. Diebold-Mariano test

Diebold-Mariano pairwise test cho thấy quantile loss giữa nhiều cặp mô hình khác biệt có ý nghĩa theo thời gian. Tổng số cặp significant cao nhất xuất hiện ở Normal VaR 1% với 3,589 cặp và Student-t VaR 5% với 3,121 cặp. FHS có số cặp significant thấp hơn, gồm 1,355 cặp ở VaR 5% và 1,264 cặp ở VaR 1%.

| VaR case | Method | Significant pairs |
|---|---|---:|
| 5% | Normal | 2,749 |
| 5% | Student-t | 3,121 |
| 5% | FHS | 1,355 |
| 1% | Normal | 3,589 |
| 1% | Student-t | 2,928 |
| 1% | FHS | 1,264 |

Diễn giải hợp lý là FHS làm giảm độ nhạy của quantile loss đối với khác biệt giữa volatility forecasts, vì phân phối residual thực nghiệm đóng vai trò điều chỉnh tail. Ngược lại, Normal và Student-t phụ thuộc trực tiếp hơn vào scale của `predict_volatility`, nên khác biệt giữa mô hình được phản ánh mạnh hơn trong loss. Kết quả DM vì vậy hỗ trợ cách dùng ba phương pháp VaR như robustness checks cho tầng distributional forecasting.

## 9. Trade-off giữa accuracy và risk-control

Bảng `accuracy_risk_tradeoff.csv` cho thấy Moirai2, Moirai-MoE và Moirai đứng đầu về forecast accuracy nhưng không đứng đầu về legacy VaR pass rate. Ngược lại, GARCH-LSTM-Hybrid đứng đầu VaR 5% pass rate nhưng lại đứng cuối theo QLIKE. Ở VaR 1%, Reformer Tier 3 đứng đầu pass rate trong khi Moirai2 vẫn là mô hình tốt nhất về forecast accuracy.

Kết quả này là bằng chứng thực nghiệm quan trọng cho luận điểm phương pháp luận của nghiên cứu. MSE, MAE và QLIKE đo khả năng khớp volatility path ở trung bình, trong khi VaR backtesting và quantile loss kiểm tra khả năng mô tả tail của conditional return distribution. Hai mục tiêu này có liên hệ nhưng không đồng nhất. Vì vậy, nếu mục tiêu cuối cùng là dự báo rủi ro phân phối hoặc risk management, việc chọn mô hình chỉ dựa trên forecast loss có thể dẫn đến lựa chọn không tối ưu.

## 10. Hàm ý phương pháp luận và hạn chế

Kết quả ủng hộ một framework đánh giá hai tầng. Tầng thứ nhất là point forecasting evaluation, bao gồm MSE, MAE, QLIKE và average forecast rank theo metric. Tầng thứ hai là distributional risk evaluation, trong đó cùng một volatility forecast được chuyển thành VaR bằng Normal, Student-t và FHS để kiểm tra coverage, independence và quantile loss. Cách báo cáo này cho phép phân biệt mô hình dự báo volatility tốt với mô hình dự báo phân phối rủi ro tốt.

Một số hạn chế cần nêu rõ khi viết paper. Thứ nhất, `log_return` và `predict_volatility` đang cùng scale và đều đã nhân 100. Thứ hai, FHS cần warmup 250 ngày nên số dòng hợp lệ thấp hơn Normal và Student-t. Thứ ba, tham số Student-t `nu` được estimate theo dataset, không theo từng model hoặc horizon. Thứ tư, VaR backtesting legacy trong `stats_by_model.csv` phản ánh Student-t VaR, còn so sánh Normal/Student-t/FHS nằm trong `var_method_comparison.csv`. Cuối cùng, DM test là pairwise và tạo nhiều so sánh, vì vậy cần diễn giải theo hướng bằng chứng hỗ trợ thay vì kết luận tuyệt đối cho từng cặp riêng lẻ.

## 11. Kết luận

Nhìn chung, Moirai-family vượt trội về độ chính xác dự báo volatility, đặc biệt khi xét đồng thời MSE, MAE, QLIKE và average forecast rank. Tuy nhiên, khi cùng các volatility forecasts này được đánh giá qua Normal, Student-t và FHS VaR, ranking theo tail-risk metrics thay đổi theo alpha và giả định phân phối. Điều này cho thấy ưu thế point forecast không đủ để bảo đảm mô hình mô tả tốt phân phối rủi ro ở tail. Các kiểm định Friedman, Nemenyi và Diebold-Mariano xác nhận rằng nhiều khác biệt giữa các mô hình là có ý nghĩa thống kê.

Kết luận chính của nghiên cứu là volatility forecast accuracy và distributional risk forecasting quality không thể thay thế cho nhau. Mô hình dự báo volatility tốt nhất không nhất thiết là mô hình dự báo phân phối tổn thất hoặc tail quantile tốt nhất. Do đó, các benchmark về volatility forecasting nên báo cáo đồng thời predictive accuracy và VaR-based distributional evaluation, đặc biệt khi mô hình được định hướng cho ứng dụng tài chính rủi ro.
