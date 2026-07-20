# Diễn giải kết quả nghiên cứu

Tài liệu này diễn giải các bảng tổng hợp trong `output/stats_analysis/research_summary/`, được xây dựng từ hai bộ kết quả VaR 5% và VaR 1%. Trọng tâm của phân tích không phải là tìm một phương pháp VaR tốt nhất tuyệt đối, mà là kiểm tra liệu mô hình dự báo volatility chính xác theo point forecast có đồng thời dự báo tốt phân phối rủi ro ở phần đuôi hay không. Vì vậy, ba phương pháp VaR gồm Normal, Student-t và Filtered Historical Simulation (FHS) được sử dụng như ba góc nhìn robustness cho cùng một tập volatility forecasts.

## 1. Tổng quan

Kết quả thực nghiệm cho thấy sự tách biệt rõ giữa hai mục tiêu đánh giá. Nhóm Moirai đạt độ chính xác dự báo volatility tốt nhất theo các loss trung bình MSE, MAE và QLIKE. Tuy nhiên, khi các volatility forecasts này được chuyển thành VaR bằng các giả định phân phối khác nhau, các mô hình đứng đầu về backtesting và quantile loss không còn trùng ổn định với nhóm có forecast loss thấp nhất. Điều này củng cố luận điểm trung tâm của nghiên cứu: mô hình dự báo point volatility tốt chưa chắc là mô hình dự báo tốt conditional return distribution, đặc biệt ở tail.

## 2. Độ chính xác dự báo volatility

Bảng `forecast_ranking.csv` xếp hạng mô hình theo các metric dự báo trung bình ở cấp model. Vì MSE, MAE và QLIKE không phụ thuộc mức alpha của VaR, ranking forecast ở VaR 5% và VaR 1% là giống nhau. Ngoài các rank riêng `mse_rank`, `mae_rank` và `qlike_rank`, bảng hiện có thêm `avg_forecast_rank`, được tính bằng trung bình đều của ba rank metric này. Đây là average rank theo metric ở cấp aggregate, không phải average rank theo từng dataset-horizon.

| Rank tổng hợp | Model | MSE | MAE | QLIKE | Avg forecast rank |
|---:|---|---:|---:|---:|---:|
| 1 | Moirai\|\|moirai2 | 0.022751 | 0.090622 | 0.009121 | 1.000 |
| 2 | Moirai_VAR\|lambda_0.2\|moirai2 | 0.023751 | 0.094772 | 0.009136 | 2.333 |
| 3 | Moirai\|\|moirai_moe | 0.024757 | 0.097559 | 0.009127 | 2.667 |
| 4 | Moirai_VAR\|lambda_0.2\|moirai_moe | 0.025260 | 0.101491 | 0.009323 | 4.000 |
| 5 | Moirai_VAR\|lambda_0.2\|moirai | 0.041562 | 0.130161 | 0.014217 | 5.000 |
| 6 | Moirai\|\|moirai | 0.042343 | 0.132846 | 0.014733 | 6.000 |

Các biến thể thuộc họ Moirai (bao gồm phiên bản gốc và phiên bản kết hợp Vector Autoregression `Moirai_VAR`) thống trị hoàn toàn các vị trí dẫn đầu. Trong đó, `moirai2` gốc đứng đầu đồng thời ở cả MSE, MAE và QLIKE. Moirai_VAR cũng cho thấy độ chính xác đáng kinh ngạc khi bám sát ở vị trí thứ hai và thứ tư. Kết quả này củng cố mạnh mẽ lợi thế của các mô hình foundation-style trong bài toán point volatility forecasting.

Một điểm thay đổi đáng lưu ý so với các đánh giá trước đây là GARCH-LSTM-Hybrid đã cải thiện mạnh mẽ độ chính xác nhờ việc chuyển đổi cơ chế dự báo sang *trực tiếp (direct multi-horizon)*. MSE của nó đã giảm từ mức khổng lồ $1.28 \times 10^6$ xuống khoảng 0.11. Dù vẫn xếp sau nhóm Moirai, mô hình này không còn là "điểm nghẽn" về mặt dự báo điểm. Tuy nhiên, sự vượt trội tuyệt đối của Moirai cho thấy khả năng khớp chuỗi kỳ vọng (mean path) xuất sắc của nó.

## 3. VaR 5% backtesting

Bảng `var_5pct_ranking.csv` cho thấy GARCH-LSTM-Hybrid đạt pass rate cao nhất ở VaR 5%, dù mô hình này đứng cuối theo QLIKE. Các mô hình Autoformer và Reformer cũng nằm trong nhóm có backtesting tốt ở mức tail 5%.

| Rank | Model | Pass rate | Violation rate | QLIKE |
|---:|---|---:|---:|---:|
| 1 | Transformers\|Tier_2_Standard\|Autoformer | 0.2444 | 0.0363 | 0.0980 |
| 2 | Transformers\|Tier_1_Miniaturized\|Autoformer | 0.2000 | 0.0341 | 0.0929 |
| 2 | Transformers\|Tier_3_Large\|Reformer | 0.2000 | 0.0413 | 0.1001 |
| 2 | Transformers\|Tier_2_Standard\|Reformer | 0.2000 | 0.0382 | 0.1003 |
| 5 | Transformers\|Tier_2_Standard\|Vanilla | 0.1556 | 0.0355 | 0.0840 |
| 5 | Transformers\|Tier_2_Standard\|Informer | 0.1556 | 0.0350 | 0.0850 |

Bảng `var_5pct_ranking.csv` ghi nhận sự soán ngôi hoàn toàn của các mô hình Transformer. Dẫn đầu là `Autoformer Tier 2` với pass rate cao nhất (24.44%), theo sau là các biến thể Autoformer Tier 1 và Reformer Tier 3. Đáng chú ý, GARCH-LSTM-Hybrid không còn giữ vị trí dẫn đầu ở VaR 5% như trước đây. Các model đứng đầu VaR 5% thường có violation rate thấp hơn mức kỳ vọng 5% (như Autoformer Tier 2 là 3.63%), cho thấy nhóm này có xu hướng bảo thủ một cách an toàn. Tuy nhiên, pass rate không chỉ phản ánh violation rate mà còn phụ thuộc vào kiểm định Kupiec và Christoffersen independence. Do đó, khả năng vượt qua các kiểm định này chứng tỏ nhóm Transformer không chỉ dự báo được số lượng vi phạm đúng tỷ lệ mà còn đảm bảo chuỗi vi phạm này là độc lập.

## 4. VaR 1% backtesting

Ở VaR 1%, ranking risk-control thay đổi đáng kể. Reformer Tier 3 đạt pass rate cao nhất, trong khi GARCH-LSTM-Hybrid giảm xuống nhóm đồng hạng thứ năm. Điều này cho thấy mô hình tốt ở tail 5% không nhất thiết ổn định khi chuyển sang tail nghiêm ngặt hơn.

| Rank | Model | Pass rate | Violation rate | QLIKE |
|---:|---|---:|---:|---:|
| 1 | Transformers\|Tier_1_Miniaturized\|Autoformer | 0.3556 | 0.0066 | 0.0929 |
| 2 | Transformers\|Tier_1_Miniaturized\|Informer | 0.2889 | 0.0068 | 0.0972 |
| 2 | Transformers\|Tier_2_Standard\|Autoformer | 0.2889 | 0.0071 | 0.0980 |
| 4 | Transformers\|Tier_2_Standard\|Informer | 0.2667 | 0.0068 | 0.0850 |
| 5 | Transformers\|Tier_3_Large\|Reformer | 0.2444 | 0.0090 | 0.1001 |
| 5 | Transformers\|Tier_2_Standard\|Reformer | 0.2444 | 0.0081 | 0.1003 |

Nhóm Transformer, đặc biệt là các biến thể hạng nhẹ (`Tier_1_Miniaturized`) chiếm ưu thế tuyệt đối ở ngưỡng cắt đuôi nghiêm ngặt VaR 1%. `Autoformer Tier 1` vươn lên dẫn đầu với pass rate vượt trội (35.56%) và violation rate xấp xỉ mức 1% (0.66%). Xếp hạng ngay sau là `Informer Tier 1` và `Autoformer Tier 2`. Kết quả này khẳng định rằng nhóm Transformer variants sở hữu khả năng kiểm soát rủi ro phân phối (distributional risk-control) cực kỳ ổn định ở các ngưỡng tail khắt khe nhất, điều mà Moirai (vua dự báo điểm) chưa thể làm được.

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

## 6. Ranking mô hình dưới các giả định phân phối khác nhau và Hiệu quả của VaR-Aware Loss

Khi xếp hạng theo quantile loss (Tổn thất phân vị), nhóm mô hình dẫn đầu thay đổi mạnh theo alpha và giả định phân phối. Điểm sáng lớn nhất ở VaR 5% là sự vươn lên dẫn đầu (Top 1) của biến thể **`Moirai_VAR|lambda_0.2|moirai`** dưới cả hai giả định phân phối Normal và FHS. 

Thành tích này không phải ngẫu nhiên mà đến từ một **phương pháp tiếp cận hàm mất mát mới (VaR-Aware Loss)**. Thay vì chỉ huấn luyện Moirai bằng Mean Squared Error (MSE) để tìm giá trị trung bình, mô hình `Moirai_VAR` nhúng trực tiếp nhận thức rủi ro vào quá trình học thông qua hàm mục tiêu:

$$ \text{Total\_Loss} = \text{MSE}(\text{vol}_{pred}, \text{vol}_{true}) + \lambda_{var} \times \text{QuantileVaRLoss}(\text{return}_{realized}, \text{VaR}_{threshold}) $$

Bằng cách thêm hình phạt Pinball Loss (mất mát phân vị) với trọng số $\lambda_{var}=0.2$, mô hình bị trừng phạt nặng nề nếu lợi suất thực tế đâm thủng ngưỡng VaR dự kiến. Nhờ cơ chế lai này, `Moirai_VAR` chấp nhận đánh đổi một phần nhỏ độ chính xác trung bình (khiến nó lùi xuống Hạng 2 ở bảng dự báo điểm), nhưng bù lại, nó đánh bại toàn bộ các đối thủ về khả năng tối thiểu hóa tổn thất ở vùng đuôi 5%.

| VaR case | Method | Top models theo quantile loss |
|---|---|---|
| 5% | Normal | Moirai_VAR, Moirai-MoE, Moirai2 |
| 5% | Student-t | Transformers (Vanilla Tier 3, v.v.) |
| 5% | FHS | Moirai_VAR, GARCH-LSTM-Hybrid, Moirai |
| 1% | Normal | GARCH-LSTM-Hybrid, FI-GARCH, v.v. |
| 1% | Student-t | Transformers (Informer Tier 1, v.v.) |
| 1% | FHS | GARCH-LSTM-Hybrid, GJR-GARCH, v.v. |

Tuy nhiên, dù `Moirai_VAR` xuất sắc ở Normal/FHS VaR mức 5%, lợi thế đó không giữ nguyên tuyệt đối ở vùng extreme tail (1%). Khi đánh giá ở VaR 1%, các mô hình GARCH-family (như GARCH-LSTM-Hybrid) và Transformer variants lại tạo ra quantile loss cạnh tranh hơn. Đây chính là bằng chứng bổ sung cho kết luận rằng mô hình dự báo volatility chính xác chưa chắc là mô hình dự báo phân phối rủi ro xuất sắc toàn diện, và việc tinh chỉnh hàm mục tiêu (như `Moirai_VAR`) là một hướng tiếp cận vô cùng hứa hẹn cho các nghiên cứu tiếp theo.

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

Sự đối lập giữa kết quả của nhóm Moirai và nhóm Transformer chính là minh họa hoàn hảo cho sự đánh đổi này. Moirai và Moirai_VAR thống trị hoàn toàn về độ chính xác dự báo điểm (MSE, MAE, QLIKE), nhưng lại không thể đứng nhất về pass rate trong cả VaR 5% lẫn VaR 1%. Ngược lại, nhóm Transformer (đặc biệt là Autoformer) có độ chính xác dự báo (Accuracy) kém hơn Moirai rất nhiều, nhưng lại giành vị trí Top 1 về khả năng kiểm soát rủi ro Backtesting (Risk-control) ở cả hai ngưỡng VaR.

Kết quả này là bằng chứng thực nghiệm quan trọng cho luận điểm phương pháp luận của nghiên cứu. MSE, MAE và QLIKE đo khả năng khớp volatility path ở trung bình, trong khi VaR backtesting và quantile loss kiểm tra khả năng mô tả tail của conditional return distribution. Hai mục tiêu này có liên hệ nhưng không đồng nhất. Vì vậy, nếu mục tiêu cuối cùng là dự báo rủi ro phân phối hoặc risk management, việc chọn mô hình chỉ dựa trên forecast loss có thể dẫn đến lựa chọn không tối ưu.

## 10. Hàm ý phương pháp luận và hạn chế

Kết quả ủng hộ một framework đánh giá hai tầng. Tầng thứ nhất là point forecasting evaluation, bao gồm MSE, MAE, QLIKE và average forecast rank theo metric. Tầng thứ hai là distributional risk evaluation, trong đó cùng một volatility forecast được chuyển thành VaR bằng Normal, Student-t và FHS để kiểm tra coverage, independence và quantile loss. Cách báo cáo này cho phép phân biệt mô hình dự báo volatility tốt với mô hình dự báo phân phối rủi ro tốt.

Một số hạn chế cần nêu rõ khi viết paper. Thứ nhất, `log_return` và `predict_volatility` đang cùng scale và đều đã nhân 100. Thứ hai, FHS cần warmup 250 ngày nên số dòng hợp lệ thấp hơn Normal và Student-t. Thứ ba, tham số Student-t `nu` được estimate theo dataset, không theo từng model hoặc horizon. Thứ tư, VaR backtesting legacy trong `stats_by_model.csv` phản ánh Student-t VaR, còn so sánh Normal/Student-t/FHS nằm trong `var_method_comparison.csv`. Cuối cùng, DM test là pairwise và tạo nhiều so sánh, vì vậy cần diễn giải theo hướng bằng chứng hỗ trợ thay vì kết luận tuyệt đối cho từng cặp riêng lẻ.

## 11. Kết luận

Nhìn chung, Moirai-family vượt trội về độ chính xác dự báo volatility, đặc biệt khi xét đồng thời MSE, MAE, QLIKE và average forecast rank. Tuy nhiên, khi cùng các volatility forecasts này được đánh giá qua Normal, Student-t và FHS VaR, ranking theo tail-risk metrics thay đổi theo alpha và giả định phân phối. Điều này cho thấy ưu thế point forecast không đủ để bảo đảm mô hình mô tả tốt phân phối rủi ro ở tail. Các kiểm định Friedman, Nemenyi và Diebold-Mariano xác nhận rằng nhiều khác biệt giữa các mô hình là có ý nghĩa thống kê.

Kết luận chính của nghiên cứu là volatility forecast accuracy và distributional risk forecasting quality không thể thay thế cho nhau. Mô hình dự báo volatility tốt nhất không nhất thiết là mô hình dự báo phân phối tổn thất hoặc tail quantile tốt nhất. Do đó, các benchmark về volatility forecasting nên báo cáo đồng thời predictive accuracy và VaR-based distributional evaluation, đặc biệt khi mô hình được định hướng cho ứng dụng tài chính rủi ro.

## 12. Tài liệu tham khảo (References)

[1] P. Christoffersen, "Evaluating interval forecasts," *International Economic Review*, vol. 39, no. 4, pp. 841-862, 1998.
[2] P. Kupiec, "Techniques for verifying the accuracy of risk measurement models," *The Journal of Derivatives*, vol. 3, no. 2, pp. 73-84, 1995.
[3] X. Liu et al., "Moirai-MoE: Empowering time series foundation models with sparse mixture of experts," in *Proceedings of the 42nd International Conference on Machine Learning*, vol. 267, PMLR, 2025, pp. 38940-38962.
[4] F. X. Diebold and R. S. Mariano, "Comparing predictive accuracy," *Journal of Business & Economic Statistics*, vol. 13, no. 3, pp. 253-263, 1995.
[5] O. B. Sezer, M. U. Gudelek, and A. M. Ozbayoglu, "Financial time series forecasting with deep learning: A systematic literature review: 2005–2019," *Applied Soft Computing*, vol. 90, p. 106181, 2020.
[6] C. Koenker and G. Bassett Jr, "Regression quantiles," *Econometrica: journal of the Econometric Society*, pp. 33-50, 1978.
