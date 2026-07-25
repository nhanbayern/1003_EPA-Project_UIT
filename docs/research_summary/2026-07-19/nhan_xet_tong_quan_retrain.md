# Diễn giải kết quả nghiên cứu

Tài liệu này diễn giải các bảng tổng hợp trong `output/stats_analysis/research_summary/`, được xây dựng từ hai bộ kết quả VaR 5% và VaR 1%. Trọng tâm của phân tích không phải là tìm một mô hình có forecast loss thấp nhất tuyệt đối, mà là kiểm tra liệu mô hình dự báo volatility chính xác theo point forecast có đồng thời tạo ra đánh giá rủi ro tail tốt khi chuyển sang VaR hay không. Vì vậy, các kết quả VaR được xem như tầng đánh giá distributional risk bổ sung cho tầng đánh giá realized volatility forecast.

## 1. Tổng quan

Kết quả thực nghiệm cho thấy sự tách biệt rõ giữa hai mục tiêu đánh giá. Nhóm Moirai đạt độ chính xác dự báo realized volatility tốt nhất theo MSE, MAE và QLIKE. Tuy nhiên, khi các volatility forecasts được chuyển thành VaR và kiểm định bằng Kupiec coverage test cùng Christoffersen independence test, nhóm đứng đầu không còn là Moirai mà chuyển sang các biến thể Transformer, đặc biệt ở VaR 1%.

Bộ kết quả gồm `991,885` dòng dự báo, `9` datasets, `19` nhóm model, `5` horizons và `855` detailed cases. Dữ liệu không thiếu `log_return` hoặc `predict_volatility`; có `1,260` dòng thiếu `true_volatility`, được loại khỏi các metric forecast khi cần.

| Case | Input rows | Datasets | Model groups | Horizons | Detailed cases | Missing log_return | Missing true_volatility | Missing predict_volatility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VaR 5% | 991,885 | 9 | 19 | 5 | 855 | 0 | 1,260 | 0 |
| VaR 1% | 991,885 | 9 | 19 | 5 | 855 | 0 | 1,260 | 0 |

Luận điểm trung tâm là: forecast accuracy và tail-risk calibration có liên hệ nhưng không đồng nhất. Một model có MSE/MAE/QLIKE thấp chưa chắc tạo ra VaR thresholds có coverage và independence tốt.

## 2. Độ chính xác dự báo volatility

Bảng `forecast_ranking.csv` xếp hạng mô hình theo các metric dự báo ở cấp aggregate model. Vì MSE, MAE và QLIKE không phụ thuộc mức alpha của VaR, forecast ranking trong hai case VaR 5% và VaR 1% là giống nhau. Cột `avg_forecast_rank` là trung bình của ba rank metric MSE, MAE và QLIKE.

| Rank tổng hợp | Model | MSE | MAE | QLIKE | Avg forecast rank |
|---:|---|---:|---:|---:|---:|
| 1 | Moirai2 | 0.022751 | 0.090622 | 0.009121 | 1.000 |
| 2 | Moirai-MoE | 0.024757 | 0.097559 | 0.009127 | 2.000 |
| 3 | Moirai | 0.042343 | 0.132846 | 0.014733 | 3.000 |
| 4 | GARCH-LSTM-Hybrid | 0.110451 | 0.236196 | 0.033966 | 4.333 |
| 5 | FI-GARCH | 0.116380 | 0.236267 | 0.030465 | 4.667 |

Moirai2 đứng đầu đồng thời ở MSE, MAE và QLIKE, nên là model mạnh nhất cho point volatility forecasting. Moirai-MoE xếp thứ hai với khoảng cách rất nhỏ ở QLIKE, cho thấy mixture-of-experts không tạo ra cải thiện rõ ràng so với Moirai2 trong aggregate forecast loss. Moirai gốc đứng thứ ba nhưng vẫn tách biệt rõ so với các nhóm GARCH và Transformer.

Điểm đáng chú ý là GARCH-LSTM-Hybrid và FI-GARCH lần lượt đứng thứ tư và thứ năm theo forecast ranking tổng hợp. Tuy nhiên, khoảng cách giữa nhóm Moirai và phần còn lại vẫn đáng kể: QLIKE của Moirai2 là `0.009121`, trong khi GARCH-LSTM-Hybrid là `0.033966` và FI-GARCH là `0.030465`. Điều này cho thấy lợi thế point forecast của Moirai-family là ổn định và không chỉ đến từ một metric đơn lẻ.

## 3. VaR 5% backtesting

Bảng `var_5pct_ranking.csv` cho thấy nhóm Transformer chiếm ưu thế khi đánh giá VaR 5% bằng joint pass rate của Kupiec và Christoffersen. Mô hình đứng đầu là Autoformer Tier 2 Standard với pass rate `0.244`.

| Rank | Model | Pass rate | Passed cases | Violation rate | QLIKE |
|---:|---|---:|---:|---:|---:|
| 1 | Autoformer Tier 2 Standard | 0.244 | 11 | 0.0362 | 0.0980 |
| 2 | Autoformer Tier 1 Miniaturized | 0.222 | 10 | 0.0344 | 0.0929 |
| 2 | Reformer Tier 3 Large | 0.222 | 10 | 0.0409 | 0.1001 |
| 4 | Informer Tier 2 Standard | 0.178 | 8 | 0.0353 | 0.0850 |
| 4 | Reformer Tier 2 Standard | 0.178 | 8 | 0.0379 | 0.1003 |

Các model đứng đầu VaR 5% đều có violation rate thấp hơn mức kỳ vọng 5%, dao động từ khoảng `0.034` đến `0.041`. Điều này cho thấy VaR thresholds có khuynh hướng bảo thủ. Tuy nhiên, nhóm Transformer vẫn đạt pass rate cao hơn vì coverage và independence cân bằng hơn ở nhiều dataset-horizon cases.

Theo model family, pass rate trung bình ở VaR 5% là:

| Branch | Pass rate | Violation rate | Kupiec p | Christoffersen p | QLIKE |
|---|---:|---:|---:|---:|---:|
| GARCH | 0.000 | 0.0218 | 0.0001 | 0.4857 | 0.0391 |
| Moirai | 0.104 | 0.0262 | 0.0190 | 0.1638 | 0.0110 |
| Transformers | 0.163 | 0.0368 | 0.0873 | 0.1344 | 0.0975 |

Kết quả này minh họa trade-off đầu tiên: Moirai có QLIKE thấp nhất nhưng không dẫn đầu VaR 5% pass rate; Transformers có QLIKE cao hơn nhưng violation rate gần alpha hơn và backtesting tốt hơn.

## 4. VaR 1% backtesting

Ở VaR 1%, sự tách biệt giữa point forecast accuracy và risk-control thể hiện rõ hơn. Reformer Tier 3 Large đạt pass rate cao nhất, với violation rate `0.0090`, gần mức kỳ vọng 1%.

| Rank | Model | Pass rate | Passed cases | Violation rate | QLIKE |
|---:|---|---:|---:|---:|---:|
| 1 | Reformer Tier 3 Large | 0.378 | 17 | 0.0090 | 0.1001 |
| 2 | Autoformer Tier 1 Miniaturized | 0.356 | 16 | 0.0061 | 0.0929 |
| 2 | Informer Tier 1 Miniaturized | 0.356 | 16 | 0.0068 | 0.0972 |
| 4 | Reformer Tier 2 Standard | 0.333 | 15 | 0.0083 | 0.1003 |
| 5 | Autoformer Tier 2 Standard | 0.311 | 14 | 0.0072 | 0.0980 |

Nhóm Transformer variants chiếm toàn bộ top 5 ở VaR 1%. Trong khi đó, Moirai2 vẫn là model có QLIKE tốt nhất nhưng chỉ đạt pass rate `0.133` (`6/45` cases). Moirai-MoE tốt hơn Moirai2 về VaR 1% pass rate (`0.244`) nhưng vẫn không cạnh tranh với nhóm Transformer dẫn đầu.

Theo model family, VaR 1% cho thấy pattern rõ hơn:

| Branch | Pass rate | Violation rate | Kupiec p | Christoffersen p | QLIKE |
|---|---:|---:|---:|---:|---:|
| GARCH | 0.017 | 0.0027 | 0.010 | 0.644 | 0.039 |
| Moirai | 0.163 | 0.0042 | 0.087 | 0.387 | 0.011 |
| Transformers | 0.281 | 0.0075 | 0.205 | 0.398 | 0.097 |

GARCH và Moirai có violation rate thấp hơn đáng kể so với target 1%, thể hiện khuynh hướng bảo thủ. Transformers có violation rate gần 1% hơn, do đó đạt Kupiec p trung bình và pass rate cao hơn. Điều này cho thấy calibration tail không chỉ phụ thuộc vào forecast loss, mà còn phụ thuộc vào cách forecast volatility phản ứng với tail events.

## 5. Ba phương pháp VaR như robustness checks

Bảng `var_method_comparison.csv` so sánh ba cách chuyển volatility forecast thành VaR: Normal, Student-t và Filtered Historical Simulation (FHS). Ở đây, ba phương pháp không nên được hiểu đơn giản là ba model VaR độc lập, mà là ba giả định phân phối dùng để kiểm tra độ bền của kết luận khi đánh giá tail risk.

| VaR case | Method | Violation rate | Abs violation error | Quantile loss |
|---|---|---:|---:|---:|
| 5% | Normal | 0.0590 | 0.0164 | 0.1413 |
| 5% | Student-t | 0.0320 | 0.0221 | 0.1472 |
| 5% | FHS | 0.0527 | 0.0040 | 0.1320 |
| 1% | Normal | 0.0274 | 0.0175 | 0.0529 |
| 1% | Student-t | 0.0060 | 0.0058 | 0.0491 |
| 1% | FHS | 0.0148 | 0.0048 | 0.0458 |

Ở VaR 5%, FHS có violation rate `0.0527`, gần alpha 5% nhất và cũng có quantile loss thấp nhất trong ba phương pháp. Normal có xu hướng underestimate risk vì tạo violation rate `0.0590`; Student-t lại bảo thủ hơn với violation rate `0.0320`.

Ở VaR 1%, Normal tạo quá nhiều violation (`0.0274`), trong khi Student-t bảo thủ hơn (`0.0060`). FHS nằm gần alpha nhất theo absolute violation error (`0.0048`) và có quantile loss thấp nhất (`0.0458`). Điều này ủng hộ việc dùng FHS như một robustness check thực nghiệm cho residual tail, đặc biệt khi phân phối return có heavy-tail và không hoàn toàn phù hợp với Gaussian assumption.

## 6. Ranking dưới các giả định phân phối khác nhau

Khi xếp hạng theo quantile loss trong từng phương pháp VaR, top models thay đổi đáng kể theo cả alpha và giả định phân phối.

| VaR case | Method | Top models theo quantile loss |
|---|---|---|
| 5% | Normal | Moirai, Moirai-MoE, Moirai2 |
| 5% | Student-t | Vanilla Tier 3, Reformer Tier 3, Vanilla Tier 2 |
| 5% | FHS | Moirai, Moirai-MoE, Moirai2 |
| 1% | Normal | FI-GARCH, GJR-GARCH, GARCH |
| 1% | Student-t | Vanilla Tier 3, Informer Tier 1, Reformer Tier 3 |
| 1% | FHS | GJR-GARCH, FI-GARCH, GARCH |

Kết quả này làm rõ rằng ưu thế forecast accuracy của Moirai không chuyển hóa đồng nhất sang mọi bài toán tail distribution. Moirai dẫn đầu quantile loss ở VaR 5% khi dùng Normal và FHS, nhưng không giữ được ưu thế ở VaR 1%. Ở extreme tail 1%, GARCH-family và một số Transformer variants trở nên cạnh tranh hơn tùy giả định phân phối.

Đây là bằng chứng bổ sung cho luận điểm phương pháp luận: nếu mục tiêu cuối cùng là distributional risk forecasting, việc chỉ chọn model theo MSE/MAE/QLIKE của volatility path là chưa đủ.

## 7. Kiểm định Friedman và Nemenyi

Friedman test xác nhận khác biệt giữa các model có ý nghĩa thống kê cho forecast metrics. Với `45` blocks và `19` model groups, p-value của MSE, MAE và QLIKE đều rất nhỏ ở cả hai VaR case.

| Metric | Friedman statistic | Friedman p |
|---|---:|---:|
| MSE | 517.893 | 1.797e-98 |
| MAE | 543.446 | 7.459e-104 |
| QLIKE | 615.804 | 3.919e-119 |

Đối với VaR metrics, toàn bộ Friedman tests trong `friedman_summary.csv` đều có ý nghĩa ở mức 5%. Điều này cho thấy khác biệt giữa model không chỉ xuất hiện ở forecast accuracy mà còn ở calibration error, Kupiec LR, independence LR và quantile loss dưới các giả định Normal, Student-t, FHS.

Nemenyi post-hoc test cũng cho thấy nhiều cặp model khác biệt có ý nghĩa. Với forecast metrics, số cặp significant là `84` cho MSE, `85` cho MAE và `92` cho QLIKE ở mỗi VaR case. Với VaR metrics, số cặp significant phụ thuộc mạnh vào phương pháp VaR: Normal thường phân biệt model rõ hơn, trong khi FHS tạo ít khác biệt hơn ở các chỉ tiêu coverage như `kupiec_lr`.

## 8. Diebold-Mariano test

Diebold-Mariano pairwise test được dùng để kiểm tra khác biệt quantile loss theo thời gian. Số cặp significant cao nhất xuất hiện ở Normal VaR 1%, trong khi FHS có số cặp significant thấp hơn ở cả hai alpha.

| VaR case | Method | Significant pairs |
|---|---|---:|
| 5% | Normal | 2,502 |
| 5% | Student-t | 2,841 |
| 5% | FHS | 1,111 |
| 1% | Normal | 3,466 |
| 1% | Student-t | 2,709 |
| 1% | FHS | 1,010 |

Một diễn giải hợp lý là Normal và Student-t phụ thuộc trực tiếp hơn vào scale của `predict_volatility`, nên khác biệt giữa mô hình phản ánh mạnh trong quantile loss. FHS sử dụng residual distribution thực nghiệm, qua đó hấp thụ một phần sai khác giữa volatility forecasts và làm số cặp significant thấp hơn. Kết quả DM vì vậy hỗ trợ cách dùng ba phương pháp VaR như robustness checks thay vì xem một phương pháp duy nhất là kết luận tuyệt đối.

## 9. Sensitivity theo horizon và dataset

Ở VaR 5%, pass rate tăng nhẹ khi horizon dài hơn, từ `0.105` ở horizon 1 lên `0.140` ở horizon 21. Tuy vậy violation rate trung bình vẫn nằm quanh `0.031-0.033`, thấp hơn target 5%, cho thấy tính bảo thủ khá ổn định theo horizon.

| Horizon | VaR 5% Pass rate | VaR 5% Violation rate | VaR 1% Pass rate | VaR 1% Violation rate |
|---:|---:|---:|---:|---:|
| 1 | 0.105 | 0.0334 | 0.228 | 0.0063 |
| 3 | 0.117 | 0.0317 | 0.211 | 0.0059 |
| 5 | 0.111 | 0.0318 | 0.211 | 0.0059 |
| 10 | 0.123 | 0.0321 | 0.187 | 0.0061 |
| 21 | 0.140 | 0.0308 | 0.199 | 0.0056 |

Ở VaR 1%, horizon 1 có pass rate cao nhất (`0.228`), nhưng các horizon còn lại không suy giảm quá mạnh. Nhìn chung, kết quả không cho thấy horizon dài tự động làm VaR calibration tốt hơn; thay vào đó, calibration phụ thuộc nhiều vào model family và dataset.

Theo dataset, VaR 1% phân hóa mạnh. `VN_INDEX`, `VN30_INDEX` và `SNP500` có pass rate cao hơn rõ rệt, trong khi `EURONEXT_100` là dataset khó nhất với pass rate bằng `0.000`.

| Dataset | VaR 1% Pass rate | VaR 1% Violation rate |
|---|---:|---:|
| VN_INDEX | 0.495 | 0.0049 |
| VN30_INDEX | 0.432 | 0.0034 |
| SNP500 | 0.379 | 0.0075 |
| DAX_40 | 0.263 | 0.0047 |
| KOSPI_INDEX | 0.137 | 0.0144 |
| NIKKEI_225 | 0.063 | 0.0055 |
| IBEX_35 | 0.053 | 0.0049 |
| SMI | 0.042 | 0.0059 |
| EURONEXT_100 | 0.000 | 0.0026 |

Dataset sensitivity này cho thấy risk calibration không phải thuộc tính chỉ của model. Cùng một model family có thể hoạt động khác nhau mạnh theo thị trường, đặc biệt ở extreme tail.

## 10. Trade-off giữa accuracy và risk-control

Bảng `accuracy_risk_tradeoff.csv` cho thấy ba model Moirai đứng đầu theo forecast accuracy nhưng không đứng đầu VaR pass rate. Ngược lại, các model Transformer có QLIKE cao hơn lại đứng đầu VaR 5% và VaR 1% backtesting.

Trade-off này có ý nghĩa phương pháp luận quan trọng. MSE và MAE đo sai số point forecast của volatility path; QLIKE nhạy hơn với sai lệch scale volatility nhưng vẫn là metric cho volatility forecast. VaR backtesting lại kiểm tra coverage và independence của tail events trong conditional return distribution. Hai tầng đánh giá này đo hai đối tượng khác nhau: một bên là realized volatility, một bên là tail quantile của return loss distribution.

Vì vậy, nếu ứng dụng cuối cùng là risk management, không nên chọn model chỉ dựa trên forecast loss. Ngược lại, nếu mục tiêu là dự báo volatility như một biến mục tiêu độc lập, Moirai2 là lựa chọn mạnh nhất trong bộ kết quả hiện tại.

## 11. Hàm ý phương pháp luận và hạn chế

Kết quả ủng hộ framework đánh giá hai tầng:

1. Tầng point forecasting: dùng MSE, MAE, QLIKE và average forecast rank để đánh giá khả năng dự báo realized volatility.
2. Tầng distributional risk evaluation: chuyển cùng volatility forecasts thành VaR bằng Normal, Student-t và FHS, sau đó kiểm tra coverage, independence và quantile loss.

Cách báo cáo này giúp phân biệt mô hình dự báo volatility tốt với mô hình tạo VaR tốt. Nó cũng tránh kết luận quá mức từ một metric đơn lẻ.

Một số hạn chế cần nêu rõ khi viết paper. Thứ nhất, VaR được suy ra từ volatility forecast qua giả định phân phối, nên chất lượng VaR không chỉ phản ánh model forecast mà còn phản ánh mapping từ volatility sang tail quantile. Thứ hai, FHS cần rolling history nên bản chất khác Normal và Student-t. Thứ ba, Student-t `nu` được estimate theo dataset, không theo từng model hoặc horizon. Thứ tư, DM test tạo nhiều pairwise comparisons, nên nên dùng như bằng chứng hỗ trợ thay vì kết luận tuyệt đối cho từng cặp riêng lẻ.

## 12. Kết luận

Nhìn chung, Moirai-family vượt trội về dự báo realized volatility, với Moirai2 đứng đầu theo MSE, MAE và QLIKE. Tuy nhiên, khi đánh giá ở tầng VaR, ranking thay đổi đáng kể: các biến thể Transformer dẫn đầu VaR 5% và VaR 1% backtesting, còn GARCH-family trở nên cạnh tranh hơn ở một số quantile-loss settings của VaR 1%.

Kết luận chính của nghiên cứu là forecast accuracy và distributional risk quality không thể thay thế cho nhau. Một model dự báo volatility tốt nhất không nhất thiết là model tạo ra tail quantile tốt nhất. Do đó, benchmark volatility forecasting trong bối cảnh tài chính nên báo cáo đồng thời point forecast metrics và VaR-based risk evaluation, đặc biệt khi mô hình được định hướng cho ứng dụng quản trị rủi ro.
