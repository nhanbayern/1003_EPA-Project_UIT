# Nhận xét khoa học cho kết quả VaR 1%

## 1. Tổng quan

Phân tích VaR 1% được sinh từ cùng pipeline `stats_analysis`, cùng dữ liệu đầu vào và cùng cách ước lượng Student-t `nu` theo dataset như VaR 5%. Khác biệt duy nhất là mức tail được chuyển từ `alpha = 0.05` sang `alpha = 0.01`. Vì vậy, các chỉ số forecast accuracy (`MSE`, `MAE`, `QLIKE`) giữ nguyên, còn các chỉ số risk (`Violation Rate`, `Kupiec`, `Christoffersen`, `Pass Rate`) được tính lại cho extreme tail 1%.

Tập kết quả VaR 1% gồm `855` detailed cases, `19` model groups, `9` datasets và `5` horizons. Số dòng hợp lệ cho risk metrics vẫn bị ảnh hưởng bởi `45,840` dòng thiếu `log_return` sau bước fill, giống cấu hình VaR 5%.

## 2. Ranking theo risk backtesting

![Pass rate ranking](../figures/pass_rate_ranking.png)

**Hình 1. Ranking `Pass Rate` cho VaR 1%.** `GARCH-LSTM-Hybrid` tiếp tục là mô hình đứng đầu, với `pass_rate = 0.422` (`19/45` cases). `Autoformer Tier_1_Miniaturized` đứng thứ hai với `pass_rate = 0.356`, tiếp theo là `Autoformer Tier_2_Standard` với `0.289`.

| Nhóm mô hình | Pass Rate | Passed Cases | QLIKE | Violation Rate | Kupiec p | Christoffersen p |
|---|---:|---:|---:|---:|---:|---:|
| GARCH / No Tier / GARCH-LSTM-Hybrid | 0.422 | 19 | 0.357 | 0.0059 | 0.194 | 0.936 |
| Transformers / Tier_1_Miniaturized / Autoformer | 0.356 | 16 | 0.093 | 0.0065 | 0.198 | 0.500 |
| Transformers / Tier_2_Standard / Autoformer | 0.289 | 13 | 0.098 | 0.0071 | 0.200 | 0.453 |
| Transformers / Tier_1_Miniaturized / Informer | 0.267 | 12 | 0.097 | 0.0067 | 0.224 | 0.437 |
| Transformers / Tier_3_Large / Reformer | 0.267 | 12 | 0.100 | 0.0089 | 0.190 | 0.307 |

So với VaR 5%, các `Violation Rate` trung bình đều giảm về vùng `0.004-0.009`, thấp hơn mức target 1% nhưng không quá xa ở nhiều model transformer. Điều này làm Kupiec p-value cải thiện rõ ở nhiều nhóm, kéo theo `Pass Rate` tăng.

## 3. Forecast accuracy vẫn không đảm bảo VaR 1%

![QLIKE ranking](../figures/qlike_ranking.png)

**Hình 2. Ranking `QLIKE` cho VaR 1%.** Nhóm Moirai vẫn có `QLIKE` thấp nhất: `moirai2 = 0.009121`, `moirai_moe = 0.009127`, `moirai = 0.014733`. Tuy nhiên, `pass_rate` của chúng chỉ đạt `0.111`, `0.200`, và `0.111`.

Điều này tiếp tục xác nhận rằng forecast loss thấp không đủ để bảo đảm VaR calibration. Ở VaR 1%, tail event hiếm hơn, nên một sai lệch nhỏ trong việc chuyển volatility forecast thành VaR threshold có thể làm thay đổi mạnh số violation quan sát được. Do đó, mô hình tốt theo `QLIKE` vẫn cần được kiểm tra bằng Kupiec và Christoffersen.

![MSE MAE log ranking](../figures/mse_mae_log_ranking.png)

**Hình 3. `MSE` và `MAE` trên log scale.** Các chỉ số forecast accuracy không đổi so với VaR 5%, do chúng không phụ thuộc `alpha`. `GARCH-LSTM-Hybrid` vẫn là outlier forecast loss nhưng lại đứng đầu VaR 1%. Kết quả này làm mạnh thêm nhận định rằng risk-control objective có thể ưu tiên mô hình khác với point-forecast objective.

## 4. Trade-off giữa `QLIKE` và `Pass Rate`

![QLIKE vs Pass Rate](../figures/qlike_vs_pass_rate.png)

**Hình 4. `QLIKE` vs `Pass Rate` cho VaR 1%.** Các điểm có `QLIKE` thấp nhất không nằm ở vùng pass rate cao nhất. `GARCH-LSTM-Hybrid` có `QLIKE = 0.357` nhưng `pass_rate = 0.422`, trong khi Moirai variants có `QLIKE` rất thấp nhưng pass rate thấp hơn đáng kể.

Suy luận khoa học là mức tail 1% không làm biến mất trade-off giữa forecast accuracy và risk reliability. Ngược lại, vì VaR 1% là extreme-tail criterion, nó càng làm rõ yêu cầu calibration tail riêng biệt.

## 5. Kupiec và Christoffersen ở VaR 1%

![Backtest pass counts](../figures/backtest_pass_counts.png)

**Hình 5. Số case pass Kupiec, Christoffersen và joint backtest.** Ở VaR 1%, p-value của Kupiec tăng ở nhiều mô hình so với VaR 5%. Ví dụ, `GARCH-LSTM-Hybrid` có `Kupiec p = 0.194` và `lr_ind_p = 0.936`, cao hơn rõ rệt so với cấu hình 5%.

Dù vậy, các mô hình GARCH truyền thống `FI-GARCH`, `GARCH`, và `GJR-GARCH` vẫn có `pass_rate = 0`. Điều này cho thấy việc chuyển sang tail 1% không tự động sửa được vấn đề calibration của các baseline này. Chúng vẫn tạo số violation không đủ phù hợp với mức kỳ vọng 1% trên các dataset-horizon cases.

## 6. Phân tích theo model family

![Branch distributions](../figures/branch_metric_distributions.png)

**Hình 6. Phân phối `QLIKE` và `Violation Rate` theo branch.** Ở VaR 1%, branch `Transformers` có `pass_rate` trung bình cao nhất (`0.228`), tiếp theo là `Moirai` (`0.141`) và `GARCH` (`0.106`). Tuy nhiên, nếu xét model đơn lẻ, `GARCH-LSTM-Hybrid` vẫn vượt trội.

| Branch | Pass Rate trung bình | Violation Rate trung bình | Kupiec p trung bình | Christoffersen p trung bình |
|---|---:|---:|---:|---:|
| GARCH | 0.106 | 0.0034 | 0.077 | 0.749 |
| Moirai | 0.141 | 0.0043 | 0.089 | 0.371 |
| Transformers | 0.228 | 0.0074 | 0.188 | 0.389 |

Transformers có `Violation Rate = 0.0074`, gần target 1% hơn GARCH và Moirai. Điều này giải thích vì sao nhóm transformer cải thiện mạnh về pass rate ở VaR 1%.

## 7. Horizon sensitivity

![Pass rate by horizon](../figures/pass_rate_by_horizon_heatmap.png)

**Hình 7. `Pass Rate` theo horizon cho VaR 1%.** Pass rate tăng nhẹ từ `0.175` ở horizon 1 lên `0.205` ở horizon 21. Mẫu hình này tương tự nhưng cao hơn VaR 5%.

| Horizon | Pass Rate | Violation Rate | Kupiec p | Christoffersen p |
|---:|---:|---:|---:|---:|
| 1 | 0.175 | 0.0063 | 0.150 | 0.498 |
| 3 | 0.181 | 0.0059 | 0.143 | 0.452 |
| 5 | 0.193 | 0.0060 | 0.167 | 0.481 |
| 10 | 0.187 | 0.0064 | 0.161 | 0.447 |
| 21 | 0.205 | 0.0058 | 0.126 | 0.434 |

Ở VaR 1%, horizon dài hơn không làm backtesting tệ đi. Điều này có thể do volatility ở horizon dài được làm mượt, làm giảm nhiễu của threshold extreme-tail.

## 8. Dataset sensitivity

![Violation rate by dataset](../figures/violation_rate_by_dataset_heatmap.png)

**Hình 8. `Violation Rate` theo dataset cho VaR 1%.** Ranking dataset thay đổi mạnh so với VaR 5%. `VN_INDEX` có pass rate cao nhất (`0.526`), trong khi `EURONEXT_100` vẫn thấp nhất (`0.021`).

| Dataset | Pass Rate | Violation Rate | Kupiec p | Christoffersen p |
|---|---:|---:|---:|---:|
| VN_INDEX | 0.526 | 0.0051 | 0.146 | 0.970 |
| VN30_INDEX | 0.358 | 0.0035 | 0.063 | 1.000 |
| DAX_40 | 0.284 | 0.0049 | 0.090 | 0.845 |
| KOSPI_INDEX | 0.179 | 0.0145 | 0.148 | 0.258 |
| SNP500 | 0.158 | 0.0067 | 0.367 | 0.268 |
| EURONEXT_100 | 0.021 | 0.0038 | 0.085 | 0.277 |

Sự thay đổi này cho thấy calibration phụ thuộc không chỉ model mà còn tail level và thị trường. Một dataset có thể dễ pass ở 5% nhưng không còn dẫn đầu ở 1%, vì tail 1% đo hành vi cực đoan hơn.

## 9. Kết luận cho VaR 1%

VaR 1% làm rõ ba điểm chính. Thứ nhất, `GARCH-LSTM-Hybrid` tiếp tục là mô hình mạnh nhất về risk backtesting. Thứ hai, nhóm Transformers cải thiện đáng kể ở extreme tail, đặc biệt Autoformer và Informer ở tier nhỏ/trung bình. Thứ ba, Moirai vẫn dẫn đầu về forecast accuracy nhưng không dẫn đầu về risk reliability.

Kết quả này củng cố kết luận chung: risk calibration phải được kiểm tra trực tiếp ở từng tail level. Không nên suy luận hiệu quả VaR 1% từ kết quả VaR 5%, và cũng không nên suy luận VaR performance từ `QLIKE`.
