# Báo cáo phương pháp Moirai2 VaR-aware full fine-tuning

Tài liệu này ghi nhận kết quả thử nghiệm phương pháp mới trong:

```text
experiments/moirai_var_aware/
```

Kết quả được phân tích từ 9 file CSV trong:

```text
output/modal_moirai_var_loss/moirai_var_full_lambda_0.2_predictions/
```

Đây là thử nghiệm **Moirai2 VaR-aware full fine-tuning**, khác với bản head-only trước đó ở chỗ backbone Moirai2 được mở khóa và cập nhật trọng số trong quá trình huấn luyện.

## 1. Mục tiêu phương pháp

Mục tiêu của phương pháp là kiểm tra liệu việc fine-tune Moirai2 bằng một objective có thành phần VaR-aware có thể cải thiện đồng thời hai mục tiêu hay không:

1. Dự báo realized volatility chính xác hơn.
2. Tạo VaR forecast tốt hơn khi volatility forecast được chuyển sang VaR.

Objective được dùng:

```text
Loss = MSE(predicted_volatility, true_volatility)
       + lambda_var * QuantileLoss(realized_return, VaR_1%)
```

Trong thử nghiệm này:

```text
lambda_var = 0.2
alpha = 0.01
VaR distribution = Student-t
nu = 4
tuning_mode = full
backbone_lr = 1e-5
head_lr = 1e-3
model = moirai2
```

## 2. Kiến trúc mô hình

Kiến trúc tổng quát:

```text
60-day return window
-> Moirai2 pretrained backbone
-> representation sequence
-> mean pooling
-> MLP volatility head
-> predicted volatility for horizons [1, 3, 5, 10, 21]
-> VaR-aware training loss
```

Đầu vào là cửa sổ 60 ngày log-return quá khứ. Chuỗi này được pad thành 64 điểm, chia thành 4 patch, mỗi patch 16 giá trị. Moirai2 encoder biến các patch này thành một chuỗi vector representation. Sau đó mô hình lấy mean pooling theo chiều thời gian để thu được một vector đặc trưng duy nhất cho mỗi sample.

Prediction head:

```text
Linear(d_model -> 256)
ReLU
Dropout(0.2)
Linear(256 -> 5)
```

Đầu ra là 5 forecast volatility tương ứng với các horizon:

```text
1, 3, 5, 10, 21
```

Ở chế độ full fine-tuning, cả backbone Moirai2 và MLP head đều được cập nhật:

```text
Moirai2 backbone: lr = 1e-5
MLP head:         lr = 1e-3
```

## 3. Khác biệt so với head-only

Head-only version:

```text
Freeze Moirai2 backbone
Train only MLP head
```

Full fine-tuning version:

```text
Unfreeze Moirai2 backbone
Train backbone + MLP head
```

Do đó, full fine-tuning có khả năng thích nghi representation của Moirai2 với dữ liệu tài chính tốt hơn, nhưng cũng có rủi ro overfit và chi phí GPU cao hơn.

## 4. Dữ liệu và output

Thử nghiệm full fine-tuning đã sinh ra 9 CSV, tương ứng 9 thị trường:

```text
DAX_40
EuroNext_100
IBEX_35
KOSPI_index
Nikkei_225
SMI
snp500
VN30_INDEX
VN_INDEX
```

Tổng hợp kiểm tra kỹ thuật:

| Hạng mục | Kết quả |
|---|---:|
| Số CSV | 9 |
| Số dòng | 42,475 |
| Số model | 1 |
| Model | moirai2 |
| Số dataset | 9 |
| Horizons | 1, 3, 5, 10, 21 |
| Missing values | 0 |
| Duplicate keys | 0 |

Schema output:

```text
dataset, branch, tier, model, time, horizon,
log_return, true_volatility, predict_volatility
```

Nhãn output:

```text
branch = Moirai_VAR_FT
tier = full_lambda_0.2
model = moirai2
```

## 5. Kết quả so với baseline Moirai2

So sánh paired theo cùng dataset, model, time và horizon với baseline `Moirai/moirai2` trong `output/merged_all_predictions.csv`.

### 5.1 Forecasting accuracy

| Metric | Baseline Moirai2 | Full fine-tune | Thay đổi |
|---|---:|---:|---:|
| MSE | 0.022862 | 0.021569 | -5.66% |
| MAE | 0.090709 | 0.087833 | -3.17% |
| QLIKE | 0.009079 | 0.008632 | -4.92% |

Full fine-tuning cải thiện cả 3 metric forecast. Cải thiện MAE có bằng chứng thống kê ở setting-level Wilcoxon:

```text
MAE p = 0.032716
```

MSE và QLIKE cũng giảm, nhưng chưa đủ mạnh ở mức kiểm định setting-level:

```text
MSE p = 0.071714
QLIKE p = 0.206792
```

Diễn giải:

```text
Full fine-tuning giúp Moirai2 học realized volatility tốt hơn baseline.
```

### 5.2 VaR quantile loss

| VaR method | Baseline | Full fine-tune | Thay đổi |
|---|---:|---:|---:|
| Normal VaR 1% | 0.048123 | 0.048504 | +0.79% |
| Normal VaR 5% | 0.135863 | 0.136091 | +0.17% |
| Student-t VaR 1% | 0.047685 | 0.047421 | -0.55% |
| Student-t VaR 5% | 0.142944 | 0.142712 | -0.16% |

Kết quả VaR không đồng nhất. Full fine-tuning làm xấu hơn Normal VaR, nhưng cải thiện nhẹ Student-t VaR.

Diễn giải:

```text
Cải thiện volatility accuracy không tự động chuyển thành cải thiện VaR dưới mọi giả định phân phối.
```

### 5.3 Violation calibration

| Method | Target | Baseline violation | Full violation | Nhận xét |
|---|---:|---:|---:|---|
| Normal 1% | 0.01 | 0.022154 | 0.023331 | Full lệch hơn |
| Normal 5% | 0.05 | 0.052384 | 0.054126 | Full lệch hơn |
| Student-t 1% | 0.01 | 0.005297 | 0.005580 | Full gần target hơn nhẹ |
| Student-t 5% | 0.05 | 0.028016 | 0.028746 | Full gần target hơn nhẹ |

Full fine-tuning cải thiện calibration theo Student-t, nhưng làm Normal calibration xấu hơn.

## 6. Kết quả so với head-only VaR-aware

Head-only là bản:

```text
Moirai2 frozen backbone + MLP head + lambda_var = 0.2
```

### 6.1 Forecasting accuracy

| Metric | Head-only | Full fine-tune | Thay đổi |
|---|---:|---:|---:|
| MSE | 0.022677 | 0.021569 | -4.89% |
| MAE | 0.093985 | 0.087833 | -6.55% |
| QLIKE | 0.008798 | 0.008632 | -1.88% |

Full fine-tuning cải thiện rõ so với head-only, đặc biệt ở MAE:

```text
MAE p = 0.000053
```

Diễn giải:

```text
Mở khóa backbone có lợi cho forecasting accuracy.
```

### 6.2 VaR quantile loss

| VaR method | Head-only | Full fine-tune | Thay đổi |
|---|---:|---:|---:|
| Normal VaR 1% | 0.047709 | 0.048504 | +1.67% |
| Normal VaR 5% | 0.135594 | 0.136091 | +0.37% |
| Student-t VaR 1% | 0.047640 | 0.047421 | -0.46% |
| Student-t VaR 5% | 0.143096 | 0.142712 | -0.27% |

Full fine-tuning tốt hơn head-only về Student-t VaR, nhưng kém hơn ở Normal VaR.

Kết quả Student-t VaR 5% có bằng chứng thống kê nhẹ:

```text
Student-t VaR 5% p = 0.035333
```

## 7. Rank với toàn bộ nhóm model

Khi đưa full fine-tuned Moirai2 vào cùng bảng rank với các mô hình baseline và head-only, kết quả forecast rất mạnh:

| Metric | Rank của full fine-tuned Moirai2 |
|---|---:|
| MSE | #1 |
| MAE | #1 |
| QLIKE | #1 |

Tuy nhiên, ở VaR quantile loss, rank chưa vượt trội:

| VaR metric | Rank xấp xỉ |
|---|---:|
| Normal VaR 1% | #9 |
| Normal VaR 5% | #9 |
| Student-t VaR 1% | #8 |
| Student-t VaR 5% | #8 |

Diễn giải:

```text
Full fine-tuned Moirai2 là model rất mạnh cho realized volatility forecasting, nhưng chưa phải model tốt nhất cho VaR forecasting.
```

## 8. Hành vi mô hình

Từ 9 CSV, có thể mô tả hành vi mô hình như sau:

1. Mô hình học tốt hơn phần scale/level của realized volatility.
2. Fine-tuning backbone giúp giảm forecast error so với cả baseline và head-only.
3. Thành phần VaR-aware hiện tại chưa đủ mạnh để tạo cải thiện tail-risk nhất quán.
4. Student-t VaR phù hợp hơn Normal VaR cho mô hình này, vì Student-t phản ánh fat-tail tốt hơn.
5. Normal VaR có xu hướng bị ảnh hưởng xấu khi volatility forecast thay đổi sau fine-tuning.

Điểm quan trọng:

```text
Mô hình đang học VaR theo cách gián tiếp: dự báo volatility trước, sau đó suy VaR từ volatility bằng giả định phân phối.
```

Do đó, nếu volatility forecast tốt hơn nhưng tail distribution chưa đúng, VaR vẫn có thể không cải thiện.

## 9. Trade-off có xứng đáng không?

Nếu mục tiêu chính là realized volatility forecasting:

```text
Trade-off này xứng đáng.
```

Lý do:

- MSE giảm 5.66% so với baseline.
- MAE giảm 3.17% so với baseline.
- QLIKE giảm 4.92% so với baseline.
- Full fine-tuned Moirai2 đứng #1 ở cả MSE, MAE và QLIKE.

Nếu mục tiêu chính là VaR/tail-risk forecasting:

```text
Trade-off này chưa đủ xứng đáng.
```

Lý do:

- Normal VaR 1% và 5% xấu hơn.
- Student-t VaR chỉ cải thiện nhẹ.
- Rank VaR chưa vượt các nhóm GARCH/Transformer.
- Thành phần VaR trong loss vẫn là regularizer phụ, không phải direct VaR output.

Kết luận đúng mức:

```text
Full fine-tuning đáng giá như một cải tiến volatility forecasting, nhưng chưa đủ để claim là cải tiến VaR forecasting vượt trội.
```

## 10. Novelty nên kết luận theo hướng nào?

Novelty của phương pháp này nên được viết theo hướng:

```text
VaR-aware fine-tuning of a time-series foundation model for realized volatility forecasting.
```

Không nên claim:

```text
Đề xuất một mô hình VaR forecasting vượt trội toàn diện.
```

Nên claim:

```text
Phương pháp cho thấy full fine-tuning Moirai2 có thể cải thiện đáng kể realized volatility forecasting, nhưng cải thiện volatility accuracy không tự động chuyển hóa thành VaR calibration/quantile-loss superiority. Kết quả này củng cố luận điểm rằng point forecasting accuracy và tail-risk forecasting là hai mục tiêu liên quan nhưng không đồng nhất.
```

Đây là một contribution có giá trị vì nó cho thấy giới hạn của hướng volatility-implied VaR.

## 11. Cơ sở khoa học cho Student-t VaR

Khi diễn giải kết quả, không nên chỉ nói "Student-t tốt hơn Normal" như một nhận xét cảm tính. Claim hợp lý hơn là:

```text
Student-t VaR có cơ sở kinh tế lượng và thực nghiệm mạnh hơn Normal VaR trong bối cảnh returns tài chính có fat tails, nhưng trong kết quả hiện tại mức cải thiện của full fine-tuned Moirai2 dưới Student-t vẫn còn nhỏ.
```

### 11.1 Financial returns thường không tuân theo Normal

Các tài liệu kinh điển về phân phối returns cho thấy returns tài chính thường có đuôi dày hơn Normal:

- Praetz (1972), "The Distribution of Share Price Changes", *The Journal of Business*, 45(1), 49-55. DOI: `10.1086/295425`.
- Blattberg và Gonedes (1974), "A Comparison of the Stable and Student Distributions as Statistical Models for Stock Prices", *The Journal of Business*, 47(2), 244-280. DOI: `10.1086/295634`.

Ý nghĩa cho nghiên cứu này:

```text
Nếu returns có fat tails, Normal VaR có thể đánh giá sai xác suất tail events. Student-t là lựa chọn hợp lý hơn Normal vì có tham số degrees of freedom để điều chỉnh độ dày của tail.
```

### 11.2 Student-t trong mô hình volatility/VaR có tiền lệ mạnh

Bollerslev (1987) là tài liệu kinh điển cho việc dùng conditional Student-t distribution trong mô hình heteroskedastic returns:

- Bollerslev (1987), "A Conditionally Heteroskedastic Time Series Model for Speculative Prices and Rates of Return", *The Review of Economics and Statistics*, 69(3), 542-547. DOI: `10.2307/1925546`.

Điểm liên hệ:

```text
Phương pháp trong nghiên cứu này cũng dự báo volatility trước, rồi chuyển volatility forecast thành VaR bằng giả định phân phối. Vì vậy, khi dữ liệu tài chính có tail dày, Student-t VaR là lựa chọn có cơ sở hơn Normal VaR.
```

### 11.3 VaR cần được đánh giá bằng backtesting/calibration, không chỉ forecast loss

Christoffersen (1998) xây dựng framework kiểm định interval forecast, nhấn mạnh conditional coverage thay vì chỉ nhìn unconditional error:

- Christoffersen (1998), "Evaluating Interval Forecasts", *International Economic Review*, 39(4), 841-862. DOI: `10.2307/2527341`.

Điểm liên hệ:

```text
Kết quả full fine-tuned Moirai2 cải thiện MSE/MAE/QLIKE nhưng không cải thiện Normal VaR. Điều này phù hợp với lập luận rằng point forecast accuracy và interval/tail forecast calibration là hai tầng đánh giá khác nhau.
```

### 11.4 Nghiên cứu VaR thực nghiệm cũng nhấn mạnh vai trò của fat-tailed distribution

Một số nghiên cứu VaR thực nghiệm cho thấy lựa chọn phân phối innovation ảnh hưởng trực tiếp đến chất lượng VaR:

- Fan, Zhang và Yu (2008), "Estimation of Value-at-Risk for energy commodities via fat-tailed GARCH models", *Energy Economics*. Bài này khảo sát tác động của fat-tailed innovation process đến one-day-ahead VaR estimates.
- "Can the Student-t distribution provide accurate value at risk?", *The Journal of Risk Finance*, 7(3), 292-300. Bài này trực tiếp khảo sát liệu VaR dựa trên Student-t distribution có capture market risk hiệu quả hơn benchmark Normal hay không.

Điểm liên hệ:

```text
Khi so sánh Normal VaR và Student-t VaR trong nghiên cứu này, kết quả Student-t tốt hơn nhẹ ở full fine-tuned Moirai2 là phù hợp với hướng tài liệu cho rằng tail assumption có ảnh hưởng trực tiếp đến VaR quantile.
```

### 11.5 Cách viết claim được hỗ trợ bởi literature

Claim nên dùng:

```text
The use of Student-t VaR is theoretically and empirically motivated by the fat-tailed nature of financial returns. Prior studies in financial econometrics show that Student-t innovations can better represent speculative returns than Gaussian innovations, and VaR backtesting literature emphasizes that distributional calibration must be evaluated separately from volatility point forecast accuracy.
```

Claim tiếng Việt:

```text
Việc dùng Student-t VaR có cơ sở khoa học vì returns tài chính thường có fat tails, trong khi Normal có tail mỏng. Các nghiên cứu kinh tế lượng tài chính kinh điển cho thấy Student-t là lựa chọn phù hợp hơn cho speculative returns; đồng thời, tài liệu backtesting VaR cho thấy forecast accuracy của volatility không đủ để đảm bảo tail-risk calibration.
```

Không nên claim quá mức:

```text
Student-t VaR luôn vượt trội hơn Normal VaR.
```

Vì trong kết quả hiện tại, Student-t chỉ cải thiện nhẹ ở full fine-tuned Moirai2:

```text
Student-t VaR 1% quantile loss cải thiện 0.55% so với baseline.
Student-t VaR 5% quantile loss cải thiện 0.16% so với baseline.
```

Do đó kết luận đúng là:

```text
Student-t VaR là lựa chọn hợp lý hơn về mặt phân phối tail cho bài toán này, nhưng bằng chứng thực nghiệm hiện tại chỉ ủng hộ mức cải thiện nhẹ, chưa đủ để khẳng định ưu thế vượt trội toàn diện.
```

## 12. Hạn chế của phương pháp hiện tại

Hạn chế chính:

1. Model không dự báo VaR trực tiếp.
2. VaR được suy từ volatility bằng Student-t hoặc Normal assumption.
3. VaR loss chỉ tác động lên horizon index 0 trong training hiện tại.
4. `lambda_var = 0.2` có thể còn thấp nếu mục tiêu là tail-risk.
5. Tail events 1% hiếm, nên gradient từ VaR loss yếu và nhiễu.
6. Full fine-tuning có thể overfit volatility signal mà không cải thiện tail calibration.

## 13. Hướng tiếp theo

Hướng cải tiến hợp lý nhất là chuyển sang **direct VaR head**:

```text
Moirai2 backbone
-> shared representation
-> volatility head
-> VaR 1% head
-> VaR 5% head
```

Loss đề xuất:

```text
Total Loss =
lambda_vol * MSE(volatility)
+ lambda_q01 * QuantileLoss(return_h, VaR_1%_h)
+ lambda_q05 * QuantileLoss(return_h, VaR_5%_h)
+ lambda_order * max(0, VaR_1% - VaR_5%)
```

Lợi ích:

- Không phụ thuộc hoàn toàn vào Normal/Student-t assumption.
- VaR được học trực tiếp từ dữ liệu.
- Có thể tối ưu đồng thời VaR 1% và VaR 5%.
- Có thể enforce constraint `VaR_1% <= VaR_5%`.

Kế hoạch thực nghiệm nên làm:

1. Direct VaR head với frozen Moirai2.
2. Direct VaR head với full fine-tuned Moirai2.
3. Ablation `lambda_q01`, `lambda_q05`.
4. So sánh với volatility-implied VaR hiện tại.

## 14. Kết luận ngắn

Kết luận cuối cho phương pháp mới:

```text
Moirai2 VaR-aware full fine-tuning là một cải tiến có giá trị cho realized volatility forecasting, đạt rank #1 theo MSE, MAE và QLIKE khi so với toàn bộ baseline. Tuy nhiên, lợi thế này chưa chuyển hóa thành ưu thế VaR forecasting toàn diện. Kết quả ủng hộ hướng nghiên cứu rằng tail-risk forecasting cần được mô hình hóa trực tiếp hơn, thay vì chỉ suy VaR từ volatility forecast.
```
