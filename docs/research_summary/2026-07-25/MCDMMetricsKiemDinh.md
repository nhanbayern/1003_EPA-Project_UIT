# Báo cáo Nghiên cứu: Hệ Metric và Kiểm định cho MCDM Risk-Sensitive

**Ngày thực hiện:** 25/07/2026  
**Chủ đề:** Chuẩn hóa hệ metric đánh giá mô hình volatility forecasting, bổ sung dynamic-tracking metrics, forecast sanity gate và kiểm định dominance cho `GARCH-Autoformer` trong pipeline MCDM.

---

## Tóm tắt

Tài liệu này mô tả hệ metric đang được dùng trong pipeline MCDM, gồm các metric forecast/risk ban đầu, các metric mới thêm để phạt mô hình dự báo phẳng, và các kiểm định mới dùng để đánh giá ưu thế của `GARCH-Autoformer`.

Script liên quan:

```text
stats_analysis/run_mcdm_evaluation.py
```

Output liên quan:

```text
output/mcdm_results/MCDMYYYYMMDDHHMMSS/
```

## 1. Mục tiêu đánh giá

Pipeline MCDM không chỉ hỏi mô hình nào có sai số dự báo nhỏ nhất. Mục tiêu hiện tại là đánh giá đồng thời:

```text
1. Forecast accuracy: mô hình có dự báo volatility sát thực tế không?
2. Dynamic tracking: mô hình có bám được hình dạng biến động theo thời gian không?
3. Risk calibration: VaR sinh từ forecast có vượt qua backtesting không?
4. MCDM dominance: GARCH-Autoformer có vượt trội hơn các mô hình còn lại theo SAW/TOPSIS không?
```

Điểm quan trọng là một mô hình có thể có VaR calibration tốt nhưng dự báo volatility gần như đường phẳng. Vì vậy pipeline mới thêm các metric phạt dynamic tracking để loại các mô hình không thật sự học được chuỗi thời gian.

## 2. Forecast accuracy metrics

### Mean Squared Error (MSE)

MSE đo sai số bình phương trung bình giữa volatility thực tế và volatility dự báo:

```text
MSE = mean((true_volatility - predict_volatility)^2)
```

Ý nghĩa:

```text
MSE càng thấp càng tốt.
MSE phạt mạnh các sai số lớn vì có bình phương.
```

Vai trò trong MCDM:

```text
MSE là cost criterion.
```

### Mean Absolute Error (MAE)

MAE đo sai số tuyệt đối trung bình:

```text
MAE = mean(abs(true_volatility - predict_volatility))
```

Ý nghĩa:

```text
MAE càng thấp càng tốt.
MAE dễ diễn giải hơn MSE vì cùng scale với volatility.
```

Vai trò trong MCDM:

```text
MAE là cost criterion.
```

### Quasi-Likelihood Loss (QLIKE)

QLIKE là metric phổ biến trong volatility forecasting vì có liên hệ với likelihood của mô hình variance. Nó thường ổn định hơn khi đánh giá forecast volatility/variance trong tài chính.

Ý nghĩa:

```text
QLIKE càng thấp càng tốt.
QLIKE phù hợp để đánh giá forecast volatility trong bối cảnh tài chính.
```

Vai trò trong MCDM:

```text
QLIKE là cost criterion.
```

## 3. Metric mới để phạt mô hình dự báo phẳng

Các metric này được thêm vì một số mô hình Transformer có thể dự báo gần như đường thẳng trong từng `dataset-horizon`, nhưng vẫn có thể đạt VaR pass rate tốt do VaR calibration tình cờ phù hợp.

Nếu chỉ dùng `MSE`, `MAE`, `QLIKE`, `pass_rate`, và `abs_violation_error`, pipeline có thể xếp hạng cao cho mô hình không bám được dynamic volatility. Vì vậy cần metric kiểm tra hình dạng chuỗi dự báo.

### Volatility Standard-Deviation Ratio Error

Metric này đo độ lệch giữa độ biến động của forecast và độ biến động của volatility thực tế:

```text
std_ratio = std(predict_volatility) / std(true_volatility)
volatility_std_ratio_error = abs(std_ratio - 1)
```

Ý nghĩa:

```text
volatility_std_ratio_error càng thấp càng tốt.
```

Diễn giải:

```text
Nếu predict_volatility biến động tương đương true_volatility:
std_ratio ≈ 1
volatility_std_ratio_error ≈ 0

Nếu predict_volatility gần như đường phẳng:
std(predict_volatility) ≈ 0
std_ratio ≈ 0
volatility_std_ratio_error ≈ 1
```

Vì vậy metric này phạt mạnh các mô hình dự báo phẳng.

Trong pipeline, metric được tính theo từng:

```text
branch / tier / model / dataset / horizon
```

sau đó lấy trung bình lên cấp model.

Lý do không tính gộp toàn bộ dữ liệu:

```text
Nếu gộp nhiều dataset-horizon lại, forecast có thể nhìn như vẫn có variance do các mức nền khác nhau giữa thị trường/horizon.
Nhưng trong từng dataset-horizon, nó vẫn có thể là đường thẳng.
Vì vậy phải tính metric ở cấp case trước rồi mới aggregate.
```

Vai trò trong MCDM:

```text
Volatility Standard-Deviation Ratio Error là cost criterion.
```

### Volatility Tracking Correlation Error

Metric này đo khả năng forecast bám hình dạng chuỗi true volatility:

```text
tracking_correlation = corr(true_volatility, predict_volatility)
tracking_correlation_error = 1 - max(tracking_correlation, 0)
```

Ý nghĩa:

```text
tracking_correlation càng cao càng tốt.
tracking_correlation_error càng thấp càng tốt.
```

Diễn giải:

```text
Nếu forecast bám tốt true volatility:
tracking_correlation gần 1
tracking_correlation_error gần 0

Nếu forecast không có tương quan hoặc tương quan âm:
tracking_correlation <= 0
tracking_correlation_error = 1
```

Metric này phạt các mô hình dự báo sai hướng hoặc không bám được biến động thời gian.

Vai trò trong MCDM:

```text
Volatility Tracking Correlation Error là cost criterion.
```

## 4. Forecast sanity gate

Ngoài việc đưa dynamic-tracking metrics vào weighted score, pipeline còn dùng một bước lọc trước khi xếp hạng:

```text
volatility_std_ratio_error <= 0.9
tracking_correlation > 0
```

Nếu mô hình không qua gate này, nó bị ghi vào:

```text
ExcludedModels.csv
```

và không được xếp hạng trong:

```text
SAWRanking.csv
TOPSISRanking.csv
CombinedMCDMRanking.csv
```

Pipeline cũng xuất thêm ảnh:

```text
VolatilityStdRatioErrorByModelTier.png
```

Ảnh này vẽ `volatility_std_ratio_error` cho từng model-tier thay vì gộp theo model family. Tên model-tier dùng convention compact PascalCase, ví dụ `AutoformerTier3`, để tránh snake_case trong deliverables hình ảnh và thuận tiện khi đưa vào báo cáo.

Lý do khoa học:

```text
Mô hình không có biến động dự báo đủ tối thiểu hoặc không có tương quan dương với true volatility không nên được xem là forecast model hợp lệ, dù VaR backtesting có thể tình cờ tốt.
```

Về mặt phương pháp luận, hai ngưỡng này không được hiểu là một bảo đảm rằng mô hình đã dự báo tốt, mà là một điều kiện sàng lọc tối thiểu để loại các mô hình không thể hiện được hành vi động học cơ bản của chuỗi volatility. Ngưỡng `volatility_std_ratio_error <= 0.9` tương đương với điều kiện `std(predict_volatility) / std(true_volatility) >= 0.1` nếu forecast có variance thấp hơn true volatility. Do đó, một mô hình dự báo gần như đường thẳng trong từng `dataset-horizon` sẽ có `std(predict_volatility) ≈ 0`, kéo `std_ratio ≈ 0` và làm `volatility_std_ratio_error ≈ 1`; trường hợp này sẽ bị loại vì vượt ngưỡng 0.9. Cách đặt ngưỡng 0.9 là cố ý tương đối lỏng: pipeline không yêu cầu forecast phải có độ dao động đúng bằng true volatility, mà chỉ yêu cầu forecast có ít nhất một mức dao động tối thiểu, khoảng 10% độ lệch chuẩn của chuỗi thực tế, để tránh việc một đường ngang được xem là mô hình dự báo hợp lệ.

Tuy nhiên, chỉ có variance không đủ để chứng minh forecast bám đúng chuỗi thời gian, vì một mô hình có thể dao động mạnh nhưng sai pha hoặc đi ngược chiều true volatility. Vì vậy điều kiện thứ hai `tracking_correlation > 0` được dùng như một ràng buộc đồng biến tối thiểu: forecast phải có tương quan dương với true volatility trong cùng `dataset-horizon`. Nếu correlation bằng 0 hoặc âm, mô hình không chứng minh được rằng khi volatility thực tế tăng thì forecast cũng có xu hướng tăng, và vì vậy không nên được đưa vào ranking MCDM, ngay cả khi các VaR backtests tình cờ cho tỷ lệ violation đẹp. Hai điều kiện này bổ sung cho nhau: điều kiện độ lệch chuẩn loại forecast phẳng, còn điều kiện tương quan loại forecast có dao động nhưng không tracking đúng hướng.

Đây là cách kiểm tra phù hợp với nguyên tắc đánh giá forecast liên tục, trong đó forecast phải được so sánh trực tiếp với observation và cần đo cả mức sai số lẫn quan hệ giữa hai chuỗi; hướng dẫn verification cho continuous forecasts cũng nhấn mạnh bản chất của đánh giá forecast liên tục là so sánh forecast và observed values, đồng thời đo mối quan hệ giữa hai đại lượng đó [DTCenter METplus, Continuous Forecast Verification](https://dtcenter.org/metplus-practical-session-guide-version-5-0/basic-verification-statistics-review/continuous-forecasts/verification-statistics-continuous-forecasts/). Hiện tượng forecast suy biến về một đường quá trơn, hoặc lặp lại một tín hiệu đơn giản thay vì học dynamic thật, cũng đã được ghi nhận trong literature về time-series forecasting. Kosma, Nikolentzos, Xu và Vazirgiannis (2022) mô tả hiện tượng neural forecasting models có xu hướng “copy the past” khi tối ưu các loss như MSE/MAE trong môi trường nhiễu và bất định; dù bài toán của họ không hoàn toàn giống forecast phẳng trong volatility, nó cung cấp một dẫn chứng trực tiếp rằng mô hình học máy cho chuỗi thời gian có thể rơi vào nghiệm suy biến đơn giản thay vì học quan hệ động học cần thiết [Kosma et al., 2022](https://arxiv.org/abs/2207.13441). Gneiting (2011) cũng chỉ ra rằng việc đánh giá point forecast bằng một scoring function như squared error hoặc absolute error có thể dẫn tới kết luận sai nếu scoring function không được khớp rõ với forecast task; điều này củng cố lý do không nên để một vài loss trung bình quyết định toàn bộ ranking khi forecast có dấu hiệu suy biến hình dạng [Gneiting, 2011](https://arxiv.org/abs/0912.0902). Trong forecast evaluation nói chung, Gneiting, Balabdaoui và Raftery (2007) cũng nhấn mạnh nguyên tắc forecast hữu ích không chỉ cần calibrated mà còn cần sharp/informative; một dự báo gần như không thay đổi có thể đạt một số tiêu chí calibration nhưng thiếu thông tin động học cần thiết để được xem là forecast tốt [Gneiting et al., 2007](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jrssb.pdf).

Trong bối cảnh volatility forecasting, các loss function như MSE và QLIKE vẫn cần được giữ vì chúng là các thước đo chuẩn để so sánh forecast volatility; Patton (2011) chỉ ra vai trò đặc biệt của MSE và QLIKE trong so sánh volatility forecasts khi volatility thực tế là latent và thường phải dùng proxy [Patton, 2011](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf). Patton và Sheppard cũng nhấn mạnh rằng đánh giá volatility/correlation forecasts phải chú ý tới việc đối tượng volatility là latent và thường cần proxy, nghĩa là việc chọn loss và diagnostic phù hợp là một phần cốt lõi của forecast evaluation chứ không chỉ là tính một chỉ số duy nhất [Patton & Sheppard, 2009](https://public.econ.duke.edu/~ap172/Patton_Sheppard_29oct07.pdf). Nhưng MSE/QLIKE chủ yếu đo độ lớn sai số hoặc ratio loss, không tự động phát hiện đầy đủ vấn đề forecast phẳng nếu mô hình được lợi từ mức dự báo trung bình tương đối ổn hoặc từ VaR calibration tình cờ. Vì vậy `volatility_std_ratio_error` và `tracking_correlation_error` được thêm như các tiêu chí kiểm tra shape/dynamics trước khi tổng hợp MCDM. Nói chính xác, forecast sanity gate không khẳng định mô hình qua gate là mô hình tốt; nó chỉ khẳng định mô hình không rơi vào lỗi tối thiểu nghiêm trọng là dự báo gần như constant hoặc không đồng biến với volatility thực tế. Sau gate, chất lượng mô hình vẫn phải được đánh giá tiếp bằng MSE, MAE, QLIKE, VaR pass rate, absolute violation error, SAW, TOPSIS và dominance tests.

## 5. Risk calibration metrics

### VaR Backtesting Pass Rate

Pipeline tính VaR backtesting theo từng case:

```text
branch / tier / model / dataset / horizon
```

Một case pass nếu:

```text
backtest_pass = kupiec_pass AND independence_pass
```

Sau đó:

```text
pass_rate = passed_cases / valid_risk_cases
```

Ý nghĩa:

```text
pass_rate càng cao càng tốt.
```

Vai trò trong MCDM:

```text
VaR 1% Backtesting Pass Rate là benefit criterion.
VaR 5% Backtesting Pass Rate là benefit criterion.
```

### Absolute Violation Error

Metric này đo tỷ lệ violation lệch khỏi mức tail alpha bao nhiêu:

```text
violation_rate = violation_count / n_risk
abs_violation_error = abs(violation_rate - alpha)
```

Với VaR 1%:

```text
alpha = 0.01
```

Với VaR 5%:

```text
alpha = 0.05
```

Ý nghĩa:

```text
abs_violation_error càng thấp càng tốt.
```

Lưu ý:

```text
abs_violation_error không kiểm tra independence.
Nó chỉ đo độ gần của violation_rate với alpha.
Vì vậy nó bổ sung cho pass_rate, không thay thế Kupiec/Christoffersen.
```

Vai trò trong MCDM:

```text
VaR 1% Absolute Violation Error là cost criterion.
VaR 5% Absolute Violation Error là cost criterion.
```

## 6. Trọng số MCDM

Pipeline hiện chạy hai cấu hình.

### Cấu hình 5,5

```text
Accuracy block = 50%
Risk block = 50%
```

Trọng số:

| Metric | Direction | Weight |
|---|---|---:|
| Mean Squared Error (MSE) | cost | 0.100 |
| Mean Absolute Error (MAE) | cost | 0.100 |
| Quasi-Likelihood Loss (QLIKE) | cost | 0.100 |
| Volatility Standard-Deviation Ratio Error | cost | 0.100 |
| Volatility Tracking Correlation Error | cost | 0.100 |
| VaR 1% Backtesting Pass Rate | benefit | 0.125 |
| VaR 1% Absolute Violation Error | cost | 0.125 |
| VaR 5% Backtesting Pass Rate | benefit | 0.125 |
| VaR 5% Absolute Violation Error | cost | 0.125 |

### Cấu hình 3,7

```text
Accuracy block = 30%
Risk block = 70%
```

Trọng số:

| Metric | Direction | Weight |
|---|---|---:|
| Mean Squared Error (MSE) | cost | 0.060 |
| Mean Absolute Error (MAE) | cost | 0.060 |
| Quasi-Likelihood Loss (QLIKE) | cost | 0.060 |
| Volatility Standard-Deviation Ratio Error | cost | 0.060 |
| Volatility Tracking Correlation Error | cost | 0.060 |
| VaR 1% Backtesting Pass Rate | benefit | 0.175 |
| VaR 1% Absolute Violation Error | cost | 0.175 |
| VaR 5% Backtesting Pass Rate | benefit | 0.175 |
| VaR 5% Absolute Violation Error | cost | 0.175 |

## 7. SAW và TOPSIS

### Simple Additive Weighting (SAW)

SAW chuẩn hóa từng metric rồi cộng theo trọng số:

```text
SAW score = sum(weight_j * normalized_score_j)
```

Với benefit metric:

```text
giá trị cao hơn tốt hơn
```

Với cost metric:

```text
giá trị thấp hơn tốt hơn
```

Ý nghĩa:

```text
SAW là phương pháp cộng điểm tuyến tính.
Nó cho phép một metric rất tốt bù một phần cho metric yếu.
```

### TOPSIS

TOPSIS chuẩn hóa vector, nhân trọng số, rồi tính khoảng cách đến:

```text
ideal best: điểm lý tưởng tốt nhất
ideal worst: điểm tệ nhất
```

Closeness coefficient:

```text
TOPSIS score = distance_to_worst / (distance_to_ideal + distance_to_worst)
```

Ý nghĩa:

```text
TOPSIS score càng cao càng tốt.
```

Khác với SAW, TOPSIS nhạy hơn với việc một mô hình nằm xa điểm lý tưởng ở một số chiều quan trọng.

## 8. Kiểm định mới: GARCH-Autoformer dominance test

Sau khi có SAW/TOPSIS score, pipeline kiểm tra `GARCH-Autoformer` có vượt trội hơn các mô hình còn lại không.

Đầu tiên, các tier của cùng một model family được lấy trung bình theo:

```text
display_group
```

Ví dụ:

```text
GARCH-Autoformer (Tier 1)
GARCH-Autoformer (Tier 2)
```

được tổng hợp thành:

```text
GARCH-Autoformer
```

Sau đó so sánh `GARCH-Autoformer` với từng model family còn lại theo:

```text
SAW Composite Score
TOPSIS Closeness Coefficient
```

Một lần thắng được định nghĩa:

```text
H1_win = 1 nếu score(GARCH-Autoformer) > score(model còn lại)
H1_win = 0 nếu ngược lại
```

### Exact binomial test

Giả thuyết:

```text
H0: P(GARCH-Autoformer thắng một model còn lại) = 0.5
H1: P(GARCH-Autoformer thắng một model còn lại) > 0.5
```

Ý nghĩa:

```text
H0 nói rằng GARCH-Autoformer không có ưu thế hệ thống; xác suất thắng chỉ như ngẫu nhiên 50/50.
H1 nói rằng GARCH-Autoformer có xác suất thắng lớn hơn 50%, tức có ưu thế MCDM.
```

Pipeline dùng one-sided exact binomial test:

```text
binomial_p_value_greater
```

Vì số model so sánh không lớn, exact binomial test là kiểm định chính nên dùng.

### One-proportion z-test

Pipeline cũng báo thêm z-test như một kiểm tra xấp xỉ:

```text
H0: p = 0.5
H1: p > 0.5
```

Trong đó:

```text
p = tỷ lệ model còn lại bị GARCH-Autoformer vượt qua
```

Thống kê kiểm định:

```text
z = (observed_win_rate - 0.5) / sqrt(0.5 * 0.5 / n)
```

Ý nghĩa:

```text
z càng lớn và p-value càng nhỏ thì tỷ lệ thắng quan sát được càng vượt xa mức 50%.
```

Lưu ý:

```text
Với n nhỏ, z-test chỉ là xấp xỉ. Exact binomial test đáng tin hơn.
```

## 9. Vì sao không dùng Friedman cho dominance test này?

Friedman phù hợp khi có ma trận nhiều block:

```text
block = dataset-horizon
columns = models
value = metric hoặc rank trong block
```

Nó kiểm tra:

```text
H0: các model có phân phối rank tương đương nhau qua nhiều block
```

Trong khi dominance test hiện tại dùng SAW/TOPSIS score đã aggregate ở cấp model family:

```text
model family -> một score SAW/TOPSIS
```

Vì không còn block lặp lại, Friedman không phù hợp cho câu hỏi:

```text
GARCH-Autoformer thắng bao nhiêu model còn lại theo score aggregate?
```

Nếu muốn dùng Friedman đúng cách, cần xây thêm pipeline block-level MCDM:

```text
1. Tính SAW/TOPSIS riêng cho từng dataset-horizon.
2. Tạo rank model trong từng block.
3. Chạy Friedman/Nemenyi trên ma trận block x model.
```

Đó là một phân tích khác, không phải dominance test trên MCDM aggregate hiện tại.
