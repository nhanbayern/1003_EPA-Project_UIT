# Bố Cục Báo Cáo: GARCH-Autoformer + Sanity Gate + MCDM Dynamic Tracking

> **Luồng tư duy:** Vấn đề & Research Gap → Cơ sở lý thuyết giải thích vấn đề → Đề xuất của chúng tôi → Cơ sở thiết kế đề xuất → Kết quả & Phân tích → Kết luận

---

## 1. ĐẶT VẤN ĐỀ VÀ RESEARCH GAP

### 1.1 Bối cảnh và Động lực nghiên cứu

Dự báo volatility là nền tảng của quản trị rủi ro tài chính (VaR, capital allocation, portfolio management). Tuy nhiên, trong thực tiễn nghiên cứu, **hai thế giới này vẫn bị đánh giá tách rời nhau:**

- **Góc nhìn forecasting:** Tối ưu MSE, MAE, QLIKE — tức là mô hình nào dự báo gần true volatility nhất.
- **Góc nhìn risk management:** Tối ưu VaR backtesting (Kupiec + Christoffersen) — tức là mô hình nào tạo ra tail-risk signal đúng nhất.

Nghiên cứu thực nghiệm trong dự án này (9 chỉ số, 5 horizon, 10+ model family) xác nhận: **mô hình dự báo tốt nhất về MSE/MAE không phải là mô hình có risk calibration tốt nhất** — đây là thực tế đã được ghi nhận bởi Christoffersen & Diebold (2000), nhưng chưa được giải quyết triệt để trong pipeline đánh giá model.

> **Trích dẫn cần có:** Christoffersen & Diebold (2000); Patton (2011); Bollerslev (1986); Zhao et al. (2024) [GARCH-LSTM]; Liu et al. (2025) [Moirai-MoE].

---

### 1.2 Research Gap — Hai lỗ hổng chưa được giải quyết trong literature

#### Gap 1 — Không có cơ chế phát hiện forecast suy biến trước khi xếp hạng

Khi đánh giá nhiều mô hình trong pipeline MCDM, các công trình hiện tại thường dùng trực tiếp MSE/MAE/QLIKE và VaR metrics để xếp hạng, **mà không kiểm tra liệu forecast có thực sự tracking được dynamics của chuỗi hay không.**

Kết quả thực nghiệm trong dự án này cho thấy một số Transformer tier lớn (Informer Tier 3, Reformer Tier 3, Vanilla Tier 3) tạo forecast gần như đường phẳng trong từng `dataset×horizon`:
- `std(predict) / std(true) ≈ 0` → không có biến động dự báo.
- `corr(predict, true) ≤ 0` → không có tương quan dương với true volatility.

Dù vậy, chúng vẫn có thể đạt **VaR violation rate ngẫu nhiên gần alpha** → tình cờ pass backtesting → được xếp vào ranking MCDM như thể là mô hình hợp lệ.

**Lỗ hổng:** Literature chưa có cơ chế *screening trước ranking* để loại các forecast suy biến dạng này.

> **Trích dẫn cần có:**
> - Gneiting et al. (2007) — forecast tốt phải calibrated *và* sharp; một forecast không thay đổi vi phạm nguyên tắc sharpness.
> - Gneiting (2011) — scoring function không matched với forecast task dẫn đến kết luận sai về chất lượng mô hình.
> - Kosma et al. (2022) — neural time-series models có xu hướng "copy the past" và rơi vào nghiệm suy biến dưới MSE/MAE loss.
> - Patton & Sheppard (2009) — diagnostic evaluation cho volatility forecasts phải đo cả magnitude lẫn relationship.
> - DTCenter METplus — verification cho continuous forecasts đòi hỏi đo cả accuracy và correlation.

#### Gap 2 — Thiếu framework MCDM tích hợp dynamic tracking và risk calibration đồng thời

Các nghiên cứu hiện tại về model selection cho volatility forecasting thường:
- **Chỉ dùng forecast accuracy** (MSE, MAE, QLIKE) để xếp hạng → bỏ qua risk calibration.
- **Hoặc chỉ dùng VaR backtesting** → bỏ qua khả năng tracking dynamics.
- Một số ít dùng MCDM (SAW, TOPSIS) nhưng **không bao gồm metric động học** — không phạt forecast phẳng.

Đặc biệt: **không có công trình nào kết hợp forecast sanity gate (loại mô hình không tracking) với MCDM risk-sensitive selection** thành một pipeline nhất quán.

> **Trích dẫn cần có:**
> - Hwang & Yoon (1981) [TOPSIS] — phương pháp MCDM gốc.
> - Kupiec (1995) — VaR coverage test.
> - Christoffersen (1998) — VaR independence test.
> - Patton (2011) — MSE và QLIKE phù hợp đánh giá volatility nhưng không đủ để phát hiện flat forecast.

---

## 2. CƠ SỞ LÝ THUYẾT — TẠI SAO FORECAST PHẲNG XẢY RA?

### 2.1 Tính chất Mean Reversion của chuỗi Volatility tài chính

Volatility tài chính có tính **mean-reverting**: sau các cú sốc, volatility có xu hướng trở về mức trung bình dài hạn (long-run mean). Đây là nền tảng lý thuyết của GARCH và các mô hình variance có điều kiện.

- **Volatility clustering** (Mandelbrot, 1963): giai đoạn volatility cao theo sau giai đoạn cao, thấp theo thấp — nhưng không tăng/giảm mãi mãi.
- **Mean reversion** trong GARCH được mã hóa bởi hệ số persistence `α + β < 1` (Bollerslev, 1986).
- Đặc tính này làm cho chuỗi volatility *trông có vẻ dự báo được* bằng một hằng số xấp xỉ long-run mean.

> **Trích dẫn cần có:** Bollerslev (1986); Engle (1982); Baillie et al. (1996) [FI-GARCH]; Mandelbrot (1963).

---

### 2.2 Mean Reversion ≠ Stationarity — Kiểm định thực nghiệm trên dataset

> ⚠️ **Điểm cốt lõi:** Đây là phần bạn đã kiểm định, cần trình bày bằng chứng thực nghiệm.

**Phương pháp:** ADF test (H0: unit root) + KPSS test (H0: stationary). Kết luận `stationary` chỉ khi **cả hai** thỏa: ADF p < 0.05 **và** KPSS p ≥ 0.05. Nhãn `mixed` khi hai test cho kết luận trái chiều.

**Kết quả tổng quát (45 chuỗi `dataset × horizon`):**

| Split | Stationary | Mixed | Non-stationary |
|-------|-----------|-------|----------------|
| **Full** | **15** (33.3%) | **30** | 0 |
| Train | 9 | 30 | 6 |
| **Val** | **0** | 22 | **23** |
| **Test** | **1** | 18 | **26** |

**Diễn giải khoa học:**
- Trên full sample, 30/45 chuỗi là `mixed` — ADF không bác bỏ mean-reverting behavior nhưng KPSS vẫn phát hiện local non-stationarity.
- **Val/Test có nhiều non-stationary hơn hẳn train** → trong giai đoạn đánh giá, chuỗi thường trải qua regime shift và volatility clustering cục bộ.
- Điều này có nghĩa: mô hình học được mean path trong train **không thể giả định cùng phân phối trong test**.

> **Trích dẫn cần có:** Dickey & Fuller (1979) [ADF]; Kwiatkowski et al. (1992) [KPSS]; Perron (1989) — structural break ảnh hưởng đến unit root tests; Hamilton (1994) — Time Series Analysis, chương kiểm định tính dừng.

---

### 2.3 Tại sao mô hình học thống kê bị "flat forecast"?

Với chuỗi volatility có tính mean-reverting, **tối ưu MSE về mặt lý thuyết có thể ra nghiệm hằng số** (conditional mean của chuỗi) nếu signal-to-noise ratio thấp hoặc mô hình không có đủ inductive bias để phân biệt giai đoạn high/low volatility.

Transformer (Autoformer, Informer, Reformer) với cửa sổ đầu vào chỉ 60 ngày trên chuỗi noise cao:
- Học được *mean path* trong tập train (nơi volatility tương đối ổn định hơn).
- Khi gặp regime shift trong test → áp dụng cùng "chiến lược quay về trung bình" → forecast là đường phẳng quanh long-run mean.
- **Kết quả thực nghiệm:** Informer Tier 3 có `std_ratio_error = 1.000`, `tracking_correlation_error = 0.985` — forecast gần như không biến động dù true volatility có những đột biến lớn.

> **Trích dẫn cần có:**
> - Chen et al. (ICML 2025) — Transformer mạnh nhất khi data có long-range structure; high noise/short sequences giảm lợi thế của attention.
> - Zeng et al. (AAAI 2023) — "Are transformers effective for time series forecasting?" — thách thức về tính hiệu quả của Transformer trong nhiều bài toán TS.
> - Lim & Zohren (2021) — survey deep learning for time-series, data hunger của Transformer.
> - Kosma et al. (2022) — degenerate solutions trong neural TS forecasting.

---

## 3. ĐỀ XUẤT CỦA CHÚNG TÔI

Ba đóng góp chính được đề xuất để giải quyết hai Gap nêu trên:

```
┌─────────────────────────────────────────────────────────────────┐
│  ĐỀ XUẤT 1: GARCH-Autoformer (Kiến trúc Hybrid Residual)       │
│  → Giải quyết trade-off accuracy vs risk calibration           │
├─────────────────────────────────────────────────────────────────┤
│  ĐỀ XUẤT 2: Forecast Sanity Gate                               │
│  → Giải quyết Gap 1: phát hiện forecast suy biến              │
├─────────────────────────────────────────────────────────────────┤
│  ĐỀ XUẤT 3: MCDM Risk-Sensitive với Dynamic Tracking Metrics   │
│  → Giải quyết Gap 2: xếp hạng đa tiêu chí toàn diện           │
└─────────────────────────────────────────────────────────────────┘
```

---

### 3.1 Đề xuất 1 — GARCH-Autoformer: Kiến trúc Lai Residual Correction

#### 3.1.1 Động lực thiết kế (The Trade-off Problem)

Dự án xác định hai cực đoan trong benchmark:

| Đặc điểm | GARCH truyền thống | Autoformer thuần túy |
|-----------|-------------------|----------------------|
| **Thế mạnh** | Bám sát đường nền (mean-reverting baseline), MSE/QLIKE ổn định | Bắt được chu kỳ và shock ngắn hạn, VaR pass rate cao hơn |
| **Điểm yếu** | Cứng nhắc, phản ứng chậm với shock cực đoan → Kupiec fail | Over-smoothing → gọt bỏ micro-oscillation → QLIKE/MSE cao |

**Mục tiêu:** Kết hợp để GARCH lo phần *mean-reverting baseline*, Autoformer lo phần *non-linear residuals*.

> **Trích dẫn cần có:**
> - Bollerslev (1986) — GARCH conditional variance.
> - Wu et al. (2021) [Autoformer] — decomposition và auto-correlation mechanism.
> - Donaldson & Kamstra (1997); Khashei & Bijari (2010) — nền tảng Neural Network học residual của ARIMA/GARCH.
> - Kim & Won (2018) — GARCH-LSTM vượt trội GARCH bằng cách để NN xử lý phần dư phi tuyến.

#### 3.1.2 Kiến trúc — Pipeline 3 Giai đoạn (Residual Correction Hybrid)

**Giai đoạn 1: Base Model — GARCH asymmetric**

Dùng mô hình GARCH bất đối xứng (GJR-GARCH hoặc EGARCH) để mô hình hóa chuỗi lợi nhuận $r_t$ và tạo dự báo phương sai nền:
$$\hat{\sigma}^2_{\text{GARCH},\, t}$$

Dự báo này *hấp thụ* toàn bộ biến động tuyến tính bình thường của thị trường → *lock* chỉ số MSE/QLIKE ở mức an toàn.

**Giai đoạn 2: Trích xuất và Học Residual bằng Autoformer**

Chuỗi phần dư (phi tuyến tính mà GARCH không mô hình hóa được) được tính:
$$e_t = \sigma^2_{\text{true},\, t} - \hat{\sigma}^2_{\text{GARCH},\, t}$$

hoặc dùng chuẩn hóa (standardized residuals):
$$z_t = \frac{r_t}{\hat{\sigma}_{\text{GARCH},\, t}}$$

Autoformer nhận $e_t$ (hoặc $z_t$) làm đầu vào thay vì chuỗi lợi nhuận gốc nhiễu loạn. Lúc này, cơ chế **Auto-Correlation** của Autoformer dồn toàn bộ năng lực tìm *hidden periodicity trong các cú sốc* — không bị phân tâm bởi mean path đã được GARCH xử lý.

$$\hat{e}_{\text{pred},\, t} = \text{Autoformer}(e_{t-1},\, e_{t-2},\, \ldots)$$

**Giai đoạn 3: Ensemble — Tổng hợp tuyến tính**

$$\hat{\sigma}^2_{\text{final},\, t} = \hat{\sigma}^2_{\text{GARCH},\, t} + \hat{e}_{\text{pred},\, t}$$

#### 3.1.3 Lợi ích kỳ vọng

- **MSE/QLIKE ổn định:** GARCH đã bám đường nền → không bao giờ over-smooth như Autoformer thuần túy.
- **VaR pass rate cải thiện:** Khi có tin tức cực đoan (black swan), GARCH phản ứng chậm nhưng Autoformer nhận diện shock trong phần dư và bơm thêm $\hat{e}_{\text{pred}}$ → VaR được phình to kịp thời → Kupiec test pass.

> **Trích dẫn cần có:**
> - Zhao et al. (2024) [GARCH-LSTM AAAI 2024] — triết lý inject financial inductive bias vào neural network.
> - Sezer et al. (2020) — survey hybrid GARCH-NN, residual learning approach.
> - Expert Systems with Applications / IEEE Access (2023-2024) — Hybrid GARCH-Transformer papers.

---

### 3.2 Đề xuất 2 — Forecast Sanity Gate

**Động lực:** Trước khi đưa bất kỳ mô hình nào vào ranking MCDM, cần kiểm tra mô hình đó có thực sự tạo ra forecast có động học hay không. Gate này giải quyết **Gap 1**.

**Hai điều kiện tối thiểu:**

| Điều kiện | Công thức | Ngưỡng | Mục đích |
|-----------|-----------|--------|----------|
| Std-ratio error | `abs(std(predict)/std(true) - 1) ≤ 0.9` | ≤ 0.9 | Loại forecast phẳng (std(predict) ≈ 0) |
| Tracking correlation | `corr(predict, true) > 0` | > 0 | Loại forecast không đồng biến với true volatility |

**Lý do thiết kế ngưỡng:**
- **Ngưỡng 0.9 có chủ ý lỏng:** Không yêu cầu forecast phải biến động bằng true, chỉ yêu cầu `std(predict)/std(true) ≥ 0.1` — tức forecast có ít nhất 10% biến động so với true volatility.
- **Tracking corr > 0 là điều kiện bổ sung:** Một mô hình có thể dao động mạnh nhưng đi ngược chiều true volatility → cũng bị loại.
- Hai điều kiện bổ sung cho nhau: std-ratio loại flat forecast, tracking correlation loại wrong-direction forecast.

**Metric tính theo case trước khi aggregate:** Mỗi metric được tính riêng cho từng `dataset × horizon`, sau đó lấy trung bình ở cấp model. Điều này tránh *variance giả* do khác mức nền giữa thị trường.

> **Trích dẫn:** Gneiting et al. (2007); Gneiting (2011); DTCenter METplus; Patton & Sheppard (2009); Kosma et al. (2022).

---

### 3.3 Đề xuất 3 — MCDM Risk-Sensitive với Dynamic Tracking Metrics

**Động lực:** Sau khi Gate loại các forecast không hợp lệ, cần xếp hạng các mô hình còn lại theo đa tiêu chí phản ánh cả accuracy, dynamics tracking, và risk calibration. Giải quyết **Gap 2**.

#### 3.3.1 Bộ metrics tích hợp (9 tiêu chí)

| Nhóm | Metric | Direction | Mô tả |
|------|--------|-----------|-------|
| Forecast accuracy | MSE | cost | Sai số bình phương trung bình |
| Forecast accuracy | MAE | cost | Sai số tuyệt đối trung bình |
| Forecast accuracy | QLIKE | cost | Quasi-likelihood loss, phù hợp latent volatility proxy |
| **Dynamic tracking** | **Volatility Std-Ratio Error** | **cost** | **Phạt forecast phẳng** |
| **Dynamic tracking** | **Tracking Correlation Error** | **cost** | **Phạt forecast sai hướng** |
| Risk calibration | VaR 1% Pass Rate | benefit | Kupiec + Christoffersen independence test |
| Risk calibration | VaR 1% Abs Violation Error | cost | `\|violation_rate - 0.01\|` |
| Risk calibration | VaR 5% Pass Rate | benefit | |
| Risk calibration | VaR 5% Abs Violation Error | cost | `\|violation_rate - 0.05\|` |

#### 3.3.2 Hai kịch bản trọng số

| Kịch bản | Accuracy block | Risk block | Mục đích |
|----------|---------------|-----------|----------|
| **5,5** | 50% | 50% | Đánh giá cân bằng accuracy-risk |
| **3,7** | 30% | 70% | Risk-sensitive deployment (ưu tiên quản trị rủi ro) |

#### 3.3.3 SAW và TOPSIS song song

- **SAW (Simple Additive Weighting):** Chuẩn hóa min-max và cộng điểm tuyến tính theo trọng số. Bù trừ mạnh — một tiêu chí rất tốt có thể kéo tiêu chí yếu.
- **TOPSIS (Hwang & Yoon, 1981):** Chuẩn hóa vector, tính khoảng cách Euclid đến ideal best và ideal worst. Phạt mạnh mô hình lệch xa điểm lý tưởng ở bất kỳ chiều nào — không bù trừ tuyến tính.

Sự đồng thuận giữa SAW và TOPSIS là bằng chứng vững chắc hơn cho ưu thế của một mô hình.

#### 3.3.4 Dominance Test — Kiểm định thống kê ưu thế GARCH-Autoformer

Sau khi có SAW/TOPSIS score ở cấp model family (trung bình qua tier), pipeline kiểm định:

- **H0:** P(GARCH-Autoformer thắng model còn lại theo SAW/TOPSIS) = 0.5
- **H1:** P > 0.5 (GARCH-Autoformer có ưu thế hệ thống)

**Kiểm định chính:** Exact binomial test one-sided (phù hợp khi n nhỏ).
**Báo thêm:** One-proportion z-test như xấp xỉ.

> **Trích dẫn:** Hwang & Yoon (1981); Kupiec (1995); Christoffersen (1998); Patton (2011); Demšar (2006).

---

## 4. GIẢI THÍCH CƠ SỞ THIẾT KẾ

### 4.1 Tại sao GARCH-Autoformer (Residual Correction) thay vì end-to-end Transformer?

- Transformer thuần túy học chuỗi gốc (rất nhiễu) → rơi vào nghiệm phẳng khi noise cao.
- GARCH-Autoformer tách bài toán thành hai phần: **phần tuyến tính (GARCH)** và **phần phi tuyến (Autoformer)** → mỗi module chỉ cần xử lý tín hiệu sạch hơn.
- Đây là triết lý **"Chia để trị"** (Divide and Conquer) trong chuỗi thời gian, có nền tảng từ Donaldson & Kamstra (1997) và Kim & Won (2018).

### 4.2 Tại sao cần Sanity Gate trước MCDM, không phải metric thông thường đủ?

- MSE/MAE không phân biệt được: mô hình có MSE = 0.20 vì forecast phẳng khác với mô hình có MSE = 0.20 vì tracking tốt nhưng có noise.
- VaR pass rate có thể tình cờ tốt khi violation rate ngẫu nhiên gần alpha.
- Gate không thay thế MCDM — nó là *bước sàng lọc tối thiểu* để loại những mô hình không nên có mặt trong bảng xếp hạng.

### 4.3 Tại sao chạy cả SAW lẫn TOPSIS?

- SAW và TOPSIS có bản chất toán học khác nhau (tuyến tính vs hình học).
- SAW cho phép bù trừ → phù hợp khi muốn đánh giá "tổng giá trị mang lại".
- TOPSIS phạt mạnh khi lệch xa ideal ở một chiều quan trọng → phù hợp khi muốn tìm "mô hình không có điểm yếu nghiêm trọng".
- Sự đồng thuận giữa hai phương pháp = bằng chứng vững chắc hơn.

> **Trích dẫn:** Hwang & Yoon (1981); Zeleny (1982) [MCDM framework]; Saaty (1980) [AHP — nếu đề cập trọng số].

---

## 5. KẾT QUẢ VÀ PHÂN TÍCH

### 5.1 Kết quả Forecast Sanity Gate

**5 mô hình bị loại (cùng nhau trong cả hai kịch bản 5,5 và 3,7):**

| Model | Std-ratio error | Tracking corr | Lý do loại |
|-------|----------------|---------------|------------|
| Reformer (Tier 3) | 1.000 | −0.00247 | Phẳng hoàn toàn + tracking âm |
| Informer (Tier 1) | 0.649 | −0.00406 | Tracking correlation âm |
| Autoformer (Tier 3) | 0.919 | +0.00677 | Std-ratio error vượt ngưỡng 0.9 |
| Vanilla (Tier 3) | 1.000 | −0.000124 | Phẳng hoàn toàn + tracking âm |
| Informer (Tier 3) | 1.000 | −0.00174 | Phẳng hoàn toàn + tracking âm |

**Nhận xét:** Các mô hình bị loại đều là Transformer tier lớn — mô hình phức tạp hơn không đồng nghĩa forecast tốt hơn trong bài toán này.

---

### 5.2 Kết quả MCDM — Cấu hình 5,5 (Accuracy 50% / Risk 50%)

| Rank | Model | Accuracy score | Risk score | SAW rank | TOPSIS rank |
|------|-------|---------------|-----------|---------|------------|
| 1 | MoiraiVaR - Moirai 2 (λ=0.2) | **0.496** | 0.180 | 1 | 3 |
| 1 | **GARCH-Autoformer (Tier 2)** | 0.223 | **0.443** | 3 | **1** |
| 3 | Moirai-MoE | 0.492 | 0.179 | 2 | 4 |

- **GARCH-Autoformer (Tier 2) đạt TOPSIS rank 1** — gần điểm lý tưởng đa chiều nhất.
- Moirai-family dẫn SAW nhờ accuracy score vượt trội, nhưng thua TOPSIS vì risk score thấp.
- **Dominance test (TOPSIS):** GARCH-Autoformer thắng **15/15** model family, exact binomial p = 0.000031.
- **Dominance test (SAW):** Thắng 11/15 (73.3%), exact binomial p = 0.059 — sát ngưỡng.

---

### 5.3 Kết quả MCDM — Cấu hình 3,7 (Accuracy 30% / Risk 70%)

| Rank | Model | SAW rank | TOPSIS rank | SAW score | TOPSIS score |
|------|-------|---------|------------|----------|-------------|
| **1** | **GARCH-Autoformer (Tier 2)** | **1** | **1** | 0.754 | 0.740 |
| 2 | Reformer (Tier 2) | 3 | 2 | 0.589 | 0.695 |
| 3 | GARCH-Autoformer (Tier 1) | 2 | 5 | 0.672 | 0.673 |

- GARCH-Autoformer **thắng rõ ràng cả SAW lẫn TOPSIS** khi ưu tiên risk.
- **Dominance test (SAW + TOPSIS):** Thắng **15/15**, exact binomial p = 0.000031 (cả hai metric).
- **Mean relative difference:** +96.7% (SAW), +102.18% (TOPSIS).

---

### 5.4 Phân tích Trade-off Accuracy vs Risk

Kết quả xác nhận hai nhóm mô hình phù hợp với hai mục tiêu deployment khác nhau:

| Nhóm | Thế mạnh | Phù hợp khi |
|------|----------|-------------|
| **Moirai-family** | Accuracy score cao (MSE thấp nhất) | Deployment cần forecast point volatility chính xác |
| **GARCH-Autoformer** | Risk score cao (VaR calibration tốt, tracking tốt) | Deployment ưu tiên risk management |

**Insight chính:** Không có ranking duy nhất — ranking phụ thuộc vào preference của nhà quản trị rủi ro.

> **Trích dẫn:** Christoffersen & Diebold (2000); Patton (2011); Fikri (2025) [regime-dependent performance]; Hwang & Yoon (1981).

---

### 5.5 Giải thích kết quả trong bối cảnh Mean Reversion và Stationarity

- Test split có **26/45 chuỗi non-stationary** (vs 6/45 trong train) → regime shift trong giai đoạn đánh giá.
- Mô hình học mean path của chuỗi mean-reverting trong train → khi gặp regime shift → forecast phẳng.
- VaR backtesting "tình cờ" tốt ở một số case → không phản ánh khả năng thực sự → **Gate cần thiết để loại trước MCDM**.
- GARCH-Autoformer ổn định hơn vì GARCH component đã encode mean-reverting behavior → Autoformer chỉ cần xử lý residuals, ít bị ảnh hưởng bởi regime shift hơn.

---

## 6. KẾT LUẬN VÀ HƯỚNG MỞ RỘNG

### 6.1 Kết luận chính

Ba đóng góp của nghiên cứu:

1. **GARCH-Autoformer (Residual Correction Hybrid):** Kiến trúc lai kết hợp inductive bias của GARCH với khả năng học residual phi tuyến của Autoformer. Mô hình đạt TOPSIS rank 1 ở cả hai kịch bản và thắng 15/15 model family ở kịch bản risk-sensitive (p = 0.000031).

2. **Forecast Sanity Gate:** Cơ chế sàng lọc hai điều kiện tối thiểu trước MCDM, loại 5/26 mô hình Transformer tier lớn có forecast gần như phẳng hoặc không đồng biến với true volatility.

3. **MCDM Risk-Sensitive với Dynamic Tracking:** Framework đánh giá 9 tiêu chí tích hợp forecast accuracy, dynamic tracking và risk calibration, vận hành hai phương pháp xếp hạng (SAW + TOPSIS) và hai kịch bản preference (5,5 và 3,7).

### 6.2 Hạn chế và Hướng mở rộng

| Hạn chế hiện tại | Hướng mở rộng tương lai |
|-----------------|------------------------|
| Dominance test ở cấp aggregate (model family) | Block-level SAW/TOPSIS theo 45 `dataset×horizon`, Friedman/Nemenyi |
| Ngưỡng gate (0.9, tracking > 0) chưa có ablation | Ablation: 0.8, 0.85, 0.9, 0.95; báo cáo model nào bị loại/giữ |
| Hai kịch bản trọng số (5,5 và 3,7) | Weight sensitivity analysis: risk weight 0→1, Pareto frontier |
| Stationarity suy ra từ split 70/15/15 | Rolling stationarity diagnostics, regime-aware evaluation |
| Chưa có Expected Shortfall (ES) | Mở rộng sang ES, CVaR, regulatory backtests (Basel III) |

---

## PHỤ LỤC: Bản đồ Trích dẫn theo Section

| Section | Trích dẫn chính |
|---------|----------------|
| **1.1** — Trade-off forecast vs risk | Christoffersen & Diebold (2000); Patton (2011); Bollerslev (1986) |
| **1.2 Gap 1** — Flat forecast problem | Gneiting 2011; Gneiting et al. 2007; Kosma et al. 2022; DTCenter METplus; Patton & Sheppard 2009 |
| **1.2 Gap 2** — MCDM thiếu dynamic tracking | Hwang & Yoon 1981; Kupiec 1995; Christoffersen 1998; Patton 2011 |
| **2.1** — Mean Reversion | Bollerslev 1986; Engle 1982; Baillie et al. 1996; Mandelbrot 1963 |
| **2.2** — ADF/KPSS tests | Dickey & Fuller 1979; Kwiatkowski et al. 1992; Perron 1989; Hamilton 1994 |
| **2.3** — Flat forecast | Chen et al. ICML 2025; Zeng et al. AAAI 2023; Lim & Zohren 2021; Kosma 2022 |
| **3.1** — GARCH-Autoformer design | Bollerslev 1986; Wu et al. NeurIPS 2021; Donaldson & Kamstra 1997; Khashei & Bijari 2010; Kim & Won 2018; Zhao et al. AAAI 2024; Sezer et al. 2020 |
| **3.2** — Sanity Gate | Gneiting et al. 2007; Gneiting 2011; DTCenter METplus; Patton & Sheppard 2009 |
| **3.3** — MCDM SAW+TOPSIS | Hwang & Yoon 1981; Zeleny 1982; Kupiec 1995; Christoffersen 1998 |
| **3.4** — Dominance Test | Demšar 2006; Zhao et al. AAAI 2024 |
| **5.4** — Trade-off analysis | Christoffersen & Diebold 2000; Patton 2011; Fikri 2025 |

---

## GHI CHÚ THỰC HIỆN

> [!IMPORTANT]
> **Phần 3 (Đề xuất của chúng tôi) là phần trung tâm của bài báo.** Cần viết đủ độ sâu để reviewer thấy rõ novelty của từng đề xuất và lý do chúng giải quyết đúng gap được nêu ở Phần 1.2.

> [!TIP]
> **Thứ tự trình bày trong Phần 3 nên là:** GARCH-Autoformer (model mới) → Sanity Gate (mechanism mới) → MCDM (framework mới). Trình bày theo thứ tự từ model-level đến pipeline-level.

> [!NOTE]
> **Nguồn số liệu:**
> - MCDM results: `output/mcdm_results/MCDM20260725170414/`
> - Stationarity: `output/stats_analysis/stationarity/stationarity_20260725_173107/`
> - Forecast ranking: `output/stats_analysis/research_summary/forecast_ranking.csv`
> - MCDM input metrics: `output/mcdm_results/MCDM20260727104251/5,5/MCDMInputMetrics.csv`

> [!WARNING]
> **Lưu ý về reviewer feedback (từ `review_mapr.md`):**
> - Reviewer 2 & 3 chỉ ra novelty framework còn hạn chế — phần Sanity Gate và Dynamic Tracking Metrics cần được nhấn mạnh như **methodological contribution** cụ thể, không chỉ là "kết hợp existing techniques".
> - Cần giải thích rõ *tại sao* tracking correlation và std-ratio error được thiết kế theo từng case trước khi aggregate — đây là điểm kỹ thuật quan trọng để tránh bị phê là heuristic đơn giản.
