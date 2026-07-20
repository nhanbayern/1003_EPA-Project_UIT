---
marp: true
theme: default
paginate: true
---

# Báo Cáo Cập Nhật Tiến Độ Nghiên Cứu
**Khung Đánh Giá Hợp Nhất Các Mô Hình Dự Báo Biến Động**

**Người trình bày:** Nguyễn Thiện Nhân, Trần Hùng Vĩ
**GVHD:** PGS. TS. Nguyễn Đình Thuân

---

## Nội dung trình bày

1. Tóm tắt góp ý từ Reviewer và Nhận diện điểm yếu
2. Các hướng khắc phục và Quá trình chạy lại mô hình
3. Phân tích và Diễn giải kết quả thực nghiệm mới
4. Thảo luận và Đề xuất Khung đánh giá hai tầng
5. Tài liệu tham khảo

---

## 1. Tóm tắt góp ý từ Reviewer

**Điểm mạnh được Reviewer ghi nhận:**
- Ý tưởng cốt lõi tốt: Đánh giá sự đánh đổi giữa *forecast accuracy* và *risk control*.
- Quy mô hệ thống benchmark rộng lớn và đa dạng.

**Tuy nhiên, vẫn tồn tại 4 nhóm điểm yếu cốt lõi cần giải quyết.**

---

## Các điểm yếu cần khắc phục

1. **Dữ liệu & Thiết lập:** Thiếu minh bạch về nguồn gốc, quy trình và kích thước mẫu (Reviewer 1, 3).
2. **Kiến trúc Moirai-MoE-GARCHs:** Kết quả yếu kém (pass rate hạng 9/10), mâu thuẫn với khẳng định trong bài (Reviewer 1, 2, 3).
3. **Phân tích & Thống kê:** Phân tích mang tính bề mặt, chỉ dùng VaR 5%, thiếu các kiểm định thống kê bắt buộc (Reviewer 2).
4. **Tính mới (Novelty) & Tái tạo:** Thiếu chi tiết siêu tham số (hyperparameters), đóng góp phương pháp luận chưa rõ ràng (Reviewer 2, 3).

---

## 2. Hướng khắc phục: Dữ liệu và Thiết lập

- **Nguồn dữ liệu:** 9 chỉ số toàn cầu từ Investing.com (DAX 40, S&P 500...) và vnstock (VN30, VN-Index).
- **Tính nhân quả (Causality):** Cửa sổ quá khứ 60 ngày, tuyệt đối không dùng dữ liệu tương lai để dự báo mục tiêu $\sigma_{t,h}$ (tránh look-ahead bias).
- **Phân tách nhận biết phân phối:** Sử dụng thuật toán chia Train/Val/Test để giữ ổn định phân phối đuôi Student-t (tham số $\nu$), thay vì chia ngẫu nhiên.

---

## Dữ liệu: Bảng kích thước mẫu (2010 - 2025)

| Index | Train | Validation | Test | Total |
| :--- | :---: | :---: | :---: | :---: |
| DAX 40 | 1983 | 1104 | 972 | 4059 |
| S&P 500 | 1988 | 1109 | 927 | 4024 |
| VN30 | 1971 | 1096 | 925 | 3992 |
| VN-Index | 1964 | 1103 | 925 | 3992 |
*(Trích xuất đại diện 4/9 dataset)*

---

## 2. Hướng khắc phục: Triển khai mô hình (1)

**Nhóm Kinh tế lượng (GARCH-family)**
- Mô hình: GARCH, GJR-GARCH, FI-GARCH (Thư viện `arch`).
- Rolling window 1.000 quan sát, giả định sai số Student-t.

**Nhóm Transformer**
- Chia 3 cấp độ (Tiers): Miniaturized, Standard, Large nhằm kiểm tra "khủng hoảng dữ liệu" với chuỗi ngắn 60 ngày.
- Dự báo trực tiếp qua `VolatilityHead`, huấn luyện 30 epochs, AdamW (LR=$10^{-3}$).

---

## 2. Hướng khắc phục: Triển khai mô hình (2)

**Mô hình Lai GARCH-LSTM**
- Tích hợp phương trình phương sai vào thẳng cổng LSTM.
- **Regularization:** Cố định bậc tự do $v=5.0$ cho loss Student-t NLL.
- **Dự báo trực tiếp đa bước** thay vì tự hồi quy.
- **Kết quả:** Sai số (MSE) giảm ngoạn mục từ $1.28 \times 10^6$ xuống $\sim 0.11$.

---

## Hướng khắc phục: Khẳng định Tính Mới (Novelty)

**Loại bỏ Moirai-MoE-GARCHs**
- Nhận thấy kiến trúc cũ kém hiệu quả, em đã thay thế bằng **Moirai_VAR**.
- Backbone Moirai (frozen) kết hợp khối hồi quy MLP 2 lớp.

**Điểm nhấn phương pháp luận: Hàm mất mát VaR-Aware Loss**
- Mô hình truyền thống chỉ tập trung học trung bình (MSE), bỏ qua vùng đuôi rủi ro.

---

## Hàm mất mát VaR-Aware (VaR-Aware Loss)

Thiết kế lại hàm mục tiêu:
$$ \text{Total\_Loss} = \text{MSE}(\text{vol}_{pred}) + \lambda_{var} \times \text{QuantileVaRLoss}(\text{VaR}_{threshold}) $$

- Trọng số $\lambda_{var}=0.2$.
- Mô hình bị ép "học" tính phân phối đuôi.
- Bị phạt bằng hàm **Pinball Loss** nếu dự báo đâm thủng ngưỡng VaR.

---

## 3. Phân tích kết quả: Độ chính xác (Point Forecast)

**Top Mô hình theo Độ chính xác Dự báo (MSE, MAE, QLIKE)**

| Rank | Model | MSE | MAE | Avg Rank |
|---:|---|---:|---:|---:|
| 1 | Moirai (moirai2) | 0.0227 | 0.0906 | 1.000 |
| 2 | Moirai_VAR (λ=0.2, moirai2) | 0.0237 | 0.0947 | 2.333 |
| 3 | Moirai (moirai_moe) | 0.0247 | 0.0975 | 2.667 |
| 4 | Moirai_VAR (λ=0.2, moirai_moe)| 0.0252 | 0.1014 | 4.000 |

---

## Đánh giá Point Forecast

- Mô hình foundation **Moirai** áp đảo toàn diện ở năng lực khớp giá trị trung bình (mean path).
- **Moirai_VAR** dù bị kéo lại bởi thành phần loss rủi ro nhưng vẫn bám rất sát ở Top đầu.
- GARCH-LSTM (bản mới) không còn là điểm nghẽn về dự báo điểm.

---

## 3. Phân tích kết quả: Backtesting (Risk-Control)

**Sự đảo ngôi tại VaR 5% và VaR 1%**

| Khung đánh giá | Vị trí Top 1 | Vị trí Top 2 |
|---|---|---|
| **VaR 5% Pass Rate** | Autoformer Tier 2 (24.44%) | Autoformer Tier 1 (20.00%) |
| **VaR 1% Pass Rate** | Autoformer Tier 1 (35.56%) | Informer Tier 1 (28.89%) |

*(Bổ sung ngưỡng VaR 1% theo yêu cầu của Reviewer)*

---

## Đánh giá Risk Control

- Dù Moirai vô địch về MSE, nhưng **Transformer** (đặc biệt là biến thể nhỏ Tier 1, 2) lại giành chiến thắng tuyệt đối về năng lực kiểm soát rủi ro phân phối.
- **Chứng minh luận điểm cốt lõi:** Mô hình dự báo biến động chính xác nhất *không đảm bảo* quản trị rủi ro vùng đuôi (Tail Risk) tốt nhất.

---

## 3. Robustness Check: 3 Phương pháp VaR

Quy đổi biến động dự báo sang VaR bằng Normal, Student-t, và FHS.

| VaR (5%) | Violation rate | Abs violation error | Quantile loss |
|---|---:|---:|---:|
| Normal | 0.0598 | 0.0172 | 0.4355 |
| Student-t | 0.0326 | 0.0217 | 0.5273 |
| FHS | 0.0527 | 0.0040 | 0.4425 |

- Student-t mang tính bảo thủ cao (ít vi phạm).
- FHS cho violation rate sát mục tiêu nhất.

---

## Hiệu quả đột phá của VaR-Aware Loss

- Ở ngưỡng VaR 5% (Normal và FHS), **Moirai_VAR** đã vươn lên vị trí **Top 1 toàn bảng** về Quantile Loss (Tổn thất phân vị).
- Việc hi sinh một lượng nhỏ MSE để đạt độ chuẩn xác xuất sắc ở vùng rủi ro đuôi khẳng định sự thành công của thiết kế VaR-Aware Loss.

---

## 3. Các Kiểm định Thống kê Bổ sung

Để giải quyết yêu cầu của Reviewer 2, hệ thống kiểm định toàn diện đã được áp dụng:

- **Friedman Test:** Khẳng định sự khác biệt giữa các mô hình về Point Forecast là có ý nghĩa cực kỳ cao (p-value < $1.62 \times 10^{-92}$).
- **Nemenyi Test:** Xác nhận có hàng chục cặp mô hình (trên 80 cặp) có độ chênh lệch dự báo mang ý nghĩa thống kê.

---

## Kiểm định Diebold-Mariano (DM)

**Số cặp mô hình khác biệt có ý nghĩa thống kê (DM Test trên Quantile Loss)**

| VaR case | Normal | Student-t | FHS |
|---|---:|---:|---:|
| 5% | 2,749 | 3,121 | 1,355 |
| 1% | 3,589 | 2,928 | 1,264 |

- **Kết luận DM Test:** Sự chênh lệch Quantile Loss giữa các mô hình là sâu sắc và có ý nghĩa thống kê, minh chứng cho sự thay thế Moirai-MoE-GARCHs bằng Moirai_VAR.

---

## 4. Thảo luận: Sự phân ly của Accuracy & Risk

Từ các bằng chứng thực nghiệm, em rút ra kết luận quan trọng:
- Mô hình khớp *mean path* xuất sắc (Moirai) không tự động trở thành mô hình khớp *tail distribution* tốt (Transformer/Moirai_VAR).
- Không thể dùng các độ đo truyền thống (MSE, MAE) làm tiêu chí độc tôn để chọn mô hình quản trị rủi ro tài chính.

---

## Đề xuất: Khung đánh giá hai tầng
**(Two-Tier Evaluation Framework)**

1. **Tier 1 (Đánh giá dự báo điểm):** 
   - Dùng MSE, MAE, QLIKE. Phù hợp bài toán định giá (pricing).
   - Đề xuất: Moirai.
2. **Tier 2 (Đánh giá rủi ro phân phối):** 
   - Dùng VaR Backtesting, Quantile Loss. Phù hợp quản trị rủi ro.
   - Đề xuất: Transformer Tiers hoặc Moirai_VAR (VaR-Aware Loss).

---

## Kết luận chung

- Mọi góp ý của Reviewer đã được giải quyết triệt để thông qua quá trình cải tiến dữ liệu và phương pháp.
- Sự ra đời của **Moirai_VAR** kết hợp tính ưu việt của foundation model và kỷ luật rủi ro (VaR-Aware) đã bổ sung hoàn hảo "Tính mới" cho nghiên cứu.
- Bài báo hiện đã đủ dữ liệu và cơ sở thống kê vững chắc để sẵn sàng submit lại.

---

## Cảm ơn mọi người đã lắng nghe!

**Q&A**

---

## Tài liệu tham khảo (1/2)

[1] T. Bollerslev, "Generalized autoregressive conditional heteroskedasticity," *Journal of Econometrics*, vol. 31, 1986.
[2] L. R. Glosten et al., "On the relation between the expected value and the volatility... ," *The Journal of Finance*, vol. 48, 1993.
[3] R. T. Baillie et al., "Fractionally integrated GARCH," *Journal of Econometrics*, vol. 74, 1996.
[4] A. Vaswani et al., "Attention is all you need," *NeurIPS*, 2017.
[5] H. Zhou et al., "Informer: Beyond efficient transformer... ," *AAAI*, 2021.
[6] H. Wu et al., "Autoformer: Decomposition transformers... ," *NeurIPS*, 2021.
[7] N. Kitaev et al., "Reformer: The efficient transformer," *ICLR*, 2020.
[8] P. Zhao et al., "From GARCH to neural network for volatility forecast," *AAAI*, 2024.

---

## Tài liệu tham khảo (2/2)

[9] X. Liu et al., "Moirai-MoE: Empowering time series foundation models... ," *ICML*, 2025.
[10] O. B. Sezer et al., "Financial time series forecasting with deep learning," *Applied Soft Computing*, 2020.
[11] B. Lim and S. Zohren, "Time-series forecasting with deep learning: a survey," *Phil. Trans. R. Soc. A*, 2021.
[12] A. Zeng et al., "Are transformers effective for time series forecasting?," *AAAI*, 2023.
[13] P. Christoffersen, "Evaluating interval forecasts," *Int. Econ. Rev.*, 1998.
[14] P. Kupiec, "Techniques for verifying the accuracy of risk measurement models," *The Journal of Derivatives*, 1995.
[15] F. X. Diebold & R. S. Mariano, "Comparing predictive accuracy," *JBES*, 1995.
[16] C. Koenker & G. Bassett Jr, "Regression quantiles," *Econometrica*, 1978.
