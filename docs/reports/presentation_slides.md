---
marp: true
theme: default
paginate: true
---

# Báo Cáo Cập Nhật Tiến Độ Nghiên Cứu
**Khám Phá Thực Nghiệm: Nghịch Lý Accuracy-Risk Trong Dự Báo Biến Động**

**Người trình bày:** Nguyễn Thiện Nhân, Trần Hùng Vĩ
**GVHD:** PGS. TS. Nguyễn Đình Thuân

---

## Nội dung trình bày

1. Tóm tắt góp ý từ Reviewer và Định hình lại nghiên cứu
2. Thiết lập Benchmark toàn diện
3. Phân tích kết quả: Sự xuất hiện của Nghịch lý Accuracy-Risk
4. Giải thích học thuật và Đề xuất Khung đánh giá
5. Tài liệu tham khảo

---

## 1. Tóm tắt góp ý từ Reviewer

**Điểm mạnh được Reviewer ghi nhận:**
- Ý tưởng đánh giá sự đánh đổi giữa *forecast accuracy* và *risk control*.
- Quy mô hệ thống benchmark rộng lớn và đa dạng.

**Tuy nhiên, các điểm yếu cốt lõi cần giải quyết:**
- Thiếu minh bạch về nguồn gốc dữ liệu, kích thước mẫu.
- **Quan trọng nhất:** Sự yếu kém và mâu thuẫn của mô hình lai đề xuất cũ (Moirai-MoE-GARCHs), làm giảm tính thuyết phục và "tính mới" (Novelty) của toàn bộ nghiên cứu.

---

## Định hình lại Cốt lõi Nghiên cứu (Novelty Shift)

Để bài báo có sức nặng học thuật cao nhất, nhóm nghiên cứu quyết định **loại bỏ hoàn toàn mô hình lai ghép Moirai-MoE-GARCHs**. 

Thay vào đó, bài báo sẽ được tái định vị thành một **Khám phá Thực nghiệm (Empirical Discovery)** về một "điểm mù" lớn khi áp dụng Foundation Models vào Quản trị rủi ro:

**Nghịch lý Accuracy-Risk:** *Một mô hình có độ chính xác dự báo (MSE) xuất sắc hoàn toàn không đảm bảo khả năng quản trị rủi ro đuôi (VaR) an toàn.*

---

## 2. Thiết lập Benchmark Toàn diện

- **Nguồn dữ liệu:** 9 chỉ số toàn cầu từ Investing.com và vnstock.
- **Tính nhân quả (Causality):** Tuyệt đối không dùng dữ liệu tương lai (chỉ dùng chuỗi 60 ngày quá khứ).
- **Thuật toán phân tách:** Giữ ổn định phân phối đuôi Student-t qua các tập Train/Val/Test để đánh giá chính xác rủi ro đuôi hẹp.

**Bảng kích thước mẫu (2010 - 2025)**

| Index | Train | Validation | Test | Total |
| :--- | :---: | :---: | :---: | :---: |
| DAX 40 | 1983 | 1104 | 972 | 4059 |
| S&P 500 | 1988 | 1109 | 927 | 4024 |
*(Trích xuất đại diện 2/9 dataset)*

---

## Ba Nhóm Mô Hình So Sánh

**1. Foundation Models (Moirai, Moirai-MoE):**
- Được pre-train trên hàng chục tỷ điểm dữ liệu.
- Mục tiêu tối ưu: Giảm thiểu sai số trung bình (MSE).

**2. Local Transformers (Autoformer, Informer, v.v.):**
- Huấn luyện cục bộ trên dữ liệu 60 ngày.
- **Đặc biệt:** Dùng hàm mất mát Student-t với bậc tự do $\nu$ động, tự động co giãn theo độ dày của đuôi rủi ro thực tế.

**3. Nhóm Kinh tế lượng & GARCH-LSTM:** Làm tham chiếu cơ sở (Baseline).

---

## 3. Phân tích kết quả: Mặt trận Dự báo (Point Forecast)

**Top Mô hình theo Độ chính xác Dự báo (MSE, MAE)**

| Rank | Model | MSE | MAE | Avg Rank |
|---:|---|---:|---:|---:|
| 1 | Moirai (moirai2) | 0.0227 | 0.0906 | 1.000 |
| 2 | Moirai (moirai_moe) | 0.0247 | 0.0975 | 2.667 |
| ... | ... | ... | ... | ... |
| 8 | Autoformer Tier 1 | 0.2241 | 0.3465 | 16.00 |

**Đánh giá:**
- **Moirai** áp đảo toàn diện ở năng lực dự báo giá trị trung bình (MSE rất thấp).
- **Transformer cục bộ** (Autoformer) có sai số cao gấp 10 lần Moirai.

---

## Phân tích kết quả: Mặt trận Quản trị rủi ro (VaR)

Tuy nhiên, khi đối chiếu sang năng lực bảo vệ rủi ro (VaR Backtesting) ở ngưỡng cực đoan 1%:

| Rank | Model | VaR 1% Pass Rate |
|---|---|---:|
| 1 | Autoformer Tier 1 | **35.56%** |
| 2 | Informer Tier 1 | 28.89% |
| ... | ... | ... |
| 8 | Moirai (moirai_moe) | 20.00% |
| 15 | Moirai (moirai2) | 11.11% |

**Nghịch lý xuất hiện:** Dù Moirai có sai số (MSE) thấp gấp 10 lần, nhưng khả năng bảo vệ rủi ro của nó lại **thua xa** Autoformer!

---

## 4. Giải thích Học thuật (1): Lệch pha Hàm Mục Tiêu

Tại sao Moirai có MSE xuất sắc nhưng VaR lại yếu?

- **Moirai (Foundation Model)** được pre-train bằng các hàm MSE/MAE để tối ưu hóa giá trị kỳ vọng (mean).
- Đặc tính của toán học MSE là "phạt" rất nặng các dự báo nằm xa trung bình. Do đó, mô hình có xu hướng **làm mượt (over-smooth)** các cú sốc biến động cực đoan.
- Kết quả: Khi thị trường biến động mạnh, Moirai dự báo "quá an toàn" và đâm thủng ngưỡng rủi ro VaR liên tục *(Gneiting, 2011)*.

---

## Giải thích Học thuật (2): Độ nhạy cục bộ vs. Foundation

Tại sao Transformer cục bộ có MSE kém nhưng VaR lại mạnh?

- Transformer có MSE cao do thiếu dữ liệu (chỉ nhìn 60 ngày), không học được xu hướng chuẩn xác.
- **Tuy nhiên:** Do có cấu trúc tinh gọn (parsimonious), nhóm Tier 1 tránh được bẫy *over-parameterization* *(Zeng et al., 2023)*.
- Nhờ hàm **Student-t động**, mô hình đạt được **độ nhạy cục bộ (local sensitivity)** cao. Tham số $\nu$ thích ứng linh hoạt, tạo ra dải rủi ro (coverage) phản ứng tức thời bao trọn vẹn các cú sốc đuôi hẹp *(Koenker, 1978)*.
- **VaR Backtesting** không quan tâm sai số MSE, nó chỉ đếm số lần hụt. Coverage rộng = Tỷ lệ Pass cao *(Christoffersen, 1998)*.

---

## Hiệu ứng Kích thước: Tier 1 vs. Tier 3

Tại sao Tier 1 (nhỏ gọn) lại đánh bại Tier 3 (hàng triệu tham số)?
- **Over-parameterization:** Dữ liệu tài chính có signal-to-noise rất thấp. Mô hình lớn bị nhiễu loạn và suy thoái nghiệm *(Zeng et al., 2023)*.
- **Bằng chứng Loss Curves (Autoformer - EuroNext 100):**

| Tier 1 (Hội tụ ổn định, bám sát) | Tier 3 (Bất ổn, dao động dữ dội) |
| :---: | :---: |
| ![Tier 1 Loss](file:///D:/UIT/1003_EPA_PROJECT/1.0.0/1003_EPA-Project_UIT/results_v2/tranformers%20based/results_v2/visualizations/Loss_Curves/Tier_1_Miniaturized/EuroNext_100_Autoformer_loss.png) | ![Tier 3 Loss](file:///D:/UIT/1003_EPA_PROJECT/1.0.0/1003_EPA-Project_UIT/results_v2/tranformers%20based/results_v2/visualizations/Loss_Curves/Tier_3_Large/EuroNext_100_Autoformer_loss.png) |

---

## Kiểm định Thống kê (Statistical Tests)

Sự khác biệt cực độ trên không phải là ngẫu nhiên. Các kiểm định thống kê đã xác nhận (p-value < 0.05):
- **Friedman Test:** Điểm số dự báo (MSE/MAE) giữa Moirai và Transformer có sự khác biệt mang ý nghĩa thống kê tuyệt đối.
- **Diebold-Mariano Test:** Xác nhận có hàng ngàn cặp (vd: 3,589 cặp ở VaR 1%) có độ chênh lệch cực lớn về Quantile Loss.

Điều này chứng minh nghịch lý Accuracy-Risk là một thực tế mang tính hệ thống.

---

## Đề xuất: Khung đánh giá hai tầng (Two-Tier)

Để không "sập bẫy" khi ứng dụng Foundation Models, chúng tôi đề xuất:

1. **Tier 1 (Đánh giá dự báo điểm):** 
   - Tiêu chí: MSE, MAE.
   - Ứng dụng: Bài toán định giá (Asset Pricing) - Nơi sự bám sát giá trị thực tế là tối quan trọng.
   - **Lựa chọn tối ưu:** Moirai Foundation.

2. **Tier 2 (Đánh giá rủi ro phân phối):** 
   - Tiêu chí: VaR Backtesting, Quantile Loss.
   - Ứng dụng: Quản trị rủi ro (Risk Management) - Nơi cần biên an toàn cao.
   - **Lựa chọn tối ưu:** Transformer Cục bộ (Student-t).

---

## Kết luận

- Bài nghiên cứu đã loại bỏ sự phụ thuộc vào một mô hình đề xuất yếu kém, chuyển hóa thành một **Khám phá Thực nghiệm (Empirical Discovery)** đắt giá.
- Phát hiện ra **Nghịch lý Accuracy-Risk** là minh chứng rõ rệt cảnh tỉnh các nhà phân tích định lượng về việc lạm dụng Foundation Models.
- Khung đánh giá hai tầng (Two-Tier Evaluation Framework) đóng vai trò kim chỉ nam vững chắc cho các nghiên cứu và ứng dụng thực tiễn trong tương lai.

---

## Cảm ơn mọi người đã lắng nghe!

**Q & A**

---

## Tài liệu tham khảo (1/2)

[1] T. Bollerslev, "Generalized autoregressive conditional heteroskedasticity," *Journal of Econometrics*, 1986.
[2] L. R. Glosten et al., "On the relation between the expected value and the volatility... ," *The Journal of Finance*, 1993.
[3] R. T. Baillie et al., "Fractionally integrated GARCH," *Journal of Econometrics*, 1996.
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
[17] T. Gneiting, "Making and evaluating point forecasts," *J. Am. Stat. Assoc.*, 2011.
