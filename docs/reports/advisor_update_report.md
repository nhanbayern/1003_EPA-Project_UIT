# Báo Cáo Cập Nhật Tiến Độ Nghiên Cứu: Khung Đánh Giá Hợp Nhất Các Mô Hình Dự Báo Biến Động

**Nội dung:** Báo cáo về việc tiếp thu góp ý từ Reviewer, các hướng khắc phục đã thực hiện, và khám phá thực nghiệm mới nhất về nghịch lý giữa dự báo điểm và quản trị rủi ro.

Dựa trên những phản hồi từ các Reviewer về bài báo của chúng ta, em đã tiến hành định hình lại toàn bộ đóng góp cốt lõi của nghiên cứu. Thay vì cố gắng đề xuất một mô hình lai ghép mới, nghiên cứu này sẽ đóng vai trò là một **khám phá thực nghiệm (Empirical Discovery)** về một điểm mù lớn khi ứng dụng Foundation Models vào Quản trị rủi ro tài chính.

## 1. Tóm tắt góp ý từ Reviewer và nhận diện điểm yếu
Qua việc đọc kỹ các nhận xét, em nhận thấy cả 3 Reviewer đều đánh giá cao ý tưởng cốt lõi của bài nghiên cứu (sự đánh đổi giữa *forecast accuracy* và *risk control*), cũng như đánh giá cao quy mô của hệ thống benchmark. Tuy nhiên, bài viết bản trước tồn tại các điểm yếu nghiêm trọng cần khắc phục:
1.  **Vấn đề Dữ liệu và Thiết lập Thực nghiệm (Reviewer 1, 3):** Chưa làm rõ nguồn gốc dữ liệu, quy trình tiền xử lý, và hoàn toàn thiếu bảng kích thước mẫu (sample sizes).
2.  **Sự mâu thuẫn của mô hình đề xuất Moirai-MoE-GARCHs (Reviewer 1, 2, 3):** Kết quả thực nghiệm cho thấy mô hình này xếp thứ 9/10 về năng lực quản trị rủi ro. Việc cố gắng bảo vệ một mô hình "lắp ghép" nhưng hiệu năng lởm khởm đã làm suy yếu toàn bộ bài báo.
3.  **Hạn chế về Tính mới (Novelty) (Reviewer 2, 3):** Framework bị đánh giá là thiếu tính mới mẻ vì chỉ ghép nối các phương pháp có sẵn. Hơn nữa, việc thiếu chi tiết thiết lập làm bài nghiên cứu khó có thể tái tạo (reproduce).

---
## 2. Các hướng khắc phục: Định hình lại cốt lõi nghiên cứu
Để giải quyết triệt để các vấn đề trên, em đã loại bỏ hoàn toàn mô hình Moirai-MoE-GARCHs. Tính mới (Novelty) của bài báo giờ đây sẽ nằm ở việc **chỉ ra Nghịch lý Accuracy-Risk (Accuracy-Risk Fallacy) giữa các dòng mô hình.**

### 2.1. Chuẩn hóa Dữ liệu và Thiết lập (Dataset & Problem Setup)
Sử dụng 9 chỉ số toàn cầu từ Investing.com và vnstock. Bài toán dự báo được định hình lại rõ ràng:
*   **Tính nhân quả (Causality):** Tại thời điểm $t$, mô hình chỉ được sử dụng cửa sổ 60 ngày quá khứ $x_t = (r_{t-60}, \dots, r_{t-1})$. Mô hình tuyệt đối không có quyền truy cập dữ liệu tương lai để tránh look-ahead bias.
*   **Phân tách nhận biết phân phối:** Sử dụng thuật toán phân tách sao cho đặc tính phân phối đuôi dày Student-t (thông qua tham số $\nu$) được giữ ổn định qua các tập Train/Val/Test.

**Bảng 1: Kích thước mẫu và phân tách cho từng dataset (2010–2025)**

| Index        |    Source     | Train | Validation | Test | Total |
| :----------- | :-----------: | :---: | :--------: | :--: | :---: |
| DAX 40       | Investing.com | 1983  |    1104    | 972  | 4059  |
| S&P 500      | Investing.com | 1988  |    1109    | 927  | 4024  |
| VN30         | Investing.com | 1971  |    1096    | 925  | 3992  |

### 2.2. Chi tiết Triển khai và Thiết lập Siêu tham số
*   **Nhóm Foundation Models (Moirai):** Các mô hình pre-trained khổng lồ (Moirai cơ bản, Moirai-MoE) được huấn luyện trên hàng chục tỷ điểm dữ liệu đa miền bằng hàm MSE. Chúng cực kỳ mạnh trong việc nắm bắt *mean path* (đường đi trung bình).
*   **Nhóm Transformer Cục bộ (Vanilla, Informer, Autoformer, Reformer):** Các mô hình này được huấn luyện độc lập trên chuỗi 60 ngày bằng hàm mất mát **Student-t NLL với bậc tự do $\nu$ động**. Sự kết hợp này cho phép Transformer tự thích nghi (thậm chí overfit) vào độ dày của đuôi rủi ro trong từng chuỗi cục bộ.
*   **Nhóm Kinh tế lượng (GARCH) & Lai ghép:** Thiết lập thư viện `arch` với rolling window 1000 và mô hình Lai GARCH-LSTM (đóng vai trò tham chiếu chuẩn).

### 2.3. Khẳng định Tính Mới (Novelty): Nghịch lý Accuracy-Risk
Sự đóng góp lớn nhất của nghiên cứu này là vạch trần quan điểm phổ biến: "Một mô hình có độ chính xác dự báo điểm (MSE/MAE) tốt thì sẽ tự động dùng tốt cho quản trị rủi ro (VaR Backtesting)". Sự xuất hiện của Foundation Models đã đẩy nghịch lý này lên mức cực đại, điều mà các nghiên cứu trước đây chưa chỉ rõ bằng thực nghiệm.

---
## 3. Phân tích và diễn giải kết quả: Khám phá thực nghiệm
Sau quá trình tinh chỉnh và chạy lại toàn bộ mô hình, kết quả đạt được đã cung cấp bằng chứng rõ ràng cho nghịch lý này.

### 3.1. Sự phân ly giữa Độ chính xác (Accuracy) và Kiểm soát rủi ro (Risk-Control)
Trên mặt trận dự báo giá trị (Point Forecast):
**Bảng 2: Top Mô hình theo Độ chính xác Dự báo (Point Forecast)**

| Rank | Model | MSE | MAE |
|---:|---|---:|---:|
| 1 | Moirai (moirai2) | 0.0227 | 0.0906 |
| 2 | Moirai (moirai_moe) | 0.0247 | 0.0975 |

Đúng như lý thuyết, mô hình khổng lồ Moirai áp đảo toàn diện ở các thang đo MSE/MAE. Trái lại, nhóm Transformer cục bộ có MSE rất tệ (khoảng $\sim 0.22$), lọt thỏm ở cuối bảng xếp hạng. 

Nhưng, khi bước sang bài toán quản trị rủi ro (VaR Backtesting):
**Bảng 3: Xếp hạng Mô hình Backtesting tại ngưỡng VaR 1% (Extreme Risk)**

| Rank | Model | VaR 1% Pass Rate |
|---|---|---:|
| 1 | Autoformer Tier 1 | 35.56% |
| 2 | Informer Tier 1 | 28.89% |
| ... | ... | ... |
| 10+ | Moirai (moirai_moe) | 20.00% |
| 15+ | Moirai (moirai2) | 11.11% |

**Nghịch lý xuất hiện:** Dù Moirai có sai số thấp (tốt) gấp 10 lần Autoformer, nhưng ở khả năng bảo vệ rủi ro (VaR Pass Rate), Autoformer (Tier 1) lại cao gần gấp đôi Moirai!

### 3.2. Lý giải học thuật cho Nghịch lý
Hiện tượng này có thể được giải thích vững chắc thông qua 3 cơ sở lý luận:
1. **Sự sai lệch hàm mục tiêu (Objective Function Divergence):** Moirai được pre-train bằng các hàm mục tiêu MSE/MAE, vốn tối ưu hóa giá trị kỳ vọng (mean). Đặc tính của MSE là "phạt" rất nặng các dự báo nằm xa trung bình, khiến mô hình có xu hướng làm mượt (over-smooth) các cú sốc cực đoan [17].
2. **Độ nhạy cục bộ vs. Tính ổn định (Local Sensitivity vs. Over-parameterization):** Các mô hình Transformer quy mô lớn thường thất bại trong dự báo chuỗi thời gian do bị nhiễu bởi cơ chế attention quá phức tạp (Zeng et al., 2023 [12]). Ở chiều ngược lại, các Transformer thu nhỏ (Tier 1) với cấu trúc tinh gọn (parsimonious) đã tránh được bẫy over-parameterization. Việc này giúp mô hình tối ưu hóa rất ổn định hàm Student-t có tham số $\nu$ động. Sự nhạy bén cục bộ (local sensitivity) này cho phép mô hình điều chỉnh linh hoạt độ dày của đuôi phân phối, tạo ra dải rủi ro (coverage) phản ứng tức thời với các cú sốc hẹp [16].
3. **Bản chất của VaR Backtesting:** Các bài test Kupiec và Christoffersen [13, 14] không quan tâm dự báo của bạn bám sát giá trị thực tế bao nhiêu (MSE), chúng chỉ đếm số lần dự báo hụt (violation) có khớp với tỷ lệ thống kê (1%) hay không. Mô hình Transformer cục bộ có dải băng tin cậy rộng (coverage) sẽ dễ dàng vượt qua bài test này hơn Foundation Models bám quá sát trung bình [13].

### 3.3. Hiệu ứng Kích thước Mô hình: Vì sao Tier 1 (Miniaturized) đánh bại Tier 3 (Large)?
Trong hệ thống Benchmark, nhóm Transformer được thiết kế thành 3 phân cấp: Tier 1 (nhỏ gọn), Tier 2 (tiêu chuẩn), và Tier 3 (quy mô lớn). Nghịch lý xuất hiện ngay trong nội bộ nhóm khi **Tier 1 liên tục giành Top 1 VaR Pass Rate (35.56%), trong khi Tier 3 hoàn toàn sụp đổ**.

Điều này được giải thích bởi **Hiện tượng Over-parameterization** trong chuỗi thời gian tài chính (Zeng et al., 2023 [12]). Dữ liệu biến động tài chính có tỷ lệ tín hiệu trên nhiễu (signal-to-noise ratio) cực thấp. Khi sử dụng Tier 3 (hàng triệu tham số) trên cửa sổ huấn luyện ngắn, mô hình bị mất phương hướng học, dẫn đến hiện tượng gradient bất ổn và suy thoái nghiệm. Ngược lại, Tier 1 đóng vai trò như một cơ chế điều chuẩn (regularization) cấu trúc, ép mô hình chỉ học những đặc trưng cốt lõi nhất.

Bằng chứng rõ ràng nhất nằm ở đồ thị hội tụ hàm mất mát (Loss Curves) của mô hình Autoformer trên chỉ số EuroNext 100:

**Hình 1: Đồ thị hàm mất mát của Tier 1 (Trái) và Tier 3 (Phải)**
| Autoformer Tier 1 (Miniaturized) | Autoformer Tier 3 (Large) |
| :---: | :---: |
| ![Tier 1 Loss](file:///D:/UIT/1003_EPA_PROJECT/1.0.0/1003_EPA-Project_UIT/results_v2/tranformers%20based/results_v2/visualizations/Loss_Curves/Tier_1_Miniaturized/EuroNext_100_Autoformer_loss.png) | ![Tier 3 Loss](file:///D:/UIT/1003_EPA_PROJECT/1.0.0/1003_EPA-Project_UIT/results_v2/tranformers%20based/results_v2/visualizations/Loss_Curves/Tier_3_Large/EuroNext_100_Autoformer_loss.png) |

Nhìn vào biểu đồ trên, Tier 1 có đường loss Validation bám sát Training và hội tụ rất êm mượt. Trong khi đó, Tier 3 bị dao động dữ dội (fluctuations), đường Validation nhảy vọt mất kiểm soát, thể hiện rõ sự bất ổn và sụp đổ của một kiến trúc bị thừa mứa tham số.

### 3.4. Các kiểm định thống kê bổ sung
Theo đúng yêu cầu của Reviewer, em đã tiến hành 3 loại kiểm định (Friedman, Nemenyi, Diebold-Mariano). Kết quả khẳng định: Sự chênh lệch khổng lồ về khả năng dự báo điểm và khả năng quản trị rủi ro giữa Moirai và Transformer đều có ý nghĩa thống kê sâu sắc (p-value < 0.05), không phải là yếu tố ngẫu nhiên.

### 3.5. Thảo luận: Đề xuất Khung Đánh Giá Hai Tầng
Từ phát hiện thực nghiệm này, đóng góp của bài báo được chốt lại thành **Khung Đánh Giá Hai Tầng (Two-Tier Evaluation Framework)**:
*   **Tier 1 (Đánh giá dự báo điểm):** Dùng MSE, MAE để đo lường. Moirai Foundation Models làm cực kỳ tốt, phù hợp cho bài toán định giá tài sản (Asset Pricing).
*   **Tier 2 (Đánh giá rủi ro phân phối):** Dùng VaR Backtesting. Transformer cục bộ tích hợp hàm phân phối linh hoạt là sự lựa chọn ưu việt, phù hợp cho bài toán quản trị rủi ro (Risk Management).

---
## 4. Tài liệu tham khảo
[1] T. Bollerslev, "Generalized autoregressive conditional heteroskedasticity," *Journal of Econometrics*, vol. 31, no. 3, pp. 307-327, 1986.
[2] L. R. Glosten, R. Jagannathan, and D. E. Runkle, "On the relation between the expected value and the volatility of the nominal excess return on stocks," *The Journal of Finance*, vol. 48, no. 5, pp. 1779-1801, 1993.
[3] R. T. Baillie, T. Bollerslev, and H. O. Mikkelsen, "Fractionally integrated generalized autoregressive conditional heteroskedasticity," *Journal of Econometrics*, vol. 74, no. 1. pp. 3-30, 1996.
[4] A. Vaswani et al., "Attention is all you need," in *Advances in Neural Information Processing Systems 30*, 2017.
[5] H. Zhou et al., "Informer: Beyond efficient transformer for long sequence time-series forecasting," in *AAAI*, 2021.
[6] H. Wu et al., "Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting," in *NeurIPS*, 2021.
[7] N. Kitaev, L. Kaiser, and A. Levskaya, "Reformer: The efficient transformer," in *ICLR*, 2020.
[8] P. Zhao, H. Zhu, W. S. H. Ng, and D. L. Lee, "From GARCH to neural network for volatility forecast," in *AAAI*, 2024.
[9] X. Liu et al., "Moirai-MoE: Empowering time series foundation models with sparse mixture of experts," in *ICML*, 2025.
[10] O. B. Sezer, M. U. Gudelek, and A. M. Ozbayoglu, "Financial time series forecasting with deep learning: A systematic literature review: 2005–2019," *Applied Soft Computing*, 2020.
[11] B. Lim and S. Zohren, "Time-series forecasting with deep learning: a survey," *Phil. Trans. R. Soc. A*, 2021.
[12] A. Zeng et al., "Are transformers effective for time series forecasting?," in *AAAI*, 2023.
[13] P. Christoffersen, "Evaluating interval forecasts," *International Economic Review*, 1998.
[14] P. Kupiec, "Techniques for verifying the accuracy of risk measurement models," *The Journal of Derivatives*, 1995.
[15] F. X. Diebold and R. S. Mariano, "Comparing predictive accuracy," *JBES*, 1995.
[16] C. Koenker and G. Bassett Jr, "Regression quantiles," *Econometrica*, 1978.
[17] T. Gneiting, "Making and evaluating point forecasts," *Journal of the American Statistical Association*, vol. 106, no. 494, pp. 746-762, 2011.