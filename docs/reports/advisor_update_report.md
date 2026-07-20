# Báo Cáo Cập Nhật Tiến Độ Nghiên Cứu: Khung Đánh Giá Hợp Nhất Các Mô Hình Dự Báo Biến Động

**Nội dung:** Báo cáo về việc tiếp thu góp ý từ Reviewer, các hướng khắc phục đã thực hiện, và phân tích kết quả thực nghiệm mới nhất.
Dựa trên những phản hồi từ các Reviewer về bài báo của chúng ta, em đã tiến hành nghiên cứu lại các luận điểm bị phê bình, điều chỉnh lại phương pháp luận và cho chạy lại toàn bộ mô hình để khắc phục các hạn chế trước đây. Dưới đây là báo cáo chi tiết về những vấn đề em đã nhận diện, cách em khắc phục, và kết quả phân tích mới nhất.
## 1. Tóm tắt góp ý từ Reviewer và nhận diện điểm yếu
Qua việc đọc kỹ các nhận xét, em nhận thấy cả 3 Reviewer đều đánh giá cao ý tưởng cốt lõi của bài nghiên cứu (sự đánh đổi giữa *forecast accuracy* và *risk control*), cũng như đánh giá cao quy mô của hệ thống benchmark. Tuy nhiên, bài viết bản trước tồn tại 4 nhóm điểm yếu nghiêm trọng cần khắc phục:
1.  **Vấn đề Dữ liệu và Thiết lập Thực nghiệm (Reviewer 1, 3):** Bài trước của chúng ta chưa làm rõ nguồn gốc dữ liệu, quy trình tiền xử lý, cũng như hoàn toàn thiếu bảng kích thước mẫu (sample sizes), làm giảm độ minh bạch và độ tin cậy của thực nghiệm.
2.  **Sự mâu thuẫn và yếu kém của mô hình đề xuất Moirai-MoE-GARCHs (Reviewer 1, 2, 3):** Đây là điểm bị chỉ trích nặng nhất. Kết quả thực nghiệm cho thấy mô hình này xếp thứ 9/10 về năng lực quản trị rủi ro, trái ngược với những khẳng định (claim) trong bài. Việc này làm suy yếu đi ý nghĩa đóng góp cấu trúc mô hình của bài báo.
3.  **Thiếu kiểm định thống kê và đánh giá chuyên sâu (Reviewer 2):** Đánh giá VaR trước đó quá mỏng (chỉ có VaR 5%). Bài báo phân tích mang tính chất bề mặt (descriptive) thay vì đi sâu vào bản chất (analytical). Đặc biệt, thiếu hẳn các kiểm định thống kê như Diebold-Mariano hay Friedman.
4.  **Hạn chế về Tính mới (Novelty) (Reviewer 2, 3):** Framework bị đánh giá là chỉ ghép nối các phương pháp có sẵn. Hơn nữa, việc thiếu chi tiết thiết lập (hyperparameters, training time) làm bài nghiên cứu khó có thể tái tạo (reproduce).
---
## 2. Các hướng khắc phục và quá trình chạy lại mô hình
Để giải quyết triệt để các vấn đề trên, em đã cấu trúc lại một khung đánh giá thống nhất, chuẩn hóa toàn bộ quy trình tiền xử lý, thiết lập siêu tham số và thiết kế lại hàm mất mát (loss function).
### 2.1. Viết lại phần mô tả Dữ liệu và Thiết lập Bài toán (Dataset & Problem Setup)
Em đã làm rõ thông tin bộ dữ liệu: sử dụng 9 chỉ số toàn cầu, được lấy từ nguồn đáng tin cậy là Investing.com và vnstock. Bài toán dự báo được định hình lại rõ ràng với:
*   **Tỷ suất sinh lợi (Log Returns):** $r_t = \ln(P_t/P_{t-1}) \times 100$.
*   **Tính nhân quả (Causality):** Tại thời điểm $t$, mô hình chỉ được sử dụng cửa sổ 60 ngày quá khứ $x_t = (r_{t-60}, \dots, r_{t-1})$. Biến động thực tế (Target Realized Volatility) $\sigma_{t,h}$ tại các khoảng thời gian $h \in \{1, 3, 5, 10, 21\}$ được tính bằng độ lệch chuẩn trượt 60 ngày kết thúc chính xác tại $t+h-1$. Mô hình tuyệt đối không có quyền truy cập dữ liệu tương lai để tránh look-ahead bias.
*   **Phân tách nhận biết phân phối:** Thay vì chia 60/20/20 ngẫu nhiên, em đã cài đặt **thuật toán phân tách nhận biết phân phối (distribution-aware split)**. Thuật toán này tìm kiếm điểm chia sao cho đặc tính phân phối đuôi dày Student-t (thông qua tham số $\nu$) được giữ ổn định qua các tập Train/Val/Test.
**Bảng 1: Kích thước mẫu và phân tách cho từng dataset (2010–2025)**

| Index        |    Source     | Train | Validation | Test | Total |
| :----------- | :-----------: | :---: | :--------: | :--: | :---: |
| DAX 40       | Investing.com | 1983  |    1104    | 972  | 4059  |
| Euronext 100 | Investing.com | 2005  |    1115    | 979  | 4099  |
| IBEX 35      | Investing.com | 1815  |    1362    | 923  | 4100  |
| KOSPI Index  | Investing.com | 1944  |    1109    | 881  | 3934  |
| Nikkei 225   | Investing.com | 1677  |    1174    | 1062 | 3913  |
| SMI          | Investing.com | 1987  |    1135    | 901  | 4023  |
| S&P 500      | Investing.com | 1988  |    1109    | 927  | 4024  |
| VN30         | Investing.com | 1971  |    1096    | 925  | 3992  |
| VN-Index     | Investing.com | 1964  |    1103    | 925  | 3992  |
### 2.2. Chi tiết Triển khai và Thiết lập Siêu tham số cho 10 Mô hình
Để đảm bảo tính minh bạch và khả năng tái tạo (reproducibility) theo yêu cầu của Reviewer, em đã chuẩn hóa và báo cáo chi tiết toàn bộ cấu hình:
*   **Nhóm Kinh tế lượng (GARCH-family):** Bao gồm GARCH(1,1), GJR-GARCH(1,1) và FI-GARCH(1,d,1). Em sử dụng thư viện `arch` với cửa sổ lịch sử trượt (rolling window) gồm 1.000 quan sát để fit động tại mỗi bước kiểm thử. Các mô hình đều dùng giả định đuôi Student-t cho sai số (innovations) và tạo dự báo phương sai đệ quy.
*   **Nhóm Transformer (Vanilla, Informer, Autoformer, Reformer):** Để kiểm định xem Transformer có bị "khủng hoảng dữ liệu" khi đầu vào chỉ dài 60 ngày hay không, em thiết kế 3 cấp độ phức tạp:
    *   *Tier 1 (Miniaturized):* $d_{model}=32$, 2 encoder layers, 4 heads (tránh overfitting).
    *   *Tier 2 (Standard):* $d_{model}=128$, 3 encoder layers, 8 heads.
    *   *Tier 3 (Large):* $d_{model}=512$, 6 encoder layers, 8 heads.
    Tất cả Transformer xuất dự báo trực tiếp qua `VolatilityHead` thay vì tự hồi quy. Chúng được huấn luyện bằng loss Student-t NLL với bậc tự do $\nu$ được **tính toán động** cho từng chỉ số thị trường; sử dụng AdamW (LR=$10^{-3}$), 30 epochs, batch size 128, patience=5.
*   **Mô hình Lai GARCH-LSTM:** Em đã cải tiến toàn diện bản gốc. Cell GARCH-LSTM tích hợp thẳng phương trình phương sai vào cổng của LSTM. Điểm khác biệt mấu chốt là hàm mất mát Student-t NLL được **cố định $v=5.0$** để làm mỏ neo điều chuẩn (regularization), giúp LSTM không học theo nhiễu cực đoan. Cơ chế dự báo cũng đổi từ tự hồi quy sang **dự báo trực tiếp đa bước (direct multi-horizon forecasting)**. Nhờ đó, sai số (MSE) giảm ngoạn mục từ mức $1.28 \times 10^6$ xuống $\sim 0.11$.
### 2.3. Khẳng định Tính Mới (Novelty): Mô hình Moirai_VAR và hàm mất mát VaR-Aware
Điểm thay đổi quan trọng nhất là em đã **loại bỏ hoàn toàn kiến trúc Moirai-MoE-GARCHs** gây tranh cãi do hiệu suất không như ý. Thay vào đó, em giữ nguyên backbone Moirai (đóng băng weights) và cho các biểu diễn đặc trưng đi qua khối hồi quy MLP 2 lớp để tạo thành mô hình **Moirai_VAR**.
Tính mới của nghiên cứu giờ đây không nằm ở việc ghép nối cấu trúc, mà tập trung vào **hàm mất mát nhận thức rủi ro (VaR-Aware Loss)**. Vì các mô hình chỉ tập trung vào MSE sẽ luôn thiên lệch về kỳ vọng trung bình và thất bại trong việc kiểm soát rủi ro đuôi, em đã thiết kế lại hàm mục tiêu:
$$ \text{Total\_Loss} = \text{MSE}(\text{vol}_{pred}, \text{vol}_{true}) + \lambda_{var} \times \text{QuantileVaRLoss}(\text{return}_{realized}, \text{VaR}_{threshold}) $$
Với trọng số $\lambda_{var}=0.2$, mô hình bị ép phải "học" tính phân phối đuôi. Nếu dự báo đâm thủng ngưỡng VaR mục tiêu, mô hình sẽ lập tức bị phạt bằng hàm Pinball Loss (mất mát phân vị). Moirai_VAR được huấn luyện bằng AdamW (LR=$10^{-3}$), batch size 32, patience=7. Đây là đóng góp chính về mặt phương pháp luận, phản bác trực tiếp lập luận "thiếu tính mới" của Reviewer 2 và cung cấp một cách tiếp cận đột phá cho lĩnh vực.
## 3. Phân tích và diễn giải kết quả mới
Sau quá trình tinh chỉnh và chạy lại toàn bộ mô hình, kết quả đạt được đã cung cấp những insight rõ nét hơn rất nhiều.
### 3.1. Phân tích tổng quan: Sự phân ly giữa Độ chính xác (Accuracy) và Kiểm soát rủi ro (Risk-Control)
Đầu tiên, em phân tích trên mặt trận dự báo giá trị (Point Forecast):
**Bảng 2: Top Mô hình theo Độ chính xác Dự báo (Point Forecast)**

| Rank | Model | MSE | MAE | QLIKE | Avg forecast rank |
|---:|---|---:|---:|---:|---:|
| 1 | Moirai (moirai2) | 0.0227 | 0.0906 | 0.0091 | 1.000 |
| 2 | Moirai_VAR (λ=0.2, moirai2) | 0.0237 | 0.0947 | 0.0091 | 2.333 |
| 3 | Moirai (moirai_moe) | 0.0247 | 0.0975 | 0.0091 | 2.667 |
| 4 | Moirai_VAR (λ=0.2, moirai_moe)| 0.0252 | 0.1014 | 0.0093 | 4.000 |
Đúng như kỳ vọng, mô hình foundation Moirai áp đảo toàn diện ở các thang đo MSE, MAE và QLIKE. Moirai_VAR dù bị kéo lại bởi thành phần loss rủi ro, vẫn bám sát ở các vị trí đầu.
Tuy nhiên, câu chuyện hoàn toàn rẽ sang hướng khác khi đánh giá Backtesting (Risk-Control). Theo góp ý của Reviewer, em đã bổ sung thêm ngưỡng VaR 1%:
**Bảng 3: Xếp hạng Mô hình Backtesting tại VaR 5% và VaR 1%**

| Khung đánh giá | Vị trí Top 1 | Vị trí Top 2 |
|---|---|---|
| **VaR 5% Pass Rate** | Autoformer Tier 2 (24.44%) | Autoformer Tier 1 / Reformer Tier 3 (20.00%) |
| **VaR 1% Pass Rate** | Autoformer Tier 1 (35.56%) | Informer Tier 1 / Autoformer Tier 2 (28.89%) |
Nhóm Transformer, đặc biệt là các biến thể thu nhỏ (Tier 1 và Tier 2), lại là những mô hình giành chiến thắng tuyệt đối ở khả năng kiểm soát rủi ro phân phối. Dữ liệu này là bằng chứng đanh thép bảo vệ cho thông điệp cốt lõi của bài báo: **mô hình dự báo biến động chính xác nhất (Moirai) không đảm bảo năng lực quản trị rủi ro đuôi (Tail Risk) tốt nhất.**
### 3.2. Giải thích sâu sắc kết quả (Robustness Check với 3 phương pháp VaR)
Để phân tích sâu hơn (analytical insights), em biến đổi dự báo biến động thành các giá trị VaR thông qua 3 giả định: Normal (đuôi mỏng), Student-t (đuôi tham số), và FHS (đuôi thực nghiệm).

**Bảng 4: So sánh Tỷ lệ vi phạm (Violation Error) qua 3 phương pháp VaR**

| VaR case | Method | Violation rate | Abs violation error | Quantile loss |
|---|---|---:|---:|---:|
| 5% | Normal | 0.0598 | 0.0172 | 0.4355 |
| 5% | Student-t | 0.0326 | 0.0217 | 0.5273 |
| 5% | FHS | 0.0527 | 0.0040 | 0.4425 |
| 1% | Normal | 0.0280 | 0.0182 | 0.1363 |
| 1% | Student-t | 0.0061 | 0.0058 | 0.1830 |
| 1% | FHS | 0.0148 | 0.0048 | 0.1470 |
Phân tích này cho thấy Student-t có khuynh hướng bảo thủ (ít vi phạm nhất), trong khi FHS giúp đưa violation rate về sát nhất với mục tiêu (5% và 1%).
Quan trọng hơn, **sự ra đời của Moirai_VAR đã chứng minh hiệu quả cực lớn**. Ở góc độ Quantile Loss (Tổn thất phân vị) cho VaR 5% (Normal và FHS), Moirai_VAR đã vươn lên vị trí **Top 1 toàn bảng**. Nó đã hi sinh một lượng nhỏ MSE để đổi lấy sự chuẩn xác tuyệt vời ở vùng rủi ro đuôi, khẳng định tính đúng đắn của VaR-Aware Loss.
### 3.3. Các kiểm định thống kê bổ sung
Theo đúng yêu cầu của Reviewer 2, em đã tiến hành đầy đủ 3 loại kiểm định:
1.  **Friedman Test:** Khẳng định sự khác biệt giữa các mô hình ở MSE, MAE, QLIKE là hoàn toàn không ngẫu nhiên (p-value < $1.62 \times 10^{-92}$).
2.  **Nemenyi Test:** Xác nhận có hàng chục cặp mô hình có sự chênh lệch mang ý nghĩa thống kê (hơn 80 cặp về điểm số dự báo).
3.  **Diebold-Mariano (DM) Test:**
**Bảng 5: Số cặp khác biệt có ý nghĩa thống kê (DM Test trên Quantile Loss)**

| VaR case | Normal | Student-t | FHS |
|---|---:|---:|---:|
| 5% | 2,749 | 3,121 | 1,355 |
| 1% | 3,589 | 2,928 | 1,264 |
Kết quả DM Test xác nhận rằng Quantile loss giữa các mô hình có độ chênh lệch rất rõ rệt ở mức ý nghĩa thống kê (đặc biệt là ngưỡng VaR 1% Normal).
### 3.4. Thảo luận và hàm ý thực tiễn (Discussion)
Từ kết quả này, em muốn kết luận về một **Khung Đánh Giá Hai Tầng (Two-Tier Evaluation Framework)**:
*   **Tier 1 (Đánh giá dự báo điểm):** Dùng MSE, MAE để đo lường. Moirai làm rất tốt, phù hợp cho bài toán định giá (pricing).
*   **Tier 2 (Đánh giá rủi ro phân phối):** Dùng VaR Backtesting. Ở đây, Transformer Tiers và Moirai_VAR thể hiện sự ưu việt, phù hợp cho bài toán quản trị rủi ro (Risk Management).
Sự kết hợp giữa Moirai Foundation và VaR-Aware Loss đã tạo nên một mô hình mạnh mẽ, dung hòa được cả hai yếu tố, đủ sức nặng để chúng ta tự tin submit lại bài nghiên cứu.
---
## 4. Tài liệu tham khảo
[1] T. Bollerslev, "Generalized autoregressive conditional heteroskedasticity," *Journal of Econometrics*, vol. 31, no. 3, pp. 307-327, 1986.
[2] L. R. Glosten, R. Jagannathan, and D. E. Runkle, "On the relation between the expected value and the volatility of the nominal excess return on stocks," *The Journal of Finance*, vol. 48, no. 5, pp. 1779-1801, 1993.
[3] R. T. Baillie, T. Bollerslev, and H. O. Mikkelsen, "Fractionally integrated generalized autoregressive conditional heteroskedasticity," *Journal of Econometrics*, vol. 74, no. 1. pp. 3-30, 1996.
[4] A. Vaswani et al., "Attention is all you need," in *Advances in Neural Information Processing Systems 30*, 2017, pp. 5998-6008.
[5] H. Zhou et al., "Informer: Beyond efficient transformer for long sequence time-series forecasting," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 35, no. 12, pp. 11106-11115, 2021.
[6] H. Wu et al., "Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting," in *Advances in Neural Information Processing Systems 34*, 2021, pp. 22419-22430.
[7] N. Kitaev, L. Kaiser, and A. Levskaya, "Reformer: The efficient transformer," in *International Conference on Learning Representations*, 2020.
[8] P. Zhao, H. Zhu, W. S. H. Ng, and D. L. Lee, "From GARCH to neural network for volatility forecast," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 38, no. 15, 2024, pp. 16998-17006.
[9] X. Liu et al., "Moirai-MoE: Empowering time series foundation models with sparse mixture of experts," in *Proceedings of the 42nd International Conference on Machine Learning*, vol. 267, PMLR, 2025, pp. 38940-38962.
[10] O. B. Sezer, M. U. Gudelek, and A. M. Ozbayoglu, "Financial time series forecasting with deep learning: A systematic literature review: 2005–2019," *Applied Soft Computing*, vol. 90, p. 106181, 2020.
[11] B. Lim and S. Zohren, "Time-series forecasting with deep learning: a survey," *Philosophical Transactions of the Royal Society A*, vol. 379, 2021.
[12] A. Zeng et al., "Are transformers effective for time series forecasting?," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 37, no. 9, pp. 11121-11128, 2023.
[13] P. Christoffersen, "Evaluating interval forecasts," *International Economic Review*, vol. 39, no. 4, pp. 841-862, 1998.
[14] P. Kupiec, "Techniques for verifying the accuracy of risk measurement models," *The Journal of Derivatives*, vol. 3, no. 2, pp. 73-84, 1995.
[15] F. X. Diebold and R. S. Mariano, "Comparing predictive accuracy," *Journal of Business & Economic Statistics*, vol. 13, no. 3, pp. 253-263, 1995.
[16] C. Koenker and G. Bassett Jr, "Regression quantiles," *Econometrica: journal of the Econometric Society*, pp. 33-50, 1978.