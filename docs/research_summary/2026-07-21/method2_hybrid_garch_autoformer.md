# Phân tích chi tiết Phương pháp 2: Kiến trúc Lai GARCH-NN (Hybrid GARCH-Autoformer)

## 1. Vấn đề cốt lõi (The Trade-off)
Trong bài toán dự báo biến động (Volatility Forecasting), việc lựa chọn kiến trúc mô hình thường dẫn đến sự đánh đổi (trade-off) giữa **Chỉ số dự báo (Forecast metrics)** và **Chỉ số rủi ro (Risk metrics)**:
*   **GARCH:** Các mô hình GARCH truyền thống bám sát đường nền (baseline conditional variance) rất tốt nhờ vào đặc tính *Mean Reversion* (đảo chiều về trung bình). Nhờ đó, chúng thường có điểm MSE và QLIKE tốt. Tuy nhiên, chúng quá cứng nhắc (tuyến tính) nên phản ứng rất chậm với các cú sốc cực đoan, dẫn đến trượt bài kiểm định rủi ro Kupiec (Violation Rate quá cao).
*   **Autoformer:** Khối Decomposition và cơ chế Auto-Correlation giúp Autoformer săn lùng các chu kỳ (periodicity) vĩ mô cực tốt. Nó bắt kịp ngay các cú sốc chu kỳ, bảo vệ dải VaR rất tốt. Tuy nhiên, nó bị dính lỗi *over-smoothing* (làm mượt quá mức), gọt bỏ các dao động vi mô $\implies$ QLIKE và MSE rất cao.

## 2. Giải pháp: "Chia để trị" (Residual Correction Hybrid)
Hướng tiếp cận SOTA (State-of-the-Art) để dung hòa cả hai là kiến trúc Lai (Hybrid) theo mô hình **Học phần dư (Residual Correction)**.

### Cơ chế hoạt động (Pipeline 2 giai đoạn)

**Giai đoạn 1: Base Model (GARCH)**
Sử dụng một mô hình GARCH bất đối xứng (như GJR-GARCH hoặc EGARCH) để mô hình hóa chuỗi lợi nhuận $r_t$.
Mô hình này sẽ trả ra dự báo nền: $\sigma^2_{garch, t}$.
Dự báo này "hấp thụ" toàn bộ các biến động tuyến tính bình thường của thị trường. Nhờ đó, ta đã "khóa" (lock) được chỉ số MSE và QLIKE ở mức an toàn.

**Giai đoạn 2: Trích xuất và Học Phần dư (Residual Learning)**
Chuỗi phần dư (những cú sốc phi tuyến tính mà GARCH bó tay) được tính bằng:
$$ e_t = \sigma^2_{true, t} - \sigma^2_{garch, t} $$
Hoặc sử dụng sai số chuẩn hóa (Standardized residuals):
$$ z_t = \frac{r_t}{\sigma_{garch, t}} $$

Thay vì yêu cầu Autoformer dự báo chuỗi lợi nhuận nhiễu loạn ban đầu, ta **feed chuỗi phần dư $e_t$ (hoặc $z_t$) vào Autoformer**.
Lúc này, mạng nơ-ron không cần bận tâm đến đường trung bình nữa. Nó dồn toàn bộ năng lực tính toán của lớp `Auto-Correlation` để tìm ra các "chu kỳ ẩn" trong các cú sốc (hidden periodicity in shocks). 
Dự báo của mạng nơ-ron: $e_{pred, t} = Autoformer(e_{t-1}, ...)$.

**Giai đoạn 3: Tổng hợp (Ensemble)**
Dự báo biến động cuối cùng là sự tổng hợp tuyến tính:
$$ \sigma^2_{final, t} = \sigma^2_{garch, t} + e_{pred, t} $$

## 3. Lợi ích mang lại
*   **Giảm QLIKE:** Không bao giờ bị over-smoothing vì phần $\sigma^2_{garch}$ đã bám sát thực tế.
*   **Vượt qua Kupiec Test:** Khi có tin tức "Thiên nga đen", GARCH phản ứng chậm, nhưng Autoformer nhận diện được cú sốc phần dư và bơm thêm $e_{pred, t}$ vào dự báo. Kết quả là dải VaR được phình to ngay lập tức, che chắn hoàn hảo cho danh mục.

## 4. Tài liệu tham khảo (References)
Phương pháp này dựa trên một loạt các nghiên cứu thành công về GARCH-NN hybrid trong giai đoạn 2020-2024:
1.  **Donaldson & Kamstra (1997), Khashei & Bijari (2010):** Các nghiên cứu nền tảng về việc dùng Neural Network để học phần dư (residuals) của chuỗi thời gian nhằm cải thiện ARIMA/GARCH.
2.  **Kim & Won (2018) - "Forecasting the volatility of stock price index":** Một trong những nghiên cứu nổi tiếng chứng minh mô hình lai GARCH-LSTM vượt trội hơn hẳn GARCH truyền thống bằng cách để LSTM xử lý phần dư phi tuyến tính.
3.  **Học thuật 2023-2024:** Các bài báo trên *Expert Systems with Applications* và *IEEE Access* về "Hybrid GARCH-Transformer" đều sử dụng chung một triết lý: Dùng GARCH trích xuất đặc trưng (Feature extraction) và dùng Transformer (hoặc Autoformer/Informer) để xử lý phần phi tuyến tính (non-linear dynamics).
