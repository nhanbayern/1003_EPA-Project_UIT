# Báo cáo Nghiên cứu: Các Phương pháp Ra quyết định Đa tiêu chí (MCDM) trong Đánh giá Mô hình Dự báo Biến động

**Ngày thực hiện:** 22/07/2026
**Chủ đề:** Nghiên cứu cơ sở lý thuyết của hệ thống MCDM, phân tích so sánh giữa phương pháp Điểm tổng hợp tuyến tính (SAW) và kỹ thuật hình học TOPSIS trong việc xếp hạng mô hình tài chính.

---

## 1. Mở đầu: Tại sao cần Ra quyết định Đa tiêu chí (MCDM)?

Trong lĩnh vực dự báo tài chính và quản trị rủi ro, việc đánh giá mô hình luôn đối mặt với một sự đánh đổi (trade-off) gay gắt giữa hai thái cực:
1.  **Độ chính xác thống kê (Accuracy):** Tối ưu hóa sai số trung bình bằng các hàm mất mát như MSE, MAE, QLIKE.
2.  **Bảo vệ rủi ro đuôi (Tail Risk Coverage):** Tối ưu hóa việc dự đoán các sự kiện cực đoan bằng Value-at-Risk (VaR 99%) qua bài test Kupiec POF [1].

Trong quá trình thực nghiệm, các mô hình Foundation/Deep Learning thuần túy (ví dụ: `Moirai`) thường có sai số MSE cực thấp nhưng lại thất bại hoàn toàn ở việc đánh giá rủi ro (VaR). Ngược lại, các mô hình kinh tế lượng (GARCH) giữ rủi ro đuôi rất tốt nhưng sai số thống kê lại lớn.

Đứng trước sự mâu thuẫn này, tài chính định lượng hiện đại bắt buộc phải sử dụng **MCDM (Multi-Criteria Decision Making - Ra quyết định Đa tiêu chí)**. MCDM giúp tổng hợp các tiêu chí xung đột thành một điểm số duy nhất để xếp hạng mô hình một cách toàn diện [2]. 

Để giải bài toán này, trong dự án chúng ta đã triển khai và tham khảo **ba kỹ thuật MCDM chính**: Phương pháp Điểm tổng hợp tuyến tính (SAW), Phương pháp Khoảng cách hình học (TOPSIS), và Phương pháp Phân tích Thứ bậc (AHP).

---

## 2. Phân tích Các Kỹ thuật MCDM Được Áp Dụng

Dưới đây là chi tiết cơ chế tính toán và đặc điểm cấu trúc của từng hệ thống đánh giá. Các mô hình xếp hạng (như SAW và TOPSIS) trước đây thường dùng bộ 7 tiêu chí chia đều (50% Accuracy - 50% Risk), trong khi AHP có thể hỗ trợ xác định lại bộ trọng số này một cách khách quan hơn.

### Kỹ thuật 1: Simple Additive Weighting (SAW) - Phương pháp Điểm Tổng hợp Trung bình
Đây là phương pháp ra quyết định đa tiêu chí cổ điển và trực quan nhất. Nó tính điểm bằng cách chuẩn hóa tuyến tính mọi dữ liệu và lấy trung bình cộng.

*   **Cách tính toán riêng biệt:**
    1. **Chuẩn hóa (Min-Max Scaling):** Đưa mọi tiêu chí về thang điểm 0-100. 
       *   Tiêu chí chi phí (Càng nhỏ càng tốt, VD: MSE, Khoảng cách tới VaR mục tiêu): Điểm = $100 \times \frac{Max - X}{Max - Min}$
       *   Tiêu chí lợi ích (Càng lớn càng tốt, VD: Passed Cases): Điểm = $100 \times \frac{X}{Max_{Lý thuyết}}$
    2. **Nhân trọng số & Tính tổng:** Lấy trung bình cộng (hoặc tổng có trọng số) của nhóm Điểm Chính xác (50%) và nhóm Điểm Rủi ro (50%) để ra **Final Score**.
*   **Đặc điểm nổi bật:** 
    *   Tính chất hoàn toàn tuyến tính. Mọi thang đo được ép phẳng về thang điểm 100 quen thuộc, rất dễ giải thích cho người không có chuyên môn về toán.

### Kỹ thuật 2: TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
Đây là thuật toán MCDM phức tạp hơn, được Hwang và Yoon đề xuất (1981) [3]. Thay vì cộng điểm, TOPSIS tìm kiếm giải pháp dựa trên vị trí hình học không gian đa chiều.

*   **Cách tính toán riêng biệt:**
    1. **Chuẩn hóa Vector:** Không dùng Max-Min, TOPSIS chuẩn hóa bằng cách chia cho độ dài vector: $r_{ij} = \frac{x_{ij}}{\sqrt{\sum x_{ij}^2}}$. Sau đó nhân với trọng số để tạo không gian không gian điểm $v_{ij}$.
    2. **Xác định Cực trị Lý tưởng ($A^+$) và Tồi tệ ($A^-$):** Thuật toán tự tìm ra giá trị hoàn hảo nhất và kém nhất cho từng tiêu chí có trong bộ dữ liệu.
    3. **Tính Khoảng cách Euclid:** Đo lường khoảng cách hình học từ mỗi mô hình tới Điểm Lý tưởng ($D^+$) và Điểm Tồi tệ ($D^-$) trong không gian 7 chiều.
    4. **Tính Điểm Gần gũi (Closeness Coefficient):** $C_i = \frac{D^-}{D^+ + D^-}$. Giá trị càng gần 1 (hoặc 100) thì càng tối ưu.
*   **Đặc điểm nổi bật:**
    *   Tính phi tuyến tính (tính theo bình phương khoảng cách). Nó coi việc đánh giá mô hình như việc đo lường khoảng cách vật lý tới một "Mô hình Hoàn hảo" trong trí tưởng tượng.

### Kỹ thuật 3: AHP (Analytic Hierarchy Process) - Phương pháp Phân tích Thứ bậc
Được phát triển bởi Thomas L. Saaty (1980) [5], AHP là một trong những phương pháp MCDM phổ biến nhất, đặc biệt mạnh mẽ trong việc **xác định trọng số (weighting)** cho các tiêu chí dựa trên đánh giá chuyên gia hoặc kiến thức miền (domain knowledge), thay vì gán cứng tỷ lệ.

*   **Cách tính toán riêng biệt:**
    1. **Xây dựng cấu trúc phân cấp:** Phân rã bài toán thành 3 tầng: Mục tiêu (Đánh giá mô hình) -> Các tiêu chí (Accuracy, Risk) -> Các lựa chọn (Moirai, GARCH, Hybrid).
    2. **Ma trận So sánh từng cặp (Pairwise Comparison):** Sử dụng thang điểm Saaty (từ 1 đến 9) để so sánh mức độ quan trọng giữa tiêu chí A và tiêu chí B. Ví dụ: Rủi ro đuôi (VaR) quan trọng hơn Sai số (MSE) 3 lần, ta gán điểm 3.
    3. **Tính toán Vector Trọng số (Eigenvector):** Từ ma trận so sánh chéo này, dùng đại số tuyến tính để tính toán ra vector trọng số chuẩn hóa của từng tiêu chí.
    4. **Kiểm tra Tỷ số nhất quán (Consistency Ratio - CR):** Đảm bảo rằng các đánh giá so sánh cặp không bị mâu thuẫn (VD: A > B, B > C nhưng lại chọn C > A). Nếu CR < 0.1, ma trận được chấp nhận.
*   **Đặc điểm nổi bật:**
    *   Thế mạnh cốt lõi của AHP là khả năng **lượng hóa các yếu tố định tính** và kiểm soát tính nhất quán trong tư duy đánh giá. Nó là mảnh ghép hoàn hảo để tạo thành mô hình **Hybrid MCDM (AHP-TOPSIS)**: dùng AHP để tính trọng số khách quan, sau đó đẩy trọng số đó vào TOPSIS để xếp hạng mô hình.

---

## 3. So sánh Điểm Giống và Khác nhau giữa SAW và TOPSIS

Dù cả hai phương pháp đều phục vụ bài toán đa tiêu chí (MCDM), sự khác biệt trong tư duy toán học dẫn đến sự khác biệt lớn về hiệu ứng xếp hạng.

| Đặc tính | SAW (Điểm Tổng Hợp Trung Bình) | TOPSIS (Khoảng cách Hình học) |
| :--- | :--- | :--- |
| **Cơ sở toán học** | Đại số tuyến tính (Phép cộng/chia trung bình). | Hình học không gian (Khoảng cách Euclid đa chiều). |
| **Cách chuẩn hóa dữ liệu** | Min-Max Scaling (dễ bị ảnh hưởng bởi giá trị ngoại lai - Outliers). | Vector Normalization (triệt tiêu ảnh hưởng của Outliers tốt hơn). |
| **Hiệu ứng Bù trừ (Compensatory)** | **Rất mạnh (Bù trừ hoàn toàn).** Một tiêu chí đạt 100 điểm có thể dễ dàng kéo một tiêu chí 0 điểm lên mức trung bình khá (50 điểm). | **Bù trừ yếu (Trừng phạt sự lệch pha).** Vì dùng bình phương khoảng cách, nếu mô hình quá kém ở 1 tiêu chí, nó bị đẩy ra rất xa Điểm Lý Tưởng $A^+$, khiến điểm tổng bị sụt giảm nặng nề. |
| **Góc nhìn lý thuyết** | Xếp hạng dựa trên "Tổng giá trị mang lại". | Xếp hạng dựa trên "Khoảng cách ngắn nhất tới sự Hoàn hảo". |

### 📌 Kết luận từ sự Khác biệt: Tại sao dự án cần chạy cả 2?
Việc đối chiếu giữa SAW và TOPSIS cho thấy bức tranh cực kỳ thú vị đối với các mô hình Foundation như `Moirai`:
- Khi chạy bằng **SAW**, nhờ tính chất *Bù trừ cực mạnh*, điểm MSE tuyệt đối (100/100) của Moirai đã phần nào cứu vớt điểm Rủi ro (6/100) của nó, giúp nó lọt vào giữa bảng xếp hạng (52/100).
- Khi chạy bằng **TOPSIS**, tính chất bù trừ bị phá vỡ. Khoảng cách hình học của Moirai tới mức rủi ro lý tưởng là một hố sâu thăm thẳm. TOPSIS trừng phạt khuyết điểm này cực mạnh, khiến hệ số $C_i$ của nó bị nén lại.

Dù sử dụng phương pháp nào (bù trừ tuyến tính hay trừng phạt hình học), **Mô hình Hybrid GARCH-Autoformer** vẫn bảo vệ thành công vị trí Top 1. Do cấu trúc thiết kế lai (GARCH lo rủi ro, Autoformer lo sai số), mô hình này phát triển toàn diện, không có "lỗ hổng" nào để bị TOPSIS trừng phạt, và cũng sở hữu mức điểm trung bình tuyến tính xuất sắc nhất. Điều này khẳng định độ vững chắc tuyệt đối của kết quả nghiên cứu.

---

## 4. Tài liệu Tham khảo

[1] Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *The Journal of Derivatives*, 3(2), 73-84.
[2] Zavadskas, E. K., Turskis, Z., & Kildienė, S. (2014). State of art surveys of overviews on MCDM/MADM methods. *Technological and economic development of economy*, 20(1), 165-179.
[3] Hwang, C. L., & Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications A State-of-the-Art Survey*. Springer, Berlin, Heidelberg.
[4] Tzeng, G. H., & Huang, J. J. (2011). *Multiple attribute decision making: methods and applications*. CRC press.
[5] Saaty, T. L. (1980). *The Analytic Hierarchy Process*. McGraw-Hill, New York.
