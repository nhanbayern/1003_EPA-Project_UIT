# Báo cáo Đánh giá Mô hình Dự báo Biến động bằng Phương pháp Ra quyết định Đa tiêu chí (MCDM)

**Ngày báo cáo:** 25/07/2026

## 1. Mở đầu
Trong lĩnh vực dự báo tài chính và quản trị rủi ro, việc đánh giá mô hình luôn đối mặt với sự đánh đổi (trade-off) giữa hai thái cực:
1. **Độ chính xác thống kê (Accuracy):** Tối ưu hóa sai số trung bình (MSE, MAE, QLIKE).
2. **Bảo vệ rủi ro đuôi (Tail Risk Coverage):** Tối ưu hóa khả năng dự đoán các sự kiện cực đoan bằng Value-at-Risk (VaR 99%) qua bài test Kupiec POF [1].

Các mô hình Foundation/Deep Learning (như `Moirai`) thường có sai số thấp nhưng đánh giá rủi ro kém, trong khi các mô hình kinh tế lượng (GARCH) giữ rủi ro đuôi rất tốt nhưng sai số thống kê lại lớn. Đứng trước sự mâu thuẫn này, tài chính định lượng hiện đại cần sử dụng **MCDM (Multi-Criteria Decision Making - Ra quyết định Đa tiêu chí)** để tổng hợp các tiêu chí xung đột thành một điểm số duy nhất, giúp xếp hạng mô hình một cách toàn diện [2].

## 2. Phương pháp luận (Các kỹ thuật MCDM được áp dụng)
Dự án áp dụng ba kỹ thuật MCDM chính:

### 2.1. Simple Additive Weighting (SAW) - Điểm tổng hợp trung bình
Đây là phương pháp MCDM cổ điển, chuẩn hóa tuyến tính mọi dữ liệu (Min-Max Scaling) về thang điểm 0-100 và lấy trung bình cộng (hoặc tổng có trọng số).
- **Đặc điểm nổi bật:** Có tính chất bù trừ hoàn toàn. Một tiêu chí đạt điểm tuyệt đối có thể dễ dàng kéo một tiêu chí kém lên mức trung bình.

### 2.2. TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
Thay vì cộng điểm, TOPSIS tìm kiếm giải pháp dựa trên vị trí hình học không gian đa chiều bằng cách chuẩn hóa vector. Thuật toán đo lường khoảng cách Euclid từ mỗi mô hình tới Điểm Lý tưởng ($A^+$) và Điểm Tồi tệ ($A^-$) [3].
- **Đặc điểm nổi bật:** Có tính phi tuyến tính, trừng phạt sự lệch pha. Mô hình quá kém ở một tiêu chí sẽ bị đẩy ra rất xa Điểm Lý Tưởng, khiến điểm tổng sụt giảm nặng nề.

### 2.3. AHP (Analytic Hierarchy Process) - Phân tích Thứ bậc
Phương pháp sử dụng ma trận so sánh từng cặp để xác định trọng số (weighting) cho các tiêu chí dựa trên đánh giá chuyên gia. AHP thường được kết hợp tạo thành Hybrid MCDM (như AHP-TOPSIS) [5].

## 3. Kết quả Đánh giá
Dựa trên kết quả chạy mô hình tại thư mục `output/mcdm_results`, bảng xếp hạng bằng hai phương pháp SAW và TOPSIS được đối chiếu như sau:

### 3.1. Phân tích dựa trên SAW (Bù trừ mạnh)

**Bảng trích xuất kết quả xếp hạng SAW (Top 5, Bottom 4 & nhóm mô hình Foundation):**

| Hạng | Mô hình | Điểm Chính Xác | Điểm Rủi Ro 1% | Điểm Rủi Ro 5% | Điểm Tổng SAW |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | HybridGARCHAutoformer | 0.2068 | 0.3433 | 0.2704 | **0.8206** |
| 2 | HybridGARCHAutoformer | 0.2068 | 0.3433 | 0.2640 | 0.8142 |
| 3 | HybridGARCHAutoformer | 0.1950 | 0.3433 | 0.2704 | 0.8087 |
| 4 | HybridGARCHAutoformer | 0.1950 | 0.3433 | 0.2640 | 0.8024 |
| 5 | Autoformer | 0.1676 | 0.3119 | 0.3154 | 0.7950 |
| ... | ... | ... | ... | ... | ... |
| 26 | moirai_moe | 0.2977 | 0.2386 | 0.1952 | 0.7315 |
| 80 | moirai | 0.2842 | 0.1871 | 0.2143 | 0.6856 |
| ... | ... | ... | ... | ... | ... |
| 149 | GARCH-LSTM-Hybrid | 0.2372 | 0.1059 | 0.0695 | 0.4127 |
| 150 | FI-GARCH | 0.2378 | 0.0000 | 0.0000 | 0.2378 |
| 151 | GARCH | 0.2214 | 0.0000 | 0.0000 | 0.2214 |
| 152 | GJR-GARCH | 0.2038 | 0.0000 | 0.0000 | 0.2038 |

- **Top 1 Bảng Xếp Hạng:** Mô hình `HybridGARCHAutoformer` đạt điểm tuyệt đối cao nhất (0.8206), thống trị các vị trí dẫn đầu, cho thấy khả năng cân bằng tốt giữa sai số thấp và dự báo rủi ro tốt.
- **Biểu hiện của các mô hình Foundation:** Các mô hình như `moirai` và `moirai_moe` đạt điểm từ khoảng 0.68 đến 0.73, nằm ở khu vực giữa bảng xếp hạng. Do SAW có khả năng bù trừ mạnh, điểm số Accuracy tuyệt vời của Moirai đã gánh vác phần nào kết quả dự báo rủi ro rất thấp của nó.
- **Cuối bảng:** Các mô hình thuần truyền thống (`GARCH-LSTM-Hybrid`, `FI-GARCH`, `GARCH`, `GJR-GARCH`) chót bảng do sai số quá lớn, điểm số chỉ đạt từ 0.20 đến 0.41.

### 3.2. Phân tích dựa trên TOPSIS (Trừng phạt lệch pha)

**Bảng trích xuất kết quả xếp hạng TOPSIS (Top 5, Bottom 4 & nhóm mô hình Foundation):**

| Hạng | Mô hình | K/c tới lý tưởng (D+) | K/c tới tồi tệ (D-) | Điểm Tổng TOPSIS |
| :--- | :--- | :--- | :--- | :--- |
| 1 | HybridGARCHAutoformer | 0.0127 | 0.0571 | **0.8174** |
| 2 | HybridGARCHAutoformer | 0.0128 | 0.0563 | 0.8148 |
| 3 | HybridGARCHAutoformer | 0.0133 | 0.0568 | 0.8099 |
| 4 | HybridGARCHAutoformer | 0.0133 | 0.0560 | 0.8072 |
| 5 | WaveletAutoformer | 0.0134 | 0.0524 | 0.7961 |
| ... | ... | ... | ... | ... |
| 74 | moirai_moe | 0.0198 | 0.0500 | 0.7157 |
| 103 | moirai | 0.0220 | 0.0484 | 0.6869 |
| ... | ... | ... | ... | ... |
| 149 | GARCH-LSTM-Hybrid | 0.0391 | 0.0344 | 0.4681 |
| 150 | FI-GARCH | 0.0544 | 0.0289 | 0.3473 |
| 151 | GARCH | 0.0546 | 0.0276 | 0.3363 |
| 152 | GJR-GARCH | 0.0548 | 0.0259 | 0.3211 |

- **Top 1 Bảng Xếp Hạng:** Mô hình `HybridGARCHAutoformer` vẫn tiếp tục duy trì thành công vị trí dẫn đầu (Top 1 với điểm số 0.8174).
- **Sự trừng phạt với các mô hình Foundation:** Với TOPSIS, tính chất bù trừ bị phá vỡ. Do Moirai có khoảng cách rất xa so với "mức rủi ro lý tưởng", khuyết điểm này bị TOPSIS trừng phạt cực mạnh làm nén lại hệ số Closeness Coefficient, khiến Moirai bị tụt lại phía sau nhiều biến thể của `Reformer` hay `Autoformer` (Ví dụ: `moirai` tụt từ hạng 80 bên SAW xuống hạng 103 bên TOPSIS).
- **Cuối bảng:** Tương tự như kết quả SAW, các mô hình GARCH truyền thống vẫn nằm ở hạng thấp nhất với điểm khoảng 0.32 đến 0.34.

## 4. Kết luận
Dù sử dụng phương pháp luận mang tính bù trừ (SAW) hay trừng phạt (TOPSIS), **HybridGARCHAutoformer** đều bảo vệ thành công vị trí Top 1. Thiết kế lai của mô hình này (GARCH quản trị rủi ro, Autoformer tối ưu hóa sai số) tạo nên sự toàn diện, không có điểm yếu để bị TOPSIS trừng phạt, và cũng sở hữu mức điểm trung bình tuyến tính xuất sắc nhất. Điều này khẳng định độ vững chắc của kiến trúc Hybrid trong dự án.

## 5. Tài liệu Tham khảo
[1] Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *The Journal of Derivatives*, 3(2), 73-84.
[2] Zavadskas, E. K., Turskis, Z., & Kildienė, S. (2014). State of art surveys of overviews on MCDM/MADM methods. *Technological and economic development of economy*, 20(1), 165-179.
[3] Hwang, C. L., & Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications A State-of-the-Art Survey*. Springer, Berlin, Heidelberg.
[4] Tzeng, G. H., & Huang, J. J. (2011). *Multiple attribute decision making: methods and applications*. CRC press.
[5] Saaty, T. L. (1980). *The Analytic Hierarchy Process*. McGraw-Hill, New York.
