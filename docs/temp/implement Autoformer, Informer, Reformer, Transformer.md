# Tài liệu Phân tích và Triển khai các mô hình Transformer cho Dữ liệu Chuỗi thời gian ngắn

Tài liệu này trình bày các vấn đề khi áp dụng các mô hình học sâu thuộc họ Transformer (Autoformer, Informer, Reformer, Vanilla Transformer) vào dự báo chuỗi thời gian tài chính ngắn (cụ thể: Lookback window $L=60$). Đồng thời, tài liệu đề xuất phương pháp tinh chỉnh (miniaturization) để giữ lại đặc trưng gốc của các thuật toán này mà không gây ra hiện tượng học vẹt (overfitting).

---

## 1. Lý do tại sao không dùng các mô hình gốc (Vanilla Models)

Các mô hình họ Transformer nguyên bản (như Informer, Autoformer) được thiết kế đặc biệt cho bài toán **LSTF (Long Sequence Time-Series Forecasting - Dự báo chuỗi thời gian siêu dài)**.

*   **Kiến trúc cồng kềnh:** Các mô hình gốc thường có kích thước rất lớn với hàng triệu tham số (ví dụ: `d_model = 512`, từ 3 đến 6 lớp Encoder/Decoder, 8 khối Multi-head Attention).
*   **Vấn đề Overfitting trên Dữ liệu ngắn:** Khi áp dụng một mạng lưới khổng lồ vào một chuỗi dữ liệu ngắn (như 60 ngày giao dịch), mô hình sẽ nhanh chóng "học vẹt" (memorize) toàn bộ các nhiễu ngẫu nhiên của thị trường thay vì học được quy luật chung. Điều này dẫn đến sai số rất thấp trên tập huấn luyện (Train) nhưng kết quả cực kỳ tệ trên tập kiểm thử (Test).
*   **Được chứng minh qua nghiên cứu:** Bài báo nổi tiếng *"Are Transformers Effective for Time Series Forecasting?"* (Zeng et al., 2023 - nhóm tác giả DLinear) đã chứng minh rằng trên các chuỗi dữ liệu ngắn hoặc ít mẫu, các Transformer lớn bị suy giảm hiệu suất nghiêm trọng và thường bị đánh bại bởi các mô hình Linear rất đơn giản. Do đó, việc bê nguyên xi cấu hình gốc vào bài toán $L=60$ là không khả thi và sai về mặt khoa học.

---

## 2. Vấn đề Dữ liệu ngắn trong Bài toán Của Chúng Ta

*   **Đặc thù bài toán:** Dự báo biến động thị trường (Volatility) với Lookback window $L=60$ ngày và Horizon $h \in \{1, 3, 5, 10, 21\}$.
*   **Kích thước nhỏ:** Một chuỗi 60 ngày tạo ra một ma trận dữ liệu rất nhỏ. Các kỹ thuật phức tạp vốn sinh ra để giảm thiểu độ phức tạp tính toán $O(L^2)$ của ma trận hàng nghìn chiều (như ProbSparse của Informer hay LSH của Reformer) bỗng trở nên không cần thiết, thậm chí gây cản trở vì chúng có thể vô tình lược bỏ những thông tin quan trọng của chuỗi vốn đã quá ngắn.
*   **Nhiễu thị trường:** Dữ liệu tài chính nổi tiếng là non-stationary (không dừng) và chứa nhiều nhiễu. Dữ liệu càng ngắn, tỷ lệ nhiễu so với tín hiệu (signal-to-noise ratio) càng cao, càng đòi hỏi mô hình phải có tính "kháng nhiễu" tốt thông qua các biện pháp Regularization mạnh.

---

## 3. Phương pháp Triển khai theo hướng Tinh gọn (Miniaturization)

Để giải quyết mâu thuẫn giữa việc "muốn chạy mô hình gốc để so sánh" và "chống overfitting cho dữ liệu ngắn", phương pháp tối ưu là **thu nhỏ siêu tham số (hyperparameters)** nhưng **giữ nguyên hoàn toàn logic thuật toán đặc trưng** của từng mạng.

Dưới đây là chi tiết cho từng mô hình:

### 3.1. Vanilla Transformer
*   **Đặc trưng gốc:** Sử dụng cơ chế Multi-head Self-Attention toàn cục (Full Attention) giúp tính toán mối tương quan giữa mọi điểm thời gian với nhau.
*   **Phương pháp tinh gọn:** 
    *   Với $L=60$, độ phức tạp tính toán của ma trận $60 \times 60$ là cực kỳ nhỏ. Do đó, **giữ nguyên hoàn toàn hàm Full Attention**.
    *   **Tinh gọn ở kích thước mạng:** Thu nhỏ chiều dữ liệu ẩn `d_model` (xuống mức 16 hoặc 32), giảm số lượng `encoder layers` (xuống 1-2 lớp), và tăng tỷ lệ Dropout (ví dụ: 0.3) để ép mô hình tập trung vào xu hướng chính.
*   **Nguồn tham khảo:** 
    *   *Tác giả:* Qingsong Wen, Tian Zhou, Chaoli Zhang, Weiqi Chen, Ziqing Ma, Junchi Yan, Liang Sun.
    *   *Tên tài liệu:* "Transformers in Time Series: A Survey" (arXiv).
    *   *Thời gian:* 2022.

### 3.2. Autoformer
*   **Đặc trưng gốc:** 
    1. **Series Decomposition:** Tách chuỗi gốc thành phần Xu hướng (Trend) và phần Chu kỳ (Seasonal).
    2. **Auto-Correlation Mechanism:** Dùng biến đổi Fourier (FFT) để tìm độ trễ có tính tự tương quan cao nhất thay vì dùng Self-Attention.
*   **Phương pháp tinh gọn:** 
    *   Khối **Decomposition** đặc biệt hiệu quả trong tài chính để loại bỏ nhiễu, do đó bắt buộc phải được **giữ nguyên**.
    *   Trong khối **Auto-Correlation**, việc tìm chu kỳ dài trên chuỗi 60 ngày sẽ gặp khó khăn. Cách tinh gọn là giữ nguyên biến đổi FFT nhưng chỉ chọn số lượng `top_k` độ trễ (delays) cực kỳ nhỏ (ví dụ $k=1$ hoặc $2$). Kết hợp thu nhỏ `d_model` tương tự Transformer.
*   **Nguồn tham khảo:** 
    *   *Tác giả gốc:* Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long.
    *   *Tên tài liệu:* "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting" (NeurIPS).
    *   *Thời gian:* 2021.
    *   *(Các biến thể mở rộng như "MD-Autoformer" (2024) cũng ủng hộ việc dùng tích chập kết hợp để hỗ trợ dữ liệu nhỏ).*

### 3.3. Informer
*   **Đặc trưng gốc:** 
    1. **ProbSparse Attention:** Kỹ thuật lấy mẫu thưa thớt dựa trên xác suất để giảm tính toán cho ma trận khổng lồ.
    2. **Self-Attention Distilling:** Khối Max-pooling để giảm một nửa chiều dài chuỗi sau mỗi layer.
*   **Phương pháp tinh gọn:** 
    *   Việc lấy mẫu ProbSparse trên chuỗi 60 ngày là không cần thiết, tuy nhiên để giữ tính "Informer", ta vẫn chạy hàm này nhưng **hạ hệ số lấy mẫu** (sampling factor $c$) xuống, hoặc cấu hình sao cho nó hoạt động gần giống Full Attention.
    *   Khối **Distilling** phải được **giữ nguyên**, vì đối với chuỗi ngắn, thao tác nén chiều này tình cờ đóng vai trò là một kỹ thuật Regularization xuất sắc giúp chống overfitting.
*   **Nguồn tham khảo:** 
    *   *Tác giả gốc:* Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang.
    *   *Tên tài liệu:* "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (AAAI).
    *   *Thời gian:* 2021.

### 3.4. Reformer
*   **Đặc trưng gốc:** Sử dụng **Locality-Sensitive Hashing (LSH) Attention**. Hàm băm LSH giúp gom các điểm dữ liệu giống nhau vào chung một "bucket" (xô) để chỉ tính Attention trong từng xô, tiết kiệm RAM.
*   **Phương pháp tinh gọn:** 
    *   Hàm băm tạo ra tính ngẫu nhiên. Với chuỗi 60 ngày, băm dữ liệu có thể chia cắt các tín hiệu liền kề. 
    *   Tuy nhiên, để bảo tồn kiến trúc Reformer cho mục đích so sánh, chúng ta vẫn **giữ nguyên thuật toán LSH**, nhưng thiết lập tham số tinh gọn: Số lượng `buckets` rất nhỏ (ví dụ: 2 hoặc 4) và số lần băm `n_hashes = 1`. Việc này ép mô hình phân cụm tín hiệu tài chính vào một số ít nhóm đại diện thay vì băm nát chuỗi.
*   **Nguồn tham khảo:** 
    *   *Tác giả gốc:* Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.
    *   *Tên tài liệu:* "Reformer: The Efficient Transformer" (ICLR).
    *   *Thời gian:* 2020.

---

**Kết luận:** Phương pháp tốt nhất cho thực nghiệm so sánh với dataset của đồ án là triển khai các mã nguồn thuật toán gốc (Vanilla Source Code), áp dụng chuẩn hóa **RevIN** cho dữ liệu tài chính, và cấu hình các mạng này ở phiên bản "thu nhỏ" (Miniaturized) cực nhẹ. Việc này giúp việc chứng minh kết quả là khách quan và khoa học nhất.
