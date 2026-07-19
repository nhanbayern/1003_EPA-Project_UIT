# Báo cáo Chi tiết: Triển khai và Cấu hình Siêu tham số các Mô hình Transformer (Version 1)

**Đường dẫn thư mục:** `model/transformer based/version_1`
**Mục tiêu:** Báo cáo này diễn giải chi tiết cách thức mã hóa (implement), các giá trị siêu tham số (hyperparameters) hiện tại, và giải thích lý do đằng sau các quyết định cấu hình này nhằm phục vụ việc dự báo Volatility với độ dài chuỗi $L=60$.

---

## 1. Cấu trúc Chung của Mạng (Common Architecture)

Mặc dù có 4 biến thể Transformer khác nhau, tất cả đều chia sẻ một pipeline tiền xử lý và hậu xử lý đồng nhất để đảm bảo tính công bằng khi so sánh.

### 1.1. Khởi tạo và Tiền xử lý
- **Đầu vào (Input):** Chuỗi lợi suất (returns) dài 60 ngày, với số lượng đặc trưng là 1 (Shape: `[batch_size, 60, 1]`).
- **Chuẩn hóa (PackedStdScaler):** Thay vì dùng RevIN vốn có thêm tham số học (learnable parameters $\gamma, \beta$), mô hình sử dụng chuẩn hóa Z-score cục bộ trên từng chuỗi (trừ Mean, chia Std) để khử tính không dừng (non-stationary) một cách tự nhiên mà không gây tăng lượng tham số.
- **Embedding:** Thay vì sử dụng Token Embedding phức tạp của NLP, đầu vào 1 chiều được ánh xạ tuyến tính (Linear mapping) thẳng lên không gian đa chiều `d_model`.
- **Positional Encoding:** Sử dụng Sine/Cosine Positional Encoding tiêu chuẩn để truyền đạt thông tin về thứ tự thời gian cho mạng.

### 1.2. Lớp Đầu ra (VolatilityHead)
Thay vì dùng một lớp Linear đơn giản, tất cả các mô hình dùng chung cấu trúc `VolatilityHead` lấy cảm hứng từ cấu trúc phân phối của MOIRAI:
- **Luồng:** `Linear(in_features, 64) -> GELU -> Linear(64, 21) -> Softplus`
- **Mục đích:** Đầu ra là Volatility (độ biến động), bắt buộc phải là một số dương. Hàm kích hoạt `Softplus` (phiên bản mượt của ReLU) ở lớp cuối cùng đảm bảo kết quả dự báo luôn lớn hơn 0, phù hợp với bản chất toán học của Volatility.
- **Đầu ra (Output):** Dự báo đồng thời cho 21 horizons (Shape: `[batch_size, 21]`).

---

## 2. Cấu hình Siêu tham số (Hyperparameters) Cốt lõi

Các siêu tham số cơ bản được đồng bộ trên toàn bộ 4 mô hình:
*   `seq_len = 60`: Độ dài chuỗi đầu vào.
*   `pred_len = 21`: Số bước thời gian dự báo tương lai.
*   `d_model = 32`: Kích thước không gian ẩn (Hidden dimension). *Giảm mạnh từ 512 của mô hình gốc.*
*   `n_heads = 4`: Số lượng "đầu" tự chú ý (Attention heads). *Giảm từ 8.*
*   `e_layers = 2`: Số lượng lớp mã hóa (Encoder layers). *Giảm từ 6.*
*   `d_ff = 128`: Kích thước mạng Feed-Forward. *Giảm từ 2048.*
*   `dropout = 0.3`: Tỷ lệ ngắt kết nối nơ-ron ngẫu nhiên. *Tăng từ mức mặc định 0.1.*

### Giải thích Lý luận (Rationale):
1.  **Tại sao `d_model = 32` và `e_layers = 2`?** Dữ liệu tài chính 60 ngày cực kỳ ngắn và chứa nhiều nhiễu ngẫu nhiên. Việc sử dụng mạng quá lớn (như `d_model=512, layers=6` - hàng triệu tham số) sẽ khiến mô hình lập tức nội suy và ghi nhớ các điểm nhiễu (overfitting). Mức `32` và `2` lớp là đủ "độ sâu" để trích xuất đặc trưng đơn giản mà không bị thừa thãi tham số (parameter bloat).
2.  **Tại sao `dropout = 0.3`?** Tài chính là môi trường noise-heavy. Mức dropout cao (30%) hoạt động như một bộ regularization mạnh, ép mô hình không được phụ thuộc vào bất kỳ một nơ-ron cụ thể nào, từ đó nâng cao tính tổng quát hóa (generalization) trên tập Test.

---

## 3. Triển khai Chi tiết Từng Mô hình

### 3.1. Vanilla Transformer
- **Triển khai:** Sử dụng trực tiếp `nn.TransformerEncoderLayer` của PyTorch.
- **Cơ chế:** Full Multi-Head Self-Attention. Nó tính toán ma trận tương quan kích thước $60 \times 60$. Với $L=60$, việc này mất chưa tới vài mili-giây, nên mô hình này giữ được sự nguyên bản hoàn hảo nhất.

### 3.2. Autoformer (Miniaturized)
- **Series Decomposition:** Được giữ nguyên. Mã nguồn thực hiện dùng khối `moving_avg` (kernel_size=5) để tách chuỗi thành 2 phần: `Trend` (Xu hướng) và `Residual` (Phần dư). Lớp Attention chỉ học trên phần dư, sau đó Trend được cộng ngược trở lại qua từng lớp.
- **Auto-Correlation:** Thay vì dùng Self-Attention, cơ chế này dùng FFT (Fast Fourier Transform) để tìm các điểm trễ (delays) có tính tự tương quan cao nhất. 
- **Tinh chỉnh cho $L=60$:** Chỉ lấy độ trễ cao nhất (`c = 1` hay `top_k = 1`). Đối với chuỗi quá ngắn, việc tìm nhiều chu kỳ dài là bất hợp lý. Thiết lập `c=1` giúp mô hình tập trung bám theo chu kỳ ngắn hạn có tín hiệu mạnh nhất.

### 3.3. Informer (Miniaturized)
- **ProbSparse Attention (Mock):** Trong thực tế, hàm chọn mẫu xác suất (ProbSparse) hoạt động không hiệu quả và gây mất tín hiệu với chuỗi quá ngắn như 60. Việc cài đặt C++ cho ProbSparse cũng làm mã nguồn cồng kềnh. Ở đây mã nguồn sử dụng khái niệm tương tự (Mock ProbSparse) tính toán ma trận Attention nhưng vẫn giữ khung cấu trúc của Informer.
- **Distilling (Max-Pooling):** Đây là đặc trưng quan trọng nhất của Informer được giữ lại nguyên vẹn. Khối `Distilling` bao gồm `Conv1d` và `MaxPool1d(stride=2)` giúp giảm một nửa độ dài chuỗi sau mỗi lớp Encoder (60 -> 30).
- **Lý do Tinh chỉnh:** Hành động gộp (pooling) này đặc biệt hữu dụng cho dữ liệu tài chính $L=60$, đóng vai trò như một bộ lọc nhiễu tần số cao (High-frequency noise filter) và tạo ra hiệu ứng Regularization cực tốt.

### 3.4. Reformer (Miniaturized)
- **LSH Attention (Mock):** Reformer gốc sử dụng Locality-Sensitive Hashing (Băm nhạy cảm vị trí) để gom nhóm các token. Tuy nhiên, việc "băm" một chuỗi 60 ngày sẽ phá vỡ hoàn toàn tính liên tục của dữ liệu tài chính cục bộ.
- **Shared Query-Key Space:** Một đặc trưng khác của Reformer là tiết kiệm bộ nhớ bằng cách buộc Query và Key chia sẻ chung một không gian trọng số. Triển khai trong mã nguồn giả lập cơ chế này bằng phép toán `qk = (q + k) / 2` trước khi tính Attention Score.
- **Lý do Tinh chỉnh:** Đảm bảo mô hình bắt chước được tính chất chia sẻ bộ nhớ (shared-weights) của Reformer mà không cần đưa vào hàm băm phức tạp vốn sẽ phá hủy tín hiệu chuỗi thời gian ngắn.

---

## 4. Tóm tắt Tổng kết
Phiên bản 1 trong thư mục `transformer based` đã thành công trong việc **hiện thực hóa khái niệm "Miniaturized Transformers" (Transformer tinh gọn)**. 
Bằng cách thu hẹp đáng kể dung lượng mạng lưới (`d_model=32`) nhưng vẫn duy trì các đặc trưng cấu trúc lõi của Autoformer (Decomposition), Informer (Distilling Pooling), Reformer (Shared Q-K), bộ mã nguồn này đã cung cấp một môi trường so sánh (baseline) cực kỳ tối ưu, khách quan và khoa học để làm tiền đề so sánh với các mô hình Foundation (như MOIRAI) trong đồ án.
