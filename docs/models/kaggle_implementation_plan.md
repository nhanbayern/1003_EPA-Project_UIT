# Kế hoạch Triển khai (Implementation Plan) - Kaggle Notebook
**Mục tiêu:** Xây dựng một tệp Jupyter Notebook (`.ipynb`) độc lập, hoàn chỉnh để chạy trên nền tảng Kaggle. Notebook này sẽ cài đặt và so sánh 4 mô hình Transformer (Vanilla, Autoformer, Informer, Reformer) theo hướng "tinh gọn" (miniaturized) nhằm dự báo độ biến động tài chính (Volatility) với cửa sổ $L=60$.

---

## 1. Môi trường và Dữ liệu (Kaggle Environment & Data)
- **Đường dẫn dữ liệu trên Kaggle:** `/kaggle/input/datasets/trnhngv/historical-price`
- **Đầu vào (Input):** Các file CSV của 9 chỉ số chứng khoán (VN30, DAX_40, ...).
- **Các thư viện cần thiết:** `torch`, `numpy`, `pandas`, `scikit-learn` (Không dùng các thư viện có sẵn mã nguồn Transformer phức tạp, sẽ code PyTorch từ đầu để đảm bảo tính tinh gọn và kiểm soát được cấu trúc mạng).

## 2. Quy trình Xử lý Dữ liệu (Data Preprocessing)
Tạo class `Dataset` trong PyTorch thực hiện:
1. Đọc dữ liệu `close`, tính tỷ suất sinh lợi ngày: $r_t = \ln(P_t / P_{t-1}) \times 100$.
2. Tính toán Target Volatility $\sigma_{t,h}$ theo cửa sổ 60 ngày như định nghĩa trong `docs/data.md`.
3. Cắt cửa sổ trượt (Sliding Windows):
   - **X (Input):** $r_{t-60}$ đến $r_{t-1}$ (shape: `[batch_size, 60, 1]`).
   - **Y (Target):** $\sigma_{t,h}$ với $h \in \{1, 3, 5, 10, 21\}$ (shape: `[batch_size, 5]`).
4. **Chia tập (Split):** Thực hiện chia Train/Val/Test theo tỷ lệ (dựa theo data.md).

## 3. Kiến trúc Mô hình Tinh gọn (Miniaturized Architectures)
Tất cả các mô hình sẽ kế thừa một bộ khung chuẩn và chỉ khác biệt ở module Attention.
### 3.1. Các thành phần chung
- **RevIN (Reversible Instance Normalization):** Chuẩn hóa đầu vào $X$ trước khi vào mạng và dùng tham số đó để giải chuẩn hóa ở đầu ra. Giúp mạng học tốt dữ liệu phi quy chuẩn.
- **Token Embedding + Positional Encoding:** Nhúng 60 ngày giá trị vào không gian `d_model = 32`.
- **Output Projection:** Lớp Linear mapping từ không gian ẩn ra 5 giá trị dự báo cho 5 horizon $h$.

### 3.2. Cấu hình cụ thể cho từng mạng
*Giảm kích thước đáng kể so với bản gốc để tránh overfitting.*
- **Vanilla Transformer:**
  - `d_model = 32`, `n_heads = 4`, `e_layers = 2`, `d_ff = 128`.
  - Giữ nguyên Full Multi-Head Attention.
- **Autoformer:**
  - Giữ khối **Series Decomposition** (trích xuất trend và seasonal).
  - **Auto-Correlation:** Chỉ tính `top_k = 1` hoặc `2` delay để tìm chu kỳ ngắn.
  - Kích thước mạng: `d_model = 32`, `e_layers = 2`.
- **Informer:**
  - **ProbSparse Attention:** Set `factor = 1` (hệ số lấy mẫu rất thấp).
  - **Distilling:** Giữ nguyên các lớp Max-pooling giữa các layer encoder để nén chiều dài chuỗi từ 60 -> 30 -> 15.
- **Reformer:**
  - **LSH Attention:** Set `n_hashes = 1`, `bucket_size = 4` hoặc `8` để tránh băm nát thông tin.

## 4. Quá trình Huấn luyện (Training & Evaluation)
- **Loss Function (Student-t NLL với $\nu$ cố định):** 
  Sử dụng hàm **Student-t Negative Log-Likelihood (NLL)** làm hàm loss chính, đảm bảo tính nhất quán khoa học với `docs/paper.md`.
  - Bậc tự do $\nu$ sẽ **không dự báo động** mà được **tính toán tĩnh trước** từ hệ số nhọn dư (excess kurtosis $k$) của tập **Train** riêng biệt cho từng chỉ số theo công thức $\nu = 4 + 6/k$.
  - Các mô hình chỉ cần dự báo 1 đầu ra duy nhất là Volatility $\hat{\sigma}_t$. Hàm Loss sẽ sử dụng giá trị $\nu$ cố định này để tính toán NLL, đảm bảo quá trình training diễn ra ổn định và tránh lỗi nổ gradient (NaN).
- **Optimizer:** AdamW với learning rate $1e-3$ hoặc $5e-4$, kết hợp Cosine Annealing LR scheduler.
- **Early Stopping:** Dừng nếu validation loss không giảm sau 10 epochs.
- **Evaluation Metrics:** MSE, MAE, và QLIKE (đặc thù cho volatility).

## 5. Tổ chức file Notebook (`.ipynb`)
Cấu trúc file `kaggle_notebook.ipynb` sẽ bao gồm các cell sau:
- **Cell 1:** Import thư viện.
- **Cell 2:** Các hàm tính toán tài chính (log returns, target volatility).
- **Cell 3:** PyTorch Dataset & Dataloaders (Tích hợp logic sliding window).
- **Cell 4:** Modules dùng chung (RevIN, Embedding, TimeFeature).
- **Cell 5:** Source code Transformer gốc thu nhỏ.
- **Cell 6:** Source code Autoformer gốc thu nhỏ.
- **Cell 7:** Source code Informer gốc thu nhỏ.
- **Cell 8:** Source code Reformer gốc thu nhỏ.
- **Cell 9:** Vòng lặp Training (Train Loop) & Testing.
- **Cell 10:** Hàm main để chạy lần lượt 4 mô hình và in ra bảng so sánh kết quả.

## 6. Tổ chức Thư mục và Định dạng Đầu ra (Outputs)
- **Tạo thư mục tự động:** Notebook sẽ chứa code để tự động tạo cấu trúc thư mục lưu kết quả trên Kaggle (ví dụ: `/kaggle/working/results/all_predictions`).
- **Định dạng tệp dự báo (Predictions):** Tương tự cấu trúc file `DAX_40_moirai_moe_predictions.csv`, mỗi mô hình trên từng chỉ số sẽ xuất ra một file `.csv` gồm các cột:
  - `time, log_return, horizon, true_volatility, predict_volatility`
- **Định dạng tệp so sánh (Metrics):** Tương tự file `comparison_metrics_tidy.csv`, bảng tổng hợp lỗi của cả 4 mô hình sẽ được lưu chung vào một file với cấu trúc cột:
  - `index, metric, horizon, Vanilla, Autoformer, Informer, Reformer`

---

### YÊU CẦU XÁC NHẬN (USER REVIEW REQUIRED)
1. **Hàm Loss:** Đã thống nhất sử dụng **Student-t NLL Loss với hằng số $\nu$** tính tĩnh trước từ tập Train của mỗi chỉ số.
2. **Framework triển khai:** Việc tự viết lại code PyTorch từ đầu trong 1 file notebook sẽ khá dài (khoảng 1000 dòng code). Bạn có đồng ý với hướng tiếp cận gộp tất cả vào 1 file `.ipynb` để chạy trực tiếp trên Kaggle không?

Nếu bạn đồng ý với kế hoạch trên, tôi sẽ bắt đầu viết code cho file `.ipynb` này.
