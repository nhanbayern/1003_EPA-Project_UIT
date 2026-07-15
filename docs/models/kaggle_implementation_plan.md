# Kế hoạch Triển khai (Implementation Plan) - Kaggle Notebook
**Mục tiêu:** Xây dựng một tệp Jupyter Notebook (`.ipynb`) độc lập, hoàn chỉnh để chạy trên nền tảng Kaggle. Notebook này sẽ cài đặt và so sánh 4 mô hình Transformer (Vanilla, Autoformer, Informer, Reformer) theo hướng "tinh gọn" (miniaturized) nhằm dự báo độ biến động tài chính (Volatility) với cửa sổ $L=60$.

---

## 1. Môi trường và Dữ liệu (Kaggle Environment & Data)
- **Đường dẫn dữ liệu trên Kaggle:** `/kaggle/input/datasets/trnhngv/historical-price`
- **Đầu vào (Input):** Các file CSV của 9 chỉ số chứng khoán (VN30, DAX_40, ...).
- **Phương thức đồng bộ (Git-Sync):**
  - Repository URL (Public): `https://github.com/nhanbayern/1003_EPA-Project_UIT.git`
  - Nhánh làm việc: `kaggle-implementation`
  - Thư mục code cục bộ: `transformer based` (đường dẫn tuyệt đối: `D:\UIT\1003_EPA_PROJECT\1.0.0\1003_EPA-Project_UIT\transformer based`)
  - Lệnh clone trên Kaggle: `!git clone -b kaggle-implementation --single-branch https://github.com/nhanbayern/1003_EPA-Project_UIT.git`
- **Các thư viện cần thiết:** `torch`, `numpy`, `pandas`, `scikit-learn` (Code PyTorch thuần từ đầu để đảm bảo tính tinh gọn và kiểm soát được cấu trúc mạng).

## 2. Quy trình Xử lý Dữ liệu (Data Preprocessing)
Tạo class `Dataset` trong PyTorch thực hiện:
1. Đọc dữ liệu `close`, tính tỷ suất sinh lợi ngày: $r_t = \ln(P_t / P_{t-1}) \times 100$.
2. Tính toán Target Volatility $\sigma_{t,h}$ theo cửa sổ 60 ngày như định nghĩa trong `docs/data.md`.
3. Cắt cửa sổ trượt (Sliding Windows):
   - **X (Input):** $r_{t-60}$ đến $r_{t-1}$ (shape: `[batch_size, 60, 1]`).
   - **Y (Target):** $\sigma_{t,h}$ cho toàn bộ $H=21$ (shape: `[batch_size, 21]`). Khi đánh giá sẽ trích xuất tại các mốc $h \in \{1, 3, 5, 10, 21\}$ theo đúng `docs/problem.md`.
4. **Chia tập (Split):** Thực hiện chia Train/Val/Test theo tỷ lệ (dựa theo data.md).

## 3. Kiến trúc Mô hình Tinh gọn (Miniaturized Architectures)
Tất cả các mô hình sẽ kế thừa một bộ khung chuẩn và chỉ khác biệt ở module Attention.
### 3.1. Các thành phần chung
- **PackedStdScaler:** Bộ chuẩn hóa động đầu vào $X$ trước khi đưa vào mạng (sử dụng thay vì RevIN để bám sát hoàn toàn Mục V trong `docs/problem.md`).
- **Token Embedding + Positional Encoding:** Nhúng 60 ngày giá trị vào không gian `d_model = 32`.
- **Output Projection:** Lớp Linear mapping từ không gian ẩn ra 21 giá trị dự báo tương ứng với $H=21$.

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

## 5. Tổ chức Thư mục Code Cục bộ (`transformer based`)
Chúng ta sẽ viết code cục bộ trong thư mục `transformer based` và đẩy lên nhánh `kaggle-implementation` để Kaggle clone về chạy. Cấu trúc thư mục gồm:
1. `models.py`: Định nghĩa 4 mô hình Transformer tinh gọn (Vanilla, Autoformer, Informer, Reformer) kèm theo bộ chuẩn hóa `PackedStdScaler`.
2. `dataset.py`: Định nghĩa lớp `VolatilityDataset` chịu trách nhiệm tính toán log return, tính toán target $\sigma_{t,h}$ và cắt cửa sổ trượt $L=60$.
3. `utils.py`: Hàm ước lượng bậc tự do $\nu$ tĩnh từ tập Train cho từng chỉ số, tính toán metrics (MSE, MAE, QLIKE), và các hàm tiện ích vẽ biểu đồ.
4. `kaggle_notebook.ipynb`: Notebook chính để chạy trên Kaggle. Notebook này sẽ clone code từ Git, nạp dữ liệu, thực thi vòng lặp training/testing trên cả 9 chỉ số và lưu trữ kết quả.

## 6. Tổ chức Thư mục và Định dạng Đầu ra (Outputs)
- **Tạo thư mục tự động:** Notebook sẽ chứa code để tự động tạo cấu trúc thư mục lưu kết quả trên Kaggle (ví dụ: `/kaggle/working/results/all_predictions`).
- **Định dạng tệp dự báo (Predictions):** Tương tự cấu trúc file `DAX_40_moirai_moe_predictions.csv`, mỗi mô hình trên từng chỉ số sẽ xuất ra một file `.csv` gồm các cột:
  - `time, log_return, horizon, true_volatility, predict_volatility`
- **Định dạng tệp so sánh (Metrics):** Tương tự file `comparison_metrics_tidy.csv`, bảng tổng hợp lỗi của cả 4 mô hình sẽ được lưu chung vào một file với cấu trúc cột:
  - `index, metric, horizon, Vanilla, Autoformer, Informer, Reformer`

---

### YÊU CẦU XÁC NHẬN (USER REVIEW REQUIRED)
1. **Hàm Loss:** Đã thống nhất sử dụng **Student-t NLL Loss với hằng số $\nu$** tính tĩnh trước từ tập Train của mỗi chỉ số.
2. **Framework triển khai:** Đã thống nhất sử dụng **Git-Sync** (nhánh `kaggle-implementation` của repo `nhanbayern/1003_EPA-Project_UIT`). Code được tổ chức dạng module hóa tại thư mục cục bộ `transformer based` để dễ phát triển và bảo trì.

Kế hoạch đã sẵn sàng, chúng ta có thể chuyển sang giai đoạn tạo các file code nháp cục bộ.
