# Báo cáo Cập nhật Mô hình Moirai VaR-aware (Ngày 21/07/2026)

Tài liệu này tổng hợp các phân tích, quyết định và thay đổi cấu hình kỹ thuật đã được thực hiện đối với thực nghiệm `moirai_var_aware` nhằm mục đích quản trị rủi ro và dự báo Value at Risk (VaR).

## 1. Phân tích Lý thuyết & Siêu tham số (Hyperparameters)

Trong quá trình review cấu hình mô hình `moirai_var_aware`, các siêu tham số quan trọng đã được mổ xẻ để làm rõ ý nghĩa và định hướng cho Hyperparameter Tuning:

### 1.1. Tham số Hàm Loss: `lambda_var`
- **Vai trò:** Kiểm soát sự đánh đổi giữa hàm mất mát của độ biến động (Volatility Loss - Student-t NLL) và hàm mất mát của VaR (VaR Quantile Loss / Pinball Loss).
- **Ý nghĩa khi điều chỉnh:** Việc tăng `lambda_var` (ví dụ từ 0.2 lên 0.5 hoặc 1.0) sẽ ép mô hình tập trung nhiều hơn vào việc tối ưu phần "đuôi" của phân phối (tail risk). Điều này giúp cải thiện kết quả backtest của VaR (giảm số lần vi phạm thực tế so với ngưỡng $\alpha = 0.01$), nhưng có thể đánh đổi bằng sự sụt giảm độ chính xác của dự báo volatility tổng thể.

### 1.2. Kỹ thuật Tối ưu (Optimization)
- **Differential Learning Rates:** Mô hình hỗ trợ hai chế độ `tuning_mode`. Ở chế độ `full`, chúng ta áp dụng tốc độ học (LR) khác biệt: `1e-3` cho phần Head (mới khởi tạo ngẫu nhiên) và `1e-5` cho phần Backbone (đã pre-train) để tránh hiện tượng *Catastrophic Forgetting* (Quên thảm khốc).
- **Batch Size:** Sử dụng `32` cho tập Train (nhằm tiết kiệm GPU VRAM khi phải tính toán đạo hàm) và tăng gấp đôi lên `64` cho Validation/Test (do không cần tính toán đạo hàm `torch.no_grad()`, giúp tăng tốc độ đánh giá).
- **Max Gradient Norm (Clip Grad = 1.0):** Cơ chế "dây an toàn" giúp cắt tỉa các vector đạo hàm quá lớn, ngăn chặn hiện tượng Bùng nổ đạo hàm (Exploding Gradients) khi mô hình tiếp xúc với các cú sốc bất thường trong chuỗi dữ liệu tài chính.

---

## 2. Các thay đổi về Code và Cấu trúc

### 2.1. Tính toán Bậc tự do Động (`dynamic_nu`)
Đã áp dụng logic tính toán bậc tự do (degrees of freedom - `nu`) của phân phối Student-t một cách tự động (tương tự như mô hình `transformer_based`), thay vì hardcode cố định `nu = 4.0`.

- **Cơ chế:** Trước khi train trên một Dataset bất kỳ, hệ thống sẽ trích xuất chuỗi tỷ suất sinh lợi (`log_return`) của tập Train và tính toán Độ nhọn (Kurtosis) thông qua `scipy.stats.kurtosis`.
- **Công thức:** 
  - Nếu $Kurtosis \le 0$ (không phải phân phối đuôi dày): $nu = 30.0$ (tiệm cận phân phối chuẩn).
  - Nếu $Kurtosis > 0$: $nu = 4.0 + \frac{6.0}{Kurtosis}$, sau đó được ghim (clip) trong khoảng an toàn `[2.1, 30.0]`.
- **Tác động:** Giúp mô hình bắt nhịp chính xác hơn với đặc tính phân phối sinh lời (đuôi dày mỏng khác nhau) của từng loại tài sản/chỉ số chứng khoán riêng biệt (ví dụ: VN-Index sẽ có `nu` khác S&P 500).
- **Files thay đổi:** `experiments/moirai_var_aware/runner.py`.

### 2.2. Tự động hóa Pipeline trên Kaggle
Đã nâng cấp Notebook `moirai_kaggle.ipynb` thành một hệ thống pipeline hoàn chỉnh để hỗ trợ thí nghiệm siêu tham số.

- **Vòng lặp đa cấu hình:** Chuyển tham số cấu hình tĩnh `LAMBDA_VAR = 0.2` thành vòng lặp quét qua danh sách `LAMBDA_VARS = [0.3, 0.4, 0.5]`. Mô hình sẽ liên tục huấn luyện qua 3 phiên bản cấu hình này.
- **Tích hợp `dynamic_nu`:** Thuật toán tính toán `nu` dựa trên Kurtosis cũng đã được cấy vào pipeline huấn luyện trên Kaggle.
- **Quản lý Output:** Tất cả các tệp dự báo (CSV) sinh ra từ nhiều cấu hình Lambda khác nhau hiện sẽ được đóng gói gọn gàng vào một tệp nén duy nhất: `moirai_var_multi_lambda_predictions.zip` để thuận tiện cho việc tải về máy local và đánh giá.
- **Files thay đổi:** `model/moirai_var_aware/moirai_kaggle.ipynb`.
