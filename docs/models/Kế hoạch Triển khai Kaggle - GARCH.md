# Kế hoạch Triển khai (Implementation Plan) - GARCH Family on Kaggle

**Mục tiêu:** Tái cấu trúc và triển khai các mô hình hệ GARCH (GARCH, GJR-GARCH, FI-GARCH, GARCH-LSTM Hybrid) để chạy trên nền tảng Kaggle. Mục đích là tạo ra kết quả so sánh công bằng với nhóm mô hình Moirai/Transformer, đồng thời khắc phục lỗi dự báo đa bước (Multi-horizon) và bảo toàn tính chất ước lượng phân phối (Density Estimation) của GARCH.

**Thư mục làm việc:** `D:\UIT\1003_EPA_PROJECT\1.0.0\1003_EPA-Project_UIT\model\GARCH based`

---

## 1. Phương pháp luận cốt lõi (Core Methodology)

### 1.1. Mục tiêu tối ưu hóa (Objective / Loss Function)
Để duy trì ưu thế về kiểm soát rủi ro vùng đuôi (VaR), chúng ta **không** ép các mô hình GARCH học bằng MSE.
* **GARCH, GJR-GARCH, FI-GARCH:** Tối ưu hóa bằng Ước lượng Hợp lý Cực đại (Maximum Likelihood Estimation - MLE) sử dụng thư viện `arch`.
* **GARCH-LSTM Hybrid:** Huấn luyện bằng Backpropagation sử dụng hàm **Student-t Negative Log-Likelihood (TLoss)** trên chuỗi lợi suất thực tế (`target_returns`), không dùng `target_variance`.

### 1.2. Cơ chế sinh dự báo Đa bước (Multi-horizon Simulation)
Thay vì xuất ra phương sai 1 ngày như code cũ, pipeline mới sẽ triển khai đệ quy (rolling simulation) để dự báo chính xác cho các chân trời $h \in \{1, 3, 5, 10, 21\}$.
* **Với Statistical GARCH:** Gọi `forecast(horizon=21)`.
* **Với GARCH-LSTM Hybrid:** Trong quá trình suy luận (Inference), mô hình sẽ dùng phương sai dự báo của bước $t$ làm input để tiếp tục dự đoán bước $t+1$, cho đến khi đạt bước 21.

### 1.3. Quy đổi Đại lượng (Variance Aggregation)
Sau khi có mảng phương sai dự báo $\sigma_{t}^2, \sigma_{t+1}^2, \dots, \sigma_{t+h-1}^2$, chúng ta thực hiện tích lũy phương sai (Variance Additivity) để quy đổi thành "Độ biến động gộp" (Volatility) tương đương với Target của mô hình Transformer:
$$ Vol_{pred, h} = \sqrt{\frac{1}{h} \sum_{i=1}^{h} \sigma_{t+i-1}^2} $$

---

## 2. Tổ chức Thư mục Mã nguồn & Kết quả (Code & Output Structure)

### 2.1. Cấu trúc Mã nguồn (Source Code)
Tại `model/GARCH based/`, cấu trúc code sẽ được tổ chức gọn gàng để dễ dàng zip và up lên Kaggle:
*   **`dataset.py`:** Chứa class DataLoader.
*   **`losses.py`:** Chứa class `TLoss` (Student-t NLL).
*   **`models/`:** Chứa `stat_models.py` và `garch_lstm.py`.
*   **`train_evaluate.py`:** Chứa vòng lặp huấn luyện và đánh giá.
*   **`kaggle_notebook.ipynb`:** File Notebook chính chạy từ A-Z trên Kaggle.

### 2.2. Cấu trúc Thư mục Kết quả (Output Directories)
Trong quá trình chạy trên Kaggle, script sẽ tự động tạo các thư mục kết quả (ví dụ: `kaggle/working/results/`) với cấu trúc như sau:
*   `results/predictions/`: Lưu dự báo dưới dạng CSV.
    * Định dạng chuẩn bắt buộc: `time, log_return, horizon, true_volatility, predict_volatility`
*   `results/visualizations/`: Lưu ảnh biểu đồ (line plots) đối chiếu `true_volatility` và `predict_volatility` ở các horizon chính (ví dụ: h=21).
*   `results/model_params/`: 
    * Lưu trọng số `.pth` đối với GARCH-LSTM.
    * Lưu log thông số fitted parameters (omega, alpha, beta...) đối với statistical GARCH.

---

## 3. Các bước thực thi trên Kaggle

1.  **Đường dẫn Dataset Input:** Code sẽ được trỏ cứng (hardcode path) vào Dataset đã upload trên Kaggle: `/kaggle/input/datasets/trnhngv/historical-price`.
2.  **Tải mã nguồn:** Upload thư mục `model/GARCH based/` lên Kaggle dưới dạng một Utility Script hoặc Upload dataset chứa code.
3.  **Môi trường:** Cài đặt thư viện `arch` (`!pip install arch`).
4.  **Chạy Pipeline:**
    *   Vòng lặp (Loop) qua 9 chỉ số.
    *   Với mỗi chỉ số, khởi tạo và train 4 mô hình.
    *   Tại tập Test, sinh dự báo đa bước, thực hiện quy đổi Căn-bậc-hai-của-thời-gian.
    *   Lưu toàn bộ kết quả vào các thư mục `predictions`, `visualizations`, `model_params` tương ứng.
