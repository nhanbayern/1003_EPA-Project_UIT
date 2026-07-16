# Kế hoạch Triển khai (Implementation Plan) - Version 2: Transformer Complexity Test

**Mục tiêu:** Xây dựng luồng thực thi (pipeline) `version_2` cho 4 mô hình Transformer (Vanilla, Autoformer, Informer, Reformer) chạy trên 3 mức độ phức tạp (Tiers). Quá trình thực thi sẽ được tiến hành trên Kaggle thông qua Git-sync từ thư mục `model/transformer based/version_2` của repository cục bộ.

## Open Questions

Không có câu hỏi mở lớn. Pipeline đã được xác định rõ. Tuy nhiên, mong người dùng xem xét kỹ cấu trúc thư mục lưu trữ kết quả để đảm bảo phù hợp với thói quen sử dụng trên Kaggle.

## Proposed Changes

Chúng ta sẽ tạo một thư mục mới hoàn toàn `version_2` để không ảnh hưởng đến code của `version_1`.

### 1. Tổ chức Thư mục Cục bộ (Local Repository)
Code cục bộ sẽ được viết tại `D:\UIT\1003_EPA_PROJECT\1.0.0\1003_EPA-Project_UIT\model\transformer based\version_2`.

#### [NEW] [models.py](file:///D:/UIT/1003_EPA_PROJECT/1.0.0/1003_EPA-Project_UIT/model/transformer%20based/version_2/models.py)
*   Chứa định nghĩa của 4 mô hình Transformer.
*   Thiết kế lại hàm `__init__` để nhận trực tiếp các tham số `d_model`, `e_layers`, `n_heads`, `d_ff`, `dropout` từ dictionary cấu hình truyền vào (thay vì gán cứng như version 1).

#### [NEW] [config.py](file:///D:/UIT/1003_EPA_PROJECT/1.0.0/1003_EPA-Project_UIT/model/transformer%20based/version_2/config.py)
*   File mới dùng để định nghĩa dictionary chứa 3 cấu hình Tiers:
    *   `Tier_1_Miniaturized`: (d_model=32, layers=2, n_heads=4)
    *   `Tier_2_Standard`: (d_model=128, layers=3, n_heads=8)
    *   `Tier_3_Large`: (d_model=512, layers=6, n_heads=8)

#### [NEW] [dataset.py](file:///D:/UIT/1003_EPA_PROJECT/1.0.0/1003_EPA-Project_UIT/model/transformer%20based/version_2/dataset.py)
*   Sử dụng lại class `VolatilityDataset` từ version 1.

#### [NEW] [utils.py](file:///D:/UIT/1003_EPA_PROJECT/1.0.0/1003_EPA-Project_UIT/model/transformer%20based/version_2/utils.py)
*   Bổ sung các hàm để tính toán Metrics (như bản cũ).
*   Thêm các hàm vẽ biểu đồ mới (Plotting functions) để xuất ra file thay vì chỉ in ra màn hình.
    *   `plot_loss_curve(train_losses, val_losses, save_path)`
    *   `plot_predictions(true, pred, horizon, save_path)`

#### [NEW] [kaggle_notebook_v2.ipynb](file:///D:/UIT/1003_EPA_PROJECT/1.0.0/1003_EPA-Project_UIT/model/transformer%20based/version_2/kaggle_notebook_v2.ipynb)
*   Notebook chính để upload/chạy trên Kaggle.
*   Notebook sẽ clone nhánh Github, import code từ `version_2` và thực thi vòng lặp: `For each index -> For each Tier -> For each Model`.

### 2. Tổ chức Thư mục Kết quả (Output Structure trên Kaggle)
Để đáp ứng yêu cầu lưu trữ tham số mô hình, kết quả và biểu đồ (visualize), đoạn code chạy trên Kaggle sẽ tự động sinh ra cấu trúc thư mục sau tại `/kaggle/working/results_v2/`:

```text
/kaggle/working/results_v2/
├── metrics/
│   ├── comparison_metrics_Tier_1.csv
│   ├── comparison_metrics_Tier_2.csv
│   └── comparison_metrics_Tier_3.csv
├── all_predictions/
│   ├── Tier_1/ (chứa các file CSV dự báo của từng model trên từng index)
│   ├── Tier_2/
│   └── Tier_3/
├── models_weights/
│   ├── Tier_1/ (chứa file .pt của các model)
│   ├── Tier_2/
│   └── Tier_3/
└── visualizations/
    ├── Loss_Curves/
    │   ├── Tier_1/ (chứa ảnh .png biểu diễn train_loss vs val_loss để minh họa overfitting)
    │   ├── Tier_2/
    │   └── Tier_3/
    └── Predictions/
        ├── Tier_1/ (chứa ảnh .png so sánh true vs pred)
        ├── Tier_2/
        └── Tier_3/
```

## Verification Plan

1. **Khởi tạo Code Cục bộ:** Sẽ tiến hành viết và hoàn thiện các file Python trong thư mục `version_2`.
2. **Review Code Cấu hình:** Đảm bảo `config.py` và vòng lặp `for` trong Notebook có thể chạy độc lập, tự động tạo cấu trúc thư mục và phân phối đúng cấu hình siêu tham số cho từng model.
3. **Mô phỏng Thử nghiệm:** (Chạy test local một epoch duy nhất nếu có thể) để xác nhận hệ thống sinh file (CSV, Model Weight, Hình ảnh PNG) hoạt động đúng như thiết kế ở phần Output.
4. **Push to Github & Run Kaggle:** Sau khi user duyệt, mã nguồn sẽ được commit/push lên nhánh Kaggle, và user chỉ cần dùng notebook để chạy thực tế.
