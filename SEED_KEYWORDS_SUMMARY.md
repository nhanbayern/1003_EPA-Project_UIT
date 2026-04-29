# Seed - Tóm Tắt Các Thông Tin Chính

## Seed là gì?
- **Seed**: Giá trị khởi tạo cho bộ sinh số ngẫu nhiên giả (RNG)
- Cùng seed + cùng code → cùng kết quả (reproducible)
- Khác seed → kết quả có thể khác nhau

## Các Keywords Liên Quan

### 1. **Random Initialization** (Khởi Tạo Ngẫu Nhiên)
- `nn.Linear(in_features, out_features)` - khởi tạo trọng số ngẫu nhiên
- `nn.Parameter(torch.zeros(...))` - khởi tạo embedding
- Seed quyết định chuỗi ngẫu nhiên được dùng để khởi tạo
- Khác seed → trọng số ban đầu khác

### 2. **Data Shuffling** (Lấy Mẫu Ngẫu Nhiên)
- `DataLoader(..., shuffle=True)` - xáo trộn dữ liệu trong mỗi epoch
- Seed quyết định thứ tự shuffle batch
- Ảnh hưởng: quá trình tối ưu có thể đi theo "con đường" khác

### 3. **Training Pipeline** (Quy Trình Huấn Luyện)
- `set_global_seed(seed)`:
  - `np.random.seed(seed)` - NumPy randomness
  - `torch.manual_seed(seed)` - PyTorch CPU randomness
  - `torch.cuda.manual_seed_all(seed)` - PyTorch GPU randomness
- Được gọi TRƯỚC mỗi lần train
- Có 10 seed: [42, 123, 202, 303, 404, 505, 606, 707, 808, 999]

### 4. **Determinism (Xác định Tính Ổn Định)**
- Mỗi seed tạo một "biến thể" khác của quá trình học
- Chạy 10 seed giúp biết model có ổn định (consistent) hay chỉ "ăn may"
- Báo cáo thường dùng: **mean ± std** trên nhiều seed

### 5. **Split/Sampling** (Chia Dữ Liệu)
- Trong code hiện tại: split 80/10/10 theo thời gian (không random)
- Sliding window được tạo tuần tự (không bị seed chi phối)
- Seed chỉ ảnh hưởng đến lấy batch trong train loader (shuffle=True)

### 6. **Model Learning** (Học Mô Hình)
- Epoch: số lần duyệt toàn bộ train set
- Weight update: cập nhật trọng số dựa gradient
- Early stopping: dừng train khi val loss không cải thiện
- Seed ảnh hưởng: khởi tạo → shuffle → gradient → update

### 7. **Inference/Prediction** (Dự Đoán)
- Model được chuyển sang `eval()` mode (deterministic)
- Không có dropout, không shuffle
- Nhưng kết quả vẫn khác vì model đã học khác (từ trọng số khác nhau ban đầu)

### 8. **Hyperparameters** (Siêu Tham Số)
- `epochs=60` (hoặc 2 ở smoke mode)
- `batch_size=64`
- `learning_rate=1e-2`
- `lr_patience=5`, `early_stopping_patience=20`
- Những thứ này cố định, seed quyết định cách học

### 9. **Deep Learning Models** (Mô Hình Học Sâu)
- **DL baselines**: Autoformer, Informer, Reformer, Transformer
- **Hybrid model**: GARCH-LSTM-Hybrid
- **Statistical models**: GARCH, GJR-GARCH, FI-GARCH (không bị seed ảnh hưởng)

### 10. **Evaluation Metrics** (Chỉ Số Đánh Giá)
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- Seed ảnh hưởng → metric khác nhau
- Chạy 10 seed lấy trung bình giảm biến động

## Tóm Tắt Pipeline Với Seed

```
For each dataset:
  ├─ Load dữ liệu (cố định)
  ├─ Split 80/10/10 theo thời gian (cố định, không bị seed ảnh hưởng)
  └─ For each seed in [42, 123, ..., 999]:
      ├─ set_global_seed(seed)  ← Khởi tạo RNG
      ├─ Khởi tạo weights ngẫu nhiên (random init)
      ├─ Tạo DataLoader với shuffle=True
      ├─ Train 60 epochs:
      │   ├─ Mỗi epoch shuffle batch (seed ảnh hưởng)
      │   ├─ Forward pass
      │   ├─ Backward pass
      │   └─ Update weights
      ├─ Chọn best model (best validation loss)
      ├─ Inference (deterministic, nhưng model khác)
      └─ Tính MAE, MSE (khác do model khác)
```

## Tại Sao Chạy 10 Seed?

1. **Độ tin cậy**: Kiểm tra sự ổn định, không phải may mắn
2. **Thống kê**: Tính mean ± std để so sánh công bằng
3. **Convention**: Thông lệ trong machine learning papers
4. **Trade-off**: 10 là cân bằng giữa tin cậy và thời gian tính

## Code Liên Quan

- **Khởi tạo seed**: `AAAI24_GARCH_NN_Reproduction/experiments/run_benchmark.py` (lines 37, 65-69, 345)
- **Model definition**: `AAAI24_GARCH_NN_Reproduction/models/dl_baselines.py` (lines 58-71)
- **DataLoader shuffle**: `AAAI24_GARCH_NN_Reproduction/models/dl_baselines.py` (line 52)
- **Split data**: `AAAI24_GARCH_NN_Reproduction/core/data_processor.py` (lines 46-80)
