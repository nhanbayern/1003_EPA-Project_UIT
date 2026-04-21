# Prompt: check and Rebuild Notebook Volatility cho baseline aaai24

## mô tả

## Mục Tiêu



## Cấu Hình Chạy

Notebook phải có biến `mode` ở đầu file:

- `mode = "smoke"`: chỉ dùng 1 dataset, `epoch = 1`
- `mode = "full"`: dùng tất cả dataset, `epoch = 60`

Chọn dataset bằng cách dùng `SMOKE_DATASET = "VN30_INDEX"` khi ở smoke mode.

### Split cố định (bắt buộc)

Mọi notebook sinh ra từ prompt này phải dùng đúng `SPLIT_COUNTS` dưới đây, không được thay đổi giá trị:

```python
SPLIT_COUNTS = {
	"VN30_INDEX": (1971, 1096, 925),
	"VN_INDEX": (1964, 1103, 925),
	"DAX_40": (1983, 1104, 972),
	"EuroNext_100": (2005, 1115, 979),
	"IBEX_35": (1815, 1362, 923),
	"KOSPI_index": (1944, 1109, 881),
	"SMI": (1987, 1135, 901),
	"snp500": (1988, 1109, 927),
	"Nikkei_225": (1677, 1174, 1062),
}
```

### 1. Log-return

Tính đúng công thức:
**Return Calculation:**
$$r_t = \ln(Close_t / Close_{t-1}) \times 100 \text{ (percent-point)}$$

### 2. True Volatility

Tính volatility 60 ngày chỉ từ quá khứ, không được leakage:

$$\sigma_t = \sqrt{\frac{1}{60} \sum_{k=1}^{60} (r_{t-k} - \bar{r}_t)^2}$$

Triển khai bằng:

- `shift(1)`
- `.rolling(window=60).std(ddof=0)`

Quan trọng: phải tính volatility trên toàn bộ chuỗi lịch sử trước, rồi mới lọc dữ liệu theo năm 2010 trở đi. Không được lọc trước rồi mới tính rolling.

Ràng buộc bắt buộc về thời gian:

- Chỉ dùng dữ liệu có `time.year >= 2010` cho train/val/test.
- Mọi split trong `SPLIT_COUNTS` phải được áp trên tập đã lọc 2010+.
- Không được dùng bất kỳ hàng nào trước năm 2010 trong huấn luyện hay xuất dự báo.

### 3. Horizon Mapping

Tạo target theo các horizon:

- `1`
- `3`
- `5`
- `10`
- `21`

Target tại horizon `h` phải là `TrueVol(i, h) = σ_{i+h}`.

## Model Requirements

### 1. Model gốc

Notebook phải dùng `MoiraiMoEModule` gốc từ thư viện hiện có trong project. Không sửa kiến trúc bên trong model gốc.

### 2. Input Pipeline

Input cho model phải build đúng format của MoiraiMoE:

- `target`
- `observed_mask`
- `sample_id`
- `time_id`
- `variate_id`
- `prediction_mask`
- `patch_size`

Không được thêm `hmm_regime` hoặc bất kỳ feature HMM nào vào input pipeline. Notebook này khác hoàn toàn với bản MOIRAI_MOE_GARCH, nơi HMM được dùng cho gating.

### 3. Loss Function

Dùng Student-t Negative Log-Likelihood, hoặc wrapper tương đương nhưng output phải bám theo volatility prediction.

### 4. Training Wrapper

Tạo một `LightningModule` tối giản để:

- build input cho model gốc
- chạy forward
- trích volatility dự báo từ distribution
- compute loss
- log `train_loss` và `val_loss`

## Output Specification

Chỉ xuất **một** CSV prediction cho mỗi lần chạy.

### Bắt buộc

- Tên file: `{model_name}_prediction.csv`
- Ví dụ: `Moirai_MoE_prediction.csv`
- Đường dẫn lưu cố định: `D:\UIT\1003_EPA_PROJECT\1003_EPA-Project_UIT\AAAI24_GARCH_NN_Reproduction\experiments\results_4`
- Phải tạo thư mục bằng `os.makedirs(..., exist_ok=True)`

### Columns bắt buộc

CSV phải có đúng các cột sau:

- `time`
- `dataset`
- `model`
- `horizon`
- `True_Volatility`
- `Pred_Volatility`
- `return_1_day`
**Return Calculation:**
$$r_t = \ln(Close_t / Close_{t-1}) \times 100 \text{ (percent-point)}$$

Không tạo thêm `results CSV`, không tạo metrics CSV, không tạo dashboard output.

## Các Lỗi Cần Tránh

### 1. Import order

Không được import local module trước khi chèn `src` vào `sys.path`.

### 2. Data leakage

Không được filter theo năm 2010 trước khi tính `True_Volatility`.
Sau khi tính xong, bắt buộc chỉ giữ dữ liệu từ năm 2010 trở đi cho toàn bộ pipeline.

### 3. In-place tensor update

Không dùng các phép gán in-place trong forward pass nếu gây lỗi autograd.

### 4. Thừa output

Không tạo:

- metrics CSV
- results CSV
- dashboard widget
- plotting cell không cần thiết

### 5. Sai nguồn True Volatility

## Yêu Cầu Chất Lượng Code

- Code tối giản, rõ ràng, có hàm tách bạch
- Tuân thủ PEP 8
- Không thêm EDA không cần thiết
- Không dùng biến thừa
- Giữ notebook ngắn, thực dụng, dễ chạy lại

## Output Cuối Cùng

Notebook hoàn chỉnh phải sinh ra một file CSV duy nhất chứa dự báo volatility theo từng dataset và horizon, với đúng schema đã nêu ở trên, và lưu tại thư mục output cố định.

