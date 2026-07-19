## Issue: GARCH-LSTM-Hybrid bị nổ dự báo ở horizon xa, làm MSE cực cao dù horizon 1 bám pattern

### Bối cảnh

Khi trực quan hóa `GARCH-LSTM-Hybrid`, dự báo ở `horizon=1` nhìn khá bám pattern volatility thật. Tuy nhiên khi horizon tăng lên, đặc biệt `horizon=10` và `horizon=21`, giá trị dự báo bị lệch rất xa, có nhiều điểm spike cực lớn. Điều này làm MSE tổng thể của model cao bất thường so với các baseline GARCH và Transformer.

Trong kết quả hiện tại từ `output/merged_all_predictions.csv`, `GARCH-LSTM-Hybrid` có:

```text
horizon=1   MSE ~0.143,      pred_max ~8.17
horizon=3   MSE ~1.03,       pred_max ~21.53
horizon=5   MSE ~6.12,       pred_max ~66.95
horizon=10  MSE ~398.26,     pred_max ~675.33
horizon=21  MSE ~6,452,464,  pred_max ~106,060
```

Ví dụ cùng một forecast origin tại `DAX_40`, ngày `2020-03-13`:

```text
h=1   true=2.2999  pred=6.0369
h=3   true=2.3754  pred=21.5268
h=5   true=2.4906  pred=66.9466
h=10  true=2.9535  pred=675.3307
h=21  true=3.1712  pred=106060.3594
```

Điều này cho thấy lỗi chính không nằm ở metric MSE đơn thuần, mà nằm ở logic sinh forecast multi-horizon hoặc tính ổn định của recursion trong GARCH-LSTM.

### File liên quan

Nhánh Kaggle GARCH-based:

- [`model/GARCH based/models/garch_lstm.py`](https://github.com/nhanbayern/1003_EPA-Project_UIT/blob/kaggle-implementation/model/GARCH%20based/models/garch_lstm.py)
- [`model/GARCH based/create_notebook.py`](https://github.com/nhanbayern/1003_EPA-Project_UIT/blob/kaggle-implementation/model/GARCH%20based/create_notebook.py)
- [`model/GARCH based/utils.py`](https://github.com/nhanbayern/1003_EPA-Project_UIT/blob/kaggle-implementation/model/GARCH%20based/utils.py)

Nhánh reproduction chính:

- [`AAAI24_GARCH_NN_Reproduction/models/garch_lstm_hybrid.py`](https://github.com/nhanbayern/1003_EPA-Project_UIT/blob/master/AAAI24_GARCH_NN_Reproduction/models/garch_lstm_hybrid.py)
- [`AAAI24_GARCH_NN_Reproduction/experiments/run_benchmark.py`](https://github.com/nhanbayern/1003_EPA-Project_UIT/blob/master/AAAI24_GARCH_NN_Reproduction/experiments/run_benchmark.py)

### Nghi vấn nguyên nhân

#### 1. Recursive multi-step forecast của GARCH-LSTM có thể không ổn định

Trong `model/GARCH based/models/garch_lstm.py`, hàm `forecast_multi_variance()` sinh forecast nhiều bước bằng recursion:

```python
eps_prev = encoder_returns[:, -1:]
forecasts = []
for h in range(max_horizon):
    sigma2_t, c_t = self.cell(eps_prev, sigma2_prev, c_t)
    sigma2_scalar = sigma2_t.mean(dim=-1, keepdim=True)
    forecasts.append(sigma2_scalar)

    eps_prev = torch.zeros_like(eps_prev)
    sigma2_prev = sigma2_scalar
```

Sau bước đầu tiên, model dùng `eps_prev = 0` và đưa chính `sigma2_scalar` dự báo vào bước tiếp theo. Nếu hệ số GARCH-LSTM học ra vùng không ổn định, variance sẽ bị khuếch đại lũy tiến qua 21 bước.

#### 2. Tham số GARCH trong cell không bị ràng buộc stationarity

Trong `model/GARCH based/models/garch_lstm.py`, các tham số được khai báo tự do:

```python
self.omega = nn.Parameter(torch.rand(1))
self.alpha = nn.Parameter(torch.rand(1))
self.beta = nn.Parameter(torch.rand(1))
self.gamma = nn.Parameter(torch.rand(1))
```

Hiện không có constraint rõ ràng cho:

```text
omega > 0
alpha >= 0
beta >= 0
gamma >= 0
alpha + beta + gamma/2 < 1
```

Trong GARCH/GJR-GARCH, nếu tổng hệ số động học quá gần hoặc vượt 1, forecast variance có thể không mean-revert và dễ explode. `torch.clamp(..., min=1e-6)` chỉ chặn giá trị âm hoặc quá nhỏ, không chặn giá trị quá lớn.

#### 3. Train path và inference path không khớp nhau

Trong `model/GARCH based/create_notebook.py`, training dùng teacher forcing:

```python
pred_var_path = model(enc_r, dec_v)
pred_var = pred_var_path[:, -1]
loss = criterion(pred_var, target_r)
```

Nhưng inference lại dùng:

```python
model.forecast_multi_variance(...)
```

Tức model được train chủ yếu cho 1-step với variance thật trong decoder window, nhưng khi test multi-step lại tự đưa forecast variance của chính nó vào các bước sau. Sự khác biệt này có thể làm lỗi nhỏ ở h=1 bị khuếch đại rất mạnh ở h=10/h=21.

#### 4. Logic rolling multi-horizon ở reproduction có offset đáng nghi

Trong `AAAI24_GARCH_NN_Reproduction/experiments/run_benchmark.py`, matrix forecast rolling được build bằng:

```python
anchor_idx = np.arange(n_anchors, dtype=int)[:, None]
horizon_offsets = np.arange(1, max_horizon + 1, dtype=int)[None, :]
return np.sqrt(pred_var_1d[anchor_idx + horizon_offsets])
```

Với định nghĩa forecast tại anchor `i`, horizon `h=1` thường phải map về forecast đầu tiên tại `i`, tức offset `h-1`, không phải `h`. Hiện tại `h=1` đang lấy `pred_var_1d[i+1]`. Cần xác minh lại convention target/prediction để tránh lệch một ngày.

Ngoài ra, cách dùng `pred_var_1d[i+h]` là lấy 1-step forecast ở origin tương lai, sau khi model đã nhìn thấy các observation trung gian. Đây không phải h-step forecast causal từ cùng một origin. Nếu so sánh với model one-shot hoặc recursive h-step thật thì logic này không công bằng.

### Test nhanh đề xuất

#### Test 1: kiểm tra explosion theo horizon

Thêm diagnostic đọc `output/merged_all_predictions.csv`, group theo `model,horizon`:

```python
import pandas as pd
import numpy as np

df = pd.read_csv("output/merged_all_predictions.csv")
g = df[df["model"].eq("GARCH-LSTM-Hybrid")].copy()

g["true_volatility"] = pd.to_numeric(g["true_volatility"], errors="coerce")
g["predict_volatility"] = pd.to_numeric(g["predict_volatility"], errors="coerce")
g["err"] = g["predict_volatility"] - g["true_volatility"]
g["sqerr"] = g["err"] ** 2

summary = g.groupby("horizon").agg(
    n=("sqerr", "count"),
    mse=("sqerr", "mean"),
    mae=("err", lambda x: np.nanmean(np.abs(x))),
    pred_median=("predict_volatility", "median"),
    pred_p95=("predict_volatility", lambda x: np.nanpercentile(x, 95)),
    pred_max=("predict_volatility", "max"),
)

print(summary)
```

Expected hiện tại: `horizon=21` có `pred_max` cực lớn và MSE tăng phi tuyến.

#### Test 2: kiểm tra cùng origin pred_h21 / pred_h1

```python
pivot = g.pivot_table(
    index=["dataset", "time"],
    columns="horizon",
    values="predict_volatility",
    aggfunc="first",
)

pivot["ratio_21_1"] = pivot[21] / pivot[1]
print(pivot["ratio_21_1"].describe())
print(pivot.sort_values("ratio_21_1", ascending=False).head(20))
```

Nếu `ratio_21_1` có giá trị hàng nghìn hoặc hàng chục nghìn, đây là bằng chứng recursion đang explode.

#### Test 3: unit test stability cho recursion

Set thủ công tham số GARCH-LSTM về vùng ổn định và không ổn định:

```python
# Stable case
alpha = 0.05
beta = 0.90
gamma = 0.00
# expected: h=21 bounded

# Unstable case
alpha = 0.20
beta = 1.10
gamma = 0.00
# expected: h=21 explode or test fail
```

Test này giúp chứng minh cần enforce stationarity constraint.

#### Test 4: kiểm tra train/inference consistency

Với cùng một input window:

```python
pred_train_path = model(enc_r, dec_v)
pred_train_1 = pred_train_path[:, -1]

pred_infer_1 = model.forecast_multi_variance(
    enc_r, dec_v, max_horizon=1
)[:, 0]

print(torch.max(torch.abs(pred_train_1 - pred_infer_1)))
```

Nếu hai giá trị lệch đáng kể, training objective và inference path không nhất quán.

#### Test 5: kiểm tra offset trong `_build_pred_matrix_point_in_time`

Dùng sentinel array:

```python
pred_var_1d = np.arange(1, 30, dtype=float) ** 2
pred_matrix = _build_pred_matrix_point_in_time(
    pred_var_1d,
    max_horizon=21,
    n_anchors=5,
)

print(pred_matrix[0, 0])
```

Nếu `horizon=1` của anchor 0 trả về `2.0` thay vì `1.0`, tức logic đang offset thêm 1 bước. Cần xác định lại target convention rồi sửa `horizon_offsets` thành `np.arange(0, max_horizon)` nếu phù hợp.

### Hướng sửa đề xuất

1. Ràng buộc tham số GARCH-LSTM bằng transform thay vì dùng parameter tự do:
   - `omega = softplus(raw_omega) + eps`
   - `alpha, beta, gamma` qua `softplus` hoặc `sigmoid`
   - normalize để `alpha + beta + gamma/2 < 1 - eps`

2. Train model bằng đúng path inference:
   - thay vì chỉ train `model.forward(...)[-1]`, train qua `forecast_multi_variance(..., max_horizon=1)` cho nhất quán;
   - hoặc tốt hơn, train multi-horizon loss cho các horizon `[1,3,5,10,21]`.

3. Thêm cap/guard diagnostic trong evaluation:
   - không nên silently xuất `predict_volatility` hàng nghìn/hàng trăm nghìn;
   - nếu `pred_vol > threshold` hoặc non-finite thì log warning kèm dataset/time/horizon.

4. Xác minh và sửa offset horizon trong `_build_pred_matrix_point_in_time()` ở reproduction branch.

5. Sau khi sửa, chạy lại smoke benchmark trên 1-2 dataset có spike nặng (`DAX_40`, `KOSPI_INDEX`) và so sánh:
   - MSE theo horizon;
   - `pred_max`, `p95`, `median`;
   - plot cùng origin h=1/h=21.
