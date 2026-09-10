# Kiến trúc train trên Modal

Tài liệu này mô tả pipeline train GPU hiện tại của project. Entrypoint hợp nhất
là [`experiments/all_models_modal/modal_app.py`](../../experiments/all_models_modal/modal_app.py).
Pipeline dùng Modal để chạy các họ model trong các container độc lập, sau đó
chuẩn hóa artifact về cùng một schema để phân tích tiếp.

## 1. Sơ đồ tổng thể

```text
Local repository
    │
    │ modal run ... --families garch,transformer,moirai,moiraivar,hybrid,wavelet
    ▼
Local entrypoint: modal_app.main()
    │
    ├─ tạo run_id = YYYYMMDD_HHMMSS
    ├─ spawn một remote call cho mỗi family
    │
    ▼
Modal remote container (mỗi family một container GPU)
    │
    ├─ Image Python 3.11 + PyTorch + arch + scientific stack
    ├─ mount source, dataset, pretrained weights
    ├─ clone Salesforce uni2ts vào /root/uni2ts
    ├─ prepare notebook Kaggle → đường dẫn Modal
    ├─ execute từng code cell bằng exec(...)
    ├─ train / validate / test
    ├─ lưu weights vào Modal Volume
    └─ normalize prediction artifacts
    │
    ▼
Modal Volume + ZIP trả về local
    │
    ├─ predictions / metrics / figures trong ZIP
    ├─ weights tải riêng từ Volume
    └─ output/<timestamp>/run_manifest.txt
```

## 2. Các thành phần chính

| Thành phần | Vai trò |
|---|---|
| `modal_app.py` | Khai báo Modal App, image, mount, remote function và orchestration |
| `NOTEBOOKS` | Ánh xạ family → notebook nguồn cần chạy |
| `_prepare_notebook()` | Loại Kaggle shell command, sửa path, inject smoke/reporting và kiểm tra target contract |
| `run_selected()` | Chạy family trong container remote; publish checkpoint |
| `experiments/moirai_var_aware/runner.py` | Runner Python riêng cho Moirai VaR-aware |
| `experiments/moirai_var_aware/train.py` | Vòng train, early stopping và validation checkpoint |
| `experiments/moirai_var_aware/data.py` | Tạo origin, input context, future-realized target và split |
| `Results` | Modal Volume lưu weights/checkpoint lâu dài |

Các family được hỗ trợ:

```text
garch       → model/GARCH based/GARCH_Kaggle_Pipeline.ipynb
transformer → model/transformer based/version_2/kaggle_notebook_v2.ipynb
moirai      → model/Morai based/notebooks/kaggle_notebook.ipynb
moiraivar   → model/moirai_var_aware/moirai_kaggle.ipynb
hybrid      → model/modify_autoformer/Hybrid GARCH-Autoformer/kaggle_notebook_hybrid.ipynb
wavelet     → model/modify_autoformer/Wavelet Transform/kaggle_notebook_wavelet.ipynb
```

## 3. Modal image và filesystem remote

`modal_app.py` dựng một image Debian Slim với Python 3.11. Các dependency chính
được cài trong image gồm:

```text
torch==2.3.1, pandas==2.1.4, numpy==1.26.4, scipy==1.11.4,
arch, matplotlib, nbformat, ipython, hydra-core, jaxtyping,
jax[cpu], huggingface_hub, safetensors, einops, gluonts,
ptwt, PyWavelets
```

Filesystem remote chuẩn:

```text
/root/project  ← source tree, bỏ qua .git, .venv, output, tmp
/root/dataset  ← dataset local
/root/weights  ← pretrained Moirai weights
/root/uni2ts   ← Salesforce uni2ts clone
/root/modal_output
    ├─ garch/
    ├─ transformer/
    ├─ moirai/
    ├─ moiraivar/
    ├─ hybrid/
    └─ wavelet/
/root/persist   ← mount của Modal Volume
```

### Kiểm tra và tạo volume `Results`

Ngay khi module Modal được load, pipeline dùng cơ chế get-or-create của Modal:

```python
RESULTS_VOLUME_NAME = "Results"
artifact_volume = modal.Volume.from_name(
    RESULTS_VOLUME_NAME,
    create_if_missing=True,
)
```

Nếu storage đã có volume tên `Results`, pipeline dùng lại volume đó; nếu chưa
có, Modal tạo volume mới. Volume được mount vào container tại `/root/persist`
để lưu checkpoint/weights theo từng `run_id`.

Remote function dùng GPU `A10G` và timeout tối đa 8 giờ:

```python
@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 8,
    volumes={"/root/persist": artifact_volume},
)
```

## 4. Cách notebook được chạy trên Modal

Notebook được giữ làm artifact/source của từng family, nhưng không chạy nguyên
xi như trên Kaggle. `_prepare_notebook()` thực hiện:

1. Đọc notebook bằng `nbformat`.
2. Bỏ các dòng shell bắt đầu bằng `!`.
3. Đổi các path `/kaggle/...` sang `/root/...` hoặc `/root/modal_output/...`.
4. Ghi đè lambda grid khi chạy MoiraiVaR.
5. Chèn reporter cho dataset, tier, model và epoch.
6. Kiểm tra không còn target rolling/RMS hoặc origin lệch.
7. Ghi notebook đã chuẩn bị vào `/root/prepared_notebooks/`.

Sau đó `run_selected()` compile và `exec` từng code cell. Vì các notebook cũ
dùng tên module chung như `dataset`, `models`, `config`, `utils`, runner xóa
các module này khỏi `sys.modules` và thay family path trước khi chạy family kế
tiếp. Đây là lớp cách ly cần thiết khi nhiều notebook chạy trong cùng một
process remote.

## 5. Luồng train và validation

### Baseline, Transformer và hybrid

Notebook của family chịu trách nhiệm:

```text
load data
  → tạo log returns
  → tạo input context
  → tạo future-realized volatility target
  → split train / validation / test
  → train model
  → chọn checkpoint tốt nhất trên validation
  → đánh giá test một lần
  → export prediction
```

Contract target dùng chung là:

```text
origin: t
input:  r[t-lookback+1 : t+1]
target(h): std(r[t+1 : t+h+1], ddof=0)
VaR return: r[t+1]
```

Historical rolling volatility hoặc GARCH conditional volatility chỉ được dùng
làm feature khi family yêu cầu, không được dùng thay cho future target.

### MoiraiVaR-aware

Runner [`experiments/moirai_var_aware/runner.py`](../../experiments/moirai_var_aware/runner.py)
thực hiện trực tiếp bằng Python:

```text
load_and_split_dataset()
  → DataLoader train / validation / test
  → load frozen hoặc fine-tuned Moirai backbone
  → train volatility head
  → validation early stopping
  → export validation và test predictions
```

Hai chế độ:

```text
head: chỉ train MLP volatility head; backbone frozen
full: fine-tune backbone với backbone_lr nhỏ hơn head learning rate
```

Loss gồm volatility loss và VaR-aware quantile loss, điều khiển bởi
`lambda_var`. Lambda sweep phải được đánh giá bằng validation để chọn cấu hình;
test chỉ dùng báo cáo cuối cùng.

## 6. Parallelism và checkpoint

Local `main()` gọi `run_selected.spawn([family], ...)` cho từng family trước khi
đợi kết quả. Vì vậy các family chạy song song trong các container độc lập.

Mỗi remote family:

1. Ghi artifact vào `/root/modal_output/<family>/`.
2. Normalize về `normalized_predictions/` theo schema chung.
3. Copy toàn bộ output sang `/root/persist/<run_id>/<family>/`.
4. Commit Modal Volume.
5. Trả ZIP chứa prediction, figure và metric artifact.

Weights không đưa vào ZIP để tránh ZIP quá lớn. Local entrypoint tải weights
riêng bằng `modal volume get` vào thư mục run local.

Nếu một family lỗi, các family đã hoàn thành vẫn được tải và lưu; lỗi được ghi
ở local orchestration.

## 7. Schema prediction sau normalize

Mọi family phải quy về các cột:

```text
dataset
branch
tier
model
time
horizon
log_return
true_volatility
predict_volatility
```

`modal_app.py` kiểm tra trước khi publish:

- đủ canonical columns;
- không rỗng;
- horizon thuộc `{1, 3, 5, 10, 21}`;
- giá trị số hữu hạn;
- volatility không âm;
- không trùng khóa `dataset/branch/tier/model/time/horizon`.

## 8. Lệnh chạy

### Smoke test

```powershell
cd "D:\UIT_LEARNING_MATERIAL\07.Research\code"
py -3.11 -m modal setup
py -3.11 -m modal run experiments/all_models_modal/modal_app.py `
  --families moiraivar `
  --smoke-test
```

Chạy một family nhỏ trước giúp kiểm tra image, GPU, path, weights và notebook
execution.

### Chạy các family chính

```powershell
py -3.11 -m modal run experiments/all_models_modal/modal_app.py `
  --families garch,transformer,moirai,moiraivar,hybrid,wavelet
```

### Chạy lambda sweep cho MoiraiVaR

```powershell
py -3.11 -m modal run experiments/all_models_modal/modal_app.py `
  --families moiraivar `
  --lambda-sweep 0,0.05,0.1,0.2,0.5,1
```

Output local nằm tại:

```text
output/<YYYYMMDD_HHMMSS>/
```

## 9. Quy trình hậu xử lý

Sau khi các family hoàn tất, dùng prediction artifacts đã normalize để:

```text
merge_canonical_predictions.py
  → run_full_var_analysis.py
  → run_mcdm_evaluation.py
```

MCDM phải nhận một artifact validation riêng để khóa eligibility/model gate;
không dùng test metrics để loại model trước khi ranking.

## 10. Giới hạn cần ghi nhận

- `uni2ts` hiện được clone bằng `--depth 1`; muốn reproducibility tuyệt đối nên
  pin commit/tag cụ thể.
- Notebook execution qua `exec` giữ tính tương thích với notebook legacy nhưng
  khó debug hơn Python package thuần.
- Dataset và pretrained weights được mount từ máy local; phải bảo đảm đúng
  phiên bản trước mỗi run.
- Smoke test kiểm tra wiring, không thay thế full training hay statistical
  validation.
