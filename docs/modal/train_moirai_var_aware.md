# Quy trinh train Moirai VaR-aware tren Modal GPU

Tai lieu nay mo ta cac cong doan train bien the Moirai VaR-aware trong:

```text
experiments/moirai_var_aware/
```

Muc tieu cua experiment la train Moirai-family bang loss VaR-aware. Code hien ho tro 2 che do:

```text
head: freeze Moirai backbone, chi train regression head.
full: fine-tune toan bo Moirai backbone + regression head.
```

Objective chung:

```text
loss = MSE(predicted_volatility, true_volatility)
       + lambda_var * QuantileLoss(realized_return, VaR_1%)
```

Trong do `lambda_var` dieu khien muc do uu tien tail risk VaR 1%.

## 1. Dieu kien truoc khi chay

Can co cac thanh phan sau tren may local:

```text
dataset/
model/Morai based/weights/
experiments/moirai_var_aware/
```

Y nghia:

- `dataset/`: du lieu goc dung chung voi baseline.
- `model/Morai based/weights/`: pretrained weights cua Moirai family.
- `experiments/moirai_var_aware/`: code experiment VaR-aware.

Khong dat file tai lieu vao `output/`. Folder `output/` chi dung de chua ket qua chay model va ket qua phan tich.

## 2. Cai dat va ket noi Modal

Mo PowerShell tai thu muc repo:

```powershell
cd "D:\UIT_LEARNING_MATERIAL\07.Research\code"
```

Cai Modal CLI:

```powershell
py -3.11 -m pip install modal
```

Kiem tra:

```powershell
py -3.11 -m modal --version
```

Dang nhap Modal:

```powershell
py -3.11 -m modal setup
```

Kiem tra token:

```powershell
py -3.11 -m modal token info
```

Neu khong dung duoc browser login, co the set token thu cong:

```powershell
py -3.11 -m modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_TOKEN_SECRET
```

## 3. Cong doan build moi truong tren Modal

File dieu khien Modal:

```text
experiments/moirai_var_aware/modal_app.py
```

Khi chay lenh `modal run`, Modal se:

1. Tao container Python 3.11.
2. Cai cac package can thiet nhu `torch`, `pandas`, `numpy`, `scipy`, `hydra-core`, `jaxtyping`, `jax[cpu]`, `gluonts`.
3. Clone source `uni2ts`:

```text
https://github.com/SalesforceAIResearch/uni2ts.git
```

vao:

```text
/root/uni2ts
```

4. Mount code experiment local vao Modal:

```text
experiments/moirai_var_aware/ -> /root/moirai_var_aware
```

5. Mount dataset local vao Modal:

```text
dataset/ -> /root/dataset
```

6. Mount Moirai weights local vao Modal:

```text
model/Morai based/weights/ -> /root/weights
```

## 4. Cong doan load Moirai backbone

Trong remote runner:

```text
experiments/moirai_var_aware/runner.py
```

Python path duoc them:

```python
sys.path.insert(0, "/root/uni2ts/src")
sys.path.insert(0, "/root")
```

Sau do `modeling.py` import backbone tu `uni2ts`:

```python
from uni2ts.model.moirai import MoiraiModule
from uni2ts.model.moirai2 import Moirai2Module
from uni2ts.model.moirai_moe import MoiraiMoEModule
```

Weights duoc load tu:

```text
/root/weights/
```

Vi du voi Moirai2:

```python
Moirai2Module.from_pretrained("/root/weights/moirai-2.0-R-small")
```

Backbone duoc freeze:

```python
param.requires_grad = False
```

Do do experiment hien tai la:

Che do mac dinh la:

```text
frozen Moirai backbone + trainable MLP volatility head
```

Neu chay voi `--tuning-mode full`, backbone Moirai duoc mo khoa va fine-tune cung MLP head.

## 5. Cong doan tao dataset train

File xu ly du lieu:

```text
experiments/moirai_var_aware/data.py
```

Moi file index trong `dataset/` duoc doc theo ten:

```text
/root/dataset/{INDEX_NAME}.csv
```

Dataset tao:

- input `x`: cua so return qua khu.
- target `y`: realized volatility theo cac horizon.
- `log_return`: realized return dung cho VaR quantile loss.

Split train/validation/test lay tu:

```text
experiments/moirai_var_aware/config.py
```

Luu y nghien cuu: dataset goc va split duoc dung chung voi baseline, nhung preprocessing/target construction can duoc doi chieu voi baseline neu muon khang dinh so sanh tuyet doi cong bang.

## 6. Cong doan train

File train:

```text
experiments/moirai_var_aware/train.py
```

Moi batch gom:

```python
batch_x, batch_y, batch_return
```

Model du bao volatility:

```python
pred = model(batch_x)
```

Loss duoc tinh bang:

```python
criterion = VarAwareVolatilityLoss(
    alpha=0.01,
    lambda_var=lambda_var,
    distribution="student_t",
    nu=4.0,
)
```

Vong train:

```python
loss, parts = criterion(pred, batch_y, batch_return)
loss.backward()
optimizer.step()
```

Voi `--tuning-mode head`, chi optimizer MLP head:

```python
optimizer = torch.optim.AdamW(model.mlp.parameters(), lr=lr, weight_decay=1e-4)
```

Voi `--tuning-mode full`, optimizer dung 2 learning rate:

```python
optimizer = torch.optim.AdamW(
    [
        {"params": model.extractor.backbone.parameters(), "lr": backbone_lr},
        {"params": model.mlp.parameters(), "lr": head_lr},
    ],
    weight_decay=1e-4,
)
```

Khuyen nghi ban dau:

```text
head_lr = 1e-3
backbone_lr = 1e-5
```

Nghia la full fine-tune cap nhat ca Moirai backbone, nhung voi learning rate nho hon head de tranh pha pretrained representation.

## 7. Chay smoke test

Chay test nho truoc de kiem tra Modal, GPU, mount dataset va weights:

```powershell
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --lambda-var 0.2 --epochs 1 --models moirai2 --datasets VN_INDEX
```

Smoke test full fine-tune:

```powershell
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --tuning-mode full --lambda-var 0.2 --epochs 1 --batch-size 8 --backbone-lr 1e-5 --models moirai2 --datasets VN_INDEX
```

Neu full fine-tune bi OOM, giam batch:

```powershell
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --tuning-mode full --lambda-var 0.2 --epochs 1 --batch-size 4 --backbone-lr 1e-5 --models moirai2 --datasets VN_INDEX
```

Neu thanh cong, output zip se duoc luu vao:

```text
output/modal_moirai_var_loss/
```

## 8. CLI quan tri Modal can dung

Kiem tra version:

```powershell
py -3.11 -m modal --version
```

Kiem tra dang nhap/token:

```powershell
py -3.11 -m modal token info
```

Mo dashboard Modal:

```powershell
py -3.11 -m modal dashboard
```

List app dang chay/da chay gan day:

```powershell
py -3.11 -m modal app list
```

Xem dashboard cua app train:

```powershell
py -3.11 -m modal app dashboard moirai-var-aware-training
```

Xem log gan nhat:

```powershell
py -3.11 -m modal app logs moirai-var-aware-training
```

Stream log khi job dang chay:

```powershell
py -3.11 -m modal app logs moirai-var-aware-training -f
```

Dung app/job neu can huy:

```powershell
py -3.11 -m modal app stop moirai-var-aware-training
```

Kiem tra usage/cost hom nay, tach theo resource CPU/GPU:

```powershell
py -3.11 -m modal billing report --for today --show-resources
```

Kiem tra usage/cost thang nay:

```powershell
py -3.11 -m modal billing report --for "this month" --show-resources
```

## 9. CLI kiem tra GPU Modal tren Windows

Modal CLI khong co lenh rieng kieu `modal gpu list`. Lenh `modal shell --gpu ...` cung khong ho tro Windows, nen tren may nay phai kiem tra GPU bang `modal run`.

Script check GPU:

```text
experiments/moirai_var_aware/gpu_check.py
```

Chay check A10G:

```powershell
py -3.11 -m modal run experiments/moirai_var_aware/gpu_check.py
```

Lenh nay se chay remote function tren Modal GPU va in:

- `torch.cuda.is_available()`
- `torch.version.cuda`
- ten GPU tu `torch.cuda.get_device_name(0)`
- output cua `nvidia-smi`

Neu muon test GPU khac, sua `gpu="A10G"` trong `gpu_check.py`, vi du:

```python
@app.function(image=image, gpu="A100", timeout=10 * 60)
```

hoac:

```python
@app.function(image=image, gpu="any", timeout=10 * 60)
```

Experiment hien tai dang cau hinh GPU trong:

```text
experiments/moirai_var_aware/modal_app.py
```

```python
@app.function(image=image, gpu="A10G", timeout=60 * 60 * 8)
```

Muon doi GPU mac dinh, sua `gpu="A10G"` thanh:

```python
gpu="A100"
```

hoac:

```python
gpu="any"
```

## 10. Chay full experiment

Chay ca 3 model Moirai-family theo che do head-only:

```powershell
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --lambda-var 0.2 --epochs 30 --models moirai,moirai2,moirai_moe
```

Chay full fine-tune 1 model truoc:

```powershell
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --tuning-mode full --lambda-var 0.2 --epochs 30 --batch-size 8 --backbone-lr 1e-5 --models moirai2
```

Chay full fine-tune ca 3 model:

```powershell
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --tuning-mode full --lambda-var 0.2 --epochs 30 --batch-size 8 --backbone-lr 1e-5 --models moirai,moirai2,moirai_moe
```

Co the chi dinh dataset:

```powershell
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --lambda-var 0.2 --epochs 30 --models moirai2 --datasets VN_INDEX,snp500
```

Lenh full nen chay sau khi smoke test da pass. Goi y thu tu:

```powershell
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --lambda-var 0.2 --epochs 1 --models moirai2 --datasets VN_INDEX
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --lambda-var 0.2 --epochs 30 --models moirai,moirai2,moirai_moe
```

## 11. Chay ablation cho novelty

De chung minh loss VaR-aware co tac dong that, can chay nhieu gia tri `lambda_var`:

```powershell
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --lambda-var 0.0 --epochs 30
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --lambda-var 0.1 --epochs 30
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --lambda-var 0.2 --epochs 30
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --lambda-var 0.5 --epochs 30
```

Y nghia:

- `lambda_var = 0.0`: chi toi uu volatility MSE, lam baseline noi bo.
- `lambda_var = 0.1`: them rang buoc VaR nhe.
- `lambda_var = 0.2`: muc can bang ban dau.
- `lambda_var = 0.5`: uu tien tail risk manh hon.

Sau do so sanh:

- forecasting accuracy: MSE, MAE, QLIKE.
- VaR backtesting: quantile loss, violation rate, Kupiec, Christoffersen.
- rank trung binh theo dataset, horizon, VaR method.

## 12. Cong doan export ket qua

File export:

```text
experiments/moirai_var_aware/export.py
```

Moi prediction CSV co schema:

```text
dataset, branch, tier, model, time, horizon, log_return, true_volatility, predict_volatility
```

Trong do:

- Head-only: `branch = Moirai_VAR`, `tier = lambda_{lambda_var}`
- Full fine-tune: `branch = Moirai_VAR_FT`, `tier = full_lambda_{lambda_var}`
- `model = moirai | moirai2 | moirai_moe`

Output duoc nen thanh zip:

```text
moirai_var_lambda_{lambda_var}_predictions.zip
moirai_var_full_lambda_{lambda_var}_predictions.zip
```

va luu local vao:

```text
output/modal_moirai_var_loss/
```

## 13. Sau khi train xong

Sau khi co prediction CSV:

1. Giai nen zip trong `output/modal_moirai_var_loss/`.
2. Merge prediction vao file prediction experiment rieng, khong ghi de `output/merged_all_predictions.csv` neu chua muon thay benchmark chinh.
3. Chay lai pipeline thong ke VaR.
4. So sanh `lambda_var = 0.0` voi cac `lambda_var > 0`.

Ket luan chi nen manh neu thoa ca hai dieu kien:

- VaR 1% quantile loss hoac calibration cai thien co y nghia.
- Forecasting accuracy khong suy giam qua lon.
