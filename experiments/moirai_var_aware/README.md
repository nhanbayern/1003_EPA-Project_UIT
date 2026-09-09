# Moirai VaR-Aware Experiment

Experiment nay train Moirai-family cho realized volatility/VaR forecasting bang objective:

```text
loss = MSE(predicted_volatility, true_volatility)
       + lambda_var * QuantileLoss(realized_return, VaR_1%)
```

Ho tro 2 che do:

```text
head: freeze Moirai backbone, chi train MLP head.
full: fine-tune toan bo Moirai backbone + MLP head.
```

Default la `head` de giu hanh vi cu.

## Structure

```text
experiments/moirai_var_aware/
  config.py      # horizons, split info
  data.py        # VolatilityDataset, split loader
  losses.py      # QuantileVaRLoss, VarAwareVolatilityLoss
  modeling.py    # Moirai extractor + MLP head
  train.py       # train/evaluate loop
  export.py      # prediction CSV schema
  runner.py      # remote orchestration
  modal_app.py   # Modal image + CLI entrypoint
  gpu_check.py   # Windows-safe GPU check
```

## Modal Setup

To reproduce the reviewer ablation (including validation/test exports), run the full lambda sweep:

```powershell
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --lambda-sweep 0,0.05,0.1,0.2,0.5,1.0 --epochs 30 --models moirai2 --datasets VN_INDEX
py -3.11 experiments/moirai_var_aware/evaluate_lambda_sweep.py --input-dir output/modal_moirai_var_loss --output-dir output/lambda_ablation
```

The evaluator tunes the scalar factor on validation for the `lambda=0` forecast and reports its untouched test performance against every trained lambda value.

```powershell
py -3.11 -m pip install modal
py -3.11 -m modal setup
py -3.11 -m modal token info
```

Check GPU:

```powershell
py -3.11 -m modal run experiments/moirai_var_aware/gpu_check.py
```

## Head-Only Training

Smoke test:

```powershell
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --lambda-var 0.2 --epochs 1 --models moirai2 --datasets VN_INDEX
```

Full head-only run:

```powershell
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --lambda-var 0.2 --epochs 30 --models moirai,moirai2,moirai_moe
```

## Full Fine-Tuning

Smoke test full fine-tune, nen bat dau voi batch nho:

```powershell
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --tuning-mode full --lambda-var 0.2 --epochs 1 --batch-size 8 --backbone-lr 1e-5 --models moirai2 --datasets VN_INDEX
```

Full fine-tune 1 model truoc:

```powershell
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --tuning-mode full --lambda-var 0.2 --epochs 30 --batch-size 8 --backbone-lr 1e-5 --models moirai2
```

Full fine-tune ca Moirai-family:

```powershell
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --tuning-mode full --lambda-var 0.2 --epochs 30 --batch-size 8 --backbone-lr 1e-5 --models moirai,moirai2,moirai_moe
```

Neu OOM, giam `--batch-size` xuong `4`.

## Output

Output zip nam o:

```text
output/modal_moirai_var_loss/
```

Head-only zip:

```text
moirai_var_lambda_0.2_predictions.zip
```

Full fine-tune zip:

```text
moirai_var_full_lambda_0.2_predictions.zip
```

Full fine-tune export dung:

```text
branch = Moirai_VAR_FT
tier = full_lambda_0.2
```

## Ablation

Head-only:

```powershell
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --lambda-var 0.0 --epochs 30
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --lambda-var 0.1 --epochs 30
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --lambda-var 0.2 --epochs 30
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --lambda-var 0.5 --epochs 30
```

Full fine-tune nen lam co chon loc vi ton GPU hon:

```powershell
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --tuning-mode full --lambda-var 0.0 --epochs 30 --batch-size 8 --models moirai2
py -3.11 -m modal run experiments/moirai_var_aware/modal_app.py --tuning-mode full --lambda-var 0.2 --epochs 30 --batch-size 8 --models moirai2
```
