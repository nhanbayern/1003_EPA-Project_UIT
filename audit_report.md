# Volatility forecasting pipeline audit

**Audit date:** 2026-09-10  
**Scope:** the requested research workspace and existing `output/final/` artifacts.  
**Mode:** read-only audit; no source or result files were modified or deleted.

## Protocol audited

For an origin `t`, `lookback=60` means `X_t = r[t-59:t+1]`. For horizon `h`, the future target is `r[t+1:t+h+1]` and the requested realized volatility is `std(r[t+1:t+h+1])`. Output `time` is the timestamp at `t`, and `log_return=r[t+1]`. Therefore `h=1,3,5,10,21` means respectively the next 1, 3, 5, 10 and 21 observations after the origin—not a backward/offset rolling window.

## Executive verdict

**BLOCKED: do not retrain or publish/compare the existing benchmark yet.** The core Transformer, modified-Autoformer and MoiraiVaR data builders use a backward-looking rolling target that overlaps the input. The Modal notebook-preparation code also injects that same wrong formula into legacy notebooks. This invalidates realized-volatility labels and all downstream MSE/MAE/QLIKE, VaR, MCDM and model-ranking results derived from those artifacts. GARCH export alignment is also not proven correct and should be rebuilt under one canonical origin-aligned schema.

## Findings table

| Pipeline | File | Dòng/công thức | Trạng thái | Mức độ | Hệ quả |
| --- | --- | --- | --- | --- | --- |
| Moirai baseline | `experiments/all_models_modal/modal_app.py` | 243–248; notebook rewrite changes target/export cells | WARNING | HIGH | The runner mutates a prepared copy, but baseline notebook source itself was not fully audited from Python-only files. Must verify generated notebook uses canonical future target before reuse. |
| Moirai baseline | `experiments/all_models_modal/modal_app.py` | 151–159, 179–193 | WARNING | MEDIUM | Normalizer assumes transformer/Moirai exports already contain origin `t` and `r[t+1]`; no assertion checks timestamp-to-dataset calendar mapping. |
| MoiraiVaR | `experiments/moirai_var_aware/data.py` | 36–42: `x=r[t-59:t+1]`, but `target_window=r[t+h-60:t+h]` | **FAIL / BLOCKER** | CRITICAL | Target overlaps input for every h; h=1 is effectively a backward window ending at `t`, not `r[t+1]`. All MoiraiVaR forecasts/losses are invalid under the requested protocol. |
| MoiraiVaR | `experiments/moirai_var_aware/data.py` | 48 | PASS | MEDIUM | `log_return` is explicitly `returns[t+1]`, subject to the target/output rows being regenerated with the corrected target. |
| MoiraiVaR split | `experiments/moirai_var_aware/data.py` | 73–88 | WARNING | HIGH | Uses `origin_position` and `end-max_horizon`, which is directionally a purge, but `origin_position` is the enumeration position of valid indices, not the raw return index. This needs an assertion and exact boundary test. |
| Transformer v1 | `model/transformer based/version_1/dataset.py` | 26, 36, 44–50 | **FAIL / BLOCKER** | CRITICAL | Input is `returns[t-60:t]` (ends at `t-1`), while target window is `returns[t+h-60:t+h]`; it overlaps input and is shifted relative to the stated origin convention. `y_ret` is also `returns[t+h-1]`, not consistently `r[t+h]`. |
| Transformer v2 | `model/transformer based/version_2/dataset.py` | 26, 36, 40–43 | **FAIL / BLOCKER** | CRITICAL | Input is origin-inclusive, but volatility target is the same overlapping backward formula. Return target is future-aligned (`t+1:t+h+1`), creating an inconsistent joint target. |
| GARCH-LSTM Hybrid | `model/GARCH based/dataset.py` | 30–43 | WARNING | HIGH | Sliding target is one-step-ahead, but `prepare_series` creates a shifted past volatility and the caller splits on raw close positions. Semantics may be causal for h=1, but this is not demonstrated for common multi-horizon outputs. |
| GARCH-LSTM Hybrid export | `model/GARCH based/utils.py` | 102–123 | **FAIL / BLOCKER** | CRITICAL | Export reconstructs `target_window=joined[target_end-60:target_end]`, excluding the endpoint and using a concatenated history whose exact index is not asserted. It does not directly compute `std(future_returns[i:i+h])`; existing `actual_vol` cannot be trusted without rebuild. |
| GARCH-Autoformer | `model/modify_autoformer/Hybrid GARCH-Autoformer/dataset.py` | 36–42 | **FAIL / BLOCKER** | CRITICAL | Same overlapping target formula as Transformer v2. GARCH features may be causal, but the supervised label is invalid. |
| Wavelet-Autoformer | `model/modify_autoformer/Wavelet Transform/dataset.py` | 36–43 | **FAIL / BLOCKER** | CRITICAL | Same overlapping target formula as Transformer v2. Wavelet preprocessing cannot rescue an invalid target. |
| GARCH/GJR/FI-GARCH | `model/GARCH based/utils.py` | 57–75 | WARNING | HIGH | `volatility=returns.rolling(60).std().shift(1)` is causal for a volatility-at-time feature, but it is a different object from requested future realized volatility. Evaluation must recompute labels from future returns, not reuse this feature. |
| AAAI24 GARCH/NN reproduction | `AAAI24_GARCH_NN_Reproduction/core/data_processor.py` | 109–120, 167–172 | WARNING | HIGH | Rolling volatility is computed from historical returns and split after preprocessing; this can be valid as a feature, not as the requested future target. Split boundaries and one-step target alignment require a dedicated corrected evaluator. |
| AAAI24 windows | `AAAI24_GARCH_NN_Reproduction/core/data_processor.py` | 177–240 | WARNING | HIGH | Single-horizon target uses `r[i+seq_len+h-1]`; multi-horizon target uses precomputed `v[i+seq_len+h-1]`. This is not demonstrably `std(r[t+1:t+h+1])` and has no explicit origin timestamp contract. |
| Modal target injection | `experiments/all_models_modal/modal_app.py` | 241–245: replaces future RMS with `target_window=r[t+h-lookback:t+h]` | **FAIL / BLOCKER** | CRITICAL | Modal retraining would reproduce the leakage in prepared notebooks. The orchestration layer itself must be fixed before any retrain. |
| Modal export | `experiments/all_models_modal/modal_app.py` | 247–252 | WARNING | HIGH | Export rewrite attempts to set `origin_time=df.time[t]` and `next_return=df.log_return[t+1]`, but uses text replacement rather than a typed schema/assertion. A missed notebook variant can silently keep old semantics. |
| Normalization | `experiments/all_models_modal/modal_app.py` | 93–201 | WARNING | HIGH | Column normalization is mostly schema-preserving, but it trusts source `time`, `horizon`, `log_return`, and true-volatility columns. There are no checks for origin mapping, duplicate `(dataset,branch,tier,model,time,horizon)` keys, or target recomputation. |
| Student-t ν | `stats_analysis/risk.py` | 1–31 | WARNING | HIGH | `stats.t.fit(arr)` is available, but this file alone does not prove `arr` is train-only. The audit found no global contract/assertion tying ν fitting to the training split. Existing ν artifacts must be traced and regenerated train-only. |
| FHS | `stats_analysis/risk.py` | 97–119 | PASS (conditional) | MEDIUM | Residual quantile uses `shift(1).rolling(...)`, which excludes the current observation and is causal if the input residual stream is origin-aligned. It is not safe if the upstream `log_return`/volatility rows are misaligned. |
| Normal VaR | `stats_analysis/risk.py` | 61–66 | PASS (conditional) | MEDIUM | Correct quantile transformation for a supplied origin-aligned volatility forecast; horizon semantics remain conditional on the row's `r[t+1]`. |
| Student-t VaR | `stats_analysis/risk.py` | 75–85 | PASS (conditional) | MEDIUM | Standardized-t variance correction is present. ν provenance remains unresolved; do not treat current results as clean until ν is proven train-only. |
| VaR evaluation | `stats_analysis/run_mcdm_evaluation.py` | 295–331 | WARNING | HIGH | Violations compare `log_return` to VaR thresholds by case. This is correct only when `log_return=r[t+1]`; no runtime assertion checks it, and h>1 rows can otherwise be misinterpreted. |
| Metrics | `ultility/metrics.py` and `stats_analysis/run_mcdm_evaluation.py` | `compute_mse_qlike`; case aggregation | **FAIL / BLOCKER** | CRITICAL | Metrics consume stored `true_volatility`; because core labels are overlapping historical windows, MSE/MAE/QLIKE are not measurements of future realized volatility. |
| MCDM SAW | `stats_analysis/run_mcdm_evaluation.py` | 570–612 | WARNING | HIGH | Min-max normalization is computed over the decision matrix; this is acceptable for descriptive ranking but must be frozen on a predeclared case/model set. No evidence here that test data was not used to choose weights/model. |
| MCDM TOPSIS | `stats_analysis/run_mcdm_evaluation.py` | 615–675 | WARNING | HIGH | Same issue: ideal best/worst and normalization use the supplied matrix. Need an explicit train/validation selection stage and untouched test ranking stage. |
| Constant-forecast filter | `stats_analysis/run_mcdm_evaluation.py` | 678–710 | PASS (conditional) | MEDIUM | Eligibility checks std-ratio error and tracking correlation, so near-constant forecasts are screened. Threshold and scope need to be predeclared and applied identically across cases. |
| MCDM common set | `stats_analysis/run_mcdm_evaluation.py` | 541–549, 666–675 | WARNING | HIGH | Inner merges enforce common rows across VaR cases and SAW/TOPSIS, but no audit manifest proves the same model/case set was fixed before test results were seen. |
| Dataset names | `experiments/all_models_modal/modal_app.py`; `model/GARCH based/utils.py` | `_dataset_key`, `_normalize_dataset_name` | WARNING | MEDIUM | Multiple normalization conventions exist (`snp500`, filename stems, family prefixes). Add a single canonical dataset registry and reject unknown aliases. |
| CSV keys | sampled `output/final/.../*_predictions.csv` | programmatic read-only scan | WARNING | MEDIUM | In the first 200 prediction CSVs checked, no duplicate keys were detected under available schema columns; this is not a proof for all files, and current key uniqueness is not enforced in loaders. |
| Existing horizons | sampled `output/final/...` | programmatic read-only scan | WARNING | HIGH | Wavelet artifacts expose `[1,3,5,10,21]`, but their labels are generated by the overlapping formula. Horizon names exist; horizon semantics are invalid until labels are rebuilt. |

## Horizon mapping audit

Under the requested protocol:

| h | Future returns | Correct target | VaR return |
|---:|---|---|---|
| 1 | `r[t+1]` | `std([r[t+1]])` (usually 0 with `ddof=0`) | `r[t+1]` |
| 3 | `r[t+1:t+4]` | `std(r[t+1:t+4])` | `r[t+1]` |
| 5 | `r[t+1:t+6]` | `std(r[t+1:t+6])` | `r[t+1]` |
| 10 | `r[t+1:t+11]` | `std(r[t+1:t+11])` | `r[t+1]` |
| 21 | `r[t+1:t+22]` | `std(r[t+1:t+22])` | `r[t+1]` |

The repeated implementation `returns[t+h-lookback:t+h]` instead represents a backward 60-observation window ending at `t+h-1`; it overlaps the input for all h≤60. It must not be described as future realized volatility.

## Blocker list

1. Correct the target construction in MoiraiVaR, Transformer v1/v2, GARCH-Autoformer and Wavelet-Autoformer.
2. Remove the wrong target injection in Modal notebook preparation.
3. Rebuild GARCH/AAAI24 evaluation labels and exports from the canonical future-return formula; do not reuse existing `actual_vol`/`true_volatility` columns.
4. Rebuild all derived metrics, VaR tables, statistical tests, SAW/TOPSIS rankings and consolidated results after corrected predictions exist.
5. Establish train-only provenance for scaler parameters and Student-t ν; current source inspection does not establish this end-to-end.

## Files that need correction before retraining

- `experiments/moirai_var_aware/data.py`
- `experiments/all_models_modal/modal_app.py`
- `model/transformer based/version_1/dataset.py`
- `model/transformer based/version_2/dataset.py`
- `model/modify_autoformer/Hybrid GARCH-Autoformer/dataset.py`
- `model/modify_autoformer/Wavelet Transform/dataset.py`
- `model/GARCH based/utils.py` and the GARCH evaluation/export caller
- `AAAI24_GARCH_NN_Reproduction/core/data_processor.py` and its evaluation caller
- the canonical evaluation/normalization layer under `stats_analysis/`

## Correct code/formula proposal

```python
# origin t, lookback=60
x = returns[t - lookback + 1 : t + 1]
future = returns[t + 1 : t + h + 1]
assert len(x) == lookback
assert len(future) == h
target_vol = np.std(future, ddof=0)
log_return = returns[t + 1]
time = timestamps[t]
```

For a split boundary, generate windows in global time but assign a sample to a split only when its origin and every target observation are inside that split's allowed evaluation interval. If the split is defined by origin ranges, purge at least `max_horizon` origins at the end of train/validation, and assert `max(target_indices) < next_split_start`.

## Do we need to retrain Modal?

**Yes, if the goal is a valid benchmark under this protocol.** Existing neural predictions should be treated as contaminated by invalid labels; rerunning only metrics is insufficient. Modal should be retrained only after target/export/schema assertions pass locally on a tiny synthetic series.

## Results that must be discarded

Discard or mark non-comparable every result whose `true_volatility` was generated by the overlapping formula, including existing MoiraiVaR, Transformer v1/v2, GARCH-Autoformer and Wavelet-Autoformer prediction CSVs; all dependent MSE/MAE/QLIKE tables; VaR backtests based on those rows; and SAW/TOPSIS/consolidated winners derived from them. GARCH/AAAI24 artifacts remain **unverified**, not publishable, until their actual-volatility alignment is reconstructed.

## Benchmark-after-fix plan

1. Add canonical sample generation and run synthetic index tests for h=1,3,5,10,21.
2. Rebuild each family with identical origin, horizon and split manifests.
3. Fit every scaler only on train observations; serialize fit range/mean/scale and assert no validation/test rows were passed to `fit`.
4. Fit Student-t ν on train residuals/returns only; store `nu_source_split=train` in artifacts.
5. Generate predictions with one schema and validate origin timestamp, `r[t+1]`, target values, horizon, and unique key.
6. Compute MSE/MAE/QLIKE against recomputed future realized volatility only.
7. Run Normal, Student-t and FHS VaR using `r[t+1]`; FHS residual windows must be strictly prior to the origin.
8. Select hyperparameters/lambda/weights on validation only. Freeze the model/case set and MCDM weights before one final test evaluation.
9. Run statistical tests and SAW/TOPSIS on the frozen, common case set; archive manifests and hashes.

## Assertions/tests to prevent recurrence

```python
assert np.array_equal(x, returns[t-59:t+1])
assert np.array_equal(future, returns[t+1:t+h+1])
assert not set(range(t-59, t+1)) & set(range(t+1, t+h+1))
assert row["time"] == timestamps[t]
assert row["log_return"] == returns[t+1]
assert row["target_end_exclusive"] == t+h+1
assert target_indices.max() < split_end
assert scaler_fit_max_index < train_end
assert nu_source_split == "train"
assert prediction_df.duplicated(KEY_COLUMNS).sum() == 0
assert set(prediction_df.horizon.unique()) <= {1,3,5,10,21}
```

Also add a test that deliberately compares the implementation output against a hand-built monotonic return sequence; the old overlapping formula must fail that test. Add a manifest field for `origin_index`, `target_indices`, `split`, `scaler_fit_range`, `nu_fit_range`, and `selection_split`.

## Audit limitations

This was a read-only static and artifact audit. Notebook code was not executed, Modal was not invoked, and the full output tree was not exhaustively loaded into memory. The sampled duplicate scan covered the first 200 matching prediction CSVs and found no duplicate keys under the available columns; this is an advisory check, not a PASS for the entire artifact tree.

## Decision gate

**Audit decision before remediation: BLOCKER — no retrain.** The requested remediation has now been applied to the source data builders/export layer, but existing artifacts remain invalid and must not be reused. Modal retraining/rebuild still requires a separate execution approval after the post-fix tests pass.

## Remediation applied after the audit

The following source files were updated after the read-only audit: MoiraiVaR, Transformer v1/v2, GARCH-Autoformer, Wavelet-Autoformer, GARCH CSV export, and Modal notebook preparation. They now use `x=returns[t-59:t+1]`, `future=returns[t+1:t+h+1]`, `std(future, ddof=0)`, and `log_return=returns[t+1]`. Existing CSV outputs were deliberately not modified.
