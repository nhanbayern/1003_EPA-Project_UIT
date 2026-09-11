# ICEBA Codebase Scientific-Setup Audit

**Date:** 2026-09-10  
**Commit audited:** `6030acb` (`fix: prevent volatility target leakage`)  
**Scope:** codebase, notebooks, data builders, split/evaluation logic, manifests, and representative output artifacts. Paper prose was used only to identify the intended experiment contract; this is not a paper-writing review.  
**Mutation policy:** read-only audit. No source, notebook, CSV, model, Modal job, or historical artifact was changed.

## Executive verdict

**BLOCKER — the experiment is not yet scientifically ready for retraining or reporting.**

The latest commit corrected the main Python target builders, but the codebase is not yet one reproducible experiment. There are still incompatible target definitions across active/stale pipelines, no end-to-end proof that every family uses the same split/origin contract, and the available benchmark artifacts were generated before the correction.

Strong evidence:

- `output/canonical_final_audited_normalized_v2/merged_all_predictions.csv` was generated at 15:42, before commit `6030acb` at 16:27.
- Its first DAX row (`time=2022-03-08`, `h=1`) has `true_volatility=1.53598266`.
- From `dataset/DAX_40.csv`, the correct future h=1 standard deviation is `0.0` (`ddof=0`); `1.53598266` is the old backward rolling target.

Thus existing model metrics, VaR results, statistical summaries, MCDM rankings, plots, and winners must not be treated as results from the corrected experiment.

## Scientific setup contract used

For forecast origin `t`, lookback 60, and horizon `h`:

```python
x = returns[t - 59 : t + 1]
future = returns[t + 1 : t + h + 1]
target_vol = np.std(future, ddof=0)
time = timestamps[t]
log_return = returns[t + 1]
```

Input and target index sets must be disjoint. The VaR observation is always `r[t+1]`. Valid horizons are `{1, 3, 5, 10, 21}`.

## Inventory and verification performed

- Inspected `experiments/`, `model/`, `AAAI24_GARCH_NN_Reproduction/`, `stats_analysis/`, `ultility/`, `dataset/`, and `output/`.
- Inspected required audit/reproduction notes under `docs/audit/` and `docs/modal/`.
- Parsed 102 Python files with `ast.parse`; zero syntax errors.
- Inspected notebook code cells directly.
- Counted 2,962 CSV files below `output/`.
- Inspected three versioned merged artifacts structurally and by raw-data recomputation.
- Ran an in-memory monotonic synthetic-series contract check.
- Inspected Git status, history, blame, and target-fix commits.

## Findings

| Area | Concrete code/artifact | Evidence | Status | Severity | Consequence | Required action |
|---|---|---|---|---|---|---|
| Main target builders | `experiments/moirai_var_aware/data.py:36-48`; Transformer v1/v2; hybrid/wavelet datasets | Main Python builders now use future `std` and `log_return=r[t+1]`. | PASS (source only) | Medium | No regenerated result proves the corrected source was used. | Add runtime assertions and rebuild. |
| Moirai baseline | `model/Morai based/notebooks/kaggle_notebook.ipynb`, cell 4 lines 31-40 | Uses `sqrt(mean(future_returns**2))` (RMS), not future standard deviation. | FAIL | BLOCKER | Baseline and MoiraiVaR solve different target tasks. | Align/remove baseline, then retrain. |
| Moirai stale source | `model/Morai based/notebooks/kaggle_notebook.py:69-80` | Uses `returns[t+h-lookback:t+h]`, a backward rolling window. | FAIL | BLOCKER | Regeneration can reintroduce target overlap. | Align source/generator/notebook and test parity. |
| Legacy Moirai notebook | `model/Morai based/notebooks/1003-moirai-based-models.ipynb`, cell 4 lines 28-43 | Input is shifted; target is backward/offset; `log_return` is `returns[t]`. | FAIL | BLOCKER | Wrong origin, target overlap, and wrong VaR return. | Exclude or repair completely. |
| AAAI24/GARCH-NN target | `AAAI24_GARCH_NN_Reproduction/core/data_processor.py:109-125,177-240` | Uses historical rolling volatility and `v[i+seq_len+h-1]`; does not compute future `std(r[t+1:t+h+1])`. | FAIL | BLOCKER | Econometric baselines are not comparable. | Rebuild labels/evaluator from raw returns. |
| GARCH exporter | `model/GARCH based/utils.py:92-129` | Current export computes future `std`, but docstring lines 95-97 still describes offset rolling volatility; caller alignment is not asserted. | WARNING | High | Correct source may still emit misaligned artifacts. | Remove stale semantics; assert calendar/index mapping. |
| Modal preparation | `experiments/all_models_modal/modal_app.py:233-253` | Corrects one exact text pattern and sets origin/next return via string replacement. | WARNING | High | Unmatched notebook variants can silently remain incompatible. | Use a shared builder and assert prepared notebook contents. |
| Split/purge | `experiments/moirai_var_aware/data.py:65-89`; notebook generators | Moirai purges by `origin_position`; other notebooks borrow 60 context rows and rely on local truncation. | WARNING | High | Split semantics are implicit and not uniformly auditable. | Emit raw origin/target indices and assert boundaries. |
| Split counts | `model/transformer based/version_2/create_notebook_v2.py:240-264` | Counts are applied after filtering/dropna; no fail-closed count check. | WARNING | High | Actual populations can differ by market/family. | Publish retained counts and reject mismatches. |
| DAX split example | `dataset/DAX_40.csv` plus current split code | Raw rows 4,567; valid origins 4,038; fixed counts sum to 4,059. Current selection retains 951 test origins vs declared 972. | WARNING | High | Benchmark population is not the declared population. | Freeze a per-dataset split manifest. |
| Target artifact | `output/canonical_final_audited_normalized_v2/merged_all_predictions.csv` | DAX h=1 stores old rolling target 1.53598266 instead of future h=1 std 0.0; artifact predates latest fix. | FAIL | BLOCKER | All dependent metrics/risk/ranks are invalid for the corrected task. | Quarantine and regenerate. |
| Benchmark size | Same artifact vs ICEBA run specification | Current merge has 2,566,350 rows and 41 configurations across 9 datasets; stored ICEBA claim is 1,322,330 rows and 26 configurations. | FAIL | High | Available output is a different experiment population. | Freeze one config manifest and rebuild. |
| Truth consistency | Current v2 merged artifact | No duplicate full keys, but 41,530 `(dataset,time,horizon)` truth groups differ across configurations; max target difference about `2.15e-7`. | WARNING | Medium | Common truth is not proven by the artifact itself. | Re-run merger validation after rebuild and document tolerance. |
| Student-t training | `experiments/moirai_var_aware/losses.py:17-23,53-62` | Standardized Student-t factor `sqrt((nu-2)/nu)` is present. | PASS (source only) | Medium | Old runs used older conventions. | Persist `nu`, formula version, and commit hash. |
| Student-t `nu` | `experiments/moirai_var_aware/runner.py:59-67`; `stats_analysis/analyzer.py:99-119` | Runner uses train samples; analyzer uses train source for fixed datasets but can fall back to merged rows if source is unavailable. | WARNING | High | Fallback may fit `nu` using test rows. | Fail closed; persist `nu_source_split=train`. |
| FHS | `stats_analysis/risk.py:97-119` | Residual history uses rows before current `t`. | PASS (conditional) | Medium | Depends on upstream row alignment. | Require canonical origin metadata before FHS. |
| VaR evaluation | `stats_analysis/risk.py:75-85,188-227` | Student-t correction and violation comparison exist. | PASS (conditional) | Medium | Depends on `r[t+1]`, train-only `nu`, and rebuilt rows. | Add a preflight contract validator. |
| MCDM selection | `stats_analysis/run_mcdm_evaluation.py:541-550,570-675,1099-1109` | SAW/TOPSIS use the supplied merged evaluation matrix; validation-only selection is not required. | FAIL | BLOCKER | Test-set ranking can become model selection. | Freeze eligibility/weights/model set on validation; rank test once. |
| Lambda/rescaling control | `experiments/moirai_var_aware/evaluate_lambda_sweep.py:18-23,35-56` | Rescaling factor is fit on validation and scored on test; code supports `{0,.05,.1,.2,.5,1}`. | PASS (implementation) | Medium | No corrected post-fix run is evidenced. | Run and archive after rebuild. |
| Reproducibility provenance | `stats_analysis/merge_canonical_predictions.py:124-140` | Manifest stores hashes/commit/seeds/command, but rows lack mandatory origin index, target indices, scaler fit range, `nu` range, and selection split. | WARNING | High | Later audit cannot prove how each row was generated. | Extend manifest/schema and make fields mandatory. |

## Per-pipeline scientific decision

| Pipeline | Decision |
|---|---|
| Moirai | **BLOCKED:** active variants use RMS or rolling targets. |
| MoiraiVaR | **Source corrected, experiment unverified:** retrain and add provenance. |
| Transformer v1/v2 | **Source corrected, experiment unverified:** split/output assertions missing. |
| GARCH-LSTM | **Not comparable yet:** exporter is closer to contract, but caller alignment/artifacts are unverified. |
| GARCH-Autoformer | **Source corrected, experiment unverified:** split assertions absent. |
| Wavelet-Autoformer | **Source corrected, experiment unverified:** old artifacts must be rebuilt. |
| GARCH/GJR/FI-GARCH and AAAI24 | **Blocked:** current reproduction target is historical rolling volatility. |
| Modal | **Blocked pending local preflight:** text patch is brittle. |
| VaR | **Conditionally sound in source:** Student-t/FHS logic is plausible; row and parameter provenance unresolved. |
| MCDM | **Blocked for model-selection claims:** no enforced validation-only gate. |

## Exact origin/horizon requirements

```python
assert len(x) == 60
assert np.array_equal(x, returns[t-59:t+1])
assert np.array_equal(future, returns[t+1:t+h+1])
assert not set(range(t-59,t+1)) & set(range(t+1,t+h+1))
assert row.time == timestamps[t]
assert row.log_return == returns[t+1]
assert row.target_end_exclusive == t+h+1
assert max(target_indices) < split_end
assert scaler_fit_max_index < train_end
assert nu_source_split == "train"
```

## Git-history assessment

- Incompatible target paths entered through the July model/pipeline commits (`984466f`, `17971a8`, `c34fe01`, `a91cb7d`, `d330b28`, `76bba0c`).
- `623987f` on 2026-09-09 added common-schema/provenance code but retained rolling targets.
- `08276c2` attempted broad alignment and a lambda sweep but did not make the codebase consistent.
- `6030acb` on 2026-09-10 corrected the main Python builders, GARCH export, and Modal replacement text.
- Current state is an **incomplete repair**: stale generators/notebooks, AAAI24 code, and all pre-fix artifacts remain.

## Required repair sequence before a valid experiment

1. Import one target builder from every active family; remove/repair RMS, rolling, and shifted Moirai/AAAI24 paths.
2. Add monotonic synthetic tests and source-vs-notebook/Modal parity tests.
3. Add split manifests with raw origin/target indices, retained counts, and purge status.
4. Verify train-only scaler and Student-t `nu` provenance.
5. Rebuild every prediction after the target fix.
6. Recompute truth from raw close prices; validate common keys before merging.
7. Select models, lambda, thresholds, eligibility, and weights using validation only.
8. Run one final frozen test evaluation, then generate metrics, VaR, statistics, and MCDM.

## Artifacts that must not be reused

Quarantine rather than overwrite:

- `output/09-09-2026/`
- `output/canonical_final/`
- `output/canonical_final_audited_normalized/`
- `output/canonical_final_audited_normalized_v2/`
- dependent `output/stats_analysis/`, FHS, winners, plots, and MCDM outputs

These may remain as legacy evidence, but not as corrected benchmark results.

## Limitations

- Torch is unavailable locally, so dataset classes were statically inspected and formulas checked in memory; no full model runtime was executed.
- Modal was not invoked and no model was retrained.
- CSV review used structural scans plus targeted raw-data recomputation; it did not load all 2,962 files simultaneously.

## Mandatory decision gate

**HOLD — REPAIR REQUIRED**

