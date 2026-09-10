The Secretariat of ICEBA2026.

-------------------------------------------------------------------------------

Review result:

Paper ID: 250

Title: MoiraiVaR: VaR-Aware Adaptation of Time-Series Foundation Models for Volatility Forecasting

Reviewing Comment:

Review 1

The paper proposes MoiraiVaR, an adaptation of pretrained Moirai-family time-series foundation models for financial volatility forecasting. A frozen backbone encodes a 60-day return context; mean pooling feeds a two-layer MLP head that predicts realized volatility at horizons {1, 3, 5, 10, 21}. The head is trained on a joint objective combining multi-horizon MSE with a Student-t Value-at-Risk pinball loss (λ_VaR = 0.2). The paper also proposes a three-layer evaluation protocol: a Forecast Sanity Gate that filters near-constant forecasts, a VaR criterion averaged over Normal, Student-t, and Filtered Historical Simulation constructions, and a Pareto plus MCDM (SAW, TOPSIS) ranking layer. Across 26 configurations, 9 equity indices, and 5 horizons, the authors report that MoiraiVaR–Moirai-MoE ranks first under both aggregation methods and both weight policies.

Strengths

- Broad and well-organised experimental sweep. Inclusion of VN-Index and VN30 covers markets that are under-represented in this literature.

- The Forecast Sanity Gate addresses a real and widely ignored failure mode. Deep models trained on volatility MSE frequently collapse to near-constant output; most published work does not test for this.

- Honest limitations section. The authors state the missing rescaling ablation, the unscaled quantile threshold, the absence of purging, and the missing econometric baselines.

- Raw metrics reported alongside composite ranks. Table 2, and the explicit statement that the winner is a compromise rather than a metric-wise dominator, are good scientific hygiene; many MCDM-based papers hide the underlying numbers.

Major Weaknesses

- The scalar-rescaling control is missing, and it is the null hypothesis rather than future work: The authors acknowledge in §5 that they did not ablate against a post-hoc rescaled baseline σ̂ × c. This is not an optional extra experiment. An asymmetric lower-tail pinball penalty primarily inflates the scale of the forecast, and VaR pass rate is monotone in forecast scale.

- The Student-t quantile is missing its variance scale factor. For a Student-t variable normalised to standard deviation σ, the correct α-quantile is σ · t_{α,ν} / √(ν/(ν−2)). At ν ≈ 5 the omitted factor is 1.29, so the threshold sits roughly 29% too far into the tail.

Critically, the same unscaled quantile enters both the training loss and the evaluation. The head is therefore trained to inflate σ̂ in order to compensate for an over-wide threshold, which means the correction may be numerically indistinguishable from the paper's headline contribution. Fixing the factor could remove the reported effect entirely.
- The GARCH results show a signature consistent with a units error: GJR–GARCH reports MSE 0.19112 against 0.02275 for Moirai 2 - a factor of 8.4 is too high.

- Backtest pass rate is the wrong class of criterion: The fraction of Kupiec/Christoffersen tests not rejected is not a calibration measure; it conflates test power with calibration quality, and low-power settings pass more often. The paper should instead report empirical violation rates against nominal α, out-of-sample quantile loss, and the Christoffersen independence component separately.

- λ_VaR is never varied: λ controls exactly the accuracy–risk trade-off the paper is about, yet only λ = 0.2 is evaluated, with no justification for the value. The frontier that matters is the one traced by sweeping λ ∈ {0, 0.05, 0.1, 0.2, 0.5, 1} on a single backbone, plotted against the rescaling frontier of W1. If the λ-frontier dominates the c-frontier, the paper has a result. This is the single experiment I would most strongly recommend.

- Novelty positioning and related work: Training an MLP head on a frozen encoder is a probe, not PEFT in the usual sense (LoRA, adapters, prefix tuning); the terminology overstates the contribution. More seriously, using a pinball loss to learn VaR is not new: Engle & Manganelli (2004), CAViaR is direct prior art and is not cited. Also missing: Corsi (2009) on HAR; Glosten, Jagannathan & Runkle (1993), used but uncited; Taylor (2019) on joint quantile/ES estimation; Zeng et al. (2023) on the real effectiveness of Transformers for time series, which is directly relevant to the degenerate-forecast concern; and the broader TSFM landscape (Chronos, TimesFM, Lag-Llama).

Other comments:

- Fig. 1 is illegible at print size; the text inside the boxes cannot be read.

- Fig. 2 shows only four points, so a reader cannot verify non-dominance. Plot all 21 eligible configurations.

- "1.3 million predictions" is used as an argument for reliability, but the h-day targets overlap heavily across t and the true unit of analysis is 45 blocks. Please rephrase; the current framing overstates the independent evidence.

- "the only architecture ranked first globally by both SAW and TOPSIS" - only one model can rank first, so "only" is trivially true. Rephrase.

- The Gate thresholds s_ŷ/s_y ≥ 0.1 and ρ̄ > 0 are unmotivated and largely duplicate the e_σ and e_ρ criteria already in the decision matrix. Explain why 0.1, or drop the redundancy.

- Report the estimated ν values and any bounding rule. The moment estimator ν = 4 + 6/κ is very unstable for heavy tails; for VN-Index, sample excess kurtosis often exceeds 10, placing ν near 4 where the estimator diverges.
