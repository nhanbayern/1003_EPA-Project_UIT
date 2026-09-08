# Báo cáo khả thi và Integrity Checkpoint 2.5

**Dự án:** Volatility forecasting + VaR-aware MCDM  
**Ngày kiểm tra:** 2026-07-28  
**Target venue:** ICEBA 2026 — *The 7th International Conference on Engineering, Physics, MEMS-Biosensors and Applications*  
**Trạng thái ARS:** **FAIL — chưa đủ điều kiện viết thành bản submission-ready**

> Ghi chú định danh: người dùng ghi “ICEBAA”; các yêu cầu trong plan (Springer Book
> Chapter, double blind, 4–6 trang) khớp với trang template của ICEBA 2026. Nếu
> “ICEBAA” là một hội nghị khác, cần thay URL/CFP chính xác trước khi dàn trang.

## 1. Kết luận điều hành

Ý tưởng nghiên cứu **có thể phát triển thành paper**, nhưng bản hiện tại **không nên
nộp**. Vấn đề không chỉ là cách viết: mã tạo target của MoiraiVaR có shortcut và
leakage theo thời gian; AutoCorrelation của GARCH-Autoformer không dùng delay đã
chọn và tham số `n_heads` không có tác dụng; kết quả MCDM cũ không được tái tạo hoàn
toàn từ CSV vừa được xác nhận là source of truth; phép kiểm định nhị thức trên 15
model family không có đơn vị quan sát độc lập phù hợp.

Đánh giá:

| Hạng mục | Điểm | Nhận định |
|---|---:|---|
| Ý tưởng/nội dung tiềm năng | 6.5/10 | Risk-sensitive evaluation trên 9 thị trường là hướng có giá trị |
| Khớp scope ICEBA | 5.5/10 | Có thể vào track Computing/Modeling/ML, nhưng tài chính không phải trọng tâm |
| Tính đúng của bằng chứng hiện tại | 2.0/10 | Có lỗi blocking ở target, split và kiến trúc |
| Khả năng tái lập | 3.0/10 | Có CSV tổng hợp nhưng thiếu manifest/log, sản phẩm dẫn xuất cũ lệch nguồn |
| Khớp định dạng | 3.0/10 | PDF hiện tại 9 trang; venue yêu cầu ngắn hơn |
| Sẵn sàng nộp ngay | **2.0/10** | Không nên nộp |
| Khả thi sau sửa và chạy lại | **7.0/10** | Khả thi nếu thu hẹp contribution và tái lập toàn pipeline |

## 2. Nguồn được chốt

Nguồn dự báo duy nhất theo xác nhận của người dùng:

`output/merged_predictions/merged_all_predictions_24_7.csv`

- SHA-256:
  `04A9D5C94180EAF4DD1ED9B78E5623383466750C4FE6F5B7609FCBBA3E3A6C66`
- 1,322,330 dòng.
- 9 thị trường.
- 5 horizons.
- 26 cấu hình, thuộc 5 branch dữ liệu:
  `GARCH`, `Transformers`, `Moirai`, `Moirai_VAR`,
  `modify_autoformer`.
- 1,170 block cấu hình × thị trường × horizon:
  \(26\times9\times5=1170\).

Các tệp trong `output/mcdm_results/` chỉ được xem là **derived artifacts**, không
phải source of truth.

## 3. Tái tính độc lập từ source of truth

Tôi đã tải CSV chuẩn, chuẩn hóa
`modify_autoformer -> modified_autoformer`, rồi tái tính forecast metrics,
Student-t VaR backtests, tracking metrics, Sanity Gate, SAW và TOPSIS trong bộ nhớ
bằng chính implementation của repository.

### 3.1 Sanity Gate

- Tổng cấu hình: 26.
- Đủ điều kiện MCDM: 21.
- Bị loại: 5.
- Năm cấu hình bị loại:
  - Reformer Tier 3.
  - Informer Tier 1.
  - Autoformer Tier 3.
  - Vanilla Tier 3.
  - Informer Tier 3.

Do đó các câu trong bản thảo nói “12 cấu hình bị loại” và
“GARCH-Autoformer bị loại” là sai so với source of truth.

### 3.2 Ranking 50:50

| Hạng tổng hợp | Cấu hình | SAW | TOPSIS | Trung bình |
|---:|---|---:|---:|---:|
| 1 | GARCH-Autoformer Tier 2 | 1 | 1 | 1.0 |
| 2 | MoiraiVaR–Moirai 2, λ=0.2 | 2 | 3 | 2.5 |
| 3 | Moirai-MoE | 3 | 4 | 3.5 |
| 4 | GARCH-Autoformer Tier 1 | 6 | 2 | 4.0 |
| 5 | MoiraiVaR–Moirai-MoE, λ=0.2 | 4 | 5 | 4.5 |

### 3.3 Ranking 30:70

| Hạng tổng hợp | Cấu hình | SAW | TOPSIS | Trung bình |
|---:|---|---:|---:|---:|
| 1 | GARCH-Autoformer Tier 2 | 1 | 1 | 1.0 |
| 2 | Reformer Tier 2 | 3 | 2 | 2.5 |
| 3 (đồng hạng) | Autoformer Tier 2 | 5 | 3 | 4.0 |
| 3 (đồng hạng) | GARCH-Autoformer Tier 1 | 2 | 6 | 4.0 |
| 3 (đồng hạng) | Wavelet-Autoformer Tier 2 | 4 | 4 | 4.0 |

### 3.4 Sai lệch với run MCDM cũ

Run `MCDM20260725180350` dùng các thống kê có manifest trỏ tới
`output/merged_all_predictions.csv`, một đường dẫn hiện không tồn tại. Manifest cũ
ghi 13 dataset, trong khi source of truth hiện có 9 thị trường.

Khi tái tính từ CSV chuẩn:

- MSE/MAE/QLIKE và tracking metrics tái tạo gần như chính xác đến sai số số thực.
- VaR 1% violation rate thay đổi ở 21/26 cấu hình.
- VaR 5% violation rate thay đổi ở 22/26 cấu hình.
- VaR pass rate thay đổi ở 2 cấu hình tại 1% và 1 cấu hình tại 5%.
- Ranking 50:50 đổi từ đồng hạng đầu thành GARCH-Autoformer Tier 2 đứng đầu rõ ràng.

Vì vậy mọi bảng/hình phải được tạo lại trực tiếp từ CSV chuẩn cùng một manifest mới.

## 4. Các lỗi blocking

### B1. MoiraiVaR có target shortcut và cross-split leakage

Trong `experiments/moirai_var_aware/data.py`:

- Dòng 30 tạo input `returns[t-lookback:t]`.
- Dòng 33–38 tạo target cho horizon \(h\) từ cửa sổ kết thúc tại
  `t+h-1` nhưng không lấy phần tử cuối.
- Với \(h=1\), target chính là độ lệch chuẩn của **đúng cửa sổ input**. Đây là
  nowcast có thể tính trực tiếp từ đầu vào, không phải one-step-ahead forecast.
- Với \(h=3\), target chỉ dịch thêm 2 returns; nói chung horizon \(h\) chỉ đưa
  \(h-1\) quan sát tương lai vào target.
- Dòng 62 tạo toàn bộ sample trước, rồi dòng 68–70 mới cắt `Subset`. Các sample
  cuối train với horizon lớn dùng returns thuộc validation, tạo cross-split leakage.
- Dòng 35–38 giữ lại các sample thiếu future horizon bằng cách dùng cửa sổ cuối
  chuỗi thay vì loại chúng.

Hệ quả: hiệu quả của MoiraiVaR, đặc biệt ở horizon 1, không đủ hợp lệ để dùng làm
claim kiến trúc hoặc so sánh dự báo.

### B2. GARCH-Autoformer không thực hiện đúng AutoCorrelation được mô tả

Trong
`model/modify_autoformer/Hybrid GARCH-Autoformer/models.py`:

- Dòng 85 chọn `delays` bằng `topk`.
- Dòng 90 đọc `delay`.
- Dòng 91 lại luôn `torch.roll(..., shifts=-1)`, không dùng `delay`.
- `n_heads` xuất hiện ở constructor dòng 116 và 136 nhưng không được truyền vào
  bất kỳ attention/head implementation nào.

Do đó:

- Không thể mô tả implementation là Autoformer time-delay aggregation đúng nghĩa.
- So sánh Tier 1/Tier 2 theo `n_heads=4/8` là sai: hyperparameter này không tác
  động đến mô hình.
- Cần sửa implementation và train lại, hoặc đổi tên thành một kiến trúc
  decomposition–FFT-inspired và mô tả trung thực cơ chế shift cố định.

Plan cũng nhầm kích thước checkpoint `.pt` với parameter count. Suy ra từ các layer
thực tế:

- Tier 1: khoảng 149,525 trainable parameters.
- Tier 2: khoảng 1,086,485 trainable parameters.

Các buffer positional encoding lớn giải thích vì sao kích thước `.pt` lớn hơn.

### B3. Output volatility của MoiraiVaR không được ràng buộc dương

`experiments/moirai_var_aware/modeling.py`, dòng 128–136, kết thúc bằng
`Linear(hidden_dim, output_dim)` mà không có Softplus/Exp.

`losses.py`, dòng 50, chỉ clamp khi tính VaR loss; MSE ở dòng 83 vẫn dùng prediction
thô. Metrics sau đó loại prediction không dương. Cách này có thể làm thay đổi tập
quan sát đánh giá và cần được sửa/ghi nhận rõ.

### B4. Kiểm định dominance nhị thức không có thiết kế suy luận phù hợp

`stats_analysis/run_mcdm_evaluation.py`, dòng 847–911:

- Gộp tier thành `display_group`.
- So một composite score của GARCH-Autoformer với 15 family scores khác.
- Xem 15 so sánh phụ thuộc trên cùng decision matrix như 15 Bernoulli trials.

Các “trial” không phải các dataset/block độc lập và các đối thủ cùng chia sẻ dữ liệu,
normalization, criteria và baseline. Vì vậy \(p=3.05\times10^{-5}\) không chứng minh
dominance tổng quát.

Nên thay bằng:

- Friedman trên 45 market–horizon blocks cho nhiều mô hình, kèm post-hoc đã hiệu
  chỉnh;
- Wilcoxon signed-rank cho so sánh hai mô hình trên block;
- bootstrap confidence intervals/effect sizes;
- sensitivity analysis cho weights và Gate thresholds.

### B5. Plan/bản thảo chứa kết quả hoặc đặc tả chưa có bằng chứng

- Không có run `MCDM20260727104251` trong workspace.
- `paper/generate_figures.py`, dòng 29, vẫn trỏ tới run này.
- Chỉ có prediction λ=0.2; không có thư mục multi-lambda
  \(\{0.3,0.4,0.5\}\) được plan mô tả.
- Default batch size của Modal runner là 32, không phải 8; chưa có run log chứng
  minh lệnh thực tế dùng 8.
- MCDM 26 cấu hình dùng MoiraiVaR head-only λ=0.2, trong khi kết quả full
  fine-tune Moirai 2 là một thí nghiệm riêng. Hai population không được trộn thành
  một ranking/claim.

### B6. Bản thảo hiện tại mâu thuẫn dữ liệu

Trong `paper/bookchapter.tex`:

- Dòng 60, 340, 452: nói 12 cấu hình bị loại; đúng là 5.
- Dòng 350, 427: nói GARCH-Autoformer bị loại; source-derived ranking cho thấy
  Tier 2 đứng hạng 1 ở cả hai weight scenarios.
- Dòng 369–383: ranking 50:50 không còn đúng sau khi tái tính từ source of truth.
- Dòng 468: tuyên bố multi-lambda “available in repository”; dữ liệu này không tồn
  tại trong workspace.
- PDF hiện có 9 trang.

## 5. Kiểm tra kết quả full fine-tune riêng

Các file full fine-tune Moirai 2 λ=0.2 có 42,475 dòng và khớp một-một với baseline
Moirai 2 trong CSV chuẩn.

| Model | MSE | MAE | QLIKE |
|---|---:|---:|---:|
| Moirai 2 full fine-tune λ=0.2 | 0.021569 | 0.087833 | 0.008632 |
| Moirai 2 baseline | 0.022862 | 0.090709 | 0.009079 |

Trên 45 market–horizon blocks:

- MSE: 26 thắng / 19 thua; Wilcoxon hai phía \(p=0.1434\).
- MAE: 26 thắng / 19 thua; Wilcoxon hai phía \(p=0.0654\);
  kiểm định một phía \(p=0.0327\).
- QLIKE: 25 thắng / 20 thua; Wilcoxon hai phía \(p=0.4136\).

Chỉ MAE đạt \(p<0.05\) khi dùng giả thuyết một phía. Tuy nhiên toàn bộ kết quả này
vẫn bị ảnh hưởng bởi lỗi target/split ở B1, nên chưa được dùng làm claim xác nhận.

## 6. Citation và originality audit

### 6.1 References

Cả 15 tài liệu tham khảo đều có thể truy vết tới publication có thật. Các vấn đề:

1. `liu2025` ghi “Liu, Y.” nhưng first author đúng là **Xu Liu**; citation đầy đủ
   là PMLR 267, 38940–38962 (2025).
2. Demšar (2006) khuyến nghị Wilcoxon cho hai phương pháp và Friedman + post-hoc
   cho nhiều phương pháp trên nhiều datasets. Dùng tài liệu này để biện minh cho
   exact binomial test trên 15 family scores là context distortion.
3. Zeng et al. (2023) lập luận temporal information loss của self-attention và
   benchmark linear models; câu “smoothing effect over long contexts” trong draft
   cần viết lại sát nguồn.
4. Autoformer gốc dùng time-delay aggregation ở sub-series level; implementation
   hiện tại không dùng selected delay nên citation không thể hợp thức hóa mô tả
   kiến trúc.
5. Gneiting (2011) hỗ trợ lập luận về scoring functions/forecast functionals,
   nhưng không trực tiếp chứng minh toàn bộ câu gộp riêng MSE, MAE và QLIKE như
   draft đang viết.

Nguồn kiểm tra chính:

- Demšar, JMLR: https://jmlr.org/papers/v7/demsar06a.html
- Moirai-MoE, PMLR: https://proceedings.mlr.press/v267/liu25an.html
- Autoformer, NeurIPS:
  https://proceedings.neurips.cc/paper_files/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html
- Zeng et al., AAAI: https://ojs.aaai.org/index.php/AAAI/article/view/26317
- Kupiec record: https://www.econbiz.de/10001223182

### 6.2 Originality screen

Đã lấy mẫu 12 đoạn đặc trưng trong Introduction, Related Work, Methodology,
Results và Conclusion, tìm exact phrase trên web. Không tìm thấy exact match có ý
nghĩa. Kết quả: **PASS ở mức heuristic screen**.

Đây không thay thế iThenticate/Turnitin. Trước khi nộp vẫn cần kiểm tra similarity
bằng công cụ của đơn vị xuất bản.

## 7. Venue-fit: ICEBA 2026

CFP chính thức liệt kê:

- Computing Science, Simulations and Modeling.
- Embedded systems, IoT, Machine Learning và Artificial Intelligence.

Vì vậy paper có thể được định vị vào track Computing/ML. Tuy nhiên scope chủ đạo
của ICEBA là engineering, physics, microelectronics, semiconductors và các ứng
dụng kỹ thuật. Một paper thuần financial econometrics có rủi ro desk rejection cao
hơn. Phần mở đầu cần nhấn mạnh **reliable AI evaluation**, dynamic forecasting và
risk-sensitive decision support, không tập trung vào trading/portfolio.

Có mâu thuẫn trên website:

- Trang Call for Papers ghi full paper 4–8 trang.
- Trang Templates ghi 4–6 trang, gồm hình, bảng và references; double blind; IEEE
  citation style; hình tối thiểu 300 dpi.

Nên áp dụng giới hạn nghiêm ngặt hơn là **6 trang tối đa** cho đến khi ban tổ chức
xác nhận. Plan cũ ghi line drawing 800–1200 dpi không phải yêu cầu ICEBA trên trang
template; 300 dpi là mức venue nêu.

Liên kết:

- CFP/subject areas:
  https://phys.hcmus.edu.vn/ICEBA2026/Call%20for%20Paper%20ICEBA2026.pdf
- Submission:
  https://iceba2026.vercel.app/call-for-papers/
- Templates:
  https://www.iceba-conference.com/submission/templates/

## 8. Hướng paper khả thi sau khi sửa

### 8.1 Tái định vị

Không nên tiếp tục title nhấn mạnh “two architectures” cho đến khi hai
implementations được sửa và retrain. Hướng an toàn và mạnh hơn:

> **Risk-Sensitive Multi-Criteria Evaluation of Volatility Forecasts Across Nine
> Equity Markets**

Nếu cả hai kiến trúc được sửa và kết quả vẫn ổn định, có thể dùng:

> **Risk-Sensitive Evaluation of Hybrid and Foundation Models for Multi-Horizon
> Volatility Forecasting**

### 8.2 Research questions

1. Cấu hình nào còn hợp lệ sau một dynamics-aware sanity gate?
2. Ranking thay đổi thế nào khi chuyển từ cân bằng accuracy–risk sang
   risk-priority?
3. Kết luận có ổn định trên 45 market–horizon blocks, weight perturbations và Gate
   thresholds hay không?

### 8.3 Contributions nên giữ

1. Protocol đánh giá chín tiêu chí kết hợp forecast quality, dynamic tracking và
   VaR calibration.
2. Benchmark 26 cấu hình trên 9 thị trường và 5 horizons với source/provenance
   được đóng băng.
3. Sensitivity và block-level statistical validation của SAW/TOPSIS rankings.

Chỉ thêm contribution kiến trúc khi code, target và experiment được sửa.

### 8.4 Bố cục 6 trang

| Phần | Ngân sách |
|---|---:|
| Abstract + keywords | 150–180 từ |
| Introduction | 400–450 từ |
| Related work + gap | 250–300 từ |
| Method + protocol | 750–850 từ |
| Experimental setup | 350–450 từ |
| Results + robustness | 650–750 từ |
| Discussion + conclusion | 300–350 từ |
| References | 12–15 nguồn |

Chỉ dùng:

- 2 hình: pipeline và ranking/sensitivity.
- 2 bảng compact: dataset/configuration và main results.
- Không dành nửa trang cho hai architecture diagram nếu architecture chưa phải
  contribution đã được xác minh.

## 9. Kế hoạch sửa bắt buộc trước khi viết submission

1. Đóng băng CSV chuẩn bằng hash, schema, row count và data manifest.
2. Sửa target của MoiraiVaR:
   - định nghĩa rõ volatility forecast target;
   - loại sample không đủ future horizon;
   - purge/embargo split boundary theo max horizon;
   - thêm positivity constraint.
3. Sửa GARCH-Autoformer:
   - dùng đúng selected delays và thiết kế multi-head thật sự; hoặc
   - đổi tên/mô tả đúng implementation;
   - train lại cả hai tiers.
4. Chạy lại mọi model bị ảnh hưởng và tạo merged CSV version mới.
5. Tái tính statistical pipeline từ đúng merged CSV và lưu:
   hash, commit, command, environment, seed, timestamp.
6. Loại binomial dominance; thay bằng block-level tests và effect sizes.
7. Chạy sensitivity:
   weights, normalization, Gate thresholds, horizon/market leave-one-out.
8. Sinh lại hình/bảng và viết paper 6 trang theo template ICEBA.

## 10. Bảy failure modes của ARS

| Failure mode | Verdict | Bằng chứng |
|---|---|---|
| 1. Implementation bug | **SUSPECTED / blocking** | Target/split MoiraiVaR; selected delay bị bỏ qua |
| 2. Citation hallucination | **Không hallucinate; có metadata/context error** | `Liu, Y.` sai; Demšar bị dùng sai ngữ cảnh |
| 3. Hallucinated result | **SUSPECTED / blocking** | Multi-lambda và run 20260727 không tồn tại |
| 4. Shortcut reliance | **SUSPECTED / blocking** | \(h=1\) target bằng std của input window |
| 5. Bug reframed as insight | **SUSPECTED** | Claim kiến trúc/performance dựa trên code lỗi |
| 6. Methodology fabrication | **SUSPECTED / blocking** | `n_heads` vô tác dụng; batch size, param count không được chứng minh |
| 7. Frame-lock | **SUSPECTED** | Cố giữ bốn contributions và năm hình trong giới hạn ngắn |

## 11. RAISE/compliance warnings

Đây là primary empirical research nên compliance ở mức cảnh báo:

- Chưa có statement về AI assistance/tool use.
- Chưa có artifact manifest đủ để tái lập.
- Chưa ghi rõ human verification/oversight.
- Chưa có data/code availability statement.
- Chưa có conflict-of-interest/funding statement trong draft.

## 12. Mandatory user checkpoint

**Integrity Checkpoint 2.5: FAIL.**

Chưa chuyển sang Stage 3 (viết submission-ready) cho đến khi người dùng xác nhận
một trong hai hướng:

- **A — Khuyến nghị:** sửa code, retrain/rerun, rồi viết lại paper ICEBA.
- **B — Bản pilot có hạ claim:** viết ngay một draft MCDM-centric từ CSV đã đóng
  băng, nêu rõ limitations và không tuyên bố hai kiến trúc đã được xác nhận.

Không nên viết lại nguyên story “hai kiến trúc mới vượt trội” từ bằng chứng hiện tại.
