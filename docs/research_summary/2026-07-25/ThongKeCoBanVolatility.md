# Báo cáo Nghiên cứu: Thống kê Cơ bản và Tính dừng của Volatility

**Ngày thực hiện:** 25/07/2026  
**Chủ đề:** Kiểm tra thống kê mô tả và tính dừng của chuỗi `true_volatility` theo từng dataset, từng horizon và từng giai đoạn thời gian `train/val/test`.

---

## Tóm tắt

Tài liệu này bổ sung phần thống kê cơ bản cho research summary, tập trung vào tính dừng của volatility trước khi diễn giải kết quả forecast và MCDM. Script liên quan:

```text
stats_analysis/check_volatility_stationarity.py
```

Input đã chạy:

```text
output/merged_predictions/merged_all_predictions_24_7.csv
```

Output đã tạo:

```text
output/stats_analysis/stationarity/stationarity_20260725_173107/
```

Do file merged predictions lặp lại cùng `true_volatility` cho nhiều mô hình, script loại trùng theo `dataset/horizon/time` trước khi kiểm định. CSV hiện tại không có cột split gốc, nên các giai đoạn `train`, `val`, `test` được suy ra theo thứ tự thời gian trong từng dataset với tỷ lệ 70%/15%/15%. Vì `true_volatility` thay đổi theo `horizon`, kiểm định được chạy riêng theo từng tổ hợp `dataset × horizon × split`, đồng thời có thêm split `full` cho toàn bộ giai đoạn quan sát.

## 1. Phương pháp kiểm định

Pipeline dùng đồng thời hai kiểm định để tránh kết luận một chiều:

```text
ADF test:
H0: chuỗi có unit root, tức không dừng.
H1: chuỗi dừng.

KPSS test:
H0: chuỗi dừng quanh mức trung bình.
H1: chuỗi không dừng.
```

Với mức ý nghĩa `alpha = 0.05`, một chuỗi chỉ được kết luận là `stationary` khi thỏa đồng thời:

```text
ADF p-value < 0.05
KPSS p-value >= 0.05
```

Nếu ADF và KPSS cho kết luận ngược nhau, script ghi nhãn `mixed`. Nhãn này rất quan trọng trong bối cảnh volatility vì chuỗi volatility tài chính thường có clustering, regime shift và các đoạn shock ngắn hạn. Khi đó một kiểm định có thể phát hiện mean reversion, trong khi kiểm định còn lại vẫn xem chuỗi có dấu hiệu không dừng cục bộ. Vì vậy `mixed` không nên bị đọc là lỗi tính toán, mà là tín hiệu cho thấy chuỗi có đặc tính động học phức tạp.

## 2. Kết quả tổng quát

Tổng cộng có 9 dataset và 5 horizon, tức 45 chuỗi `dataset × horizon` trên full sample. Kết quả full sample cho thấy chỉ có 15/45 chuỗi được kết luận stationary, tương đương 33.33%. Không có chuỗi full sample nào bị kết luận `non_stationary` theo cả hai kiểm định; 30/45 chuỗi còn lại là `mixed`.

| Split | Mixed | Non-stationary | Stationary |
|---|---:|---:|---:|
| Full | 30 | 0 | 15 |
| Train | 30 | 6 | 9 |
| Val | 22 | 23 | 0 |
| Test | 18 | 26 | 1 |

Kết quả theo split ngắn hơn cho thấy tính dừng không ổn định qua thời gian. Trong train split có 9 chuỗi stationary, nhưng val split không có chuỗi nào stationary và test split chỉ có 1 chuỗi stationary. Điều này hàm ý volatility trong giai đoạn validation/test có nhiều dấu hiệu regime change hoặc local non-stationarity hơn full sample. Vì vậy khi đánh giá mô hình, không nên chỉ dựa vào thống kê toàn kỳ; cần kiểm tra hiệu suất theo từng giai đoạn và từng dataset.

## 3. Thống kê full sample theo dataset

Bảng dưới đây lấy trung bình `mean` và `std` của volatility trên 5 horizon trong full sample, đồng thời đếm số horizon thuộc từng kết luận stationarity.

| Dataset | Số horizon | Mean volatility | Std volatility | Stationary | Mixed | Non-stationary |
|---|---:|---:|---:|---:|---:|---:|
| DAX_40 | 5 | 1.138801 | 0.563730 | 0 | 5 | 0 |
| EuroNext_100 | 5 | 1.038867 | 0.522665 | 0 | 5 | 0 |
| IBEX_35 | 5 | 1.046061 | 0.322235 | 0 | 5 | 0 |
| KOSPI_index | 5 | 1.153157 | 0.321026 | 3 | 2 | 0 |
| Nikkei_225 | 5 | 1.269092 | 0.476396 | 5 | 0 | 0 |
| SMI | 5 | 0.828717 | 0.265068 | 2 | 3 | 0 |
| VN30_INDEX | 5 | 1.226565 | 0.410425 | 5 | 0 | 0 |
| VN_INDEX | 5 | 1.188819 | 0.418176 | 0 | 5 | 0 |
| snp500 | 5 | 1.083297 | 0.551989 | 0 | 5 | 0 |

Hai dataset có tính dừng rõ nhất trên full sample là `Nikkei_225` và `VN30_INDEX`, vì cả 5/5 horizon đều đạt điều kiện ADF-KPSS. `KOSPI_index` có 3/5 horizon stationary, `SMI` có 2/5 horizon stationary. Các dataset còn lại đều rơi vào nhóm `mixed` trên toàn bộ 5 horizon, nghĩa là không đủ bằng chứng nhất quán để kết luận stationary theo cả hai kiểm định.

## 4. Các chuỗi stationary trên full sample

Danh sách các chuỗi `dataset × horizon` được kết luận stationary trên toàn bộ giai đoạn:

| Dataset | Horizon | N obs | ADF p-value | KPSS p-value | Mean | Std |
|---|---:|---:|---:|---:|---:|---:|
| KOSPI_index | 1 | 1382 | 0.000005 | 0.053509 | 1.158995 | 0.341272 |
| KOSPI_index | 3 | 1382 | 0.000004 | 0.054282 | 1.157294 | 0.335659 |
| KOSPI_index | 5 | 1382 | 0.000005 | 0.053504 | 1.155595 | 0.329952 |
| Nikkei_225 | 1 | 1549 | 0.005645 | 0.100000 | 1.266183 | 0.476735 |
| Nikkei_225 | 3 | 1549 | 0.005532 | 0.100000 | 1.267005 | 0.476650 |
| Nikkei_225 | 5 | 1549 | 0.005587 | 0.100000 | 1.267842 | 0.476552 |
| Nikkei_225 | 10 | 1549 | 0.005326 | 0.100000 | 1.269938 | 0.476296 |
| Nikkei_225 | 21 | 1549 | 0.005177 | 0.100000 | 1.274493 | 0.475748 |
| SMI | 10 | 1411 | 0.011983 | 0.072092 | 0.825149 | 0.253958 |
| SMI | 21 | 1411 | 0.013564 | 0.100000 | 0.814719 | 0.234036 |
| VN30_INDEX | 1 | 1175 | 0.007834 | 0.052524 | 1.227475 | 0.411884 |
| VN30_INDEX | 3 | 1175 | 0.019834 | 0.057309 | 1.226804 | 0.411105 |
| VN30_INDEX | 5 | 1175 | 0.018534 | 0.062143 | 1.226099 | 0.410271 |
| VN30_INDEX | 10 | 1175 | 0.020068 | 0.071834 | 1.225058 | 0.409095 |
| VN30_INDEX | 21 | 1175 | 0.020899 | 0.078079 | 1.227387 | 0.409772 |

## 5. Nhận xét khoa học

Kết quả stationarity cho thấy volatility trong bộ dữ liệu không đồng nhất giữa các thị trường và giữa các giai đoạn. Một số thị trường như `Nikkei_225` và `VN30_INDEX` có bằng chứng stationary nhất quán trên full sample, nhưng nhiều thị trường lớn như `DAX_40`, `EuroNext_100`, `snp500`, `VN_INDEX` và `IBEX_35` lại rơi hoàn toàn vào nhóm `mixed`. Điều này phù hợp với đặc điểm volatility tài chính: chuỗi có thể mean-reverting trong dài hạn nhưng vẫn có volatility clustering, structural break hoặc regime shift khiến kiểm định stationarity không đồng thuận.

Việc val/test split có nhiều `non_stationary` hơn train split là tín hiệu cần chú ý khi diễn giải kết quả mô hình. Nếu giai đoạn đánh giá có phân phối volatility khác train, mô hình dự báo đường trung bình hoặc dự báo quá trơn có thể tình cờ đạt một số chỉ số calibration nhưng không thật sự học được biến động động học. Vì vậy kết quả stationarity củng cố quyết định bổ sung `volatility_std_ratio_error`, `tracking_correlation_error` và forecast sanity gate vào MCDM pipeline.

Kết luận thực nghiệm là không nên giả định toàn bộ volatility dataset là stationary. Khi báo cáo kết quả, cần nêu rõ stationarity được kiểm tra theo từng `dataset × horizon × split`, và ranking MCDM nên được đọc như kết quả dưới điều kiện dữ liệu có tính dừng không đồng nhất. Các file chi tiết cần dùng khi trích số liệu:

```text
output/stats_analysis/stationarity/stationarity_20260725_173107/volatility_stationarity_by_dataset_horizon_split.csv
output/stats_analysis/stationarity/stationarity_20260725_173107/volatility_stationarity_summary.md
output/stats_analysis/stationarity/stationarity_20260725_173107/stationarity_run_metadata.json
```
