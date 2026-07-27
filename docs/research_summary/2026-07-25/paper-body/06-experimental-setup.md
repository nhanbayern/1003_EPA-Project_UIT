# 4. Thiết lập thực nghiệm

## 4.1. Phạm vi dữ liệu

Thực nghiệm bao phủ 9 bộ dữ liệu chỉ số thị trường:

| Khu vực đại diện | Bộ dữ liệu |
|---|---|
| Việt Nam | `VN_INDEX`, `VN30_INDEX` |
| Châu Á ngoài Việt Nam | `KOSPI_index`, `Nikkei_225` |
| Châu Âu | `DAX_40`, `EuroNext_100`, `IBEX_35`, `SMI` |
| Hoa Kỳ | `snp500` |

Mỗi bộ dữ liệu được đánh giá ở 5 chân trời \(h\in\{1,3,5,10,21\}\), tạo thành 45 tổ hợp `dataset × horizon`. Các tiêu chí bám động học đều có 45 trường hợp hợp lệ cho mỗi cấu hình. Số dòng dùng để tính khả năng bám chuỗi khác nhau giữa các nhánh kết quả: 63.535 dòng cho nhóm GARCH, 50.755 dòng cho Transformer và các Autoformer sửa đổi, và 42.475 dòng cho Moirai/MoiraiVaR. Vì vậy, kết quả được diễn giải ở cấp cấu hình và trường hợp tổng hợp; khác biệt số quan sát giữa các nhánh được xem là một hạn chế về tính so sánh tuyệt đối.

## 4.2. Chia dữ liệu theo thời gian

Tệp dự báo tổng hợp được mô tả trong báo cáo nguồn không chứa nhãn phân đoạn gốc. Riêng phân tích tính dừng loại các bản ghi lặp của `true_volatility` theo `dataset/horizon/time`, sắp xếp theo thời gian và suy ra ba đoạn:

\[
\text{train}:\text{validation}:\text{test}=70\%:15\%:15\%.
\]

Phép chia này được sử dụng để chẩn đoán ADF/KPSS, không được trình bày như giao thức huấn luyện gốc của tất cả mô hình. Các bảng MCDM sử dụng kết quả dự báo và kiểm định hậu nghiệm đã được tổng hợp theo 45 trường hợp; tài liệu nguồn không cung cấp thêm ngày bắt đầu–kết thúc, cửa sổ huấn luyện, siêu tham số tối ưu hoặc quy tắc tái huấn luyện. Do đó, bài báo giới hạn phần tái lập ở pipeline đánh giá và không suy diễn các chi tiết huấn luyện chưa được ghi nhận.

## 4.3. Chẩn đoán tính dừng

Tính dừng được kiểm tra riêng cho mỗi `dataset × horizon × split` bằng ADF và KPSS ở mức ý nghĩa 0,05:

- ADF: \(H_0\) là chuỗi có nghiệm đơn vị, tức không dừng.
- KPSS: \(H_0\) là chuỗi dừng quanh trung bình.

Một chuỗi chỉ được gắn nhãn `stationary` khi đồng thời có \(p_{\mathrm{ADF}}<0{,}05\) và \(p_{\mathrm{KPSS}}\geq0{,}05\). Khi hai kiểm định không đồng thuận, chuỗi được gắn nhãn `mixed`. Nhãn `non_stationary` biểu thị cả hai kiểm định cùng ủng hộ kết luận không dừng theo quy tắc của pipeline.

## 4.4. Mô hình đề xuất và mô hình đối sánh

Bảng đầu vào chứa 26 cấu hình thuộc 5 nhánh và 16 họ hiển thị:

| Vai trò | Nhánh | Cấu hình |
|---|---|---|
| Mô hình đề xuất | GARCH-Autoformer | GARCH-Autoformer Tier 1 và Tier 2 |
| Mô hình đề xuất | MoiraiVaR | MoiraiVaR–Moirai, MoiraiVaR–Moirai 2 và MoiraiVaR–Moirai-MoE với \(\lambda=0{,}2\) |
| Mô hình đối sánh | GARCH | GARCH, GJR-GARCH, FI-GARCH, GARCH-LSTM-Hybrid |
| Mô hình đối sánh | Transformer | Autoformer, Informer, Reformer và Vanilla ở Tier 1, Tier 2, Tier 3 |
| Mô hình đối sánh | Autoformer sửa đổi khác | Wavelet-Autoformer Tier 1 và Tier 2 |
| Mô hình đối sánh | Moirai | Moirai, Moirai 2, Moirai-MoE |

GARCH-Autoformer và MoiraiVaR là hai mô hình do nhóm nghiên cứu đề xuất. Trong phân tích thống kê hiện có, GARCH-Autoformer là đối tượng được kiểm tra ưu thế riêng. Tài liệu nguồn không mô tả số tầng, số tham số, cơ chế kết hợp, biểu thức hàm mất mát căn chỉnh hoặc chi phí tính toán; vì vậy các tier và mô hình nền được giữ như các cấu hình thực nghiệm độc lập trong bảng xếp hạng. Chỉ GARCH-Autoformer Tier 1/Tier 2 được gộp khi thực hiện kiểm định ưu thế ở cấp họ mô hình.

## 4.5. Tiêu chí đánh giá

Mỗi cấu hình được biểu diễn bởi chín tiêu chí:

| Nhóm | Tiêu chí | Chiều tối ưu |
|---|---|---|
| Độ chính xác | MSE | Thấp hơn |
| Độ chính xác | MAE | Thấp hơn |
| Độ chính xác | QLIKE | Thấp hơn |
| Bám động học | Sai số tỷ lệ độ lệch chuẩn | Thấp hơn |
| Bám động học | Sai số tương quan bám chuỗi | Thấp hơn |
| Rủi ro VaR 1% | Tỷ lệ vượt backtest | Cao hơn |
| Rủi ro VaR 1% | Sai số vi phạm tuyệt đối | Thấp hơn |
| Rủi ro VaR 5% | Tỷ lệ vượt backtest | Cao hơn |
| Rủi ro VaR 5% | Sai số vi phạm tuyệt đối | Thấp hơn |

Cổng kiểm tra được áp dụng trước khi chuẩn hóa và xếp hạng. Sau sàng lọc, SAW và TOPSIS được chạy riêng cho kịch bản 50:50 và 30:70. Kết quả chính được lấy trực tiếp từ lần chạy [9]:

```text
output/mcdm_results/MCDM20260725180350/
```

## 4.6. Quy tắc báo cáo kết quả

Nghiên cứu báo cáo hai cấp kết quả:

1. **Cấp cấu hình:** giữ nguyên tier và biến thể, dùng để xác định top mô hình theo SAW, TOPSIS và thứ hạng trung bình.
2. **Cấp họ mô hình:** lấy trung bình qua các tier có mặt trong bảng xếp hạng, dùng cho biểu đồ so sánh và kiểm định ưu thế của GARCH-Autoformer với 15 họ còn lại.

Ngưỡng ý nghĩa cho kiểm định ưu thế là 0,05. Kiểm định nhị thức chính xác một phía là kết quả chính; z-test một tỷ lệ chỉ được trình bày như kiểm tra xấp xỉ. Các số thập phân trong bảng được làm tròn khi trình bày, trong khi kiểm định sử dụng giá trị đầy đủ từ tệp CSV.
