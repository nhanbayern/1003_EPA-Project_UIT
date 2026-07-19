# Định hướng Refactor Bài báo và Mã nguồn (Sau Review)

Tài liệu này trình bày chi tiết kế hoạch định hình lại bài báo và sửa đổi mã nguồn (tập trung vào GARCH/GARCH-LSTM) để giải quyết các chỉ trích cốt lõi từ hội nghị. Kế hoạch này xoay quanh 2 hành động chính: Giải thích sự phân kỳ bằng Cơ chế tối ưu hóa, và sửa lỗi dự báo Đa bước (Multi-horizon).

---

## Bối cảnh (Context)
Bài báo bị Reject/Weak Reject với các lý do chính:
1. Thiếu chiều sâu phân tích về lý do tại sao mô hình có sai số dự báo (Accuracy) tốt nhất lại thất bại trong việc kiểm soát rủi ro (VaR Risk Control). (Reviewer 2)
2. Mô hình mới đề xuất (Moirai-MoE-GARCHs) hoạt động quá kém, gây xao nhãng và không đóng góp vào kết luận chính. (Reviewer 3)

Để cứu vãn bài báo, chúng ta sẽ loại bỏ hoàn toàn mô hình "Moirai-MoE-GARCHs" ra khỏi cốt truyện, biến bài báo thành một nghiên cứu phân tích thuần túy (Empirical Analysis) về **Sự đánh đổi (Trade-off) giữa Accuracy và Risk Calibration**. 

Dưới đây là 2 hành động cụ thể để hiện thực hóa hướng đi này:

---

## HÀNH ĐỘNG 1: Giải mã Sự phân kỳ thông qua Cơ chế Tối ưu hóa (Objective Functions)

### Hướng đi này là gì?
Thay vì cố gắng đồng nhất hóa hàm Loss của tất cả các mô hình, chúng ta sẽ **cố tình giữ nguyên sự khác biệt** trong cách huấn luyện của chúng hiện tại:
- **Nhóm Transformer (Autoformer, Informer, Moirai):** Được huấn luyện bằng hàm Hồi quy khoảng cách (MSE/MAE) trên nhãn là Rolling Volatility.
- **Nhóm GARCH & GARCH-LSTM:** Được huấn luyện bằng Ước lượng Hợp lý cực đại (Maximum Likelihood Estimation - MLE) thông qua hàm Negative Log-Likelihood (NLL) trên chuỗi Lợi suất (Returns).

Chúng ta sẽ dùng chính sự khác biệt này làm "Chìa khóa phân tích" (Analytical Key) để trả lời câu hỏi của Reviewer 2.

### Đi như thế nào? (Cách triển khai)
Trong phiên bản bài báo sửa đổi, cần bổ sung một tiểu mục (Section) phân tích chuyên sâu có tên: *"The Impact of Objective Functions: Regression vs. Density Estimation"*. 
- **Bước 1:** Trình bày rõ ràng công thức Loss của cả 2 nhóm.
- **Bước 2:** Lập luận rằng các mô hình Hồi quy (Train bằng MSE) chỉ cố gắng dự đoán giá trị trung bình có điều kiện (conditional mean) của Proxy (Rolling Volatility). Việc này giúp tối thiểu hóa chỉ số RMSE/MAE (Accuracy), nhưng hoàn toàn bỏ qua hình dáng của đuôi phân phối lợi suất.
- **Bước 3:** Lập luận rằng nhóm GARCH (Train bằng NLL) là các mô hình Ước lượng Mật độ (Density Estimators). Chúng học toàn bộ hình dáng phân phối xác suất của lợi suất (đặc biệt là vùng đuôi - tails, thông qua phân phối Student-t). Vì Value-at-Risk (VaR) bản chất là bài toán tính phân vị đuôi (Tail Quantile), mô hình Density Estimator tất yếu sẽ kiểm soát rủi ro chính xác hơn, bất chấp việc nó không giỏi "khớp" (fit) đường cong Rolling Volatility bằng mạng Transformer.

### Tại sao lại như thế? (Lý do cốt lõi)
Nếu chúng ta "chữa cháy" bằng cách ép GARCH-LSTM train bằng MSE, mô hình này sẽ mất đi năng lực nhận diện vùng đuôi phân phối (Fat-tails) và sẽ rớt hạng thảm hại ở bài test VaR. Giữ nguyên hiện trạng là cách duy nhất bảo vệ được luận điểm khoa học độc đáo nhất của bài báo: *"Sự chính xác trong dự báo đường cong không đảm bảo sự bao quát rủi ro vùng đuôi"*.

### Nguồn tham khảo (References)
Luận điểm này có cơ sở toán học vững chắc từ kinh tế lượng:
1. **Christoffersen, P., & Diebold, F. X. (2000).** *How relevant is volatility forecasting for financial risk management?* The Review of Economics and Statistics. (Nền tảng lý thuyết về sự khác biệt giữa dự báo điểm và dự báo phân phối rủi ro).
2. **Wu, Y., et al. (2021).** *Unified GARCH-Recurrent Neural Networks in Financial Volatility Forecasting.* (Minh chứng cho việc dùng NLL/Likelihood để tối ưu hóa mạng nơ-ron giúp cải thiện dự báo phân phối lợi suất).

---

## HÀNH ĐỘNG 2: Sửa lỗi Mô phỏng Đa bước (Multi-horizon Simulation) cho họ GARCH

### Hướng đi này là gì?
Hiện tại, mã nguồn của các baseline Thống kê (GARCH, GJR-GARCH, FI-GARCH) và Deep Learning lai (GARCH-LSTM Hybrid) đang bị **hardcode chỉ dự báo 1 bước tới (horizon $h=1$)**. Trong khi đó, bài toán yêu cầu dự báo và so sánh ở các bước $h \in \{1, 3, 5, 10, 21\}$. Hành động này nhằm sửa lỗi logic trong codebase để sinh ra kết quả so sánh công bằng.

### Đi như thế nào? (Cách triển khai trong Code)
Thay vì chỉ dùng hàm `forecast_next_variance()`, chúng ta phải lập trình cơ chế **Rolling Forecast (Mô phỏng đệ quy)** cho $h$ ngày.
* **Đối với GARCH thống kê (`stat_baselines.py`):**
  Sử dụng hàm `forecast(horizon=h)` của thư viện `arch`. Thư viện này sẽ trả về vector phương sai dự báo cho $h$ ngày tới: $[\sigma_{t}^2, \sigma_{t+1}^2, \dots, \sigma_{t+h-1}^2]$.
* **Đối với GARCH-LSTM (`garch_lstm_hybrid.py`):**
  Viết thêm vòng lặp autoregressive. Tại bước $t+i$, lấy phương sai dự báo của bước trước làm input để đệ quy tính tiếp phương sai cho bước sau cho đến khi đủ $h$ bước.

**Quy đổi đại lượng (Cực kỳ quan trọng):**
Do GARCH xuất ra phương sai 1 ngày ($\sigma^2$), nhưng Target của bài toán (và của Transformer) là "Độ biến động gộp/trượt" (Volatility). Theo tính chất cộng dồn của phương sai (Variance Additivity trong tài chính), phương sai của chuỗi $h$ ngày bằng tổng phương sai của từng ngày. Do đó, để tạo ra giá trị tương đương với Target, chúng ta cần:
1. Tính tổng (Sum) phương sai của $h$ ngày: $V_{total} = \sum_{i=1}^{h} \sigma_{t+i-1}^2$
2. Chia trung bình và lấy căn bậc 2 để ra Volatility: $Vol_{pred} = \sqrt{\frac{V_{total}}{h}}$
3. Ghi giá trị này vào file kết quả để đem ra so sánh MAE/MSE/QLIKE với Transformer.

### Tại sao lại như thế? (Lý do cốt lõi)
Nếu không làm bước này, chúng ta đang so sánh "Độ lệch chuẩn 1 ngày" của GARCH với "Độ lệch chuẩn 21 ngày" của Transformer ở chân trời $h=21$. Điều này là sai lầm cơ bản về Toán học và sẽ bị Reviewer bắt lỗi ngay lập tức nếu họ kiểm tra code hoặc dữ liệu thô. Việc cộng dồn phương sai đảm bảo tính quy chuẩn (Scaling) của các đại lượng trước khi chấm điểm bằng các metric tĩnh.

### Nguồn tham khảo (References)
1. **Tsay, R. S. (2010).** *Analysis of Financial Time Series* (Chương 3: Volatility Models). Nêu rõ nguyên tắc "Thời gian hóa phương sai" (Temporal Aggregation of Volatility) theo quy tắc Căn-bậc-hai-của-thời-gian (Square-root of time rule).
2. Tài liệu thiết kế gốc của đồ án `docs/data.md` công thức (3) (Chứng minh sự cần thiết phải đồng nhất khung thời gian cho biến mục tiêu).
