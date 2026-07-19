# Phân tích Yêu cầu và Thiết kế Kế hoạch: Pipeline Thử nghiệm Độ phức tạp (Version 2)

**Đường dẫn dự kiến triển khai:** `model/transformer based/version_2`
**Mục tiêu:** Xây dựng một pipeline thực nghiệm (empirical test) để kiểm chứng giả thuyết: *Liệu việc tăng độ phức tạp (sức chứa - capacity) của các mô hình Transformer trên dữ liệu chuỗi thời gian ngắn ($L=60$) sẽ giúp mô hình học được các đặc trưng phi tuyến phức tạp hơn, hay chỉ đơn thuần gây ra hiện tượng học vẹt (overfitting)?*

Tài liệu này không chứa mã nguồn, chỉ trình bày cơ sở lý thuyết, phân tích học thuật và đề xuất các mức siêu tham số cụ thể cho `version_2`.

---

## 1. Cơ sở Học thuật (Academic Justification)

Trước khi cấu hình các mức siêu tham số, chúng ta cần căn cứ vào các nghiên cứu uy tín trong ngành để hiểu rõ mối quan hệ giữa độ phức tạp của mạng (Network Complexity) và nguy cơ Overfitting trong Time Series.

### 1.1. Nghịch lý giữa Không gian ẩn (Latent Space) và Lượng dữ liệu
*   **Phân tích:** `d_model` là chiều của không gian ẩn. Trong NLP, `d_model` thường là 512, 1024 để biểu diễn sự phong phú của ngữ nghĩa từ vựng. Tuy nhiên, trong dữ liệu tài chính với chuỗi ngắn ($L=60$), việc đưa 1 giá trị vô hướng (scalar return) lên không gian 512 chiều tạo ra hiện tượng **"Lời nguyền chiều dữ liệu" (Curse of Dimensionality)**. Lượng tham số quá lớn so với lượng thông tin thực tế của 60 điểm dữ liệu khiến mô hình ghi nhớ (memorize) nhiễu thay vì học quy luật.
*   **Nguồn trích dẫn:** Wen, Q., et al. (2022). *"Transformers in Time Series: A Survey"*. Bài khảo sát này chỉ ra rằng mặc dù Transformer rất mạnh, nhưng khả năng tổng quát hóa của nó phụ thuộc cực lớn vào lượng dữ liệu. Khi thiếu dữ liệu, mạng lớn sẽ overfit.

### 1.2. Sự dư thừa của các cơ chế phức tạp
*   **Phân tích:** Tăng số lượng lớp (`e_layers`) và số lượng đầu chú ý (`n_heads`) làm tăng độ sâu và khả năng trích xuất đa luồng. Tuy nhiên, Zeng et al. (2023) đã chứng minh thực nghiệm rằng các cơ chế Multi-Head Self-Attention phức tạp thường không hiệu quả bằng một lớp Linear đơn giản trên nhiều bộ dữ liệu chuỗi thời gian. Mạng càng sâu, khả năng bảo toàn thứ tự thời gian (temporal ordering) càng giảm, và overfitting càng dễ xảy ra do không gian tìm kiếm (search space) quá lớn.
*   **Nguồn trích dẫn:** Zeng, A., et al. (2023). *"Are Transformers Effective for Time Series Forecasting?"*. (Thường được biết đến qua mô hình DLinear). Đây là bài báo mang tính bước ngoặt, trực tiếp đặt câu hỏi về việc over-engineering (thiết kế quá phức tạp) của Transformer trong Time Series.

---

## 2. Đề xuất Pipeline: Các Mức Độ phức tạp (Complexity Tiers)

Dựa trên phân tích học thuật, `version_2` sẽ được thiết kế thành một vòng lặp pipeline chạy qua 3 mức độ phức tạp khác nhau. Kết quả (Metrics, Training/Validation Loss) sẽ được log lại riêng biệt để so sánh.

### Tier 1: Tinh gọn (Miniaturized - Mốc Baseline)
Đây là cấu hình đã được triển khai ở `version_1`, nhằm mục đích chống overfitting tối đa.
*   **Siêu tham số:**
    *   `d_model = 32`
    *   `e_layers = 2`
    *   `n_heads = 4`
    *   `d_ff = 128`
    *   `dropout = 0.3`
*   **Lý do chọn:** Không gian biểu diễn nhỏ nhắn, ép mô hình phải tìm ra xu hướng mấu chốt, loại bỏ hoàn toàn khả năng ghi nhớ chi tiết nhiễu. Dropout cao ngăn chặn sự phụ thuộc vào các nơ-ron cục bộ.

### Tier 2: Tiêu chuẩn (Standard/Medium)
Đây là cấu hình cân bằng, thường thấy trong các bài báo áp dụng Deep Learning cho các dataset cỡ trung bình.
*   **Siêu tham số:**
    *   `d_model = 128`
    *   `e_layers = 3`
    *   `n_heads = 8`
    *   `d_ff = 512`
    *   `dropout = 0.2`
*   **Lý do chọn:** Tăng kích thước không gian ẩn lên 128 chiều, cho phép mô hình thử biểu diễn các tương quan phi tuyến phức tạp hơn giữa 60 ngày. Tuy nhiên, độ sâu 3 lớp vẫn giữ mức độ kiểm soát nhất định. Dropout giảm nhẹ để mô hình tự tin hơn vào các kết nối đã học.

### Tier 3: Gốc / Đồ sộ (Original/Large)
Đây là cấu hình được sử dụng trong các bài báo gốc của Informer, Autoformer khi họ chạy trên các tập dữ liệu cực lớn (ETTh, Weather).
*   **Siêu tham số:**
    *   `d_model = 512`
    *   `e_layers = 6`
    *   `n_heads = 8` (hoặc 16)
    *   `d_ff = 2048`
    *   `dropout = 0.1` (Mặc định gốc)
*   **Lý do chọn:** Tái hiện chính xác sức chứa (capacity) nguyên thủy của các thuật toán. Mục đích của mức này **không phải để hy vọng nó dự báo tốt nhất**, mà để làm "thuốc thử" (stress test). Nó được kỳ vọng sẽ minh họa rõ rệt nhất hiện tượng Overfitting: Training Loss sẽ giảm về mức cực kỳ nhỏ (tiệm cận 0), nhưng Validation Loss sẽ nhanh chóng phân kỳ (diverge) và tăng vọt sau một vài epoch.

---

## 3. Giả thuyết Thực nghiệm (Experimental Hypotheses)

Khi thực thi `version_2`, chúng ta kỳ vọng quan sát được các hiện tượng sau để ghi vào báo cáo đồ án:

1.  **Đường cong Loss (Loss Curves):** 
    *   Ở **Tier 3**, chúng ta sẽ thấy sự chênh lệch (gap) khổng lồ giữa đường Train Loss (rất thấp) và đường Val Loss (rất cao). Điều này chứng minh bằng số liệu lập luận của Zeng et al. (2023).
    *   Ở **Tier 1**, khoảng cách giữa Train Loss và Val Loss sẽ hẹp nhất, cho thấy mô hình không học vẹt, dù kết quả tổng thể không xuất sắc như MOIRAI.
2.  **Kết quả Test (Metrics):**
    *   Nhiều khả năng **Tier 1** hoặc **Tier 2** sẽ cho ra kết quả Test (MAE, MSE, Q-LIKE) tốt nhất trong 3 Tiers, trong khi **Tier 3** sẽ cho kết quả Test tệ nhất do mô hình đã bị "nhiễu" đánh lừa hoàn toàn.

## 4. Cấu trúc Dự kiến của Code Version 2
Để đáp ứng yêu cầu này, `version_2` sẽ không hard-code siêu tham số nữa mà sẽ thiết lập hệ thống Configuration:
*   Định nghĩa một mảng/dictionary chứa 3 `config_dict` (tương ứng 3 Tiers).
*   Chạy vòng lặp ngoài cùng: `for tier_name, config in configs.items():`
*   Huấn luyện, đánh giá và lưu kết quả vào các file tách biệt, ví dụ: `comparison_metrics_Tier1.csv`, `comparison_metrics_Tier2.csv`, `comparison_metrics_Tier3.csv`.

**Bước tiếp theo:** Sau khi bạn xem xét và đồng ý với thiết kế lý thuyết của 3 Tier trên, chúng ta có thể bắt tay vào bước tạo mã nguồn (code implementation) cho thư mục `version_2`.
