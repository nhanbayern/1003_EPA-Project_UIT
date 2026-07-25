### I. BÀI TOÁN (THE PROBLEM)
* **Tên bài toán:** Dự báo độ biến động tài chính đa bước thời gian (Multi-horizon Financial Volatility Forecasting).
* **Mục tiêu:** Sử dụng các thông tin tỷ suất sinh lợi trong quá khứ để dự báo độ biến động thực tế (Realized Volatility) tại các thời điểm tương lai $t+h-1$ với các chân trời dự báo $h \in \{1, 3, 5, 10, 21\}$ ngày.
* **Nguyên tắc nhân quả (Causal Setup):** Tại ngày đưa ra dự báo $t$, mô hình hoàn toàn không được tiếp cận bất kỳ thông tin nào của ngày $t$ và tương lai sau ngày $t$. Dữ liệu tương lai chỉ được phép sử dụng làm biến mục tiêu (target) để tính toán hàm mất mát (loss) khi huấn luyện và đánh giá.

---

### II. DỮ LIỆU THÔ (RAW DATA)
* **Tập dữ liệu:** Gồm 9 chỉ số chứng khoán toàn cầu nằm trong thư mục [dataset](file:///d:/UIT/1003_EPA_PROJECT/1.0.0/MoiraiMoGARCHs/src/dataset) (`VN30_INDEX`, `VN_INDEX`, `DAX_40`, `EuroNext_100`, `IBEX_35`, `KOSPI_index`, `SMI`, `snp500`, `Nikkei_225`).
* **Cấu trúc tệp dữ liệu thô:** Định dạng CSV, chứa chuỗi thời gian lịch sử từ năm 2010 trở đi. Ví dụ cấu trúc của chỉ số DAX 40:
  * `time`: Ngày giao dịch (Yyyy-Mm-Dd).
  * `close`: Giá đóng cửa của chỉ số.
  * `return_1_day` (tùy chọn): Tỷ suất sinh lợi được tính sẵn hoặc tính động từ cột `close`.

---

### III. CÁCH TÍNH TOÁN CÁC BIẾN (VARIABLE CALCULATION)

#### 1. Tỷ suất sinh lợi ngày (1-day log-return - $r_t$)
Để đảm bảo tính ổn định thống kê và tính chất cộng dồn theo thời gian trong toán học tài chính, tỷ suất sinh lợi được tính dưới dạng log-return phần trăm:
$$r_t = \ln\left(\frac{\text{Close}_t}{\text{Close}_{t-1}}\right) \times 100$$
*(Nhân với 100 để đưa về đơn vị điểm phần trăm (percentage points), ví dụ: mức thay đổi 0.005 sẽ được biểu diễn thành 0.5%).*

#### 2. Biến động thực tế mục tiêu (True Volatility - $\sigma_t$)
Độ biến động thực tế ngày tại thời điểm $t$ được định nghĩa bằng độ lệch chuẩn trượt (rolling standard deviation) của 60 ngày log-return lịch sử kết thúc vào ngày $t-1$:
$$\sigma_t = \sqrt{\frac{1}{W} \sum_{i=1}^{W} (r_{t-i} - \bar{r}_t)^2}$$
Trong đó:
* $W = 60$ là độ rộng cửa sổ tính toán (volatility window).
* $\bar{r}_t$ là trung bình tỷ suất sinh lợi của cửa sổ trượt: $\bar{r}_t = \frac{1}{W} \sum_{i=1}^{W} r_{t-i}$.
* **Dịch chuyển 1 bước (`shift(1)`):** Giá trị độ biến động $\sigma_t$ tại ngày $t$ chỉ được tính dựa trên chuỗi tỷ suất sinh lợi từ ngày $t-60$ đến ngày $t-1$. Điều này đảm bảo tính nhân quả tuyệt đối: tại thời điểm $t$, thông tin của ngày $t$ chưa xảy ra và không được đưa vào tính toán volatility mục tiêu của ngày đó.

---

### IV. CỬA SỔ THỜI GIAN (WINDOWS)

Mỗi mẫu huấn luyện (sample) trích xuất bằng phương pháp cửa sổ trượt (sliding window) chứa hai phân vùng thời gian:

```
Thời gian: [  t-60,  t-59,  ...,  t-1  ] [  t,  t+1,  ...,  t+20  ]
Cửa sổ:   |<----- Lookback (L=60) ----->| |<--- Horizon (H=21) --->|
Dữ liệu:           Input Returns                 Target Volatility
```

1. **Cửa sổ lịch sử (Lookback Window / Context Length - $L$):** $L = 60$ ngày.
   * Đầu vào mô hình là vector chứa 60 ngày tỷ suất sinh lợi lịch sử gần nhất:
     $$X_t = [r_{t-60}, r_{t-59}, \dots, r_{t-1}] \quad \in \mathbb{R}^{\text{batch} \times 60 \times 1}$$
2. **Cửa sổ dự báo (Forecast Horizon / Prediction Length - $H$):** $H = 21$ ngày.
   * Mô hình thực hiện dự báo đồng thời cho 21 ngày tương lai. Các mốc đánh giá chính để so sánh số liệu giữa các mô hình là $h \in \{1, 3, 5, 10, 21\}$.

---

### V. QUY CHUẨN QUY MÔ (SCALE)

Bài toán tài chính cực kỳ nhạy cảm với quy mô (scale) của dữ liệu. Repo hiện tại xử lý vấn đề này qua hai cơ chế:

1. **Quy mô tỷ suất sinh lợi phần trăm (Percentage scale):** Như công thức ở Mục III, log-return được nhân với $100$. Nếu giữ nguyên đơn vị gốc (ví dụ: $0.0015$), giá trị bình phương hoặc độ lệch chuẩn sẽ cực kỳ nhỏ ($0.00000225$), gây lỗi triệt tiêu gradient (gradient vanishing) khi huấn luyện mạng neural sâu.
2. **Bộ chuẩn hóa động (`PackedStdScaler`):** 
   Trong quá trình huấn luyện, lớp `PackedStdScaler` được tích hợp sẵn trong cấu trúc mô hình (`MoiraiModule` / `MoiraiMoEModule`). Bộ scaler này tự động chuẩn hóa chuỗi trả về trong cửa sổ quá khứ theo công thức:
   $$r_{\tau}^{\text{scaled}} = \frac{r_{\tau} - \text{loc}}{\text{scale}} \quad \text{với } \tau \in [t-60, t-1]$$
   Trong đó:
   * $\text{loc}$ là giá trị trung bình (mean) của chuỗi returns đầu vào trong cửa sổ context.
   * $\text{scale}$ là độ lệch chuẩn (standard deviation) của chuỗi returns đầu vào trong cửa sổ context.
   * Việc chuẩn hóa động này giúp đưa đầu vào của mọi batch về dạng phân phối chuẩn hóa có trung bình bằng 0 và phương sai bằng 1 trước khi đưa vào các lớp attention của Transformer.

---

### VI. BIẾN MỤC TIÊU DỰ BÁO ĐA BƯỚC (MULTI-HORIZON TARGET VARIABLE)

Tại thời điểm $t$, đối với mỗi bước tương lai $h \in \{1, 2, \dots, 21\}$, biến mục tiêu thực tế mà mô hình cần dự đoán ($\text{Target}_{t,h}$) là giá trị độ biến động thực tế tại ngày tương lai $t+h-1$:
$$\text{Target}_{t,h} = \sigma_{t+h-1} = \text{std}(r_{t+h-60 : t+h-1})$$

**Minh họa tính chất không rò rỉ dữ liệu trong dự báo:**
* **Tại bước dự báo $h=1$ (1 ngày tới):**
  $$\text{Target}_{t,1} = \sigma_{t} = \text{std}(r_{t-60 : t-1})$$
  *(Hoàn toàn sử dụng dữ liệu returns đã xảy ra trong quá khứ để làm đích dự báo).*
* **Tại bước dự báo $h=21$ (21 ngày tới):**
  $$\text{Target}_{t,21} = \sigma_{t+20} = \text{std}(r_{t-40 : t+19})$$
  *(Sử dụng dữ liệu returns thực tế của tương lai từ ngày $t$ đến ngày $t+19$ làm đích dự báo. Mô hình khi chạy thực tế chỉ được nhận input là 60 ngày quá khứ kết thúc ở ngày $t-1$, do đó mô hình bắt buộc phải học cách ánh xạ từ thông tin quá khứ để đoán trước được giá trị độ biến động tương lai này).