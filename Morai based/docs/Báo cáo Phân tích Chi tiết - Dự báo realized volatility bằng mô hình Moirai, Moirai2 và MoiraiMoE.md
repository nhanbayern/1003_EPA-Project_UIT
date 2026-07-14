# Báo cáo Phân tích Chi tiết: Dự báo realized volatility bằng mô hình Moirai, Moirai2 và MoiraiMoE

Tài liệu này cung cấp phân tích chi tiết nhằm thiết lập khung nghiên cứu thực nghiệm so sánh hiệu suất giữa 3 mô hình nền tảng chuỗi thời gian (**Moirai**, **Moirai2**, **MoiraiMoE**) trên cùng một bài toán dự báo độ biến động tài chính, dựa trên các thông số dữ liệu và phương pháp được mô tả trong [data.md](file:///d:/UIT/1003_EPA_PROJECT/1.0.0/Morai%20based/docs/data.md).

---

## 1. Bài toán / Vấn đề (The Problem)

### 1.1. Tên bài toán
Dự báo độ biến động tài chính đa bước thời gian (Multi-horizon Financial Volatility Forecasting).

### 1.2. Mục tiêu dự báo
Sử dụng thông tin tỷ suất sinh lợi trong quá khứ để dự báo độ biến động thực tế (Realized Volatility) ký hiệu là $\sigma_{t,h}$ tại các thời điểm tương lai ứng với các chân trời dự báo (horizons) $h \in \{1, 3, 5, 10, 21\}$ ngày.

### 1.3. Nguyên tắc nhân quả & Thiết lập Causal Setup
Để đảm bảo tính thực tiễn và loại bỏ hoàn toàn hiện tượng rò rỉ dữ liệu tương lai (data leakage):
*   Tại thời điểm đưa ra dự báo $t$, mô hình chỉ được phép tiếp cận chuỗi tỷ suất sinh lợi quá khứ trong cửa sổ lịch sử (Lookback Window) có độ dài $L = 60$:
    $$x_{t} = (r_{t-60}, r_{t-59}, \dots, r_{t-1}) \quad \in \mathbb{R}^{60}$$
*   Mục tiêu dự báo $\sigma_{t,h}$ được xây dựng hoàn toàn từ các tỷ suất sinh lợi tương lai bắt đầu từ thời điểm $t$ đến ngày kết thúc chân trời dự báo $t+h-1$. Mô hình hoàn toàn không được tiếp cận bất kỳ thông tin nào của ngày $t$ và tương lai sau ngày $t$ tại thời điểm dự đoán.

---

## 2. Tập dữ liệu (Dataset)

Nghiên cứu sử dụng dữ liệu lịch sử từ năm **2010 đến 2025** của **9 chỉ số chứng khoán** đại diện cho thị trường toàn cầu và Việt Nam:
*   **Chỉ số quốc tế (Investing.com):** DAX 40 (Đức), Euronext 100 (Châu Âu), IBEX 35 (Tây Ban Nha), KOSPI Index (Hàn Quốc), Nikkei 225 (Nhật Bản), SMI (Thụy Sĩ), S&P 500 (Mỹ).
*   **Chỉ số Việt Nam (vnstock):** VN30, VN-Index.

### 2.1. Cấu trúc tệp dữ liệu thô (Raw Data Schema)
Các tệp CSV chứa 6 cột giá và khối lượng cơ bản:
$$\{\text{time, open, high, low, close, volume}\}$$

### 2.2. Kiểm định Đặc trưng phân phối (Distribution Diagnostics)
*   **Kiểm định Jarque-Bera:** Bác bỏ hoàn toàn giả thuyết phân phối chuẩn (Gaussian distribution) ở mức ý nghĩa 5% cho tất cả 9 bộ dữ liệu trên cả tập train-validation và tập test (p-value bằng 0 đến giới hạn máy tính).
*   **Khớp phân phối Student-t:** Ước lượng tham số bậc tự do $\nu$ (degree-of-freedom) của mô hình Student-t cho kết quả cực kỳ phù hợp:
    *   Tập Train: $\nu \in [2.706, 4.926]$ (fit p-value $\in [0.242, 0.842]$)
    *   Tập Validation: $\nu \in [2.434, 3.772]$ (fit p-value $\in [0.359, 0.970]$)
    *   Tập Test: $\nu \in [2.385, 5.963]$ (fit p-value $\in [0.507, 0.960]$)
*   **Kết luận phân phối:** Nghiên cứu bác bỏ giả định phân phối chuẩn và sử dụng phân phối Student-t làm giả định phân phối cố định xuyên suốt quá trình huấn luyện và đánh giá rủi ro.

---

## 3. Quy trình Tiền xử lý và Phân tách Dữ liệu

### 3.1. Phân tách tập dữ liệu (Distribution-Aware Data Split)

> [!NOTE]
> **Giải pháp tối ưu cho dữ liệu khởi động (Burn-in Data):**
> Mặc dù tổng số mẫu thử nghiệm chính thức (từ 2010 đến 2025) khớp hoàn toàn với số lượng mẫu trong Table I, tập dữ liệu thô thực tế bắt đầu từ năm 2008 (hoặc 2009 đối với VN30). 
> Trong quy trình tiền xử lý, toàn bộ phần dữ liệu trước ngày 01/01/2010 được giữ lại làm giai đoạn **khởi động (Burn-in)**. Điều này giúp chúng ta có đủ 60 ngày tỷ suất sinh lợi lịch sử trong quá khứ làm lookback window ($x_t$) cho các mẫu đầu tiên của năm 2010 mà không bị trống dữ liệu hoặc phải áp dụng phương pháp điền nội suy.

Để việc đánh giá không bị ảnh hưởng bởi sự dịch chuyển phân phối theo thời gian, quy trình áp dụng phân tách dữ liệu nhận biết phân phối (Distribution-Aware Split):
1.  **Thuật toán tìm kiếm điểm cắt:** Tìm kiếm cặp điểm cắt thời gian $(i, j)$ để phân tách chuỗi thành 3 tập (Train, Validation, Test) sao cho độ dài tối thiểu của mỗi tập đạt ít nhất $\max(30, 0.1n)$ với $n$ là tổng số mẫu.
2.  **Ước lượng tham số phân phối:** Trên mỗi phân đoạn, tính toán độ nhọn vượt mức (excess kurtosis $k$) của tỷ suất sinh lợi, từ đó ước lượng bậc tự do $\nu$ của phân phối Student-t qua công thức:
    $$\nu = 4 + \frac{6}{k}$$
3.  **Điều kiện chấp nhận:**
    *   Các giá trị $\nu_{\text{train}}, \nu_{\text{val}}, \nu_{\text{test}}$ phải hữu hạn.
    *   Mỗi giá trị $\nu$ không được lệch quá 20% so với giá trị trung bình $\bar{\nu}$ của cả ba phân đoạn.
4.  **Tiêu chí tối ưu:** Chọn điểm cắt $(i, j)$ giảm thiểu tối đa tổng độ lệch của ba giá trị $\nu$ so với giá trị trung bình:
    $$\min_{i,j} \sum_{p \in \{\text{train, val, test}\}} |\nu_p - \bar{\nu}|$$
5.  **Phương án dự phòng (Fallback):** Nếu không tồn tại cặp điểm cắt nào thỏa mãn điều kiện trên, hệ thống tự động rơi về tỷ lệ phân tách mặc định là 60/20/20.

Kích thước mẫu thực tế sau khi phân tách được chi tiết trong bảng dưới đây:

| Chỉ số | Nguồn | Tập Train | Tập Validation | Tập Test | Tổng số mẫu |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DAX 40** | Investing.com | 1983 | 1104 | 972 | 4059 |
| **Euronext 100** | Investing.com | 2005 | 1115 | 979 | 4099 |
| **IBEX 35** | Investing.com | 1815 | 1362 | 923 | 4100 |
| **KOSPI Index** | Investing.com | 1944 | 1109 | 881 | 3934 |
| **Nikkei 225** | Investing.com | 1677 | 1174 | 1062 | 3913 |
| **SMI** | Investing.com | 1987 | 1135 | 901 | 4023 |
| **S&P 500** | Investing.com | 1988 | 1109 | 927 | 4024 |
| **VN30** | vnstock | 1971 | 1096 | 925 | 3992 |
| **VN-Index** | vnstock | 1964 | 1103 | 925 | 3992 |

---

### 3.2. Tạo mẫu bằng Cửa sổ trượt (Sliding Window)

Với mỗi chuỗi thời gian, ta áp dụng một cửa sổ trượt tịnh tiến theo từng bước thời gian ngày $t$:

```
Quá khứ (Lookback L=60)                    Tương lai (Horizon H=21)
[ t-60,  t-59,  ...,  t-1 ]                [ t,  t+1,  ...,  t+20 ]
| <--- Returns Đầu vào (x_t) ---> |        | <--- Volatility Mục tiêu (sigma_t,h) ---> |
```

#### Bước 1: Tính tỷ suất sinh lợi ngày ($r_t$)
$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) \times 100$$

#### Bước 2: Tạo vector đầu vào ($x_t$)
Tại ngày dự báo $t$, vector đầu vào chứa 60 ngày tỷ suất sinh lợi lịch sử gần nhất:
$$x_t = [r_{t-60}, r_{t-59}, \dots, r_{t-1}] \quad \in \mathbb{R}^{60}$$

#### Bước 3: Tính toán biến mục tiêu realized volatility tương lai ($\sigma_{t,h}$)
Với mỗi chân trời $h \in \{1, 3, 5, 10, 21\}$, độ biến động thực tế mục tiêu được tính toán trực tiếp trên cửa sổ tương lai $[t, t+h-1]$:
$$\sigma_{t,h} = \sqrt{\frac{1}{h} \sum_{i=0}^{h-1} (r_{t+i} - \bar{r}_{t,h})^2}$$
Trong đó tỷ suất sinh lợi trung bình tương lai tương ứng là:
$$\bar{r}_{t,h} = \frac{1}{h} \sum_{i=0}^{h-1} r_{t+i}$$

> [!IMPORTANT]
> **Nhận xét quan trọng về tính nhân quả:**
> Thiết lập này hoàn toàn loại bỏ sự chồng chéo (overlap) giữa cửa sổ dữ liệu đầu vào ($[t-60, t-1]$) và cửa sổ tính target tương lai ($[t, t+h-1]$). Điều này đảm bảo mô hình dự báo biến động thực tế bằng cách học cách ánh xạ từ thông tin quá khứ sang tương lai, chứ không phải tái cấu trúc lại biến động từ thông tin hiện tại.

---

## 4. Khung quản lý huấn luyện (Training Management Framework)

Để thực hiện so sánh hiệu năng của 3 mô hình nền tảng một cách công bằng nhất, ta thiết lập một khung quản lý huấn luyện đồng bộ (Unified Representation Framework):

```
       [Returns quá khứ x_t] (L=60)
                 │
      ┌──────────┼──────────┐
      ▼          ▼          ▼
   [Moirai]  [Moirai2]  [MoiraiMoE]   <--- (Frozen Backbones)
      │          │          │
      ▼          ▼          ▼
   [Embeds]  [Embeds]   [Embeds]      <--- (Avg-pooled representations)
      │          │          │
      └──────────┼──────────┘
                 ▼
         [MLP Regression Head]        <--- (Trainable Parameters)
                 │
                 ▼
  [Dự báo Volatility 5 chiều] 
  (h = 1, 3, 5, 10, 21 tương lai)
```

### 4.1. Cơ chế đóng băng Backbone (Frozen Backbones)
Chúng ta tải các trọng số pre-trained chính thức từ Hugging Face Hub:
*   Moirai: `Salesforce/moirai-1.1-R-base` (hoặc `small`)
*   Moirai2: `Salesforce/moirai-2.0-R-small`
*   MoiraiMoE: `Salesforce/moirai-moe-1.0-R-small`

Toàn bộ các trọng số của lớp Transformer của cả 3 mô hình đều được đóng băng (`requires_grad = False`). Chúng đóng vai trò là bộ trích xuất đặc trưng chuỗi thời gian (Feature Extractors).

### 4.2. Trích xuất đặc trưng & Pooling
1.  Đưa chuỗi returns đầu vào $x_t \in \mathbb{R}^{60}$ qua Encoder của từng mô hình.
2.  Đầu ra từ lớp ẩn cuối cùng của Encoder sẽ có dạng ma trận đặc trưng $\mathbf{H}_t \in \mathbb{R}^{N_T \times d_{\text{model}}}$, với $N_T$ là số lượng tokens sau khi phân mảnh (patching) và $d_{\text{model}}$ là kích thước ẩn của mô hình (ví dụ: $d_{\text{model}} = 768$ đối với bản Base).
3.  Áp dụng phép toán **Average Pooling** trên chiều tokens $N_T$ để thu về vector biểu diễn cố định kích thước cho mỗi mẫu dữ liệu:
    $$\mathbf{z}_t = \text{Mean-Pool}(\mathbf{H}_t) \quad \in \mathbb{R}^{d_{\text{model}}}$$

### 4.3. Đầu hồi quy dùng chung (Shared MLP Regression Head)
Thiết lập một đầu hồi quy chung cho cả 3 mô hình để đảm bảo tính so sánh công bằng:
*   **Kiến trúc:** Một mạng nơ-ron truyền thẳng (MLP) gồm 2 lớp:
    *   Lớp 1: Tuyến tính $\mathbb{R}^{d_{\text{model}}} \to \mathbb{R}^{256}$ + Kích hoạt GeLU + Dropout (0.2).
    *   Lớp 2: Tuyến tính $\mathbb{R}^{256} \to \mathbb{R}^5$ (ứng với 5 giá trị dự báo cho 5 horizons).
*   **Hàm mất mát huấn luyện:** Mean Squared Error (MSE) tính trực tiếp trên vector dự đoán $\hat{\mathbf{y}}_t$ so với $\mathbf{y}_t = [\sigma_{t,1}, \sigma_{t,3}, \sigma_{t,5}, \sigma_{t,10}, \sigma_{t,21}]$.
*   **Quy trình quản lý:**
    *   Huấn luyện đầu MLP trên tập Train.
    *   Đánh giá sớm và thực hiện **Early Stopping** dựa trên loss của tập Validation.
    *   Báo cáo hiệu năng cuối cùng trên tập Test.

---

## 5. Phương pháp Tối ưu hóa hàm Loss theo gốc

Hiểu rõ phương pháp tối ưu hóa hàm loss nguyên bản (trong pha pre-training) của từng mô hình giúp giải thích cách thức chúng xây dựng không gian biểu diễn ẩn để trích xuất đặc trưng:

### 5.1. Moirai (1.0/1.1) & MoiraiMoE: Phân phối Xác suất Tham số (Parametric Distribution NLL Loss)
Cả hai mô hình này đều sử dụng phương pháp **dự báo phân phối xác suất tham số**. Mô hình không dự báo trực tiếp giá trị chuỗi thời gian mà dự báo các tham số đầu ra của một phân phối giả định $\mathcal{D}$.
*   **Hàm mất mát mục tiêu:** Negative Log-Likelihood (NLL).
*   **Công thức toán học:**
    $$\mathcal{L}_{\text{NLL}}(\Theta) = -\sum_{\tau} \ln f(y_{\tau} \,|\, \Theta_{\tau})$$
    Trong đó:
    *   $y_{\tau}$ là giá trị thực tế của chuỗi thời gian tại thời điểm $\tau$.
    *   $f(\cdot | \Theta_{\tau})$ là hàm mật độ xác suất (Probability Density Function - PDF) của phân phối đã chọn.
    *   $\Theta_{\tau}$ là vector các tham số phân phối được mô hình dự báo tại thời điểm $\tau$. Đối với phân phối Student-t, $\Theta_{\tau} = (\mu_{\tau}, \sigma_{\tau}, \nu_{\tau})$ tương ứng với vị trí (loc), quy mô (scale) và bậc tự do (degrees of freedom).
*   **MoiraiMoE:** Áp dụng cùng hàm lỗi NLL nhưng cơ chế lan truyền ngược được phân phối qua các cổng định tuyến (routing gates) đến các chuyên gia (experts) khác nhau dựa trên entropy và cân bằng tải (load balancing loss).

### 5.2. Moirai2: Hồi quy Phân vị (Quantile Pinball Loss)
Khác với thế hệ đầu tiên, Moirai2 từ bỏ cách tiếp cận phân phối tham số và chuyển sang **dự báo phân vị (quantile forecasting)** cho một tập hợp cố định các mức phân vị $\tau \in \{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\}$.
*   **Hàm mất mát mục tiêu:** Pinball Loss (Quantile Loss).
*   **Công thức toán học:**
    Với mỗi mức phân vị $\tau \in (0, 1)$, Pinball Loss tại một điểm dự báo được định nghĩa là:
    $$\mathcal{L}_{\text{Pinball}}(y, \hat{y}_{\tau}; \tau) = \max\Big( \tau (y - \hat{y}_{\tau}), (\tau - 1)(y - \hat{y}_{\tau}) \Big)$$
    Hay viết dưới dạng tương đương:
    $$\mathcal{L}_{\text{Pinball}}(y, \hat{y}_{\tau}; \tau) = \begin{cases} 
      \tau (y - \hat{y}_{\tau}) & \text{nếu } y \ge \hat{y}_{\tau} \\
      (1 - \tau) (\hat{y}_{\tau} - y) & \text{nếu } y < \hat{y}_{\tau} 
    \end{cases}$$
    Trong đó:
    *   $y$ là giá trị thực tế của chuỗi thời gian.
    *   $\hat{y}_{\tau}$ là giá trị phân vị $\tau$ mà mô hình dự đoán.
*   **Tổng loss tối ưu hóa:** Là giá trị trung bình của Pinball Loss trên toàn bộ các mức phân vị $\tau$ và toàn bộ các bước thời gian trong cửa sổ dự báo:
    $$\mathcal{L}_{\text{Moirai2}} = \frac{1}{|Q| \cdot H} \sum_{\tau \in Q} \sum_{k=1}^{H} \mathcal{L}_{\text{Pinball}}(y_{t+k}, \hat{y}_{t+k, \tau}; \tau)$$
    *Ý nghĩa:* Giúp Moirai2 dự báo không phụ thuộc vào bất kỳ giả định phân phối tham số cụ thể nào, mang lại độ dẻo dai và chính xác cao hơn khi đối mặt với dữ liệu thực tế có phân phối phức tạp hoặc đa phương thức (multimodal).
