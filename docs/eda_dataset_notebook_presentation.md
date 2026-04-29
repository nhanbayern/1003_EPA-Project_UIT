# Trình bày học thuật: EDA dữ liệu chỉ số thị trường (2010-2025)

Mặc dù theo giả thuyết thị trường hiệu quả, các biến số thị trường thường được coi là ngẫu nhiên và khó dự báo, nhưng biến động (volatility) - thước đo của sự không chắc chắn trong các hoạt động tài chính - lại sở hữu những đặc tính thống kê đặc thù được gọi là "các sự thật định hình" (stylized facts), giúp tăng khả năng dự báo. Trái ngược với tính chất "bước đi ngẫu nhiên" của giá đóng cửa, sự thay đổi của biến động có xu hướng duy trì theo cụm (volatility clustering), tăng cao hơn khi thị trường sụt giảm so với khi tăng trưởng (asymmetric effect) và có tác động lâu dài đến sự tiến hóa của chính nó (long memory). Chính vì vậy, việc thiết lập mối quan hệ tương đương giữa các mô hình GARCH truyền thống với mạng nơ-ron (GARCH-NN) cho phép chúng ta tích hợp các quy luật phi ngẫu nhiên này vào cấu trúc học máy hiện đại như LSTM, từ đó cải thiện hiệu quả dự báo biến động rõ rệt so với việc sử dụng các mô hình thống kê hay mạng nơ-ron một cách riêng lẻ.

## 1. Mục tiêu nghiên cứu
Tài liệu này tóm tắt nội dung và cách phân tích trong notebook EDA đối với bộ dữ liệu chỉ số thị trường, với mục tiêu:
- Chuẩn hóa và đồng bộ dữ liệu theo cùng một cửa sổ thời gian.
- Mô tả đặc tính thống kê của mức giá đóng cửa (Close) và lợi suất logarit.
- Đánh giá mức độ biến động và rủi ro dưới góc nhìn phân phối.
- Tạo cơ sở cho các bước mô hình hóa biến động trong các nghiên cứu tiếp theo.

## 2. Phạm vi dữ liệu
- Nguồn dữ liệu: các file chỉ số thị trường trong thư mục dataset.
- Cửa sổ quan sát: từ ngày 2010-01-01 đến hết ngày 2025-12-31.
- Tần suất: dữ liệu theo ngày giao dịch.

Trong cửa sổ trên, mỗi thị trường được căn chỉnh theo các mốc thời gian hợp lệ để đảm bảo tính nhất quán khi so sánh.

## 3. Tiền xử lý và biến đổi
### 3.1. Làm sạch và chuẩn hóa
- Chuyển cột thời gian về định dạng thời gian.
- Chuyển cột giá đóng cửa về định dạng số.
- Loại bỏ quan sát lỗi, trùng lặp mốc thời gian, sắp xếp theo thứ tự thời gian tăng dần.

### 3.2. Đại lượng lợi suất sử dụng trong nghiên cứu
Nghiên cứu chỉ sử dụng **log return**, không sử dụng simple return.

Công thức:

$$
log\_return_t = \ln\left(\frac{Close_t}{Close_{t-1}}\right)
$$

Ý nghĩa:
- Phù hợp cho phân tích chuỗi thời gian tài chính vì có tính cộng dồn theo thời gian.
- Ổn định hơn khi đánh giá biến động và các tính chất phân phối.

## 4. Hệ thống chỉ số và phương pháp EDA

### 4.1. Nhóm thống kê mô tả cho Close
- Số quan sát, trung bình, độ lệch chuẩn.
- Giá trị nhỏ nhất, trung vị, giá trị lớn nhất.
- Các phân vị quan trọng (5%, 95%).

Mục đích: mô tả mặt bằng giá và biên độ dao động của từng thị trường.

### 4.2. Nhóm thống kê mô tả cho Log Return
- Số quan sát, trung bình, độ lệch chuẩn.
- Các phân vị sau (1%, 5%, 95%, 99%).
- Độ lệch (skewness), độ nhọn (kurtosis).

Mục đích:
- Kiểm tra tính bất đối xứng của phân phối.
- Đánh giá độ dày đuôi phân phối (tail risk) và đặc điểm cực trị.

### 4.3. Chỉ số biến động
- Biến động ngày: độ lệch chuẩn của log return.
- Biến động năm hóa:

$$
\sigma_{annual} = \sigma_{daily} \sqrt{252}
$$

Mục đích: so sánh mức độ biến động giữa các thị trường trên cùng một thang đo.

### 4.4. Chỉ số rủi ro đuôi phân phối
- VaR lịch sử mức 95% và 99% (dựa trên phân vị của log return).

Mục đích: định lượng mức tổn thất biên có thể xảy ra ở các ngưỡng xác suất quan trọng.

### 4.5. Chỉ số drawdown trên chuỗi giá
- Tính drawdown từ giá đóng cửa và cực đại lưu động.
- Báo cáo max drawdown trong giai đoạn quan sát.

Mục đích: phân tích mức sụt giảm lớn nhất từ đỉnh, phản ánh rủi ro giảm sâu kéo dài.

## 5. Trực quan hóa và dashboard tương tác
Dashboard cho phép thao tác để phân tích và so sánh:
- Chọn chế độ một thị trường hoặc hai thị trường để đối chiếu.
- Chọn biến cần xem phân phối: Close, Log Return hoặc Volatility.
- Thay đổi khoảng thời gian con trong cửa sổ 2010-2025.
- Điều chỉnh số cột histogram để quan sát hình dạng phân phối.

Kết quả hiển thị:
- Biểu đồ phân phối dạng histogram chồng lấp để so sánh trực tiếp.
- Bảng thống kê tóm tắt ứng với lựa chọn hiện tại.

## 6. Giá trị học thuật và hướng diễn giải
Khung EDA này hỗ trợ:
- Nhận diện sự khác biệt cấu trúc biến động giữa các thị trường phát triển và mới nổi.
- Đánh giá tail risk thông qua VaR và các phân vị cực biên.
- Làm rõ tính chất phân phối không chuẩn (nếu có) qua skewness và kurtosis.
- Tạo nền tảng cho các mô hình dự báo biến động, quản trị rủi ro, và so sánh hiệu năng mô hình sau này.

## 7. Cách tạo biến mục tiêu True_Volatility trong benchmark
True_Volatility được xây dựng theo realized volatility từ log return quá khứ, không sử dụng giá tương lai.

### 7.1. Bước tạo return
Trong `AAAI24_GARCH_NN_Reproduction/core/data_processor.py`, return được tính theo:

$$
r_t = \ln\left(\frac{Close_t}{Close_{t-1}}\right)
$$

Sau đó chuỗi return được nhân hệ số 100 trong cùng file, nên đơn vị được biểu diễn theo percent-point.

### 7.2. Bước tạo volatility mục tiêu
Trong `AAAI24_GARCH_NN_Reproduction/core/data_processor.py`, cửa sổ mặc định là 60 (`DEFAULT_VOL_WINDOW`) và được dùng trong benchmark qua `AAAI24_GARCH_NN_Reproduction/experiments/run_benchmark.py`.

Trong hàm tạo target ở `AAAI24_GARCH_NN_Reproduction/experiments/run_benchmark.py`, code dùng `shift(1)` kết hợp `rolling(...).std(ddof=0)`, tương ứng:

$$
\sigma_t = \sqrt{\frac{1}{60}\sum_{k=1}^{60}(r_{t-k} - \bar{r}_t)^2}
$$

Do có `shift(1)`, volatility tại thời điểm $t$ chỉ dùng thông tin return quá khứ đến $t-1$.

### 7.3. Gán vào cột True_Volatility theo horizon
Target matrix được tạo trong `AAAI24_GARCH_NN_Reproduction/experiments/run_benchmark.py`, sau đó lấy theo các horizon $h \in \{1,3,5,10,21\}$ để ghi vào cột `True_Volatility` trong bảng dự báo chi tiết.

Về mặt chỉ số:

$$
TrueVol(i, h) = \sigma_{i+h}
$$

Ghi chú: để tính được các điểm đầu của tập test, pipeline nối lịch sử train và validation làm ngữ cảnh rolling trước khi tính target trên đoạn test.

## 8. Kết quả định lượng từ notebook EDA
Các bảng dưới đây được tổng hợp từ output của các cell phân tích dữ liệu và thống kê trong notebook.

### 8.1. Độ bao phủ dữ liệu sau tiền xử lý
| Dataset | Obs Close | Obs Log Return | Obs Volatility | Start | End |
|---|---:|---:|---:|---|---|
| DAX_40 | 4059 | 4058 | 3998 | 2010-01-04 | 2025-12-30 |
| EuroNext_100 | 4099 | 4098 | 4038 | 2010-01-04 | 2025-12-31 |
| IBEX_35 | 4100 | 4099 | 4039 | 2010-01-04 | 2025-12-31 |
| KOSPI_index | 3934 | 3933 | 3873 | 2010-01-04 | 2025-12-30 |
| Nikkei_225 | 3913 | 3912 | 3852 | 2010-01-04 | 2025-12-30 |
| SMI | 4023 | 4022 | 3962 | 2010-01-04 | 2025-12-30 |
| VN30_INDEX | 3992 | 3991 | 3931 | 2010-01-04 | 2025-12-31 |
| VN_INDEX | 3992 | 3991 | 3931 | 2010-01-04 | 2025-12-31 |
| snp500 | 4024 | 4023 | 3963 | 2010-01-04 | 2025-12-31 |

Nhận xét nhanh:
- Sau khi tạo log return và volatility (window 60, shift(1)), mỗi thị trường mất thêm khoảng 60 quan sát đầu cho cột volatility.
- Các chuỗi đều có độ dài đủ lớn (xấp xỉ 3.8k-4.0k quan sát volatility) để phân tích phân phối và rủi ro ổn định.

### 8.2. Bảng thống kê rủi ro và phân phối log return
| Dataset | Annualized Volatility | VaR 99% | Max Drawdown | Skewness | Kurtosis |
|---|---:|---:|---:|---:|---:|
| DAX_40 | 0.194336 | -0.034838 | -0.387794 | -0.482311 | 7.583796 |
| EuroNext_100 | 0.177955 | -0.033435 | -0.379130 | -0.659796 | 8.773266 |
| IBEX_35 | 0.212416 | -0.036637 | -0.512677 | -0.563693 | 11.520543 |
| KOSPI_index | 0.171808 | -0.030956 | -0.438979 | -0.441024 | 6.638427 |
| Nikkei_225 | 0.212959 | -0.036718 | -0.317989 | -0.491524 | 7.687328 |
| SMI | 0.151053 | -0.026751 | -0.312247 | -0.851884 | 9.154651 |
| VN30_INDEX | 0.194221 | -0.040547 | -0.481387 | -0.675733 | 3.775288 |
| VN_INDEX | 0.187228 | -0.038546 | -0.452633 | -0.774307 | 3.885045 |
| snp500 | 0.173704 | -0.032208 | -0.339250 | -0.614741 | 13.573547 |

### 8.3. Bảng thống kê biến Volatility (realized proxy)
| Dataset | Vol Mean | Vol Std | Vol Q95 | Vol Max |
|---|---:|---:|---:|---:|
| DAX_40 | 1.123319 | 0.475551 | 1.946423 | 3.475726 |
| EuroNext_100 | 1.024170 | 0.448044 | 1.813406 | 3.313875 |
| IBEX_35 | 1.221281 | 0.530016 | 2.317355 | 3.491330 |
| KOSPI_index | 0.988332 | 0.404790 | 1.689232 | 2.885552 |
| Nikkei_225 | 1.258139 | 0.442546 | 2.191910 | 2.821688 |
| SMI | 0.881012 | 0.355809 | 1.567358 | 2.612815 |
| VN30_INDEX | 1.138510 | 0.392602 | 1.909099 | 2.301074 |
| VN_INDEX | 1.097384 | 0.380292 | 1.815624 | 2.189591 |
| snp500 | 0.961727 | 0.516925 | 1.839889 | 3.998411 |

### 8.4. Insight rút ra từ kết quả
- Theo annualized volatility, nhóm biến động cao nhất là Nikkei_225 (0.212959) và IBEX_35 (0.212416), trong khi SMI thấp nhất (0.151053).
- Theo VaR 99%, rủi ro đuôi trái mạnh nhất thuộc về VN30_INDEX (-0.040547) và VN_INDEX (-0.038546), cho thấy mức lỗ cực biên tiềm năng lớn hơn so với phần lớn thị trường còn lại.
- Theo max drawdown, IBEX_35 giảm sâu nhất (-0.512677), sau đó là VN30_INDEX (-0.481387) và VN_INDEX (-0.452633), phản ánh rủi ro suy giảm kéo dài đáng kể.
- Tất cả thị trường đều có skewness âm, hàm ý phân phối log return lệch trái; độ lệch trái mạnh hơn ở SMI (-0.851884), VN_INDEX (-0.774307), VN30_INDEX (-0.675733).
- Kurtosis đều lớn hơn 3, xác nhận đặc tính đuôi dày; snp500 (13.573547) và IBEX_35 (11.520543) nổi bật về mức độ cực trị.
- Với biến volatility proxy, Nikkei_225 và IBEX_35 có mức trung bình cao (1.258139 và 1.221281), còn snp500 có đỉnh volatility lớn nhất (Vol Max = 3.998411), cho thấy xuất hiện các cú sốc ngắn hạn mạnh.

## 9. Kết luận
Notebook EDA đã được thiết kế theo định hướng thống kê - rủi ro cho dữ liệu chỉ số thị trường trong giai đoạn 2010-2025, sử dụng log return làm biến trung tâm cho phân tích biến động và phân phối. Cấu trúc này phù hợp cho trình bày học thuật và mở rộng sang các bài toán dự báo/quản trị rủi ro tài chính.
