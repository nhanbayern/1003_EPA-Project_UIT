# Phương pháp nghiên cứu

Phần này trình bày chi tiết về quy trình xử lý dữ liệu, thiết lập bài toán, và các thông số triển khai cụ thể cho 10 mô hình dự báo biến động (volatility) được đánh giá trong nghiên cứu này. Phương pháp luận được xây dựng nhằm đảm bảo một khung đánh giá thống nhất, nơi các mô hình từ các hệ tư tưởng khác nhau — kinh tế lượng, học sâu, mô hình lai, và mô hình nền tảng (foundation models) — đều được huấn luyện và kiểm thử trong các điều kiện hoàn toàn đồng nhất.

## 1. Dữ liệu và Thiết lập bài toán

### 1.1. Nguồn dữ liệu và Tiền xử lý
Nghiên cứu sử dụng 9 chỉ số thị trường chứng khoán lớn: DAX 40, Euronext 100, IBEX 35, KOSPI Index, Nikkei 225, SMI, S&P 500, VN30, và VN-Index. Để đảm bảo tính nhất quán, toàn bộ dữ liệu lịch sử về giá và khối lượng giao dịch hàng ngày của cả 9 chỉ số từ năm 2010 đến năm 2025 đều được thu thập độc quyền từ nền tảng Investing.com.

Tỷ suất sinh lợi logarit hàng ngày (tính theo phần trăm) được tính toán từ giá đóng cửa như sau:
$$ r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) \times 100 $$
trong đó $P_t$ là giá đóng cửa tại ngày $t$.

### 1.2. Bài toán dự báo và Tính nhân quả
Bài toán dự báo được thiết lập dưới dạng dự báo đa bước (multi-horizon prediction). Tại bất kỳ thời điểm dự báo $t$ nào, các mô hình được cung cấp một cửa sổ dữ liệu quá khứ gồm 60 ngày tỷ suất sinh lợi:
$$ x_t = (r_{t-60}, r_{t-59}, \dots, r_{t-1}) $$

Mục tiêu là dự báo biến động thực tế (realized volatility) tại 5 khoảng thời gian trong tương lai (horizons): $h \in \{1, 3, 5, 10, 21\}$. Để đảm bảo tính nhân quả chặt chẽ và ngăn ngừa rò rỉ dữ liệu (data leakage), biến động thực tế mục tiêu tại horizon $h$ được định nghĩa là độ lệch chuẩn trượt (rolling standard deviation) của 60 ngày, kết thúc chính xác tại thời điểm $t+h-1$:
$$ \sigma_{t,h} = \sqrt{\frac{1}{60}\sum_{i=1}^{60}(r_{t+h-i} - \bar{r}_{t,h})^2} $$
trong đó $\bar{r}_{t,h}$ là trung bình mẫu của tỷ suất sinh lợi trong cửa sổ 60 ngày đó. Điều quan trọng là tại thời điểm dự báo $t$, các mô hình hoàn toàn không được tiếp cận với bất kỳ dữ liệu tỷ suất sinh lợi nào từ thời điểm $t$ trở đi.

### 1.3. Phân tách dữ liệu dựa trên phân phối
Tỷ suất sinh lợi tài chính luôn có đặc điểm đuôi dày (heavy tails), dẫn đến việc bác bỏ phân phối chuẩn (p-value của kiểm định Jarque-Bera $\approx 0$). Do đó, nghiên cứu này áp dụng phân phối Student-t cho việc mô hình hóa rủi ro. Tập dữ liệu được chia theo trình tự thời gian thành các tập huấn luyện (training), kiểm định (validation), và kiểm thử (test). Để bảo toàn các đặc tính thống kê của chuỗi thời gian qua các tập dữ liệu, một thuật toán nhận biết phân phối được sử dụng để xác định các điểm cắt, đảm bảo rằng tham số bậc tự do (degrees of freedom - $\nu$) của phân phối Student-t ước lượng được duy trì ổn định giữa ba tập dữ liệu này.

## 2. Triển khai mô hình và Siêu tham số

Khung đánh giá bao gồm ba nhóm mô hình: các mô hình kinh tế lượng truyền thống, các kiến trúc học sâu đầu-cuối (Transformer và Hybrid), và các mô hình nền tảng chuỗi thời gian đã được huấn luyện trước (Moirai).

### 2.1. Các mô hình Kinh tế lượng (Họ GARCH)
Các mô hình kinh tế lượng nắm bắt rõ ràng hiện tượng phân cụm biến động (volatility clustering) và tính bền vững bằng cách mô hình hóa động học của phương sai có điều kiện [1].
- **Mô hình:** GARCH(1,1) tiêu chuẩn [1], GJR-GARCH(1,1) (kết hợp phản ứng bất đối xứng với các cú sốc tiêu cực) [2], và FI-GARCH(1,d,1) (tích hợp phân số cho các quá trình có bộ nhớ dài) [3].
- **Triển khai:** Được cài đặt bằng thư viện Python `arch`. Để đảm bảo khả năng hội tụ trên các chuỗi thời gian tài chính dài, một cửa sổ lịch sử trượt (rolling window) gồm 1.000 quan sát gần nhất được sử dụng để khớp (fit) các mô hình một cách linh hoạt tại mỗi bước kiểm thử.
- **Giả định & Đầu ra:** Các mô hình giả định phân phối Student-t cho các sai số (innovations). Chúng được thiết lập với phương trình trung bình bằng 0 (`mean="Zero"`). Các dự báo phương sai có điều kiện đa bước $\hat{\sigma}_{t+h}^2$ được tạo ra theo phương pháp đệ quy, và biến động dự báo cuối cùng được trích xuất là $\sqrt{\hat{\sigma}_{t+h-1}^2}$ cho mỗi horizon $h$.

### 2.2. Các mô hình Học sâu dựa trên Transformer
Kiến trúc Transformer tận dụng cơ chế tự chú ý (self-attention) [4] để nắm bắt các phụ thuộc dài hạn trong chuỗi thời gian mà không cần xử lý tuần tự đệ quy.
- **Mô hình:** Vanilla Transformer [4], Informer (sử dụng ProbSparse attention để tăng hiệu suất) [5], Autoformer (kết hợp phân rã chuỗi và tự tương quan) [6], và Reformer (sử dụng locality-sensitive hashing) [7].
- **Cấu hình:** Các mô hình được thiết lập với độ dài chuỗi đầu vào (sequence length) là 60, độ dài nhãn (label length) là 30, và độ dài dự báo đầu ra là 21. Chiều ẩn (`d_model`) được đặt là 128, với 4 lớp encoder.
- **Huấn luyện:** Các mô hình ánh xạ trực tiếp chuỗi tỷ suất sinh lợi đầu vào tới mục tiêu biến động đa bước. Chúng được huấn luyện bằng hàm mất mát Student-t Negative Log-Likelihood (NLL) tùy chỉnh, hàm này vốn tương thích với đặc tính đuôi dày của tỷ suất sinh lợi tài chính. Quá trình tối ưu hóa được thực hiện bằng thuật toán AdamW (Learning Rate = $10^{-3}$, Weight Decay = $10^{-4}$) trong 30 epochs với batch size là 128 và cơ chế dừng sớm (early stopping patience = 5).

### 2.3. Mô hình lai GARCH-LSTM
Mô hình lai GARCH-LSTM [8] kết hợp tính quy nạp (inductive bias) tài chính của các mô hình kinh tế lượng với khả năng học biểu diễn phi tuyến của mạng nơ-ron.
- **Kiến trúc:** Một cell hồi quy tùy chỉnh (`GARCH_LSTM_Cell`) nhúng trực tiếp phương trình cập nhật phương sai có điều kiện của GARCH vào cơ chế cổng (gating) của LSTM. Cell này nhận cả tỷ suất sinh lợi trước đó và phương sai trước đó làm đầu vào để cập nhật trạng thái ẩn và tính toán phương sai hiện tại.
- **Huấn luyện và Dự báo:** Mô hình được huấn luyện để dự báo 1 bước ($h=1$) bằng kỹ thuật teacher forcing (cung cấp phương sai lịch sử thực tế trong quá trình huấn luyện). Hàm mất mát là Student-t NLL. Trong quá trình suy luận (inference), các dự báo đa bước lên đến $h=21$ được tạo ra theo hướng tự hồi quy (autoregressively) bằng cách nạp lại phương sai vừa dự báo vào cell, đồng thời giả định tỷ suất sinh lợi kỳ vọng trong tương lai là 0.

### 2.4. Các mô hình nền tảng dựa trên Moirai
Các mô hình Moirai là các mô hình nền tảng chuỗi thời gian quy mô lớn, được huấn luyện trước bởi Salesforce [9], sử dụng kiến trúc masked encoder hỗ trợ kích thước patch thay đổi.
- **Mô hình:** Moirai 1.0, Moirai 2.0, và Moirai-MoE (Mixture of Experts). Các phiên bản tham số `small` được sử dụng.
- **Triển khai:** Khả năng biểu diễn phổ quát của Moirai được tận dụng qua mô hình trích xuất đặc trưng (feature extraction). Các chuỗi tỷ suất sinh lợi 60 ngày được chèn (padding) thêm để đạt 64 ngày nhằm khớp với kích thước patch cố định là 16 (tạo ra 4 patches mỗi chuỗi). Backbone được huấn luyện trước sẽ bị đóng băng (frozen), và các biểu diễn gộp (pooled representations) của nó được đưa vào một nhánh hồi quy Multi-Layer Perceptron (MLP) ở phía sau.
- **Nhánh Hồi quy & Huấn luyện:** MLP bao gồm hai lớp tuyến tính (chiều ẩn 256) với hàm kích hoạt ReLU và Dropout (0.2), ánh xạ các biểu diễn này tới 5 horizons mục tiêu. MLP được tinh chỉnh (fine-tuned) bằng hàm mất mát Mean Squared Error (MSE), bộ tối ưu hóa AdamW (LR = $10^{-3}$), batch size 32 (khi train) / 64 (khi eval), và cơ chế dừng sớm (patience = 7 epochs).

## 3. Tài liệu tham khảo

[1] T. Bollerslev, "Generalized autoregressive conditional heteroskedasticity," *Journal of Econometrics*, vol. 31, no. 3, pp. 307-327, 1986.
[2] L. R. Glosten, R. Jagannathan, and D. E. Runkle, "On the relation between the expected value and the volatility of the nominal excess return on stocks," *The Journal of Finance*, vol. 48, no. 5, pp. 1779-1801, 1993.
[3] R. T. Baillie, T. Bollerslev, and H. O. Mikkelsen, "Fractionally integrated generalized autoregressive conditional heteroskedasticity," *Journal of Econometrics*, vol. 74, no. 1. pp. 3-30, 1996.
[4] A. Vaswani et al., "Attention is all you need," in *Advances in Neural Information Processing Systems 30*, 2017, pp. 5998-6008.
[5] H. Zhou et al., "Informer: Beyond efficient transformer for long sequence time-series forecasting," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 35, no. 12, pp. 11106-11115, 2021.
[6] H. Wu et al., "Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting," in *Advances in Neural Information Processing Systems 34*, 2021, pp. 22419-22430.
[7] N. Kitaev, L. Kaiser, and A. Levskaya, "Reformer: The efficient transformer," in *International Conference on Learning Representations*, 2020.
[8] P. Zhao, H. Zhu, W. S. H. Ng, and D. L. Lee, "From GARCH to neural network for volatility forecast," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 38, no. 15, 2024, pp. 16998-17006.
[9] X. Liu et al., "Moirai-MoE: Empowering time series foundation models with sparse mixture of experts," in *Proceedings of the 42nd International Conference on Machine Learning*, vol. 267, PMLR, 2025, pp. 38940-38962.
