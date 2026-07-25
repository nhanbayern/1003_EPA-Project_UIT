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
Kiến trúc Transformer tận dụng cơ chế tự chú ý (self-attention) [4] để nắm bắt các phụ thuộc dài hạn trong chuỗi thời gian mà không cần xử lý tuần tự đệ quy. Tuy nhiên, các kiến trúc Transformer thường đòi hỏi một lượng lớn dữ liệu (data hunger) để hội tụ tốt, đặc biệt là khi học tự chú ý trên các chuỗi ngắn. Để kiểm định ranh giới khả năng mở rộng (scalability) của Transformer khi chỉ được cung cấp một cửa sổ dữ liệu cực ngắn (60 ngày), nghiên cứu này thiết lập 3 cấp độ độ phức tạp (Tiers):
- **Tier 1 (Miniaturized):** $d_{model}=32$, 2 lớp encoder, 4 heads (phù hợp để tránh overfitting trên dữ liệu ngắn).
- **Tier 2 (Standard):** $d_{model}=128$, 3 lớp encoder, 8 heads.
- **Tier 3 (Large):** $d_{model}=512$, 6 lớp encoder, 8 heads (để kiểm tra xem mô hình lớn có bị "thắt cổ chai" do data hunger hay không).

- **Mô hình:** Vanilla Transformer [4], Informer [5], Autoformer [6], và Reformer [7].
- **Cấu hình:** Cửa sổ đầu vào là 60 ngày. Nhờ cấu trúc ánh xạ trực tiếp, đầu ra của Transformer sử dụng một khối hồi quy đa tầng (VolatilityHead) xuất trực tiếp dự báo cho cả 21 horizons mục tiêu thay vì phải dự báo tự hồi quy.
- **Huấn luyện và Hàm mất mát:** Để xử lý tính chất đuôi dày của phân phối tài chính, các mô hình được huấn luyện bằng hàm mất mát Student-t Negative Log-Likelihood (NLL). Đáng chú ý, bậc tự do (degrees of freedom - $\nu$) của hàm NLL được **tính toán động** cho từng chỉ số dựa trên đặc trưng thống kê của tập dữ liệu. Điều này cho phép Transformer – vốn có khả năng biểu diễn lớn – tinh chỉnh linh hoạt theo hành vi đuôi của từng thị trường. Quá trình tối ưu sử dụng AdamW (LR = $10^{-3}$) trong 30 epochs, batch size 128 và early stopping (patience = 5).

### 2.3. Mô hình lai GARCH-LSTM
Mô hình lai GARCH-LSTM [8] kết hợp tính quy nạp (inductive bias) tài chính của các mô hình kinh tế lượng với khả năng học biểu diễn phi tuyến của mạng nơ-ron hồi quy.
- **Kiến trúc:** Một cell hồi quy tùy chỉnh (`GARCH_LSTM_Cell`, hidden_size = 16) nhúng trực tiếp phương trình cập nhật phương sai có điều kiện của GARCH vào cơ chế cổng (gating) của LSTM. Cell này nhận tỷ suất sinh lợi trước đó ($\epsilon_{t-1}$) và phương sai trước đó ($\sigma^2_{t-1}$) làm đầu vào để cập nhật trạng thái ẩn và tính toán phương sai hiện tại.
- **Huấn luyện và Hàm mất mát:** Khác với Transformer sử dụng $\nu$ động, GARCH-LSTM được huấn luyện bằng hàm Student-t NLL với bậc tự do **cố định $v=5.0$**. Trong lý thuyết tài chính, lợi suất thường có phân phối Student-t với bậc tự do từ 4 đến 6. Việc cố định $v=5.0$ đóng vai trò như một cơ chế điều chuẩn (regularization) [10], giúp ngăn cản cấu trúc hồi quy phức tạp của LSTM bị overfit vào các tín hiệu nhiễu cực đoan trong tập dữ liệu.
- **Dự báo:** Thay vì dự báo theo hướng tự hồi quy (autoregressively) truyền thống, kiến trúc đã được nâng cấp thành **dự báo trực tiếp (direct multi-horizon forecasting)**. Cell GARCH-LSTM xử lý tuần tự qua cửa sổ lịch sử 60 ngày để trích xuất trạng thái ẩn cuối cùng ($c_t$). Trạng thái này đóng vai trò là biểu diễn nén của cả quá trình sinh lợi và phương sai trong quá khứ, sau đó được truyền qua một khối hồi quy đa tầng (`VolatilityHead` với các lớp Linear, GELU, Softplus - tương tự như cấu trúc xuất của Transformer) để xuất trực tiếp dự báo cho cả 21 horizons mục tiêu cùng lúc. Phương pháp này giúp mô hình hoàn toàn tránh được hiện tượng tích lũy sai số (error accumulation) khi phải dự báo xa vào tương lai.

### 2.4. Các mô hình nền tảng dựa trên Moirai
Các mô hình Moirai là các mô hình nền tảng chuỗi thời gian quy mô lớn, được huấn luyện trước bởi Salesforce [9], sử dụng kiến trúc masked encoder hỗ trợ kích thước patch thay đổi.
- **Mô hình:** Moirai 1.0, Moirai 2.0, và Moirai-MoE (Mixture of Experts). Các phiên bản tham số `small` được sử dụng.
- **Triển khai:** Khả năng biểu diễn phổ quát của Moirai được tận dụng qua mô hình trích xuất đặc trưng (feature extraction). Các chuỗi tỷ suất sinh lợi 60 ngày được chèn (padding) thêm để đạt 64 ngày nhằm khớp với kích thước patch cố định là 16 (tạo ra 4 patches mỗi chuỗi). Backbone được huấn luyện trước sẽ bị đóng băng (frozen), và các biểu diễn gộp (pooled representations) của nó được đưa vào một nhánh hồi quy Multi-Layer Perceptron (MLP) ở phía sau.
- **Nhánh Hồi quy & Huấn luyện:** MLP bao gồm hai lớp tuyến tính (chiều ẩn 256) với hàm kích hoạt ReLU và Dropout (0.2), ánh xạ các biểu diễn này tới 5 horizons mục tiêu. MLP được tinh chỉnh (fine-tuned) bằng hàm mất mát Mean Squared Error (MSE), bộ tối ưu hóa AdamW (LR = $10^{-3}$), batch size 32 (khi train) / 64 (khi eval), và cơ chế dừng sớm (patience = 7 epochs).

## 2.5. Lập luận cho sự lựa chọn Mô hình và Cấu hình triển khai (Motivation for Model Selection & Implementations)

Việc thiết lập các cấu hình như 3 Tiers của Transformer hay việc lai tạo GARCH-LSTM xuất phát từ những thách thức cốt lõi trong phân tích chuỗi thời gian tài chính:

**Khủng hoảng dữ liệu (Data Hunger) trong các mô hình Attention:** Các kiến trúc Transformer thuần túy có xu hướng phụ thuộc vào lượng lớn dữ liệu để tối ưu hóa cơ chế tự chú ý (self-attention) [11]. Khi bị ép phải học trên chuỗi quá khứ cực ngắn (60 ngày), các Transformer có nguy cơ không nắm bắt được thông tin ngữ cảnh hoặc dễ bị overfitting. Việc triển khai 3 Tiers (từ Miniaturized đến Large) đóng vai trò như một công cụ đánh giá nhằm trả lời câu hỏi: *Liệu việc tăng độ phức tạp (scalability) của Transformer có giúp khai thác tốt hơn chuỗi 60 ngày, hay sự thiếu hụt độ dài chuỗi sẽ khiến mô hình Tier 3 cồng kềnh trở nên kém hiệu quả hơn cả Tier 1?* [12]

**Sự bù đắp thông tin qua Mô hình Lai (Hybrid Models):** Để giải quyết vấn đề nhiễu (high noise) và phi tuyến tính của dữ liệu chứng khoán, mô hình lai GARCH-LSTM không cố gắng học hoàn toàn từ dữ liệu thô như Transformer, mà tiêm trực tiếp (inject) "mã gien" của mô hình kinh tế lượng (GARCH) vào cấu trúc Neural Network [10]. Việc này đi kèm với quyết định cố định hàm mất mát Student-t ở $v=5.0$, đóng vai trò như một mỏ neo (anchor) chuẩn hóa [10]. Bằng cách này, GARCH-LSTM có thể tinh lọc trạng thái ẩn cốt lõi về biến động một cách chuẩn xác, làm nền tảng vững chắc để khối `VolatilityHead` phóng chiếu (project) trực tiếp ra các dự báo đa bước xa trong tương lai. Điều này mang lại hiệu suất ổn định mà không cần đòi hỏi kích thước tham số khổng lồ như các mô hình Transformer hay Foundation Models (Moirai).

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
[10] O. B. Sezer, M. U. Gudelek, and A. M. Ozbayoglu, "Financial time series forecasting with deep learning: A systematic literature review: 2005–2019," *Applied Soft Computing*, vol. 90, p. 106181, 2020.
[11] B. Lim and S. Zohren, "Time-series forecasting with deep learning: a survey," *Philosophical Transactions of the Royal Society A*, vol. 379, 2021.
[12] A. Zeng et al., "Are transformers effective for time series forecasting?," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 37, no. 9, pp. 11121-11128, 2023.
