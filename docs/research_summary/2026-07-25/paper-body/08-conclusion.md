# 6. Kết luận

Nghiên cứu đề xuất hai hướng kiến trúc dự báo biến động và một khung đánh giá nhạy cảm với rủi ro. GARCH-Autoformer kết hợp thiên kiến quy nạp GARCH với Autoformer ở hai cấu hình Tier 1/Tier 2. MoiraiVaR căn chỉnh ba mô hình nền Moirai, Moirai 2 và Moirai-MoE theo mục tiêu rủi ro với cấu hình \(\lambda=0{,}2\). Để so sánh các mô hình trong một không gian chung, khung MCDM kết hợp chín tiêu chí thuộc ba nhóm: độ chính xác, bám động học và hiệu chỉnh VaR. Một cổng kiểm tra hợp lý loại dự báo thiếu biên độ hoặc không có tương quan dương trước khi áp dụng SAW và TOPSIS.

Trên 26 cấu hình, 9 thị trường và 5 chân trời, cổng kiểm tra loại 5 cấu hình suy biến và giữ lại 21 cấu hình. Ở kịch bản 50:50, hai mô hình đề xuất cùng chiếm vị trí cao nhất theo hai cơ chế khác nhau: MoiraiVaR–Moirai 2 đứng đầu SAW, GARCH-Autoformer Tier 2 đứng đầu TOPSIS, và hai cấu hình đồng hạng nhất theo thứ hạng MCDM trung bình. Khi trọng số rủi ro tăng lên 70%, GARCH-Autoformer Tier 2 đứng đầu đồng thời SAW và TOPSIS. Ở cấp họ mô hình, GARCH-Autoformer thắng 15/15 đối thủ theo cả hai điểm trong kịch bản 30:70, với \(p=0{,}000031\) theo kiểm định nhị thức chính xác một phía.

So sánh trực tiếp cho thấy GARCH-Autoformer cải thiện MSE, MAE, QLIKE và khả năng bám động học so với Autoformer ở cùng tier. Với MoiraiVaR, lợi ích rõ nhất xuất hiện trên mô hình nền Moirai 2: các chỉ số VaR và điểm MCDM đều cải thiện dù MSE tăng nhẹ. Hai mô hình nền còn lại không cho cải thiện đồng đều, vì vậy chưa thể kết luận căn chỉnh với \(\lambda=0{,}2\) luôn có lợi.

Kết quả cốt lõi là không tồn tại một thứ hạng tách rời mục tiêu triển khai. MoiraiVaR–Moirai 2 phù hợp hơn với bối cảnh cân bằng độ chính xác–rủi ro, trong khi GARCH-Autoformer là ứng viên mạnh nhất khi ưu tiên hiệu chỉnh rủi ro. Cổng chẩn đoán động học cũng cho thấy một mô hình có backtest VaR chấp nhận được không nên được chọn nếu chuỗi dự báo gần như phẳng hoặc không bám đúng hướng biến động.

Các hướng nghiên cứu tiếp theo gồm:

1. Công bố đặc tả ở cấp tầng, cơ chế kết hợp và hàm mục tiêu huấn luyện đầy đủ cho GARCH-Autoformer và MoiraiVaR.
2. Thực hiện ablation thành phần GARCH–Autoformer và so sánh từng tier theo cùng ngân sách tham số.
3. Quét \(\lambda\in\{0;0{,}1;0{,}2;0{,}5\}\) cho từng mô hình nền Moirai, đồng thời so sánh tinh chỉnh lớp đầu ra với tinh chỉnh toàn bộ.
4. Kiểm tra độ nhạy của cổng kiểm tra hợp lý theo nhiều ngưỡng và trực quan hóa chuỗi thực–dự báo của các mô hình bị loại.
5. Quét trọng số rủi ro từ 0 đến 1, xây dựng Pareto frontier và vùng ổn định của mô hình hạng nhất.
6. Tính SAW/TOPSIS riêng trên 45 block, sau đó chạy Friedman–Nemenyi và Wilcoxon/sign test.
7. Phân tích hiệu suất theo lớp `stationary`, `mixed`, `non_stationary` và theo nhóm thị trường.

Những mở rộng trên sẽ cho phép chuyển bằng chứng hiện tại từ ưu thế ở cấp tổng hợp sang kết luận bền vững hơn ở cấp thị trường–chân trời, đồng thời làm rõ đóng góp kiến trúc độc lập của hai mô hình đề xuất.
