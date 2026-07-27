# 2. Công trình liên quan

## 2.1. Đánh giá độ chính xác của dự báo biến động

MSE và MAE là hai thước đo phổ biến cho dự báo điểm. MSE nhấn mạnh các sai số lớn do sử dụng bình phương sai lệch, trong khi MAE giữ cùng thang đo với biến động và thường dễ diễn giải hơn. Tuy nhiên, việc lựa chọn thước đo không trung tính: Gneiting chỉ ra rằng kết luận về chất lượng dự báo có thể thay đổi nếu hàm chấm điểm không phù hợp với đại lượng đích và nhiệm vụ dự báo [1]. Do đó, một mô hình đứng đầu theo một hàm mất mát không nhất thiết là lựa chọn tốt nhất cho mọi mục tiêu.

Đối với dự báo biến động, khó khăn tăng lên vì biến động thực tế là đại lượng tiềm ẩn và thường được thay thế bằng một đại diện. Patton phân tích việc so sánh dự báo biến động khi đại diện không hoàn hảo và nhấn mạnh vai trò của các hàm mất mát bền vững, trong đó MSE và QLIKE là hai lựa chọn quan trọng [3]. Patton và Sheppard cũng xem việc lựa chọn đại diện, hàm mất mát và quy trình đánh giá là phần cốt lõi của so sánh dự báo biến động và tương quan [4]. Từ đó, nghiên cứu hiện tại giữ đồng thời MSE, MAE và QLIKE thay vì dựa trên một thước đo duy nhất.

## 2.2. Hiệu chỉnh rủi ro và tính cung cấp thông tin

Một dự báo xác suất hoặc dự báo rủi ro không chỉ cần được hiệu chỉnh mà còn phải đủ sắc nét và cung cấp thông tin. Gneiting, Balabdaoui và Raftery phân biệt hai thuộc tính này trong đánh giá dự báo xác suất [2]. Trong bối cảnh VaR, tỷ lệ vi phạm gần mức danh nghĩa là cần thiết nhưng chưa đủ: các vi phạm còn cần thỏa điều kiện backtest về độ phủ và tính độc lập.

Khung trong nghiên cứu sử dụng hai loại tiêu chí bổ sung nhau. Tỷ lệ vượt backtest VaR yêu cầu một trường hợp đồng thời vượt qua kiểm định Kupiec và kiểm định độc lập. Sai số vi phạm tuyệt đối đo khoảng cách giữa tỷ lệ vi phạm quan sát và mức đuôi danh nghĩa. Tiêu chí thứ nhất là một tiêu chí lợi ích và bao gồm điều kiện độc lập; tiêu chí thứ hai là tiêu chí chi phí, liên tục và dễ tổng hợp, nhưng không thay thế backtest.

## 2.3. Chẩn đoán khả năng bám động học

Đánh giá dự báo liên tục về bản chất là so sánh trực tiếp giá trị dự báo với quan sát và xem xét quan hệ giữa hai chuỗi [5]. Tuy vậy, các hàm mất mát trung bình như MSE, MAE hoặc QLIKE không trực tiếp bảo đảm rằng dự báo tái hiện được biên độ và hướng thay đổi theo thời gian. Trong bộ kết quả của nghiên cứu, một số cấu hình Transformer kích thước lớn tạo dự báo gần như hằng trong từng tổ hợp thị trường–chân trời. Những cấu hình này vẫn có thể cho tỷ lệ VaR trông hợp lý ở một số trường hợp, tạo ra nguy cơ được xếp hạng quá cao nếu hệ tiêu chí thiếu chẩn đoán hình dạng chuỗi.

Nghiên cứu giải quyết vấn đề này bằng hai chỉ số: sai số tỷ lệ độ lệch chuẩn, dùng để nhận diện dự báo thiếu biên độ, và sai số tương quan bám chuỗi, dùng để phạt dự báo không đồng biến với biến động thực tế. Hai chỉ số vừa tham gia điểm MCDM, vừa tạo thành một cổng sàng lọc tối thiểu. Cổng này không chứng minh mô hình đạt chất lượng cao; nó chỉ loại các trường hợp vi phạm điều kiện động học cơ bản.

## 2.4. Lựa chọn mô hình đa tiêu chí

Lựa chọn mô hình dự báo biến động là một bài toán đa mục tiêu vì độ chính xác, khả năng bám động học và hiệu chỉnh rủi ro có thể xung đột. Simple Additive Weighting (SAW) cung cấp một điểm tổng hợp tuyến tính, trong đó ưu thế ở một tiêu chí có thể bù cho bất lợi ở tiêu chí khác. Technique for Order Preference by Similarity to Ideal Solution (TOPSIS) đánh giá đồng thời khoảng cách đến nghiệm lý tưởng tốt nhất và tệ nhất, do đó nhấn mạnh tính cân bằng giữa các chiều đánh giá.

Khoảng trống mà nghiên cứu tập trung gồm hai lớp. Ở lớp mô hình, cần kết nối thiên kiến quy nạp biến động của GARCH với năng lực mô hình hóa chuỗi của Autoformer, đồng thời cần căn chỉnh mô hình nền tảng Moirai theo mục tiêu rủi ro thay vì chỉ tối ưu độ chính xác dự báo [8]. Ở lớp đánh giá, cần một quy trình thống nhất gồm: (i) sàng lọc dự báo suy biến; (ii) tổng hợp chín tiêu chí độ chính xác–động học–rủi ro; (iii) thay đổi trọng số theo ưu tiên triển khai; và (iv) kiểm tra ưu thế ở cấp họ mô hình. Việc sử dụng đồng thời SAW và TOPSIS cũng giúp phát hiện trường hợp một mô hình đạt điểm cộng tuyến tính cao nhưng không gần nghiệm lý tưởng đa chiều.

## 2.5. Vị trí của nghiên cứu

So với đánh giá chỉ dựa trên hàm mất mát, nghiên cứu bổ sung điều kiện động học và VaR. So với đánh giá chỉ dựa trên kiểm định hậu nghiệm, nghiên cứu ngăn dự báo gần như phẳng đi thẳng vào bảng xếp hạng. So với một thứ hạng cố định, nghiên cứu báo cáo hai kịch bản ưu tiên và phân tích sự chuyển dịch giữa các họ mô hình. Hai kiến trúc đề xuất đại diện cho hai hướng bổ sung: lai ghép GARCH–Autoformer và căn chỉnh Moirai theo rủi ro. Vì tài liệu nguồn chưa cung cấp sơ đồ tầng, biểu thức hàm mục tiêu huấn luyện đầy đủ, phân tích loại bỏ theo ngưỡng, quét liên tục trọng số hoặc kiểm định theo 45 khối, bài báo chỉ mô tả các thành phần kiến trúc được nguồn xác nhận và không suy diễn cơ chế ghép nối chưa được ghi nhận.
