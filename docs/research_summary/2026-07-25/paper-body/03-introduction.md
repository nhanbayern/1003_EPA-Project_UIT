# 1. Giới thiệu

## 1.1. Bối cảnh

Dự báo biến động là một thành phần quan trọng trong đo lường rủi ro tài chính, phân bổ vốn và xây dựng ngưỡng Value-at-Risk (VaR). Một hệ thống dự báo được sử dụng trong thực tế phải đáp ứng đồng thời nhiều yêu cầu: sai số dự báo nhỏ, phản ứng được với sự thay đổi của biến động theo thời gian và tạo ra mức VaR có tần suất vi phạm phù hợp. Các yêu cầu này không hoàn toàn đồng nhất. Một mô hình có thể tối ưu tốt Mean Squared Error (MSE), Mean Absolute Error (MAE) hoặc Quasi-Likelihood Loss (QLIKE) nhưng chưa chắc tạo VaR được hiệu chỉnh tốt. Theo chiều ngược lại, một dự báo biến động gần như không đổi vẫn có thể tình cờ đạt tỷ lệ vi phạm VaR chấp nhận được ở một số trường hợp.

Khó khăn trên trở nên rõ hơn khi dữ liệu biến động không đồng nhất theo thị trường, chân trời và giai đoạn thời gian. Phân tích nguồn của nghiên cứu trên 9 chỉ số thị trường và 5 chân trời cho thấy chỉ 15/45 chuỗi `dataset × horizon` được kết luận dừng trên toàn mẫu; 30/45 chuỗi còn lại nhận nhãn `mixed` do kiểm định Augmented Dickey–Fuller (ADF) và Kwiatkowski–Phillips–Schmidt–Shin (KPSS) không đồng thuận. Ở các đoạn được suy ra theo thứ tự thời gian, tập validation không có chuỗi nào dừng và tập test chỉ có 1 chuỗi dừng [7]. Vì vậy, một tiêu chí sai số trung bình đơn lẻ khó phản ánh đầy đủ khả năng sử dụng mô hình trong điều kiện biến động có chuyển chế độ hoặc không dừng cục bộ.

Các nghiên cứu về đánh giá dự báo cũng chỉ ra rằng kết luận phụ thuộc vào scoring function được lựa chọn [1], trong khi dự báo hữu ích cần được xem xét cả về hiệu chỉnh và mức độ cung cấp thông tin [2]. Đối với biến động tiềm ẩn phải quan sát qua đại diện, việc lựa chọn hàm mất mát như MSE và QLIKE càng cần được thực hiện thận trọng [3], [4]. Những nhận định này tạo cơ sở để chuyển câu hỏi từ “mô hình nào có sai số thấp nhất?” sang “mô hình nào phù hợp nhất với ưu tiên triển khai cụ thể?”.

## 1.2. Khoảng trống nghiên cứu

Trong phạm vi hệ thống thực nghiệm được khảo sát, ba khoảng trống cần được xử lý.

Thứ nhất, xếp hạng chỉ dựa trên MSE, MAE và QLIKE không phát hiện đầy đủ dự báo suy biến về đường quá phẳng. Một mô hình có thể dự báo gần mức trung bình, hạn chế một phần sai số lớn, nhưng không theo kịp các đợt tăng giảm của biến động.

Thứ hai, đánh giá chỉ dựa trên tỷ lệ vượt backtest VaR cũng chưa đủ. Tỷ lệ vi phạm gần mức danh nghĩa không đồng nghĩa với việc chuỗi dự báo biến động có hình dạng hợp lý; đồng thời sai số tuyệt đối của tỷ lệ vi phạm không kiểm tra tính độc lập của các vi phạm.

Thứ ba, một thứ hạng cố định che khuất sự đánh đổi giữa độ chính xác và rủi ro. Nhà nghiên cứu quan tâm tới chất lượng dự báo điểm và nhà quản trị rủi ro ưu tiên hiệu chỉnh VaR có thể hợp lý khi chọn các mô hình khác nhau. Do đó, quy trình đánh giá cần biểu diễn rõ ưu tiên ra quyết định và kiểm tra độ nhất quán giữa nhiều phương pháp xếp hạng.

## 1.3. Mục tiêu nghiên cứu

Nghiên cứu hướng tới đề xuất hai hướng kiến trúc dự báo biến động — GARCH-Autoformer và MoiraiVaR — đồng thời xây dựng một khung lựa chọn nhạy cảm với rủi ro để đánh giá chúng. Khung này trả lời bốn câu hỏi:

1. Làm thế nào kết hợp độ chính xác, khả năng bám động học và hiệu chỉnh VaR trong một hệ tiêu chí thống nhất?
2. Làm thế nào loại sớm các dự báo gần như hằng hoặc không đồng biến với biến động thực tế?
3. Thứ hạng thay đổi ra sao khi ưu tiên chuyển từ cân bằng độ chính xác–rủi ro sang nhạy cảm hơn với rủi ro?
4. Hai mô hình đề xuất thể hiện thế mạnh nào dưới các ưu tiên khác nhau, và GARCH-Autoformer có ưu thế nhất quán so với các họ mô hình còn lại theo SAW và TOPSIS hay không?

GARCH-Autoformer được xây dựng theo hướng lai ghép thiên kiến quy nạp GARCH với Autoformer và có hai cấu hình Tier 1/Tier 2. MoiraiVaR được xây dựng theo hướng căn chỉnh rủi ro cho các mô hình nền Moirai, Moirai 2 và Moirai-MoE với \(\lambda=0{,}2\). Nghiên cứu không đặt mục tiêu chứng minh hai kiến trúc là tối ưu phổ quát hoặc đạt trạng thái tốt nhất trên mọi điều kiện. Trọng tâm là thiết kế mô hình ở mức được tài liệu nguồn xác nhận, phương pháp lựa chọn và bằng chứng thực nghiệm trong hai kịch bản ưu tiên cụ thể.

## 1.4. Các đóng góp chính

Các đóng góp của nghiên cứu gồm:

1. Đề xuất GARCH-Autoformer theo hướng kết hợp thiên kiến quy nạp GARCH với Autoformer nhằm cân bằng khả năng bám biến động và hiệu chỉnh rủi ro.
2. Đề xuất MoiraiVaR theo hướng căn chỉnh mô hình nền tảng chuỗi thời gian với mục tiêu rủi ro, triển khai trên ba mô hình nền ở cấu hình \(\lambda=0{,}2\).
3. Đề xuất một khung MCDM gồm chín tiêu chí, bao phủ độ chính xác dự báo, khả năng bám động học và hiệu chỉnh VaR ở hai mức đuôi 1% và 5%.
4. Bổ sung cổng kiểm tra hợp lý của dự báo dựa trên sai số tỷ lệ độ lệch chuẩn và tương quan bám chuỗi, qua đó loại các cấu hình có dự báo suy biến trước khi xếp hạng.
5. Kết hợp SAW và TOPSIS trong hai kịch bản 50:50 và 30:70, sau đó kiểm định ưu thế ở cấp họ mô hình bằng kiểm định nhị thức chính xác một phía.
6. Diễn giải kết quả trong bối cảnh tính dừng không đồng nhất của 45 chuỗi thị trường–chân trời, từ đó giới hạn tuyên bố khoa học ở đúng phạm vi mà dữ liệu hỗ trợ.

Phần còn lại của bài báo được tổ chức như sau. Phần 2 trình bày các hướng đánh giá dự báo liên quan và xác lập khoảng trống. Phần 3 mô tả khung phương pháp, hệ tiêu chí, cổng sàng lọc, SAW, TOPSIS và kiểm định ưu thế. Phần 4 trình bày dữ liệu, mô hình đối sánh và thiết lập thực nghiệm. Phần 5 báo cáo kết quả, phân tích sự đánh đổi và các hạn chế. Phần 6 kết luận và đề xuất hướng nghiên cứu tiếp theo.
