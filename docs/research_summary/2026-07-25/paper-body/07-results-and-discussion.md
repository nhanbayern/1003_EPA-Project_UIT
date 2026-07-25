# 5. Kết quả và thảo luận

## 5.1. Đặc điểm tính dừng của dữ liệu

Kết quả ADF/KPSS cho thấy tính dừng thay đổi mạnh theo giai đoạn [7]:

| Giai đoạn | Mixed | Không dừng | Dừng |
|---|---:|---:|---:|
| Toàn mẫu | 30 | 0 | 15 |
| Train suy ra 70% | 30 | 6 | 9 |
| Validation suy ra 15% | 22 | 23 | 0 |
| Test suy ra 15% | 18 | 26 | 1 |

Trên toàn mẫu, 15/45 chuỗi, tương đương 33,33%, được kết luận dừng; 30/45 chuỗi còn lại thuộc nhóm `mixed`. Nikkei 225 và VN30 có 5/5 chân trời dừng, KOSPI có 3/5 và SMI có 2/5. DAX 40, EuroNext 100, IBEX 35, VN-Index và S&P 500 đều có 5/5 chân trời thuộc nhóm `mixed`.

Kết quả theo các phân đoạn ngắn hơn thận trọng hơn đáng kể: validation không có chuỗi dừng, còn test chỉ có một chuỗi dừng. Điều này không trực tiếp chứng minh mô hình nào tốt hơn, nhưng cho thấy bảng xếp hạng được hình thành trong bối cảnh có chuyển chế độ hoặc không dừng cục bộ. Đây là lý do các tiêu chí bám động học cần được xem như thành phần bổ sung cho hàm mất mát trung bình.

## 5.2. Kết quả cổng kiểm tra hợp lý

Trong 26 cấu hình đầu vào, 5 cấu hình không đạt cổng kiểm tra và bị loại khỏi cả SAW, TOPSIS lẫn bảng xếp hạng kết hợp [9]:

| Cấu hình | Sai số tỷ lệ độ lệch chuẩn | Tương quan bám chuỗi | Nguyên nhân |
|---|---:|---:|---|
| Reformer Tier 3 | 1,000000 | -0,002470 | Sai số độ lệch chuẩn vượt ngưỡng; tương quan không dương |
| Informer Tier 1 | 0,648742 | -0,004056 | Tương quan không dương |
| Autoformer Tier 3 | 0,919195 | 0,006766 | Sai số độ lệch chuẩn vượt ngưỡng |
| Vanilla Tier 3 | 1,000000 | -0,000124 | Sai số độ lệch chuẩn vượt ngưỡng; tương quan không dương |
| Informer Tier 3 | 1,000000 | -0,001735 | Sai số độ lệch chuẩn vượt ngưỡng; tương quan không dương |

![Sai số tỷ lệ độ lệch chuẩn theo cấu hình mô hình](../../../../output/mcdm_results/MCDM20260725180350/5,5/VolatilityStdRatioErrorByModelTier.png)

**Hình 1.** Sai số tỷ lệ độ lệch chuẩn theo model–tier; đường đứt đoạn biểu thị ngưỡng 0,9.

Ba cấu hình Tier 3 có sai số gần 1 và tương quan xấp xỉ 0 hoặc âm, phù hợp với hiện tượng dự báo gần như hằng. Autoformer Tier 3 vẫn có tương quan dương nhỏ nhưng sai số tỷ lệ độ lệch chuẩn 0,919195, nên bị loại. Informer Tier 1 cho thấy vai trò bổ sung của điều kiện tương quan: biên độ dự báo vượt điều kiện tối thiểu nhưng hướng bám chuỗi không dương. Như vậy, hai điều kiện của gate phát hiện hai loại suy biến khác nhau.

Sau sàng lọc, 21 cấu hình được giữ lại. Danh sách bị loại giống nhau trong hai kịch bản trọng số vì cổng kiểm tra được áp dụng trước bước gán trọng số.

## 5.3. Phân tích hai mô hình đề xuất

### 5.3.1. GARCH-Autoformer so với Autoformer

Bảng dưới so sánh trực tiếp GARCH-Autoformer với Autoformer ở cùng tier.

| Cấu hình | MSE | MAE | QLIKE | Sai số tỷ lệ độ lệch chuẩn | Tương quan bám chuỗi |
|---|---:|---:|---:|---:|---:|
| Autoformer Tier 1 | 0,224107 | 0,346533 | 0,092884 | 0,466337 | 0,055386 |
| **GARCH-Autoformer Tier 1** | **0,156506** | **0,275386** | **0,067463** | **0,134507** | **0,560461** |
| Autoformer Tier 2 | 0,233564 | 0,356766 | 0,098034 | 0,450731 | 0,061937 |
| **GARCH-Autoformer Tier 2** | **0,169022** | **0,291126** | **0,081926** | **0,130530** | **0,529503** |

Ở cả hai tier, GARCH-Autoformer có MSE, MAE, QLIKE và sai số tỷ lệ độ lệch chuẩn thấp hơn Autoformer, đồng thời tương quan bám chuỗi cao hơn rõ rệt. Đặc biệt, tương quan tăng từ 0,055386 lên 0,560461 ở Tier 1 và từ 0,061937 lên 0,529503 ở Tier 2. Kết quả này phù hợp với mục tiêu đưa thiên kiến quy nạp GARCH vào Autoformer để cải thiện hành vi động học.

| Cấu hình | Pass VaR 1% | AVE VaR 1% | Pass VaR 5% | AVE VaR 5% |
|---|---:|---:|---:|---:|
| Autoformer Tier 1 | **0,355556** | 0,003776 | **0,200000** | 0,015712 |
| **GARCH-Autoformer Tier 1** | 0,288889 | **0,001443** | 0,133333 | **0,010110** |
| Autoformer Tier 2 | 0,288889 | 0,002818 | **0,244444** | 0,013729 |
| **GARCH-Autoformer Tier 2** | **0,422222** | **0,000613** | 0,133333 | **0,006860** |

Ở VaR 1%, Tier 2 đề xuất vừa có pass rate cao hơn, vừa có sai số vi phạm thấp hơn Autoformer Tier 2. Ở VaR 5%, GARCH-Autoformer có pass rate thấp hơn Autoformer nhưng AVE tốt hơn ở cả hai tier. Hai chỉ số không mâu thuẫn: AVE chỉ đo tỷ lệ vi phạm gần mức danh nghĩa, trong khi pass rate còn yêu cầu điều kiện độc lập. Điều này minh họa vì sao không nên thay một chỉ số bằng chỉ số còn lại.

### 5.3.2. MoiraiVaR so với mô hình nền Moirai

Tác động của căn chỉnh MoiraiVaR phụ thuộc mô hình nền:

| Cấu hình | MSE | Pass 1% | AVE 1% | Pass 5% | AVE 5% | SAW 50:50 | SAW 30:70 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moirai | 0,042343 | 0,111111 | **0,005824** | 0,111111 | **0,023550** | **0,607221** | **0,489394** |
| MoiraiVaR–Moirai | **0,041562** | 0,111111 | 0,005959 | 0,111111 | 0,024355 | 0,602238 | 0,480941 |
| Moirai 2 | **0,022751** | 0,111111 | 0,006036 | 0,111111 | 0,024017 | 0,650040 | 0,510056 |
| **MoiraiVaR–Moirai 2** | 0,023751 | **0,155556** | **0,005895** | **0,133333** | **0,023401** | **0,676720** | **0,550383** |
| Moirai-MoE | **0,024757** | **0,222222** | **0,005686** | 0,088889 | **0,023862** | **0,670016** | **0,544817** |
| MoiraiVaR–Moirai-MoE | 0,025260 | 0,200000 | 0,006017 | 0,088889 | 0,024333 | 0,654304 | 0,523431 |

MoiraiVaR–Moirai 2 là biến thể thành công nhất: dù MSE tăng nhẹ so với Moirai 2, cả tỷ lệ vượt kiểm định và AVE ở hai mức VaR đều cải thiện, làm điểm SAW tăng từ 0,650040 lên 0,676720 trong kịch bản 50:50 và từ 0,510056 lên 0,550383 trong kịch bản 30:70. Đây là bằng chứng thực nghiệm rõ nhất cho lợi ích của căn chỉnh theo rủi ro.

Ngược lại, MoiraiVaR–Moirai chỉ cải thiện nhẹ MSE nhưng giảm điểm MCDM; MoiraiVaR–Moirai-MoE cũng không vượt Moirai-MoE ở các chỉ số trình bày. Vì chỉ có \(\lambda=0{,}2\), chưa thể kết luận căn chỉnh VaR luôn cải thiện mọi mô hình nền. Kết quả ủng hộ thiết kế MoiraiVaR như một hướng kiến trúc tiềm năng, đồng thời chỉ ra nhu cầu phân tích loại bỏ theo \(\lambda\) và chế độ tinh chỉnh.

## 5.4. Xếp hạng ở kịch bản cân bằng 50:50

Sáu cấu hình dẫn đầu theo thứ hạng MCDM trung bình [9]:

| Hạng kết hợp | Cấu hình | Điểm accuracy | Điểm risk | Hạng SAW | Điểm SAW | Hạng TOPSIS | Điểm TOPSIS |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **MoiraiVaR–Moirai 2 (\(\lambda=0{,}2\))** | 0,496282 | 0,180438 | 1 | 0,676720 | 3 | 0,589468 |
| 1 | **GARCH-Autoformer Tier 2** | 0,223168 | 0,443182 | 3 | 0,666350 | 1 | 0,669217 |
| 3 | Moirai-MoE | 0,491507 | 0,178509 | 2 | 0,670016 | 4 | 0,586912 |
| 4 | **GARCH-Autoformer Tier 1** | 0,253919 | 0,371223 | 6 | 0,625142 | 2 | 0,640853 |
| 5 | **MoiraiVaR–Moirai-MoE (\(\lambda=0{,}2\))** | 0,490744 | 0,163560 | 4 | 0,654304 | 5 | 0,572443 |
| 6 | Moirai 2 | 0,500000 | 0,150040 | 5 | 0,650040 | 6 | 0,556233 |

Hai mô hình đề xuất tạo ra hai phương án đồng hạng nhất nhưng theo các cấu trúc điểm khác nhau. MoiraiVaR–Moirai 2 dẫn đầu SAW nhờ điểm accuracy 0,496282 và vẫn duy trì một phần đóng góp từ khối risk. GARCH-Autoformer Tier 2 chỉ đứng thứ ba theo SAW nhưng đứng đầu TOPSIS nhờ điểm risk 0,443182 và cấu hình cân bằng hơn giữa các chiều. GARCH-Autoformer Tier 1 đứng thứ hai TOPSIS, cho thấy lợi thế không chỉ xuất hiện ở một tier.

![Bản đồ đánh đổi giữa điểm độ chính xác và điểm hiệu chỉnh rủi ro ở kịch bản 50:50](../../../../output/mcdm_results/MCDM20260725180350/5,5/AccuracyRiskTradeoff.png)

**Hình 2.** Điểm thành phần độ chính xác và rủi ro ở cấp họ mô hình trong kịch bản 50:50.

Hình 2 thể hiện hai cụm rõ. Moirai/MoiraiVaR nằm ở vùng điểm accuracy cao nhưng risk thấp; GARCH-Autoformer nằm ở vùng accuracy trung bình và risk cao nhất. Các mô hình GARCH truyền thống có điểm accuracy trung bình nhưng điểm risk rất thấp trong hệ tiêu chí hiện tại. Kết quả bác bỏ cách diễn giải một chiều rằng mô hình có forecast error tốt nhất đương nhiên là mô hình triển khai rủi ro tốt nhất.

![So sánh thứ hạng SAW và TOPSIS theo họ mô hình ở kịch bản 50:50](../../../../output/mcdm_results/MCDM20260725180350/5,5/SAWTOPSISRankComparison.png)

**Hình 3.** So sánh hạng SAW và TOPSIS ở cấp họ mô hình trong kịch bản 50:50; hạng thấp hơn là tốt hơn.

Ở cấp họ, GARCH-Autoformer đứng hạng 3 theo SAW nhưng hạng 1 theo TOPSIS. Khoảng cách giữa hai thứ hạng phản ánh bản chất phương pháp: SAW thưởng mạnh cho điểm accuracy cao của Moirai-family, trong khi TOPSIS ưu tiên phương án gần nghiệm lý tưởng đa tiêu chí hơn.

## 5.5. Xếp hạng ở kịch bản nhạy cảm rủi ro 30:70

Khi trọng số rủi ro tăng từ 50% lên 70%, top sáu thay đổi rõ [9]:

| Hạng kết hợp | Cấu hình | Điểm accuracy | Điểm risk | Hạng SAW | Điểm SAW | Hạng TOPSIS | Điểm TOPSIS |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **GARCH-Autoformer Tier 2** | 0,133901 | 0,620455 | 1 | 0,754355 | 1 | 0,740389 |
| 2 | Reformer Tier 2 | 0,026559 | 0,561948 | 3 | 0,588507 | 2 | 0,694979 |
| 3 | **GARCH-Autoformer Tier 1** | 0,152351 | 0,519713 | 2 | 0,672064 | 5 | 0,673411 |
| 4 | Autoformer Tier 2 | 0,033116 | 0,537603 | 5 | 0,570720 | 3 | 0,678577 |
| 4 | Wavelet-Autoformer Tier 2 | 0,040504 | 0,532792 | 4 | 0,573296 | 4 | 0,675658 |
| 6 | Autoformer Tier 1 | 0,039716 | 0,494650 | 8 | 0,534366 | 6 | 0,657734 |

GARCH-Autoformer Tier 2 đứng hạng 1 tuyệt đối theo cả SAW và TOPSIS, với điểm lần lượt là 0,754355 và 0,740389. GARCH-Autoformer Tier 1 vẫn đứng hạng 2 SAW và hạng 3 kết hợp. Hai cấu hình đề xuất vì vậy cùng nằm trong top ba kết hợp.

MoiraiVaR–Moirai 2 giảm xuống hạng kết hợp 7 dù đứng hạng 6 SAW. Sự dịch chuyển này không có nghĩa dự báo của mô hình trở nên kém hơn; điểm accuracy và risk đã được nhân lại theo ưu tiên mới. Kết quả chỉ ra rằng lợi thế accuracy của Moirai-family không đủ bù khoảng cách về risk khi trọng số rủi ro đạt 70%.

![So sánh thứ hạng SAW và TOPSIS theo họ mô hình ở kịch bản 30:70](../../../../output/mcdm_results/MCDM20260725180350/3,7/SAWTOPSISRankComparison.png)

**Hình 4.** So sánh hạng SAW và TOPSIS ở cấp họ mô hình trong kịch bản 30:70.

Ở cấp họ, GARCH-Autoformer đứng hạng 1 SAW và hạng 2 TOPSIS. Autoformer đứng gần nhất theo TOPSIS, nhưng GARCH-Autoformer vẫn có điểm TOPSIS trung bình theo họ cao hơn: 0,706900 so với 0,668155.

## 5.6. Kiểm định ưu thế

Kết quả kiểm định GARCH-Autoformer so với 15 họ mô hình còn lại:

| Kịch bản | Thước đo | Số thắng | Tỷ lệ thắng | \(p\) nhị thức chính xác | \(z\) | \(p_z\) | Chênh lệch tương đối trung bình |
|---|---|---:|---:|---:|---:|---:|---:|
| 50:50 | SAW | 11/15 | 73,33% | 0,059235 | 1,807392 | 0,035351 | +51,62% |
| 50:50 | TOPSIS | 15/15 | 100,00% | 0,000031 | 3,872983 | 0,000054 | +43,52% |
| 30:70 | SAW | 15/15 | 100,00% | 0,000031 | 3,872983 | 0,000054 | +96,70% |
| 30:70 | TOPSIS | 15/15 | 100,00% | 0,000031 | 3,872983 | 0,000054 | +102,18% |

Ở kịch bản 50:50, SAW cung cấp bằng chứng gần ngưỡng nhưng chưa đủ để bác bỏ \(H_0\) ở mức 5% theo kiểm định chính xác, vì \(p=0{,}059235\). Dù z-test cho \(p_z=0{,}035351\), kết luận chính phải ưu tiên kiểm định chính xác do cỡ mẫu nhỏ. Bốn họ có điểm SAW cao hơn điểm trung bình theo họ của GARCH-Autoformer là Moirai 2, Moirai-MoE, MoiraiVaR–Moirai 2 và MoiraiVaR–Moirai-MoE.

Theo TOPSIS ở kịch bản 50:50, GARCH-Autoformer thắng 15/15 họ và kiểm định nhị thức cho \(p=0{,}000031\). Kết quả này củng cố nhận định rằng kiến trúc lai có vị trí gần nghiệm lý tưởng hơn khi cần cân bằng nhiều chiều.

Ở kịch bản 30:70, GARCH-Autoformer thắng toàn bộ 15 họ theo cả SAW và TOPSIS. Chênh lệch tương đối trung bình đạt 96,70% theo SAW và 102,18% theo TOPSIS. Bằng chứng thống kê vì vậy mạnh nhất trong bối cảnh ưu tiên rủi ro, đúng với giả thuyết thiết kế và kết quả xếp hạng cấu hình.

## 5.7. Thảo luận

### 5.7.1. Hai mô hình đề xuất có thế mạnh bổ sung

GARCH-Autoformer thể hiện ưu thế rõ ở khả năng bám động học và khối rủi ro. So với Autoformer cùng tier, mô hình giảm cả ba hàm mất mát, giảm sai số tỷ lệ độ lệch chuẩn và tăng tương quan bám chuỗi. Tier 2 đạt tỷ lệ vượt kiểm định VaR 1% cao nhất trong các cấu hình đầu vào là 0,422222, đồng thời đạt AVE VaR 1% thấp 0,000613. Tổ hợp này giải thích vì sao mô hình đứng đầu TOPSIS ở cả hai kịch bản và đứng đầu SAW khi trọng số rủi ro tăng.

MoiraiVaR cho một bức tranh tinh tế hơn. Nền Moirai vốn có độ chính xác và khả năng bám chuỗi rất mạnh; do đó hiệu quả của căn chỉnh phụ thuộc mức đánh đổi với hiệu chỉnh rủi ro. Biến thể MoiraiVaR–Moirai 2 cải thiện cả bốn chỉ số VaR được trình bày so với mô hình nền và đứng đầu SAW 50:50. Tuy nhiên, hai mô hình nền còn lại không có cải thiện đồng nhất. Điều này cho thấy \(\lambda=0{,}2\) chưa phải một cấu hình có thể mặc định chuyển sang mọi mô hình nền.

### 5.7.2. Thứ hạng là một hàm của ưu tiên triển khai

Kết quả không hỗ trợ một bảng xếp hạng duy nhất cho mọi mục tiêu. Với trọng số cân bằng, MoiraiVaR–Moirai 2 và GARCH-Autoformer Tier 2 đồng hạng nhất nhưng vì các lý do khác nhau. Với 70% trọng số rủi ro, GARCH-Autoformer vượt lên rõ rệt. Do đó, lựa chọn mô hình nên đi kèm tuyên bố ưu tiên: ưu tiên độ chính xác, cân bằng hoặc nhạy cảm với rủi ro.

SAW và TOPSIS cũng không hoàn toàn thay thế nhau. SAW có tính bù trừ tuyến tính mạnh hơn và ưu tiên các cấu hình Moirai ở kịch bản 50:50. TOPSIS nhất quán đặt GARCH-Autoformer gần nghiệm lý tưởng. Việc báo cáo cả hai giúp tránh kết luận phụ thuộc một cơ chế tổng hợp duy nhất.

### 5.7.3. Ý nghĩa của cổng kiểm tra hợp lý

Cổng kiểm tra loại 5/26 cấu hình, tương đương 19,23% đầu vào. Nếu không có bước này, các cấu hình có tỷ lệ vượt kiểm định VaR tương đối khả quan nhưng dự báo gần như hằng vẫn có thể tham gia chuẩn hóa và ảnh hưởng nghiệm lý tưởng. Cổng vì vậy hoạt động như một điều kiện hợp lệ tối thiểu trước MCDM. Tuy nhiên, một cấu hình qua cổng chưa chắc tốt; bằng chứng là nhiều mô hình qua cổng vẫn nằm cuối bảng do điểm rủi ro thấp.

### 5.7.4. Diễn giải trong điều kiện không dừng cục bộ

Tỷ lệ chuỗi không dừng tăng mạnh ở validation/test suy ra cho thấy mô hình phải hoạt động trong điều kiện phân phối biến động thay đổi. Các kết quả bám chuỗi của GARCH-Autoformer và họ Moirai có ý nghĩa thực nghiệm trong bối cảnh này, nhưng phân tích hiện tại mới tổng hợp trên toàn bộ 45 trường hợp. Chưa có bằng chứng rằng một mô hình duy trì ưu thế riêng trong nhóm `stationary`, `mixed` hay `non_stationary`.

## 5.8. Hạn chế

Nghiên cứu có sáu hạn chế chính.

Thứ nhất, tài liệu nguồn chỉ mô tả hai kiến trúc đề xuất ở mức thành phần chức năng. Chưa có sơ đồ tầng, cơ chế kết hợp GARCH–Autoformer, biểu thức hàm mục tiêu MoiraiVaR, số tham số, chi phí tính toán hoặc giao thức tinh chỉnh. Vì vậy, phần kiến trúc trong bài không thể thay thế đặc tả triển khai.

Thứ hai, cổng kiểm tra sử dụng một ngưỡng duy nhất 0,9 mà chưa có phân tích loại bỏ ở 0,8; 0,85; 0,9 và 0,95. Ngưỡng hiện tại là quy tắc sàng lọc thực nghiệm chứ chưa phải ngưỡng tối ưu.

Thứ ba, chỉ có hai kịch bản trọng số. Chưa có quét trọng số rủi ro liên tục, Pareto frontier, vùng ổn định thứ hạng hoặc tương quan Spearman/Kendall giữa nhiều kịch bản.

Thứ tư, kiểm định ưu thế sử dụng một điểm tổng hợp cho mỗi họ mô hình. Kết quả 15/15 là bằng chứng hỗ trợ ở cấp tổng hợp, chưa thay thế Friedman/Nemenyi hoặc Wilcoxon/kiểm định dấu trên 45 khối thị trường–chân trời.

Thứ năm, tỷ lệ chia 70/15/15 chỉ được suy ra cho phân tích tính dừng vì tệp tổng hợp thiếu nhãn phân đoạn gốc. Tài liệu không cung cấp đầy đủ quy trình huấn luyện và tái huấn luyện của từng mô hình.

Thứ sáu, số dòng dùng để tính khả năng bám chuỗi khác nhau giữa nhánh GARCH, Transformer và Moirai. Mặc dù mỗi cấu hình đều có 45 trường hợp, chênh lệch phạm vi quan sát cần được kiểm soát chặt hơn trong một bộ chuẩn tái lập hoàn toàn.
