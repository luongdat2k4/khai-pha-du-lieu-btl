# Plan 01 — Đặc tả bài toán và cơ sở lý thuyết

## 1. Mục tiêu và sản phẩm

Chuyển yêu cầu trong `note.md` thành đặc tả có thể thực hiện và bảo vệ trước giảng viên. Đầu ra dự kiến: `docs/problem-definition.md`, `docs/theory.md`, bảng liên hệ KDD và bản thảo chương lý thuyết trong báo cáo.

Người chính: TV2 tổng hợp phần thuật toán, TV1 phần dữ liệu, TV3 phần đánh giá; cả nhóm cùng hiểu và giải thích được quy trình.

## 2. Phát biểu bài toán

- Đối tượng: một hồ sơ khách hàng trong bộ IBM Telco Customer Churn.
- Vector đầu vào `x`: 19 thuộc tính về khách hàng, dịch vụ, hợp đồng và chi phí.
- Nhãn `y`: `Churn=Yes → 1`, `Churn=No → 0`. Lớp dương luôn là khách rời bỏ.
- Kết quả: nhãn dự đoán, điểm `predict_proba` tương ứng lớp 1, mức ưu tiên chăm sóc và phiên bản mô hình.
- Người dùng: nhân viên chăm sóc khách hàng hoặc người trình diễn bài tập lớn.
- Quyết định được hỗ trợ: xếp hạng khách cần xem xét chăm sóc; nhân viên vẫn quyết định hành động.

Điểm 0,8 là đầu ra mô hình cho lớp rời bỏ, không tự động có nghĩa 80% khách như vậy chắc chắn rời bỏ trong tháng tới. Cần đánh giá hiệu chỉnh xác suất và dữ liệu theo thời gian trước khi đưa ra cách diễn giải đó.

### Việc phải chốt ở M0

1. Dùng dữ liệu tĩnh để giải bài toán phân loại nhãn đã có.
2. Không coi `customerID` là đặc trưng; vẫn giữ để truy vết và lưu trữ.
3. Không dùng `Churn` trong đầu vào khi dự đoán.
4. Giữ đúng ba mô hình yêu cầu; thêm DummyClassifier chỉ làm mốc so sánh.
5. Giữ ngưỡng 0,5 cho kết quả chính; thử ngưỡng khác là thí nghiệm mở rộng có quy trình riêng.
6. Không tự thêm dự báo doanh thu giữ chân hay hiệu quả chiến dịch vì dữ liệu chưa cung cấp chứng cứ đó.

## 3. Bản đồ lý thuyết theo giáo trình

Các số trang dưới đây là số trang in trong nội dung giáo trình, không phải dòng Markdown. Bản chuyển đổi có công thức và hình bị OCR sai; khi viết báo cáo cần trình bày lại công thức đúng, không chép các chuỗi OCR.

| Nguồn trong tài liệu | Nội dung cần đọc và viết | Áp dụng trong đề tài |
|---|---|---|
| Chương 1, mục 1.1–1.3, khoảng tr. 8–13 | Khái niệm khai phá dữ liệu, bài toán phân loại, quy nạp | Học bộ phân lớp từ hồ sơ có nhãn để áp dụng cho hồ sơ mới |
| Chương 1, mục 1.4, khoảng tr. 14–16 | Các bước phát hiện tri thức KDD | Bố cục quy trình và báo cáo |
| Chương 4, mục 4.1, khoảng tr. 119–131 | Đối tượng/thuộc tính, kiểu dữ liệu, thiếu, nhiễu, trùng, lấy mẫu, chọn/tạo thuộc tính, chuyển đổi | Audit CSV, bỏ ID, điền thiếu, phân tầng, one-hot, scaling |
| Chương 4, mục 4.3, khoảng tr. 145–162 | Phân loại, cây quyết định, Hunt, Gini, entropy, gain, điều kiện dừng | Giải thích Decision Tree và chống quá khớp |
| Phần “Đánh giá các mô hình phân loại” trong mục 4.3, khoảng tr. 162–166 | Confusion matrix, hạn chế accuracy, precision/recall/F, holdout, cross-validation | Thiết kế 80/20, CV 5 fold, chọn F1 lớp 1 |

Logistic Regression, Random Forest, ROC-AUC, pipeline scikit-learn, calibration và vận hành mô hình là **phần bổ sung theo yêu cầu đề tài**, không trình bày như thể giáo trình đã diễn giải đầy đủ. Khi triển khai, nhóm bổ sung nguồn gốc thuật toán và tài liệu chính thức của scikit-learn cho API thực tế sử dụng, ghi tài liệu tham khảo và phiên bản thư viện.

Không cần triển khai luật kết hợp mục 4.2 hoặc phân cụm mục 4.4 vì không phục vụ trực tiếp yêu cầu phân loại này.

## 4. Ánh xạ quy trình KDD

| Bước KDD | Công việc cụ thể | Bằng chứng phải có |
|---|---|---|
| Hiểu lĩnh vực và mục tiêu | Định nghĩa churn, chi phí bỏ sót và cảnh báo nhầm | Đặc tả bài toán và giới hạn dữ liệu |
| Chọn dữ liệu đích | Chọn CSV, nhãn và 19 đầu vào | Data dictionary, nguồn và checksum CSV |
| Làm sạch, tiền xử lý | Chuyển kiểu, xử lý thiếu/trùng/bất nhất | Audit trước/sau và quy tắc xử lý |
| Chuyển đổi dữ liệu | One-hot, scaling cho LR, bỏ ID | Pipeline có thể tái lập |
| Chọn chức năng và thuật toán | Phân loại có giám sát, ba mô hình | Lý do chọn từng mô hình |
| Khai phá | Huấn luyện và tìm tham số bằng CV | Log thực nghiệm và bảng CV |
| Đánh giá và biểu diễn | Metric, confusion matrix, ROC, diễn giải | Báo cáo test và hình minh họa |
| Sử dụng tri thức | Streamlit, xếp hạng nguy cơ, cập nhật | Demo và quy trình vận hành |

Lưu ý: KDD là quy trình tổng thể; huấn luyện mô hình là một bước trong quy trình. SQLite của ứng dụng không đồng nghĩa đã xây dựng kho dữ liệu.

## 5. Nội dung lý thuyết của ba mô hình

### 5.1. Decision Tree — phần gắn trực tiếp giáo trình

Trình bày cấu trúc nút gốc, nút trong, nhánh điều kiện và nút lá; xây dựng đệ quy từ tập huấn luyện; chọn phép chia làm giảm độ hỗn tạp. Viết và giải thích:

- `Gini(t) = 1 - Σ p(k|t)^2`.
- `H(t) = -Σ p(k|t) log2 p(k|t)`, quy ước `0 log2 0 = 0`.
- `Gain = H(parent) - Σ (n_child/n_parent) H(child)`.
- Độ giảm Gini được tính tương tự bằng độ hỗn tạp cha trừ tổng có trọng số của các con.

Nút chỉ chứa một lớp có Gini/entropy bằng 0; hai lớp 50/50 có độ hỗn tạp cao. Chọn phép chia có độ giảm hỗn tạp lớn. Bản OCR có câu diễn đạt không nhất quán về “đồng nhất”; dùng định nghĩa và công thức này để giải thích.

Phải có một ví dụ tính tay nhỏ: nút cha có 10 mẫu gồm 5 Yes/5 No, phép chia tạo hai con 4 Yes/1 No và 1 Yes/4 No. Gini cha 0,5; Gini mỗi con 0,32; độ giảm 0,18. Đây là ví dụ minh họa, không phải số liệu Telco.

Giải thích quá khớp và cách hạn chế: `max_depth`, `min_samples_leaf`, `min_samples_split`, `ccp_alpha`. Phân biệt giới hạn sớm và tỉa cành sau huấn luyện.

scikit-learn DecisionTreeClassifier dùng cây nhị phân theo họ CART; chọn entropy không biến nó thành ID3 hay C4.5. Giáo trình đề cập Hunt/ID3/C4.5 để học nguyên lý; báo cáo phải gọi đúng thuật toán thực tế cài đặt.

### 5.2. Logistic Regression — mô hình tuyến tính làm đối chứng

- `p(y=1|x) = 1 / (1 + exp(-(b + wᵀx)))`.
- Học bằng tối ưu log-loss có regularization; `C` kiểm soát nghịch mức regularization.
- Dù tên có “Regression”, đầu ra được sử dụng cho bài toán phân loại.
- Chuẩn hóa các biến số giúp tối ưu và so sánh hệ số hợp lý hơn; biến danh mục phải được mã hóa.
- Hệ số phản ánh liên hệ có điều kiện trong mô hình, không chứng minh quan hệ nhân quả. Đặc trưng tương quan và mã hóa ảnh hưởng cách đọc hệ số.

### 5.3. Random Forest — tập hợp nhiều cây

- Huấn luyện nhiều cây trên mẫu bootstrap và lựa chọn ngẫu nhiên tập con đặc trưng khi tìm split.
- Gộp đầu ra của các cây để dự đoán; lấy xác suất lớp của mô hình để tính ROC-AUC.
- So sánh với một cây: thường giảm phương sai nhưng phức tạp hơn và khó giải thích trực tiếp hơn.
- Tham số chính: số cây, độ sâu, số mẫu tối thiểu ở lá và số đặc trưng được xét.
- Không giả định chắc chắn tốt hơn LR; kết luận dựa trên cùng split và metric.

## 6. Lý thuyết đánh giá và ý nghĩa nghiệp vụ

Quy ước ma trận để khớp với cách đọc phổ biến trong scikit-learn: hàng là thực tế, cột là dự đoán, thứ tự lớp `[0, 1]`; ma trận là `[[TN, FP], [FN, TP]]`. Giáo trình trình bày thứ tự Yes/No khác; phải ghi nhãn trục rõ ràng khi chuyển cách biểu diễn.

- TP: khách thật sự rời bỏ được phát hiện.
- FN: khách rời bỏ bị bỏ sót; cơ hội chăm sóc có thể mất.
- FP: khách không rời bỏ bị cảnh báo; có thể tốn công chăm sóc.
- TN: khách không rời bỏ được nhận diện đúng.
- `Precision = TP/(TP+FP)`; `Recall = TP/(TP+FN)`.
- `F1 = 2*Precision*Recall/(Precision+Recall)`.
- `Accuracy = (TP+TN)/N`; chỉ là chỉ số phụ do mất cân bằng lớp.
- ROC-AUC đo khả năng xếp hạng từ điểm lớp 1 trên các ngưỡng; không tính từ nhãn 0/1.
- PR-AUC/AP có thể bổ sung để nhìn rõ lớp rời bỏ; ghi rõ dùng Average Precision hay diện tích tích phân PR.

F1 cân bằng precision và recall nhưng không đưa trực tiếp chi phí tiền tệ vào tối ưu. Nếu chưa có chi phí FN/FP và ngân sách chăm sóc, chỉ bàn về đánh đổi và minh họa top-k; không tuyên bố đã tối ưu lợi nhuận.

Holdout là tập kiểm tra độc lập cuối cùng. CV là cách so sánh/lựa chọn trong tập phát triển. Trung bình CV và độ lệch chuẩn theo fold mô tả độ ổn định trong thiết kế này, không tự động là khoảng tin cậy.

## 7. Checklist hoàn thành

- [ ] Có định nghĩa nhãn, 19 đầu vào, đầu ra, người dùng và giới hạn.
- [ ] Có sơ đồ KDD gắn với file kết quả dự kiến.
- [ ] Có công thức đúng, ví dụ Gini tính tay và phân biệt CART với C4.5.
- [ ] Có lý do chọn ba mô hình và giải thích lớp dương.
- [ ] Phân biệt kiến thức trong giáo trình và phần bổ sung.
- [ ] Mỗi thành viên giải thích được vì sao không chọn theo accuracy và không dùng test để tuning.
