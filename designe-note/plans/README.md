# Kế hoạch bài tập lớn: Dự đoán khách hàng rời bỏ dịch vụ viễn thông

Ngày lập: 15/11/2026. Trạng thái: kế hoạch triển khai, chưa huấn luyện mô hình và chưa có kết quả thực nghiệm.

## 1. Căn cứ và phạm vi

- Yêu cầu gốc: [note.md](../note.md).
- Giáo trình: [DMDWv1.2.docx.md](../../doc-tmp/DMDWv1.2.docx.md), trọng tâm Chương 1 về KDD và Chương 4, đặc biệt mục 4.1 và 4.3.
- Dữ liệu đã kiểm tra trực tiếp: [Telco-Customer-Churn.csv](../../telco-customer-churn-on-icp4d/data/Telco-Customer-Churn.csv).
- Nhóm 5 người; Python, scikit-learn, Streamlit; ba mô hình Decision Tree, Logistic Regression, Random Forest.
- Phân chia 80% phát triển / 20% kiểm thử, xác nhận chéo phân tầng 5 fold trên phần phát triển; chọn theo F1 của lớp `Churn=Yes`.
- Ứng dụng nhập một khách, dự đoán CSV, lưu dữ liệu mới, cập nhật nhãn thực tế và có quy trình huấn luyện lại theo tháng khi đủ dữ liệu.

Đây là bài tập lớn thuộc **Khai phá dữ liệu**. Trọng tâm chấm và trình bày cần là quy trình phát hiện tri thức, chất lượng dữ liệu, thuật toán phân loại, thiết kế thực nghiệm, đánh giá và diễn giải. SQLite chỉ phục vụ lưu trữ ứng dụng; không mở rộng thành bài toán xây kho dữ liệu, OLAP hay hệ thống doanh nghiệp.

## 2. Thông tin đã xác minh và giả định cần ghi rõ

| Nội dung | Trạng thái |
|---|---|
| 7.043 dòng, 21 cột | Đã kiểm tra CSV |
| `Churn=No`: 5.174; `Churn=Yes`: 1.869 | Đã kiểm tra; tỷ lệ rời bỏ khoảng 26,54% |
| `TotalCharges` chứa 11 chuỗi trống/khoảng trắng | Đã kiểm tra; phải chuyển số và xử lý thiếu |
| Không có `customerID` trùng | Đã kiểm tra; vẫn phải kiểm tra trùng toàn dòng và trùng đặc trưng khi triển khai |
| 19 thuộc tính đầu vào sau khi bỏ ID và nhãn | Suy ra từ schema 21 cột |
| Lịch 4 tuần, khoảng 160–200 giờ công cả nhóm | Đề xuất để lập kế hoạch; điều chỉnh theo hạn nộp thật |
| Chạy cục bộ bằng Streamlit + SQLite | Đề xuất kiến trúc vừa phạm vi |
| Ngưỡng nhãn mặc định 0,5 | Quyết định thực nghiệm chính để giữ phép so sánh đơn giản, công bằng |
| Điều kiện dữ liệu mới và ngưỡng chấp nhận mô hình | Chính sách demo đề xuất, không phải thông số có sẵn của IBM |

CSV hiện có là tập dữ liệu tĩnh, không có lịch sử thời gian đầy đủ để chứng minh khả năng dự đoán trước đúng một tháng. Vì vậy báo cáo chỉ kết luận khả năng phân loại nhãn Churn trên dữ liệu này. Quy trình vận hành hàng tháng là phần thiết kế/mô phỏng; khi dùng thực tế cần định nghĩa thời điểm chụp hồ sơ, thời hạn quan sát và nguồn nhãn.

## 3. Danh sách các plan

| Plan | Nội dung | Người chính | Điều kiện bắt đầu |
|---|---|---|---|
| [01 — Bài toán và lý thuyết](01-bai-toan-va-co-so-ly-thuyet.md) | Đặc tả, KDD, công thức, liên hệ giáo trình | Cả nhóm; TV2 tổng hợp thuật toán | Bắt đầu ngay |
| [02 — Dữ liệu và tiền xử lý](02-du-lieu-va-tien-xu-ly.md) | Audit, chia tập, EDA, schema, pipeline | TV1 | Đặc tả nhãn và đầu vào |
| [03 — Huấn luyện và đánh giá](03-huan-luyen-va-danh-gia.md) | Baseline, 3 mô hình, 5-fold CV, chọn và test | TV2 + TV3 | Split và pipeline ổn định |
| [04 — Kiến trúc và hợp đồng tích hợp](04-kien-truc-va-tich-hop.md) | Cấu trúc mã, giao diện hàm, artifact, lệnh chạy | TV5 | Chốt schema sơ bộ |
| [05 — Ứng dụng Streamlit](05-ung-dung-streamlit.md) | Nhập đơn, CSV, kết quả, lịch sử và nhãn | TV4 | Hợp đồng inference; có thể dùng bản giả lập trước |
| [06 — Lưu trữ và huấn luyện lại](06-luu-tru-va-huan-luyen-lai.md) | SQLite, nhãn, dữ liệu đủ điều kiện, candidate/champion | TV5 + TV3 | Mô hình và dữ liệu đầu vào ổn định |
| [07 — Tiến độ và phân công](07-tien-do-va-phan-cong.md) | Lịch 4 tuần, việc của 5 người, phụ thuộc, rủi ro | Cả nhóm | Bắt đầu ngay |
| [08 — Kiểm thử, báo cáo và demo](08-kiem-thu-bao-cao-va-demo.md) | Nghiệm thu, báo cáo, slide, demo, bàn giao | Cả nhóm; TV5 ghép | Viết dần từ tuần 1 |

## 4. Thứ tự thực hiện

1. Đọc Plan 01, chốt phạm vi, quy ước nhãn và vai trò.
2. TV1 thực hiện Plan 02; TV5 dựng khung theo Plan 04. TV2 và TV3 soạn lý thuyết thuật toán.
3. Khi schema ổn định, TV4 làm UI dựa trên hợp đồng chung. Dữ liệu giả cho UI phải được đánh dấu, không đưa vào kết quả nghiên cứu.
4. TV2 và TV3 thực hiện Plan 03, chọn mô hình bằng CV rồi mới mở tập test.
5. Ghép pipeline thật vào ứng dụng; TV5 hoàn thiện Plan 06.
6. Hoàn thành kiểm thử, báo cáo, slide và demo theo Plan 08.

Luồng phụ thuộc chính: `đặc tả → dữ liệu/split → pipeline → CV/chọn mô hình → test cuối → artifact → ứng dụng`. Lưu trữ và UI có thể phát triển sớm theo schema, nhưng nghiệm thu phải dùng pipeline thật.

## 5. Các mốc nghiệm thu

| Mốc | Điều kiện hoàn thành |
|---|---|
| M0 — Khóa thiết kế | Chốt schema 19 đặc trưng, quy trình split/CV, metric, hợp đồng tích hợp |
| M1 — Dữ liệu sẵn sàng | Có audit, danh sách ID chia tập, pipeline xử lý thiếu/mã hóa/chuẩn hóa, EDA trên phần phát triển |
| M2 — Thực nghiệm hoàn tất | Có baseline + 3 mô hình, CV mean/std, quyết định chọn mô hình bằng CV, báo cáo test sau khi khóa lựa chọn |
| M3 — Ứng dụng hoàn tất | Nhập đơn và CSV cho cùng kết quả; lưu hồ sơ và cập nhật nhãn được |
| M4 — Chu trình cập nhật | Chạy được trường hợp thiếu dữ liệu, đủ dữ liệu mô phỏng, giữ mô hình cũ và thay mô hình đạt tiêu chí; rollback được |
| M5 — Sẵn sàng nộp | Chạy lại được từ hướng dẫn; báo cáo/slide/demo thống nhất với artifact |

Không đặt trước cam kết “F1 phải đạt 0,85” hay “Random Forest chắc chắn tốt nhất”. Kết quả phải do thực nghiệm xác định. Nếu mô hình không hơn baseline, phân tích trung thực và tìm lỗi trên dữ liệu phát triển, không sửa quy trình để làm đẹp test.

## 6. Quy tắc xuyên suốt

- Tập test gốc được giữ riêng; không dùng để chọn đặc trưng, tham số, ngưỡng hay quyết định cập nhật mô hình hàng tháng.
- Imputer, encoder, scaler được fit trong từng fold qua pipeline; ứng dụng tải cả pipeline đã fit.
- Nhãn dự đoán và điểm nguy cơ không phải nhãn thực tế; hồ sơ chưa có nhãn thật không được đưa vào huấn luyện lại.
- Mọi biểu đồ, số liệu và kết luận phải gắn với dữ liệu, split và phiên bản mô hình cụ thể.
- Phân biệt nội dung từ giáo trình, kiến thức mở rộng, lựa chọn kỹ thuật của nhóm và kết quả thực nghiệm.
- Các đường dẫn mã, lệnh và artifact trong bộ plan là thiết kế sẽ triển khai, chưa có nghĩa đã tồn tại trong workspace.
