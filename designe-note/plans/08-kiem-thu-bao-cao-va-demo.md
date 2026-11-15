# Plan 08 — Kiểm thử, báo cáo, slide và demo

Người phụ trách: cả nhóm kiểm tra phần mình; TV5 tổ chức chạy tích hợp; TV3 đối chiếu metric; TV4 điều phối demo. Mục tiêu: sản phẩm chạy được, kết quả có căn cứ và mọi người bảo vệ được phương pháp.

## 1. Phạm vi kiểm thử có ý nghĩa

Ưu tiên các lỗi có thể làm sai kết luận hoặc làm đứt luồng dùng. Không cần viết test cho từng dòng UI hay kiểm lại phép toán đã do thư viện thực hiện. Tách kiểm tra tự động quan trọng khỏi checklist thao tác demo.

### Dữ liệu và phương pháp

| ID | Tình huống | Kết quả mong đợi |
|---|---|---|
| D01 | CSV nguồn giữ nguyên | Hash khớp audit; số mẫu/cột được ghi đúng |
| D02 | Split dữ liệu | Không giao ID; đủ hai lớp; fold validation không giao phần train của fold |
| D03 | Input chứa ID và Churn | Không có hai cột này trong features của pipeline |
| D04 | TotalCharges là khoảng trắng | Chuyển thiếu và được điền bằng thống kê train, không crash |
| D05 | Dữ liệu validation có giá trị số cực khác train | Imputer/scaler vẫn giữ thống kê đã fit từ train; không tự fit lại |
| D06 | Nhãn thiếu hoặc giá trị lạ | Không được coi là No và không âm thầm đưa vào train |
| D07 | Dummy dự đoán toàn No | F1 lớp Yes bằng 0; precision/recall xử lý chia 0 nhất quán |
| D08 | Tính ROC-AUC | Dùng score lớp 1, mapping đúng với classes_ |
| D09 | Bảng lựa chọn model | Dựa CV, có config và run_id; test không tham gia quyết định |

### Inference, ứng dụng và lưu trữ

| ID | Tình huống | Kết quả mong đợi |
|---|---|---|
| A01 | Cùng khách nhập form và CSV | Score/label khớp trong sai số số học |
| A02 | CSV có dòng lỗi giữa batch | Giữ row ID, lỗi đúng dòng; kết quả hợp lệ không ghép sai |
| A03 | Thiếu cột, cột trùng, file rỗng, sai kiểu | Báo lỗi dễ sửa, không đưa dữ liệu sai vào model |
| A04 | InternetService=No nhưng TechSupport=Yes | Validation chặn hoặc yêu cầu sửa, không âm thầm sửa trái ý người dùng |
| A05 | Không có artifact hoặc version không tương thích | Không dự đoán; báo cách khắc phục, không train ngầm |
| A06 | Lưu cùng thao tác hai lần do rerun | Không sinh bản ghi trùng; có ID xác nhận |
| A07 | Cập nhật nhãn thật rồi sửa nhãn | Lịch sử event còn đủ, dự đoán gốc không thay đổi |
| A08 | DB ghi lỗi | Không báo lưu thành công, không để transaction dở |
| A09 | Export rồi đọc lại | Đúng số dòng, ID, nhãn, score và tiếng Việt |
| A10 | Đổi active model | Cache đổi theo version; dự đoán mới có version mới |

### Huấn luyện lại

| ID | Tình huống | Kết quả mong đợi |
|---|---|---|
| R01 | Nhiều hồ sơ nhưng chưa có nhãn thật | Skip, không học từ dự đoán |
| R02 | Đủ tổng mẫu nhưng thiếu một lớp hoặc thiếu nhóm | Skip/giải thích rõ, không làm CV lỗi |
| R03 | Dữ liệu test gốc hoặc synthetic lọt vào kho mới | Loại khỏi huấn luyện/đánh giá chính, ghi lý do |
| R04 | Cùng khách có nhiều snapshot | Không xuất hiện ở cả train và comparison |
| R05 | Candidate kém | Champion và active pointer giữ nguyên |
| R06 | Candidate đạt bằng fixture kiểm tra nhánh | Active đổi đúng, ghi rõ fixture không phải chứng cứ chất lượng mô hình |
| R07 | Train/ghi artifact lỗi giữa chừng | Champion vẫn tải được và phục vụ được |
| R08 | Rollback | Quay lại version cũ, kết quả trên batch chuẩn khớp bản cũ |
| R09 | Hai lần retrain đồng thời | Chỉ một run được thực thi; run còn lại có trạng thái rõ |

Chạy test trên DB tạm và artifact tạm, không phá DB demo chuẩn. Kiểm thử run-to-run phải kiểm tra config/hash/split, không ép từng byte artifact giống nhau khi thư viện không đảm bảo điều đó.

## 2. Bằng chứng cần lưu

- `reports/data_audit.md`: nguồn, hash, chất lượng, quy tắc xử lý, số mẫu trước/sau.
- `reports/experiments/`: config, CV results, seed, môi trường và thời gian.
- `reports/model_selection.md`: quyết định trước test.
- `reports/final_evaluation.md`: metric test, biểu đồ và phân tích giới hạn.
- `reports/retraining_demo.md`: dữ liệu demo thật/mô phỏng, gate, so sánh, quyết định và rollback.
- `reports/test_checklist.md`: pass/fail, ngày kiểm tra và vấn đề còn lại.
- Artifact model, metadata, schema version và hướng dẫn chạy từ môi trường mới.

Ảnh chụp UI là bằng chứng demo; không thay bảng metric sinh bằng mã. Video dự phòng cũng phải dùng đúng model version và số liệu trong báo cáo.

## 3. Dàn ý báo cáo đề xuất

Số trang dưới đây chỉ là định hướng nếu giảng viên chưa có mẫu riêng. Ưu tiên quy định nộp thực tế của lớp.

| Phần | Nội dung bắt buộc | Người soạn | Dự kiến |
|---|---|---|---|
| Mở đầu | Bối cảnh churn, mục tiêu, phạm vi, đóng góp và phân công | Cả nhóm, TV5 ghép | 2–3 trang |
| Cơ sở lý thuyết | KDD, tiền xử lý, phân loại, cây và Gini/entropy, LR, RF, metric và CV | TV2 + TV3, TV1 bổ sung | 6–8 trang |
| Dữ liệu và phương pháp | Dataset, dictionary, audit, EDA, 80/20, pipeline, search và giao thức chọn model | TV1 + TV2 + TV3 | 5–7 trang |
| Thực nghiệm và thảo luận | Baseline, bảng CV, model được chọn, test, confusion matrix/ROC, diễn giải và lỗi | TV3 + TV2 | 5–7 trang |
| Ứng dụng và cập nhật | Kiến trúc, luồng đơn/CSV, schema lưu trữ, nhãn, retrain, candidate/champion | TV4 + TV5 | 4–6 trang |
| Kết luận và hạn chế | Kết quả thực tế, giới hạn dữ liệu tĩnh, xác suất, retrain mô phỏng, hướng phát triển | Cả nhóm | 1–2 trang |
| Tài liệu và phụ lục | Giáo trình, nguồn dataset, nguồn thuật toán/API, cấu hình, cách chạy, test checklist | Cả nhóm | Theo nhu cầu |

Không chèn toàn bộ source code vào thân báo cáo. Dùng đoạn mã ngắn khi cần giải thích pipeline hoặc giao thức; phần đầy đủ ở repo/phụ lục.

### Các câu cần trả lời rõ trong báo cáo

1. Bài toán thuộc loại khai phá nào và đầu ra hỗ trợ quyết định gì?
2. Vì sao loại customerID và giữ lớp dương Yes?
3. 11 TotalCharges trống được xử lý ra sao, thống kê điền thiếu lấy từ đâu?
4. Vì sao dùng F1 lớp Yes, tại sao accuracy cao có thể gây hiểu nhầm?
5. Train/validation fold/test khác nhau thế nào, chọn model ở bước nào?
6. Cây thực tế dùng thuật toán nào, Gini/entropy ảnh hưởng ra sao?
7. Model tốt nhất theo CV có kết quả test thế nào, chênh lệch được hiểu thế nào?
8. Điểm nguy cơ có được hiệu chỉnh chưa, có đủ chứng cứ nói “rời bỏ tháng tới” không?
9. Nhãn thật đến từ đâu, khi nào đủ điều kiện huấn luyện lại?
10. Vì sao candidate không tự động thay champion sau mọi lần train?

## 4. Slide và phân vai thuyết trình

Đề xuất 12–15 slide cho 12–15 phút, điều chỉnh theo thời lượng giảng viên cho phép:

1. Tên đề tài, thành viên và bài toán.
2. Mục tiêu hỗ trợ chăm sóc và phạm vi dữ liệu.
3. Quy trình KDD.
4. Dataset, audit và điểm đáng chú ý.
5. EDA chọn lọc, mỗi hình có kết luận.
6. Pipeline và chống rò rỉ.
7. Decision Tree và ví dụ split nhỏ.
8. LR/RF và lý do so sánh.
9. Thiết kế 80/20 + CV 5 fold + metric.
10. Bảng CV và quyết định chọn model.
11. Test, confusion matrix/ROC và giới hạn.
12. Kiến trúc ứng dụng, lưu nhãn và retrain.
13. Demo trực tiếp hoặc chuyển sang ứng dụng.
14. Kết luận, hạn chế và hướng phát triển.

TV1 trình bày dữ liệu/tiền xử lý; TV2 cây và LR; TV3 RF và đánh giá; TV4 ứng dụng; TV5 lưu nhãn, retrain và tích hợp. Nếu chia lời nói làm đứt mạch, gộp thành ba phần lớn và để mỗi người trả lời chuyên môn của mình khi hỏi đáp.

## 5. Kịch bản demo 8–10 phút

| Thời lượng | Thao tác | Điều cần người xem hiểu |
|---|---|---|
| 1 phút | Mở tổng quan và kết quả model | Bài toán, version thật, model chọn bằng CV |
| 2 phút | Nhập một hồ sơ và dự đoán | Nhãn, score, threshold và giới hạn diễn giải |
| 2 phút | Upload CSV gồm ca hợp lệ và một ca lỗi | Validation, giữ row ID, xếp hạng và tải kết quả |
| 1 phút | Lưu một batch, mở lịch sử | Snapshot và dự đoán gắn đúng model version |
| 1 phút | Nhập nhãn thực tế cho một snapshot demo | Nhãn thật tách khỏi dự đoán và có nguồn |
| 1–2 phút | Chạy retrain thiếu dữ liệu, xem một run demo đủ dữ liệu | Gate, candidate/champion, quyết định có căn cứ |
| 1 phút | Đổi/rollback phiên bản demo đã kiểm tra | Model active có thể quản lý và khôi phục |

Không huấn luyện lại search dài ngay trước lớp nếu mất nhiều phút. Có thể chuẩn bị run/artifact trước, chạy gate/dry-run trực tiếp và mở báo cáo đã lưu. Phải nói rõ bước nào vừa chạy, bước nào là kết quả chuẩn bị trước, dữ liệu nào mô phỏng.

Chuẩn bị hai fixture để kiểm tra nhánh chấp nhận/từ chối, không cam kết một lần retrain ngẫu nhiên chắc chắn cải thiện. Chuẩn bị bản DB demo riêng có thể khôi phục mà không đụng DB kết quả chính.

## 6. Checklist nộp bài cuối

- [ ] Có README chạy trên môi trường mới và danh sách phụ thuộc đã pin.
- [ ] Có CSV nguồn hoặc hướng dẫn vị trí/nguồn theo điều kiện phân phối, kèm hash và ghi công phù hợp.
- [ ] Có mã nguồn, config, manifest chia tập và script tái tạo kết quả.
- [ ] Có artifact model cuối và metadata; UI dùng đúng artifact đó.
- [ ] Có baseline + ba mô hình, bảng CV, quyết định chọn và test độc lập.
- [ ] Có báo cáo đủ phần lý thuyết từ giáo trình và phần mở rộng được dẫn nguồn riêng.
- [ ] Có app nhập đơn/CSV, lưu hồ sơ, nhãn thật và demo chu trình retrain.
- [ ] Có kiểm thử các lỗi quan trọng và không còn lỗi chặn demo.
- [ ] Có slide, dữ liệu demo, kịch bản và video/ảnh dự phòng.
- [ ] Số liệu/threshold/model version giữa app, báo cáo và slide khớp nhau.
- [ ] Không trình bày dữ liệu mô phỏng, fixture hoặc metric chưa chạy như kết quả thực tế.
- [ ] Có danh sách đóng góp của cả 5 người và các giới hạn chưa giải quyết.

Điều kiện hoàn thành toàn bài: người khác chạy được theo hướng dẫn, giải thích được cách chọn mô hình, truy vết được dữ liệu/kết quả và hiểu đúng giới hạn của ứng dụng.
