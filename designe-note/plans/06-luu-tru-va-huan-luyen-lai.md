# Plan 06 — Lưu dữ liệu mới, nhãn thực tế và huấn luyện lại

Người chính: TV5. TV3 phụ trách tiêu chí so sánh, TV1 kiểm tra dữ liệu, TV4 tích hợp UI. Mục tiêu: demo được một chu trình cập nhật có kiểm soát và không làm sai đánh giá ban đầu. Đây là phần vận hành bổ sung; không thay thế trọng tâm khai phá dữ liệu của bài tập lớn.

## 1. Những khái niệm phải tách biệt

- **Hồ sơ/snapshot**: giá trị đặc trưng tại một thời điểm trước khi biết kết quả thực tế.
- **Dự đoán**: nhãn và score do một phiên bản mô hình tạo ra cho snapshot đó.
- **Nhãn thực tế**: thông tin kết quả từ nguồn xác nhận, có thời điểm và định nghĩa quan sát.
- **Champion**: mô hình đang phục vụ. **Candidate**: mô hình mới đang được kiểm chứng.
- **Tập test gốc**: 20% dùng cho báo cáo nghiên cứu; không dùng để quyết định promote mỗi tháng.
- **Tập so sánh cập nhật**: dữ liệu được giữ riêng khỏi train cho cả champion và candidate trong một kỳ đánh giá.

Tuyệt đối không dùng `churn_prediction` làm `actual_churn`. Thiếu nhãn không có nghĩa khách chưa rời bỏ.

## 2. Thiết kế SQLite

Đường dẫn dự kiến: `data/runtime/churn.db`. Bật kiểm tra khóa ngoại; thực hiện thao tác liên quan bằng transaction. Tạo migration/schema version để tránh UI và DB lệch nhau.

| Bảng | Trường chính dự kiến | Vai trò |
|---|---|---|
| `customers` | internal_customer_id PK, external_customer_id, created_at | Định danh nội bộ, external ID có thể thiếu |
| `customer_snapshots` | snapshot_id PK, customer_id FK, observed_at, features_json, schema_version, source, is_synthetic, payload_hash, created_at | Đóng băng 19 đặc trưng lúc dự đoán |
| `prediction_batches` | batch_id PK, source, idempotency_key UNIQUE, created_at | Quản lý lưu đơn/CSV và chống lưu lặp |
| `predictions` | prediction_id PK, batch_id FK, snapshot_id FK, model_version FK, churn_score, predicted_label, threshold, predicted_at | Lưu riêng dự đoán |
| `actual_label_events` | event_id PK, snapshot_id FK, actual_churn, label_observed_at, label_source, recorded_at, supersedes_event_id, note | Lịch sử nhãn và sửa nhãn |
| `model_versions` | model_version PK, run_id, artifact_path, schema_version, metrics_json, status, created_at | Theo dõi model đã đăng ký |
| `training_runs` | run_id PK, as_of, snapshot_path, snapshot_hash, config_json, champion_version, candidate_version, status, decision, reason, started_at, finished_at | Nhật ký huấn luyện lại |
| `training_membership` | run_id FK, snapshot_id FK, partition, label_event_id FK | Biết mẫu và nhãn nào đã dùng train/validation/comparison |

`features_json` phù hợp quy mô demo nếu được kiểm tra schema khi ghi/đọc. Nếu cần truy vấn từng đặc trưng thường xuyên, có thể chuyển sang 19 cột tường minh; không làm cả hai bản khác nhau rồi không đồng bộ.

Trạng thái model: candidate/active/archived/rejected. Lưu nhãn bằng event mới thay vì sửa trực tiếp sự kiện cũ. Chọn nhãn hiệu lực theo quy tắc rõ ràng tại thời điểm `as_of`; không dùng nhãn được bổ sung sau thời điểm chốt dữ liệu.

## 3. Luồng lưu và cập nhật nhãn

1. Validation đầu vào trước inference và trước ghi DB.
2. Khi người dùng chọn lưu: tạo/liên kết customer, tạo snapshot bất biến, tạo batch/prediction trong một transaction.
3. Tạo idempotency key từ thao tác người dùng hoặc batch token; hai lần rerun cùng thao tác không tạo hai snapshot.
4. Khi có kết quả thật: chọn đúng snapshot, nhập nhãn và nguồn/thời điểm; kiểm tra quan sát không sớm hơn snapshot.
5. Nếu sửa nhãn: thêm event mới có lý do; giữ dự đoán gốc để so sánh.
6. Đánh dấu dữ liệu mô phỏng ngay lúc nhập; mặc định loại khỏi huấn luyện/đánh giá chính.

Trong bộ dữ liệu gốc không có timestamp đầy đủ để thiết lập horizon. Demo có thể dùng nhãn nhập thủ công theo quy ước đã ghi, nhưng phải nói rõ đây là minh họa kỹ thuật. Khi chuyển vận hành thật, cần thống nhất “churn trong khoảng H sau thời điểm snapshot” và chỉ gán No khi đã đủ thời hạn quan sát H. Không tự mặc định H=30 ngày là định nghĩa sẵn của bộ CSV.

## 4. Gate trước huấn luyện hàng tháng

Chính sách đề xuất cho demo, cấu hình được và cần ghi rõ là do nhóm đặt:

| Điều kiện | Cách kiểm tra | Không đạt thì |
|---|---|---|
| Đến kỳ chạy | Ngày 01 hàng tháng lúc 02:00 giờ Việt Nam hoặc chạy thủ công mô phỏng | Ghi lý do skip nếu chưa đến kỳ; demo có cờ bỏ qua lịch |
| Có dữ liệu mới | Ít nhất 200 snapshot mới có nhãn hợp lệ kể từ lần train thành công | Skip, giữ mô hình cũ |
| Đủ hai lớp | Ít nhất 30 Yes và 30 No trong tập mới đủ điều kiện | Skip; không cố chạy CV trên lớp quá ít |
| Schema hợp lệ | Đủ features, đúng kiểu/miền, nhãn thật, đã đủ thời gian quan sát theo quy ước | Cách ly dòng lỗi, tính lại gate |
| Không trùng/rò rỉ | Không trùng snapshot, không dùng test gốc, không trùng entity giữa các partition cần độc lập | Sửa phân chia hoặc dừng run |
| Có tập so sánh độc lập | Đủ mẫu mỗi lớp và không có trong train của cả hai model | Đợi thêm dữ liệu hoặc chỉ tạo candidate, chưa promote |

200 và 30 mỗi lớp chỉ giúp demo có điều kiện tối thiểu, không bảo đảm độ tin cậy thống kê hay chất lượng nghiệp vụ. Gate phải kiểm tra lại số mẫu **sau** làm sạch và partition; không chỉ kiểm tra tổng số rồi mặc định 5-fold luôn hợp lệ.

## 5. Dữ liệu cho một kỳ cập nhật

1. Chốt `as_of`; xuất snapshot, nhãn hiệu lực, nguồn, nhóm khách và hash bất biến.
2. Tập nền để train bắt đầu từ **80% phát triển gốc**. Dữ liệu test gốc tiếp tục loại khỏi toàn bộ kho cập nhật.
3. Với dữ liệu thật mới, dùng snapshot trước nhãn; xử lý nhiều lần quan sát cùng khách bằng group theo customer. Không để cùng khách ở hai phía train/comparison.
4. Chọn một phần dữ liệu mới chưa từng dùng để fit champion làm comparison, ưu tiên kỳ thời gian mới nhất. Chọn kích thước theo số lượng thực tế và kiểm tra đủ cả hai lớp; nếu dữ liệu chỉ vừa chạm gate có thể vẫn chưa đủ để promote.
5. Phần còn lại bổ sung vào dữ liệu nền để train candidate. Tập comparison không tham gia fit preprocessing, tuning hay threshold.
6. Trong tập train candidate, CV phải giữ các snapshot cùng khách trong cùng fold. Dùng chia theo nhóm có phân tầng khi khả thi; kiểm tra số nhóm/lớp mỗi fold. Nếu thiếu nhóm, giảm độ phức tạp thiết kế hoặc dừng thay vì gây rò rỉ.
7. Với dữ liệu có thời gian thực, ưu tiên phân chia theo thời gian để phản ánh dự đoán tương lai; ghi rõ khác biệt so với CV phân tầng của thực nghiệm CSV tĩnh ban đầu.

Dữ liệu comparison được đánh dấu đã dùng đánh giá. Không dùng đi dùng lại một holdout nhỏ để chọn model vô hạn. Kỳ sau cần lứa so sánh mới chưa dùng fit các model đang được so sánh; nếu không có thì giữ champion. Sau khi promote, giữ candidate đúng bản đã kiểm tra, không tự refit thêm comparison rồi coi vẫn là artifact đã được đánh giá.

## 6. Huấn luyện candidate và so sánh

- Bản đầu giữ họ mô hình/không gian tìm kiếm đã chốt; retrain toàn bộ pipeline từ dữ liệu đủ điều kiện, không chỉ fit classifier trên đầu vào đã scale cũ.
- Tạo thư mục candidate mới, không ghi đè champion.
- Nếu tuning lại thì dùng CV bên trong train candidate; giữ threshold 0,5 như nghiên cứu chính.
- Chạy champion và candidate trên **cùng** comparison set với cùng mapping, threshold và scorer.
- Lưu precision, recall, F1 lớp 1, ROC-AUC, confusion matrix, số mẫu từng lớp và thời gian dự đoán.

Chính sách promote demo đề xuất:

1. Candidate có F1 cao hơn champion ít nhất 0,01 tuyệt đối.
2. Recall candidate không giảm quá 0,02 tuyệt đối so với champion.
3. Schema/artifact/inference checks đều đạt; không có NaN hoặc score ngoài [0,1].
4. Ghi lại cả metric và lý do chấp nhận/từ chối. Nếu tập so sánh nhỏ hoặc khác biệt thiếu vững chắc, giữ champion và ghi “chưa đủ bằng chứng”.

Các mốc 0,01/0,02 là chính sách minh họa, không phải kiểm định ý nghĩa thống kê. Không dùng test gốc để ép candidate vượt champion. Nếu không đạt, candidate vẫn lưu để truy vết nhưng không kích hoạt.

## 7. Kích hoạt, lỗi và rollback

1. Kiểm tra bundle candidate bằng một batch chuẩn trước khi active.
2. Đăng ký metadata/version; cập nhật con trỏ active một cách nguyên tử, tránh DB và file pointer chỉ sang hai phiên bản khác nhau. Có bước đối soát và phục hồi nếu cập nhật giữa chừng thất bại.
3. Xóa/invalidate cache theo version để UI tải đúng model; dự đoán đang xử lý tiếp tục gắn đúng version đã tải.
4. Giữ nguyên artifact của champion cũ và lịch sử kích hoạt.
5. Rollback bằng đổi active về bản đã biết hoạt động; không cần retrain.
6. Nếu train lỗi, mất file, thiếu nhãn hay comparison chỉ có một lớp: log lỗi và giữ model cũ, không thay bằng artifact dở dang.
7. Dùng khóa chống hai tác vụ retrain cùng chạy. Run đã hoàn tất không bị tính lặp thành dữ liệu mới.

Lệnh dự kiến: `python scripts/retrain.py --as-of <timestamp> --dry-run`. Sau khi luồng thủ công hoạt động ổn định, cấu hình Windows Task Scheduler theo giờ Việt Nam, trỏ đúng Python trong `.venv` và working directory dự án; lưu log. Trong demo có thể chạy bằng tay để minh họa chu kỳ tháng, không cần chờ đến tháng sau.

## 8. Kịch bản demo và giới hạn

- A: ít hơn gate → báo chưa đủ dữ liệu, champion không đổi.
- B: đủ dữ liệu mô phỏng → tạo candidate và báo cáo so sánh; dữ liệu có cờ synthetic, không gộp vào kết quả khoa học.
- C: candidate kém → giữ champion.
- D: candidate đạt điều kiện → kích hoạt; dự đoán mới hiển thị version mới; rollback về version cũ được.

Nếu thiếu dữ liệu thật, dùng fixture kết quả rõ nhãn để kiểm tra nhánh promote/reject, hoặc dữ liệu tổng hợp trong DB demo riêng. Không sửa nhãn hay chọn seed để cố tạo một kết quả cải thiện rồi trình bày như kết quả thực tế. Dữ liệu tái sử dụng từ CSV chỉ có thể minh họa phần mềm; không chứng minh mô hình cải thiện ngoài mẫu.

Điều kiện nghiệm thu: hồ sơ/nhãn/dự đoán tách biệt; snapshot có nguồn và phiên bản; không học từ nhãn giả; run tái lập được; cùng comparison set cho cả hai model; có skip/reject/promote/rollback; test gốc không tham gia vòng cập nhật.
