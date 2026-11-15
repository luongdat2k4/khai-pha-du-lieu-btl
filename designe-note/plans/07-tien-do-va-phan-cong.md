# Plan 07 — Tiến độ, phân công 5 người và quản lý rủi ro

## 1. Giả định lập lịch

Chưa có hạn nộp và lịch rảnh của nhóm, nên dùng lịch tương đối **4 tuần**, không gán ngày nộp giả định. Tổng công sức dự kiến 160–200 giờ cho 5 người, khoảng 32–40 giờ/người. Đây là ước lượng để chia việc, sẽ cập nhật sau tuần 1.

Nếu thời gian ngắn hơn, thu hẹp tính năng nâng cao và tuning trước; vẫn giữ đủ ba mô hình, chia tập đúng, CV, test, nhập đơn/CSV và minh họa lưu nhãn/retrain. Không cắt quy tắc chống rò rỉ để chạy nhanh.

## 2. Phân công theo đúng hướng trong note

| Thành viên | Trách nhiệm chính | Đầu ra sở hữu | Người rà soát | Ước lượng |
|---|---|---|---|---|
| TV1 | Dữ liệu, audit, split, EDA, schema và tiền xử lý | `data.py`, `schema.py`, `preprocessing.py`, audit, EDA, data dictionary | TV2 kiểm tra pipeline, TV3 kiểm tra split | 32–40 giờ |
| TV2 | Decision Tree, Logistic Regression, lý thuyết phân loại | Config/search hai mô hình, phần cây/LR, hình cây/hệ số | TV3 soát scorer và kết quả | 32–40 giờ |
| TV3 | Random Forest, đánh giá, so sánh và chọn model | RF config, `evaluation.py`, bảng CV/test, quyết định model, tiêu chí cập nhật | TV1 soát dữ liệu, TV2 soát lý thuyết | 32–40 giờ |
| TV4 | Streamlit và trải nghiệm demo | `app.py`, form, upload, bảng kết quả, lịch sử/nhãn | TV5 soát tích hợp, TV1 soát input | 32–40 giờ |
| TV5 | Lưu dữ liệu mới, retrain và ghép hệ thống | Storage, registry, retraining, scripts, README chạy, gói demo | TV3 soát so sánh, TV4 soát luồng dùng | 32–40 giờ |

TV5 dễ quá tải vì vừa làm vận hành vừa tích hợp. TV3 nhận phần metric/gate so sánh; TV4 nhận UI nhập nhãn; TV1 hỗ trợ snapshot/validation. Cả nhóm viết báo cáo phần mình; không dồn tất cả việc viết và ghép cho TV5 vào cuối kỳ.

## 3. Kế hoạch theo tuần

### Tuần 1 — Chốt thiết kế và dữ liệu

| Thời điểm | Công việc | Người | Đầu ra/tiêu chí |
|---|---|---|---|
| Ngày làm việc 1 | Đọc note và Plan 01; thống nhất nhãn, metric, phạm vi, vai trò | Cả nhóm | M0: đặc tả và giao thức được ghi lại |
| Ngày 1–2 | Audit CSV, kiểm tra thiếu/trùng/bất nhất, tạo dictionary | TV1 | Báo cáo có số thật và quy tắc xử lý |
| Ngày 1–2 | Khung package, config, hợp đồng inference/storage | TV5 + TV4 | Cấu trúc và schema kết quả chung |
| Ngày 2–3 | Split 80/20, manifest, tạo 5 fold | TV1 + TV3 | ID hai tập không giao, lưu seed/hash |
| Ngày 2–4 | Viết lý thuyết DT/LR/RF, metric, ví dụ Gini | TV2 + TV3 | Bản thảo lý thuyết gắn giáo trình |
| Ngày 3–5 | EDA tập phát triển, pipeline và smoke run baseline | TV1 + TV2 + TV3 | M1: dữ liệu/pipeline sẵn sàng |
| Ngày 3–5 | Form/CSV skeleton với schema thật, output giả lập được đánh dấu | TV4 | UI bắt lỗi thiếu/sai đầu vào |
| Cuối tuần | Review schema, dữ liệu và rò rỉ | Cả nhóm | Không còn khác biệt mapping/cột |

### Tuần 2 — Thực nghiệm mô hình và ứng dụng cơ bản

| Thời điểm | Công việc | Người | Đầu ra/tiêu chí |
|---|---|---|---|
| Ngày 1–3 | Search DT/LR; lưu cv_results | TV2 | Cấu hình tốt nhất của hai họ |
| Ngày 1–3 | Search RF, đồng bộ evaluator | TV3 | Cấu hình RF và bảng CV chung |
| Ngày 1–3 | DB/schema, lưu snapshot/prediction, chống lưu lặp | TV5 | Lưu/đọc bằng transaction |
| Ngày 1–4 | Form/CSV hoàn chỉnh theo hợp đồng; xuất kết quả | TV4 | P0 dự đoán đơn/batch sẵn ghép |
| Ngày 3–4 | Chọn model theo CV, khóa tham số/ngưỡng | TV2 + TV3, TV1 review | `model_selection.md` trước test |
| Ngày 4–5 | Đánh giá test cuối, đóng gói bundle | TV3 + TV2 | M2: metric, hình và artifact thật |
| Ngày 5 | Ghép model thật vào app; đối chiếu đơn/batch | TV4 + TV5 | Kết quả khớp module inference |

### Tuần 3 — Lưu nhãn, huấn luyện lại và kiểm thử tích hợp

| Thời điểm | Công việc | Người | Đầu ra/tiêu chí |
|---|---|---|---|
| Ngày 1–2 | Lịch sử, cập nhật nhãn, nguồn/thời điểm | TV4 + TV5 | M3: dùng được toàn bộ P0 |
| Ngày 1–3 | Snapshot, điều kiện retrain, partition chống trùng nhóm | TV5 + TV1 | Skip đúng, dữ liệu có provenance |
| Ngày 2–4 | Candidate/champion cùng comparison, chính sách promote | TV5 + TV3 | Báo cáo quyết định và model registry |
| Ngày 3–4 | Kiểm tra skip/reject/promote/rollback bằng fixture demo | TV5 + TV3 + TV4 | M4: đủ nhánh vận hành |
| Ngày 3–5 | Kiểm thử tích hợp và lỗi input/artifact/DB | Cả nhóm | Sửa các lỗi chặn demo |
| Ngày 4–5 | Viết phương pháp, kết quả và hạn chế vào báo cáo | Mỗi người phần mình | Bản thảo đầy đủ, không còn số giả |

### Tuần 4 — Hoàn thiện, chạy lại và bảo vệ

| Thời điểm | Công việc | Người | Đầu ra/tiêu chí |
|---|---|---|---|
| Ngày 1–2 | Chạy từ môi trường mới, đối chiếu artifact/metric | TV1 + TV5 | README đủ để người khác chạy |
| Ngày 1–3 | Hoàn thiện báo cáo, tài liệu tham khảo và slide | Cả nhóm | Số liệu giữa các tài liệu khớp |
| Ngày 2–3 | Demo đúng luồng, ghi video dự phòng | TV4 + TV5 | Không phụ thuộc Internet cho inference |
| Ngày 3–4 | Diễn tập hỏi đáp: pipeline, CV, F1, cây, retrain | Cả nhóm | Mỗi người giải thích được toàn quy trình |
| Ngày 4 | Đóng băng phiên bản nộp; sao lưu DB demo/artifact | TV5 | Danh mục file và version cuối |
| Ngày 5 | Dự phòng sửa lỗi cần thiết và kiểm tra bàn giao | Cả nhóm | M5: gói nộp hoàn chỉnh |

## 4. Quản lý đầu việc và bàn giao

- Mỗi đầu việc có người chính, đầu ra cụ thể, phụ thuộc, ngày dự kiến và điều kiện đạt. Dùng bảng task đơn giản hoặc issue của nhóm.
- Mỗi buổi làm ghi ngắn: đã có kết quả gì, vướng gì, ai đang chờ đầu ra nào. Các thay đổi schema phải báo ngay.
- Họp ghép 2 lần/tuần khoảng 20–30 phút; không chờ đến cuối tuần 3 mới ghép UI và model.
- Module dùng chung chỉ có một nguồn chân lý; không chép các bản preprocessing vào notebook của từng người rồi để lệch nhau.
- Mỗi lần chạy thí nghiệm có run_id, config và log. Không đổi dữ liệu/seed mà không ghi nhận.
- Khi bàn giao code phải kèm một ví dụ đầu vào/đầu ra và cách chạy; người nhận thử trước khi đánh dấu hoàn thành.

## 5. Các rủi ro chính

| Rủi ro | Dấu hiệu | Cách xử lý | Người theo dõi |
|---|---|---|---|
| Rò rỉ dữ liệu | Scale trước split, chọn theo test, ID lọt vào X | Pipeline theo fold, review manifest và feature names | TV1 + TV3 |
| Lệch schema giữa UI và train | Form chạy nhưng điểm khác CSV hoặc lỗi one-hot | Dùng cùng validator/inference; test đơn/batch | TV1 + TV4 |
| Không đủ nhãn thật để retrain | Nhiều hồ sơ mới nhưng actual label trống | Demo gate và fixture có đánh dấu; không chế nhãn | TV5 |
| Tuning quá lâu | RF chiếm máy, search chồng luồng | Giới hạn số cấu hình, đo thử và giới hạn song song | TV3 |
| Kết quả kém kỳ vọng | F1 thấp, recall gần 0 | Kiểm tra nhãn/pipeline/imbalance trên development; báo trung thực | TV2 + TV3 |
| Chậm tích hợp | Mỗi người có output/cột riêng | Khóa hợp đồng Plan 04 từ tuần 1 | TV5 |
| Thiếu thời gian viết | Báo cáo bắt đầu sát hạn | Viết lý thuyết tuần 1, phương pháp tuần 2, kết quả tuần 3 | Cả nhóm |
| Model/DB lỗi sát demo | Đường dẫn cứng, cache cũ, thư viện lệch | Pin môi trường, chạy sạch, sao lưu và video dự phòng | TV4 + TV5 |
| Vận hành lấn phần học thuật | Dành quá nhiều thời gian UI/retrain | Bảo vệ mốc M2; bỏ P2 và hệ thống triển khai phức tạp | Cả nhóm |

## 6. Nếu chỉ có khoảng 2 tuần

Tuần A: dành hai ngày audit/split/pipeline và khung tích hợp; ba ngày CV ba mô hình, chọn model, test và ghép dự đoán đơn/CSV. Tuần B: lưu snapshot/nhãn, retrain chạy thủ công với gate và so sánh, kiểm thử, báo cáo, slide/demo.

Cắt trước: SHAP, slider, biểu đồ tương tác phức tạp, deployment cloud, tối ưu threshold/calibration, feature engineering và tuning mở rộng. Không cắt: lý thuyết cây, audit, split/CV đúng, F1 lớp 1, test độc lập, artifact tái dùng, minh bạch dữ liệu mô phỏng.
