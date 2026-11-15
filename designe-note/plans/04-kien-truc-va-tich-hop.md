# Plan 04 — Kiến trúc mã nguồn và hợp đồng tích hợp

Người chính: TV5. Cả nhóm thống nhất hợp đồng này ở M0 để có thể làm song song. Đây là cấu trúc đề xuất cho mã mới, chưa phải mô tả các file hiện có.

## 1. Kiến trúc vừa phạm vi

Luồng offline: `CSV gốc → validation/cleaning → split manifest → pipeline + CV → chọn mô hình → test → artifact`.

Luồng ứng dụng: `form/CSV → validation chung → pipeline đang active → kết quả → SQLite/export`.

Luồng cập nhật: `snapshot đã có nhãn thật → kiểm tra đủ dữ liệu → train candidate → so sánh với champion → quyết định → đổi active model hoặc giữ nguyên`.

Streamlit gọi thư viện Python nội bộ; chưa cần FastAPI, microservice, Docker hay cloud cho bản bắt buộc. SQLite đủ cho demo cục bộ một nhóm nhỏ; nếu chuyển sang nhiều người dùng đồng thời thì phải đánh giá lại kiến trúc.

## 2. Cấu trúc thư mục dự kiến

```text
hoc-lai/
├── app.py
├── requirements.txt
├── README.md
├── src/telco_churn/
│   ├── __init__.py
│   ├── schema.py            # cột, miền, kiểu, kiểm tra
│   ├── data.py              # đọc, chuyển kiểu định trước, audit, split
│   ├── preprocessing.py     # preprocessor theo mô hình
│   ├── training.py          # cấu hình và tìm tham số
│   ├── evaluation.py        # metric, biểu đồ, bảng
│   ├── inference.py         # dự đoán dùng chung
│   ├── storage.py           # SQLite, transaction, migration
│   ├── registry.py          # active version, kiểm tra artifact, rollback
│   └── retraining.py        # gate, snapshot, candidate, so sánh
├── scripts/
│   ├── audit_data.py
│   ├── make_splits.py
│   ├── train.py
│   ├── evaluate.py
│   ├── init_db.py
│   └── retrain.py
├── notebooks/               # khám phá và giải thích, không chứa logic duy nhất
├── configs/
│   ├── experiment.json
│   └── retraining.json
├── data/
│   ├── splits/
│   ├── samples/             # CSV mẫu, gồm cả trường hợp lỗi
│   ├── snapshots/           # dữ liệu huấn luyện lại đóng băng
│   └── runtime/churn.db
├── artifacts/models/<version>/
├── artifacts/active_model.json
├── reports/{figures,experiments}/
├── docs/
├── tests/
└── designe-note/plans/       # bộ kế hoạch này
```

Dataset gốc và ví dụ IBM đang ở `telco-customer-churn-on-icp4d/`; giữ làm nguồn tham khảo. Không phụ thuộc dịch vụ IBM Cloud/Flask từ ví dụ cũ khi yêu cầu đã chọn scikit-learn + Streamlit. Đọc license và ghi công nguồn dữ liệu/mã thực sự tái sử dụng trong báo cáo.

## 3. Hợp đồng giữa các module

| Hàm dự kiến | Đầu vào | Đầu ra/điều kiện |
|---|---|---|
| `validate_records(frame, mode)` | DataFrame thô; mode training/inference | Dòng hợp lệ đã chuẩn hóa định trước + danh sách lỗi có row/column/code/message |
| `build_preprocessor(model_kind)` | logistic/tree/forest | Transformer chưa fit, tên biến thống nhất |
| `load_model_bundle(version=None)` | Phiên bản hoặc active | Pipeline + metadata; kiểm tra schema/version |
| `predict_records(frame, bundle)` | 19 features, ID tùy chọn ngoài X | Một dòng kết quả cho mỗi dòng hợp lệ; giữ thứ tự và khóa truy vết |
| `save_prediction_batch(inputs, outputs, source)` | Snapshot input + kết quả | batch_id, snapshot_id, prediction_id; transaction |
| `record_actual_label(snapshot_id, label, observed_at, source)` | Nhãn thực tế tách biệt | Lưu lịch sử thay đổi; không ghi đè dự đoán |
| `train_experiment(config, manifest)` | Cấu hình + split đã khóa | run_id, logs, model bundle, CV summary |
| `run_retraining(as_of, dry_run)` | Thời điểm chốt dữ liệu và cờ demo | run_id, gate result, candidate metrics, quyết định và lý do |

Tên hàm có thể điều chỉnh khi code nhưng việc thay đổi phải cập nhật cả bên gọi và tài liệu trong cùng lần tích hợp. Module inference tuyệt đối không gọi fit.

### Schema kết quả dự đoán

| Trường | Ý nghĩa |
|---|---|
| `input_row_id` | Vị trí dòng trong batch; không bị mất khi CSV thiếu ID |
| `customerID` | ID gốc hoặc ID nội bộ cấp khi lưu |
| `churn_prediction` | 0/1 |
| `churn_label` | No/Yes để hiển thị |
| `churn_score` | Điểm lớp 1 trong [0,1] |
| `priority_level` | Thấp/Trung bình/Cao theo chính sách hiển thị |
| `threshold` | Ngưỡng dùng tạo nhãn |
| `model_version` | Phiên bản artifact |
| `predicted_at` | Thời điểm dự đoán, lưu UTC và hiển thị giờ Việt Nam |

Không lấy cứng cột xác suất thứ hai nếu chưa đối chiếu `classes_`; tìm đúng vị trí của lớp 1 và kiểm tra mapping khi tải artifact.

## 4. Hành vi lỗi và dữ liệu

- Thiếu cột bắt buộc: trả danh sách thiếu, không đưa DataFrame vào pipeline để chờ lỗi kỹ thuật.
- Thừa cột: liệt kê cột không sử dụng; tách ID; từ chối `Churn` ở luồng dự đoán và hướng người dùng sang chức năng nhập nhãn.
- Sai giá trị ô: đánh dấu dòng/cột và lý do; không làm lệch thứ tự kết quả.
- Không có mô hình hoặc artifact không tương thích: dừng chức năng dự đoán, nêu thao tác khắc phục; không tự train trong UI.
- Không đọc được DB: thông báo việc lưu thất bại; kết quả dự đoán có thể tải xuống, không báo đã lưu thành công.
- Dự đoán batch thành công chỉ ghi khi người dùng chọn lưu; dùng khóa idempotency để tránh Streamlit rerun sinh bản ghi trùng.

## 5. Môi trường và cách chạy dự kiến

Chọn một bản Python 3 tương thích với toàn bộ thư viện sau khi smoke test, sau đó pin các phiên bản chạy thành công. Thành phần: pandas, numpy, scikit-learn, scipy theo phụ thuộc, matplotlib, seaborn, streamlit, joblib, pytest; SQLite dùng thư viện chuẩn. Không tự ghi phiên bản chưa được kiểm tra là phiên bản đã hỗ trợ.

Ví dụ lệnh trong PowerShell khi mã đã được triển khai:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/audit_data.py --config configs/experiment.json
.\.venv\Scripts\python.exe scripts/make_splits.py --config configs/experiment.json
.\.venv\Scripts\python.exe scripts/train.py --config configs/experiment.json
.\.venv\Scripts\python.exe scripts/evaluate.py --run-id <run_id>
.\.venv\Scripts\python.exe scripts/init_db.py
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe -m pytest
```

Config chỉ đường dẫn CSV nguồn tương đối từ root dự án, seed, schema version, split, CV, metric và output directory. Tách lệnh train và evaluate để tập test không bị mở vô tình trong lúc tuning. `<run_id>` là placeholder phải thay bằng lần chạy thật.

## 6. Quy trình tích hợp

1. TV5 tạo khung package, cấu hình và hợp đồng inference.
2. TV1 bàn giao schema + pipeline; TV2/TV3 cùng thử một batch mẫu.
3. TV4 dựng UI với output theo schema; mock nằm trong chế độ phát triển, tắt khi nghiệm thu.
4. TV2/TV3 bàn giao bundle thật; TV5 kiểm tra tương thích và kích hoạt phiên bản đầu.
5. TV4/TV5 kiểm tra form, CSV, DB và model version hiển thị khớp nhau.
6. Đóng băng schema trước demo; thay đổi sau mốc này phải kiểm tra lại các đường tích hợp bị ảnh hưởng.

Mỗi người sở hữu module rõ ràng; thay đổi file chung do người phụ trách tích hợp rà soát. Không chạy lệnh xóa hoặc ghi đè dữ liệu/model cũ khi chạy lại. Snapshot, runtime DB và artifact lớn có quy tắc bỏ qua Git; vẫn phải có gói bàn giao artifact và hướng dẫn tái tạo.

## 7. Điều kiện hoàn thành

- Có thể chạy từ root dự án dù đường dẫn chứa dấu tiếng Việt và khoảng trắng.
- Một đầu vào đi qua form, CSV và hàm inference cho cùng nhãn/điểm.
- Không có đường dẫn tuyệt đối theo máy một thành viên trong logic chính.
- Có metadata/schema version và lỗi tương thích dễ hiểu.
- Cài môi trường mới, chạy lại được train → evaluate → app theo README.
