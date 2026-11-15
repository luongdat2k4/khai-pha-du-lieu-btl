# Cấu trúc thư mục dự án bài tập lớn

**Đề tài:** Dự đoán khách hàng rời bỏ dịch vụ viễn thông.

**Công nghệ dự kiến:** Python, pandas, scikit-learn, Streamlit và SQLite.

Tài liệu này mô tả cấu trúc đề xuất để triển khai dự án theo [yêu cầu ban đầu](note.md) và [bộ kế hoạch chi tiết](plans/README.md). Cây bên dưới là thiết kế đích, không có nghĩa tất cả thư mục và file đã được tạo. Hiện workspace đã có `designe-note/`, `doc-tmp/`, `telco-customer-churn-on-icp4d/` và `cuop/`.

## 1. Cây thư mục tổng thể

```text
hoc-lai/
│
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
│
├── configs/
│   ├── experiment.json
│   └── retraining.json
│
├── src/
│   └── telco_churn/
│       ├── __init__.py
│       ├── schema.py
│       ├── data.py
│       ├── preprocessing.py
│       ├── training.py
│       ├── evaluation.py
│       ├── inference.py
│       ├── storage.py
│       ├── registry.py
│       └── retraining.py
│
├── ui/
│   ├── overview.py
│   ├── single_prediction.py
│   ├── batch_prediction.py
│   ├── history.py
│   ├── model_results.py
│   └── model_management.py
│
├── scripts/
│   ├── audit_data.py
│   ├── make_splits.py
│   ├── train.py
│   ├── evaluate.py
│   ├── init_db.py
│   └── retrain.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_decision_tree.ipynb
│   ├── 03_logistic_regression.ipynb
│   ├── 04_random_forest.ipynb
│   └── 05_model_comparison.ipynb
│
├── data/
│   ├── splits/
│   │   └── split_manifest.csv
│   ├── samples/
│   ├── snapshots/
│   └── runtime/
│       └── churn.db
│
├── artifacts/
│   ├── active_model.json
│   └── models/
│       └── <model_version>/
│           ├── pipeline.joblib
│           ├── metadata.json
│           ├── cv_summary.json
│           └── test_metrics.json
│
├── reports/
│   ├── data_audit.md
│   ├── model_selection.md
│   ├── final_evaluation.md
│   ├── retraining_demo.md
│   ├── test_checklist.md
│   ├── experiments/
│   └── figures/
│       ├── eda/
│       └── evaluation/
│
├── docs/
│   ├── problem-definition.md
│   ├── theory.md
│   ├── data_dictionary.md
│   └── architecture.md
│
├── tests/
│   ├── test_data_validation.py
│   ├── test_data_splitting.py
│   ├── test_inference.py
│   ├── test_storage.py
│   └── test_retraining.py
│
├── deliverables/
│   ├── report/
│   ├── slides/
│   └── demo/
│
├── designe-note/                         # Đã có
│   ├── note.md
│   ├── cau-truc-thu-muc.md
│   └── plans/
│
├── doc-tmp/                              # Đã có
│   └── DMDWv1.2.docx.md
│
├── telco-customer-churn-on-icp4d/         # Đã có; chỉ liệt kê phần liên quan
│   ├── data/
│   │   └── Telco-Customer-Churn.csv
│   ├── notebooks/
│   ├── examples/
│   ├── flaskapp/
│   ├── doc/
│   ├── README.md
│   └── LICENSE
│
└── cuop/                                # Đã có: tài liệu báo cáo tham khảo
```

`<model_version>` là ký hiệu cho thư mục phiên bản thực tế, ví dụ `model_001`; không tạo tên thư mục chứa dấu `<` hoặc `>` trên Windows. Cây đã lược bỏ một số file của nguồn IBM và các thư mục công cụ như `.git/`, `.venv/`, `__pycache__/`.

## 2. Các file ở thư mục gốc

| File | Chức năng |
|---|---|
| `README.md` | Điểm bắt đầu cho người chạy dự án: giới thiệu, cài môi trường, vị trí dữ liệu, lệnh huấn luyện, đánh giá và mở ứng dụng. |
| `requirements.txt` | Danh sách thư viện với phiên bản đã kiểm tra chạy thành công để các thành viên sử dụng môi trường nhất quán. |
| `.gitignore` | Quy định các file không đưa vào Git, như môi trường ảo, cache, DB chạy thử và file tạm. |
| `app.py` | Điểm khởi chạy Streamlit, tổ chức điều hướng và gọi các màn hình trong `ui/`. |

Không đặt toàn bộ tiền xử lý, huấn luyện và giao diện trong `app.py`; các chức năng này có module riêng để kiểm tra và dùng lại.

## 3. `configs/` — Cấu hình chạy

Tập trung các lựa chọn có thể thay đổi mà không phải sửa logic mã nguồn.

- `experiment.json`: đường dẫn dữ liệu, seed, tỷ lệ chia 80/20, CV 5 fold, metric, không gian tham số và nơi lưu kết quả.
- `retraining.json`: lịch chạy, điều kiện dữ liệu đủ nhãn, tiêu chí so sánh candidate/champion và cấu hình chế độ demo.

Sử dụng đường dẫn tương đối tính từ root dự án. Mỗi lần huấn luyện phải lưu lại cấu hình thực tế đã dùng để tái lập kết quả.

## 4. `src/telco_churn/` — Mã xử lý chính

Đây là thư viện nội bộ dùng chung cho script, notebook và giao diện. Quy tắc tiền xử lý và dự đoán có một bản triển khai thống nhất tại đây.

| Module | Trách nhiệm |
|---|---|
| `__init__.py` | Khai báo package Python và các thành phần công khai nếu cần. |
| `schema.py` | Định nghĩa 19 đặc trưng, ID, nhãn, kiểu dữ liệu, miền danh mục và kiểm tra đầu vào. |
| `data.py` | Đọc CSV, chuyển kiểu theo quy tắc định trước, kiểm tra chất lượng và tạo phân chia dữ liệu. |
| `preprocessing.py` | Tạo bộ điền thiếu, one-hot encoding và chuẩn hóa phù hợp từng mô hình. |
| `training.py` | Xây pipeline, huấn luyện Decision Tree, Logistic Regression, Random Forest và tìm tham số bằng CV. |
| `evaluation.py` | Tính precision, recall, F1, ROC-AUC; tạo confusion matrix, biểu đồ và bảng so sánh. |
| `inference.py` | Tải và sử dụng pipeline đã fit để dự đoán một hoặc nhiều hồ sơ theo cùng hợp đồng đầu vào/đầu ra. |
| `storage.py` | Thao tác SQLite: lưu snapshot hồ sơ, dự đoán, batch và lịch sử nhãn thực tế. |
| `registry.py` | Đăng ký phiên bản mô hình, xác định phiên bản active, kiểm tra tương thích và rollback. |
| `retraining.py` | Kiểm tra điều kiện dữ liệu, đóng băng snapshot, huấn luyện candidate và phối hợp so sánh với champion. |

`inference.py` không tự huấn luyện mô hình. Preprocessor đã fit phải được đóng gói cùng classifier để ứng dụng không tự tính lại median, danh mục mã hóa hoặc thống kê chuẩn hóa.

## 5. `ui/` — Giao diện Streamlit

Tách màn hình khỏi mã xử lý để dễ phân công và chỉnh sửa.

| File | Màn hình/chức năng |
|---|---|
| `overview.py` | Tổng quan đề tài, mục tiêu, dữ liệu và phiên bản mô hình. |
| `single_prediction.py` | Form nhập một khách, kiểm tra hợp lệ, hiển thị nhãn và điểm nguy cơ. |
| `batch_prediction.py` | Upload CSV, xem lỗi theo dòng, dự đoán batch và tải kết quả. |
| `history.py` | Tra cứu hồ sơ/dự đoán đã lưu, nhập và xem lịch sử nhãn thực tế. |
| `model_results.py` | Hiển thị kết quả CV, quyết định chọn mô hình và đánh giá test đã lưu. |
| `model_management.py` | Xem điều kiện retrain, lịch sử lần chạy và trạng thái các phiên bản mô hình. |

Các file này được `app.py` gọi để hiển thị; chỉ tạo thư mục `ui/` không tự làm Streamlit nhận diện chúng thành các trang. Giao diện gọi hàm từ `src/telco_churn/`, không viết một bản preprocessing khác hoặc huấn luyện lại mỗi lần trang rerun.

## 6. `scripts/` — Các điểm chạy bằng dòng lệnh

Mỗi script nhận tham số/cấu hình và gọi module trong `src/`. Script không chứa bản sao logic của thư viện.

| Script | Khi sử dụng |
|---|---|
| `audit_data.py` | Khi kiểm tra chất lượng dữ liệu đầu vào và tạo báo cáo audit. |
| `make_splits.py` | Khi tạo split manifest dùng chung cho thực nghiệm. |
| `train.py` | Khi chạy baseline, ba mô hình và tìm tham số trên phần phát triển. |
| `evaluate.py` | Khi đã khóa lựa chọn mô hình và cần đánh giá trên test. |
| `init_db.py` | Khi khởi tạo hoặc chuẩn bị schema cơ sở dữ liệu ứng dụng. |
| `retrain.py` | Khi chạy chu kỳ cập nhật thủ công, theo lịch hoặc ở chế độ demo/dry-run. |

Tách `train.py` và `evaluate.py` để tránh vô tình xem test trong lúc lựa chọn tham số. Hướng dẫn thiết lập import package và lệnh chạy từ root sẽ nằm trong README khi triển khai.

## 7. `notebooks/` — Khám phá và giải thích

Notebook phục vụ xem dữ liệu, thử nghiệm có kiểm soát và trình bày lý do lựa chọn phương pháp:

- `01_data_exploration.ipynb`: audit, phân bố nhãn và EDA trên phần phát triển.
- `02_decision_tree.ipynb`: minh họa cây, tham số và kết quả Decision Tree.
- `03_logistic_regression.ipynb`: Logistic Regression, hệ số và ảnh hưởng của chuẩn hóa.
- `04_random_forest.ipynb`: Random Forest và diễn giải độ quan trọng đặc trưng.
- `05_model_comparison.ipynb`: đọc kết quả đã lưu, so sánh và trực quan hóa.

Kết quả chính phải tái tạo được bằng script/config. Không để notebook trở thành nơi duy nhất chứa logic cần cho ứng dụng; không tự chia dữ liệu khác nhau trong notebook của từng thành viên.

## 8. `data/` — Dữ liệu phục vụ quy trình và dữ liệu phát sinh

| Thư mục | Nội dung | Quy tắc |
|---|---|---|
| `splits/` | Manifest ghi ID thuộc tập phát triển/test, kèm thông tin phiên bản hoặc tham chiếu metadata. | Dùng chung cho mọi mô hình; không chia lại tùy ý để chọn kết quả đẹp. |
| `samples/` | CSV mẫu upload, mẫu lỗi và hồ sơ minh họa. | Ghi rõ dữ liệu tổng hợp/mô phỏng; không nhầm với test độc lập. |
| `snapshots/` | Dữ liệu và nhãn hiệu lực đã chốt cho từng lần huấn luyện lại. | Có nguồn, thời điểm, hash và phân chia; giữ bất biến theo lần chạy. |
| `runtime/` | SQLite và dữ liệu phát sinh khi dùng ứng dụng. | Nhãn thật tách khỏi dự đoán; DB demo có thể tách riêng DB kết quả chính. |

CSV gốc tiếp tục ở `telco-customer-churn-on-icp4d/data/`, chưa cần sao chép sang `data/raw/`. Giữ nguyên nguồn gốc và hash để tránh hai bản dữ liệu khác nhau bị dùng lẫn.

Không lưu một bảng đã chuẩn hóa trên toàn bộ dữ liệu rồi dùng cho CV; thống kê tiền xử lý phải được fit trong từng training fold. Tập test gốc không được nhập vào kho dữ liệu mới để huấn luyện lại.

## 9. `artifacts/` — Mô hình và metadata dùng khi dự đoán

Đây là đầu ra có thể tải bằng chương trình, được tạo sau huấn luyện.

- `active_model.json`: con trỏ tới phiên bản đang phục vụ ứng dụng.
- `models/<model_version>/pipeline.joblib`: pipeline đã fit gồm tiền xử lý và classifier.
- `metadata.json`: thuật toán, tham số, schema, mapping nhãn, threshold, seed, hash dữ liệu và phiên bản thư viện.
- `cv_summary.json`: kết quả CV dùng lựa chọn mô hình.
- `test_metrics.json`: đánh giá test cuối của thực nghiệm tương ứng.

Mỗi phiên bản được lưu riêng, không ghi đè bản đang chạy. `test_metrics.json` có sau bước đánh giá cuối; candidate retrain chưa đánh giá test gốc không cần có file này. Kết quả comparison của retrain được lưu theo run riêng và tham chiếu trong metadata, không giả thành kết quả test gốc.

## 10. `reports/` — Kết quả thực nghiệm và bằng chứng

| Thành phần | Nội dung |
|---|---|
| `data_audit.md` | Chất lượng dữ liệu, số thiếu/trùng/lỗi, quy tắc và số mẫu trước/sau xử lý. |
| `model_selection.md` | Quyết định chọn mô hình dựa trên CV, được ghi trước khi mở test. |
| `final_evaluation.md` | Kết quả test, phân tích FN/FP và các giới hạn. |
| `retraining_demo.md` | Gate, dữ liệu thật/mô phỏng, so sánh candidate/champion và quyết định. |
| `test_checklist.md` | Các kiểm tra đã chạy, pass/fail và vấn đề còn lại. |
| `experiments/` | Config thực tế, log, bảng kết quả từng fold/cấu hình và run_id. |
| `figures/eda/` | Biểu đồ khám phá dữ liệu. |
| `figures/evaluation/` | Confusion matrix, ROC, PR và hình so sánh mô hình. |

`reports/` chứa kết quả và nhận xét từ các lần chạy. Các số liệu phải có nguồn thực nghiệm, không điền trước số giả để hoàn thiện giao diện hay báo cáo.

## 11. `docs/` — Tài liệu đặc tả và phương pháp

- `problem-definition.md`: đối tượng, đầu vào, nhãn, đầu ra, người dùng và giới hạn bài toán.
- `theory.md`: KDD, tiền xử lý, cây quyết định, LR, RF, metric và CV; phân biệt nội dung giáo trình với phần bổ sung.
- `data_dictionary.md`: ý nghĩa, kiểu, miền giá trị, đơn vị và quy tắc thiếu của từng cột.
- `architecture.md`: kiến trúc module, hợp đồng tích hợp, luồng dữ liệu và schema lưu trữ.

`docs/` mô tả hệ thống và phương pháp đã chốt; `designe-note/plans/` mô tả công việc cần thực hiện. Khi thiết kế thay đổi, cập nhật tài liệu tương ứng để người đọc biết cách hệ thống thực sự hoạt động.

## 12. `tests/` — Kiểm thử các phần có ảnh hưởng đến tính đúng

| File | Nội dung kiểm tra chính |
|---|---|
| `test_data_validation.py` | Thiếu cột, sai kiểu, danh mục lạ, dữ liệu thiếu và bất nhất dịch vụ. |
| `test_data_splitting.py` | ID không giao giữa các tập, nhóm khách không rò rỉ, mapping nhãn đúng. |
| `test_inference.py` | Đơn/batch nhất quán, đúng xác suất lớp 1, không fit lại khi dự đoán. |
| `test_storage.py` | Lưu/đọc snapshot, transaction, chống lưu lặp và lịch sử nhãn. |
| `test_retraining.py` | Gate thiếu dữ liệu, partition, giữ/thay mô hình và rollback. |

Kiểm thử dùng DB và thư mục tạm, không xóa hoặc ghi đè dữ liệu demo chuẩn. Checklist thao tác giao diện và bằng chứng kiểm thử nằm trong `reports/`.

## 13. `deliverables/` — Gói sản phẩm nộp

- `report/`: báo cáo hoàn chỉnh dạng Word/PDF theo yêu cầu của giảng viên.
- `slides/`: slide bảo vệ.
- `demo/`: kịch bản, ảnh, video dự phòng và hướng dẫn trình diễn.

Nội dung ở đây được tổng hợp từ kết quả đã kiểm chứng trong `reports/` và tài liệu trong `docs/`. Metric, threshold và model version trên slide/báo cáo phải khớp mô hình chạy trong ứng dụng.

## 14. Các thư mục đã có

| Thư mục | Vai trò và cách sử dụng |
|---|---|
| `designe-note/` | Yêu cầu gốc, tài liệu cấu trúc này và bộ kế hoạch. Bắt đầu đọc từ `plans/README.md`. |
| `doc-tmp/` | Tài liệu môn học, trong đó có bản Markdown giáo trình. Dùng làm cơ sở lý thuyết; lưu ý công thức/hình có thể bị lỗi OCR. |
| `telco-customer-churn-on-icp4d/` | Nguồn dữ liệu và ví dụ IBM đã có. Giữ CSV gốc; đọc license và ghi công phần thực sự tái sử dụng. Mã mới không cần phụ thuộc luồng Flask/IBM Cloud trong ví dụ. |
| `cuop/` | Hiện có PDF báo cáo tham khảo. Nếu sử dụng thì kiểm tra và dẫn nguồn; kết quả của dự án phải đến từ thực nghiệm của nhóm. |

## 15. Luồng phối hợp giữa các thư mục

### Thực nghiệm ban đầu

```text
CSV nguồn + configs/
        ↓
scripts/ gọi src/telco_churn/
        ↓
data/splits/ → pipeline + CV → chọn mô hình → đánh giá test
        ↓
artifacts/ + reports/
```

### Dự đoán và thu thập dữ liệu mới

```text
app.py → ui/ → validation và inference trong src/
                         ↑
                 artifacts/ đang active
                         ↓
              kết quả hiển thị / tải CSV
                         ↓ khi chọn lưu
                 data/runtime/churn.db
```

### Huấn luyện lại

```text
DB có nhãn thực tế → kiểm tra gate → data/snapshots/
                                      ↓
                             train candidate
                                      ↓
                       so sánh với champion
                                      ↓
                 lưu reports/ và artifacts/ phiên bản mới
                                      ↓ nếu đạt tiêu chí
                         cập nhật active_model.json
```

## 16. Phân công quản lý theo nhóm 5 người

| Thành viên | Phần phụ trách chính |
|---|---|
| TV1 | `schema.py`, `data.py`, `preprocessing.py`, EDA, data dictionary, audit và split. |
| TV2 | Decision Tree, Logistic Regression, notebook tương ứng và phần lý thuyết thuật toán. |
| TV3 | Random Forest, `evaluation.py`, bảng CV/test, chọn mô hình và tiêu chí so sánh cập nhật. |
| TV4 | `app.py`, `ui/`, nhập đơn/CSV, kết quả và giao diện cập nhật nhãn. |
| TV5 | Storage, registry, retraining, script tích hợp và README chạy dự án. |

`training.py`, config và schema là phần dùng chung: thống nhất giao diện trước khi chia việc; tránh để nhiều người sửa đồng thời mà không phối hợp. Mỗi người viết tài liệu, kiểm thử và phần báo cáo cho module mình phụ trách.

## 17. Quy tắc tổ chức và thứ tự tạo

1. Tạo trước khung `src/`, `configs/`, `scripts/`, `ui/`, `tests/` và README; chốt schema/hợp đồng dự đoán.
2. Thực hiện audit, split, EDA và pipeline; các thư mục kết quả được tạo khi có đầu ra thật.
3. Huấn luyện, so sánh và đánh giá để sinh `artifacts/` và `reports/`.
4. Ghép ứng dụng với pipeline thật, bổ sung DB và quy trình cập nhật.
5. Hoàn thiện `docs/` và gói `deliverables/` theo phiên bản cuối.

Đưa mã nguồn, config, tài liệu, test và dữ liệu mẫu nhỏ phù hợp vào Git. Môi trường ảo, cache, DB runtime, snapshot chứa dữ liệu khách mới và artifact lớn cần chính sách lưu trữ riêng; nếu không đưa vào Git thì vẫn phải có hướng dẫn tái tạo hoặc gói bàn giao. Giữ lại những artifact nhỏ và kết quả cần thiết để người chấm kiểm tra theo quy định nộp bài.

Không cần tạo file rỗng cho toàn bộ cây ngay từ đầu. Có thể triển khai dần theo [tiến độ và phân công](plans/07-tien-do-va-phan-cong.md); chi tiết hợp đồng module nằm trong [Plan 04 — Kiến trúc và tích hợp](plans/04-kien-truc-va-tich-hop.md).
