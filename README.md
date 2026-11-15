# Dự Đoán Khách Hàng Rời Bỏ Dịch Vụ Viễn Thông (Telco Customer Churn Prediction)

Hệ thống ứng dụng khai phá dữ liệu hỗ trợ dự báo nguy cơ rời bỏ của khách hàng thuê bao viễn thông, phân nhóm ưu tiên chăm sóc và quản lý chu trình cập nhật mô hình kiểm chuẩn có lưu vết dữ liệu.

Được thiết kế và triển khai theo chuẩn giao ước kiến trúc tại [Plan 04 — Kiến trúc mã nguồn và hợp đồng tích hợp](designe-note/plans/04-kien-truc-va-tich-hop.md).

---

## 1. Thông Tin Đề Tài & Phân Công Nhóm

- **Đề tài:** Khai phá dữ liệu viễn thông nhằm dự đoán khách hàng rời bỏ dịch vụ.
- **Bộ dữ liệu:** IBM Telco Customer Churn (7.043 mẫu, 21 cột dữ liệu thô, 19 đặc trưng mô hình hóa).
- **Mô hình nghiên cứu:** Dummy Baseline, Decision Tree, Logistic Regression, Random Forest.
- **Champion Model:** **Logistic Regression** (CV F1 = 0.6334, CV Recall = 0.8012, Test F1 = 0.6277).
- **Công nghệ chính:** Python 3.11, scikit-learn, pandas, Streamlit, SQLite3.

### Phân Công Thành Viên
- **TV1:** Dữ liệu, kiểm toán, tiền xử lý và thẩm định schema (`src/telco_churn/schema.py`, `data.py`).
- **TV2:** Huấn luyện Decision Tree, Logistic Regression và tìm kiếm siêu tham số (`src/telco_churn/training.py`).
- **TV3:** Huấn luyện Random Forest, so sánh mô hình, tính toán độ đo (`src/telco_churn/evaluation.py`).
- **TV4:** Xây dựng giao diện ứng dụng web Streamlit đa màn hình (`ui/`, `app.py`).
- **TV5:** Kiến trúc hệ thống, hợp đồng tích hợp, lưu trữ SQLite, kiểm chuẩn và cập nhật mô hình (`src/telco_churn/storage.py`, `registry.py`, `retraining.py`).

---

## 2. Hướng Dẫn Cài Đặt Môi Trường

Khuyến nghị sử dụng Python 3.11 trong môi trường ảo:

```powershell
# Khởi tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo (PowerShell trên Windows)
.\.venv\Scripts\Activate.ps1

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

---

## 3. Quy Trình Chạy Tái Lập Toàn Bộ Hệ Thống (End-to-End Pipeline)

Thực hiện tuần tự các lệnh sau từ thư mục gốc của dự án:

```powershell
# 1. Kiểm toán chất lượng và tính toàn vẹn của tập dữ liệu gốc
python scripts/audit_data.py --config configs/experiment.json

# 2. Tạo tập chia mẫu phân tầng đã khóa (80% Dev / 20% Test)
python scripts/make_splits.py --config configs/experiment.json

# 3. Huấn luyện 5-Fold Stratified CV, tìm siêu tham số và đóng gói Champion Model
python scripts/train.py --config configs/experiment.json

# 4. Mở tập kiểm thử Test Set 20% độc lập và xuất báo cáo nghiệm thu
python scripts/evaluate.py --run-id model_001

# 5. Khởi tạo cơ sở dữ liệu SQLite runtime và đăng ký mô hình active
python scripts/init_db.py

# 6. Kiểm tra cổng cập nhật mô hình và thử nghiệm dry-run
python scripts/retrain.py --dry-run

# 7. Khởi chạy giao diện ứng dụng web Streamlit
streamlit run app.py

# 8. Thực thi toàn bộ bộ kiểm thử đơn vị tự động
pytest
```

---

## 4. Cấu Trúc Thư Mục Dự Án

```text
hoc-lai/
├── app.py                         # Điểm khởi chạy ứng dụng web Streamlit
├── requirements.txt               # Các thư viện phụ thuộc
├── README.md                      # Hướng dẫn dự án
│
├── configs/
│   ├── experiment.json            # Cấu hình chia mẫu, metric và mô hình
│   └── retraining.json            # Cấu hình cổng kiểm soát cập nhật
│
├── src/telco_churn/               # Package thư viện lõi
│   ├── schema.py                  # Định nghĩa và thẩm định 19 đặc trưng
│   ├── data.py                    # Nạp dữ liệu, kiểm toán, băm SHA-256
│   ├── preprocessing.py           # Pipeline tiền xử lý scikit-learn
│   ├── training.py                # Huấn luyện CV, tìm tham số, chọn champion
│   ├── evaluation.py              # Đánh giá chỉ số, vẽ biểu đồ ROC/PR/CM
│   ├── inference.py               # Nạp bundle, dự đoán, tính mức ưu tiên
│   ├── storage.py                 # Lưu trữ SQLite, transaction, snapshot, audit
│   ├── registry.py                # Quản lý active model, rollback
│   └── retraining.py              # Cổng an toàn, candidate training, so sánh
│
├── ui/                            # Phân hệ giao diện người dùng Streamlit
│   ├── overview.py                # Tổng quan bài toán và mức độ rủi ro
│   ├── single_prediction.py       # Nhập liệu dự đoán cho 1 khách hàng
│   ├── batch_prediction.py        # Dự đoán danh sách theo lô qua tệp CSV
│   ├── history.py                 # Xem lịch sử dự đoán và cập nhật nhãn thực tế
│   ├── model_results.py           # Bảng kết quả CV, test metrics và đồ thị
│   └── model_management.py        # Kiểm tra gate, chạy dry-run/demo, hoàn tác
│
├── scripts/                       # Các kịch bản CLI
│   ├── audit_data.py              # Kiểm toán dữ liệu
│   ├── make_splits.py             # Phân chia mẫu
│   ├── train.py                   # Huấn luyện mô hình
│   ├── evaluate.py                # Đánh giá tập test
│   ├── init_db.py                 # Khởi tạo SQLite
│   └── retrain.py                 # Huấn luyện lại / kiểm tra gate
│
├── data/
│   ├── splits/split_manifest.csv  # Bảng khóa phân chia 80% Dev và 20% Test
│   ├── samples/                   # CSV mẫu thử nghiệm
│   └── runtime/churn.db           # SQLite database
│
├── artifacts/
│   ├── active_model.json          # Con trỏ phiên bản active
│   └── models/model_001/          # Pipeline, metadata, cv_summary, test_metrics
│
├── reports/                       # Báo cáo kết quả và minh chứng
│   ├── data_audit.md              # Báo cáo kiểm toán dữ liệu
│   ├── model_selection.md         # Báo cáo lựa chọn mô hình qua CV
│   ├── final_evaluation.md        # Báo cáo kiểm thử độc lập
│   └── figures/                   # Biểu đồ EDA và đánh giá
│
├── docs/                          # Tài liệu kiến trúc và lý thuyết
└── tests/                         # Bộ kiểm thử đơn vị pytest
```

---

## 5. Kiểm Thử Hệ Thống (Automated Testing)

Toàn bộ các module tuân thủ quy tắc hợp đồng Plan 04 được bảo vệ bởi bộ unit tests toàn diện:
- `tests/test_data_validation.py`: Kiểm tra schema, kiểu dữ liệu, các ràng buộc phụ thuộc dịch vụ.
- `tests/test_data_splitting.py`: Kiểm tra tỷ lệ 80/20, tính phân tầng của nhãn `Churn`, chống rò rỉ mẫu giữa các fold.
- `tests/test_inference.py`: Kiểm tra nạp model bundle, tính đơn điệu của ngưỡng threshold, định dạng 9 cột đầu ra, phân tầng ưu tiên (Thấp/TB/Cao).
- `tests/test_storage.py`: Kiểm tra giao dịch SQLite, tính bất biến của snapshot, cơ chế chống ghi trùng lặp (`idempotency_key`), lưu vết lịch sử nhãn thực tế (`supersedes_event_id`).
- `tests/test_retraining.py`: Kiểm tra logic cổng điều kiện (Gate checks), cơ chế so sánh Candidate vs Champion trên cùng holdout set, chế độ dry-run và thao tác rollback.

Chạy kiểm thử:
```powershell
pytest
```
