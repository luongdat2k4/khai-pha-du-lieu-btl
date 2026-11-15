# Báo cáo Lựa chọn Mô hình dựa trên Kiểm định chéo (Model Selection Report)

> [!IMPORTANT]
> **Khóa giao thức trước khi đánh giá Test:** Báo cáo này được lập và lưu trữ **TRƯỚC KHI** mở tập kiểm thử độc lập (Test Set). Toàn bộ quá trình tìm kiếm siêu tham số và lựa chọn mô hình chỉ sử dụng 80% tập phát triển (Development Set: 5.634 mẫu) thông qua 5-Fold Stratified Cross-Validation.

- **Thời điểm thực hiện (UTC):** `2026-11-15T13:50:43.791092+00:00`
- **Mã băm dữ liệu gốc (CSV SHA-256):** `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`
- **Mã băm Split Manifest:** `465d56b572b3bfbfdfe9157a3d0f1f9f40caa49b01a326471dcd807e35db1c12`
- **Môi trường:** Python `3.11.9` | scikit-learn `1.9.1` | OS `Windows-10-10.0.26200-SP0`

---

## 1. Bảng tổng hợp kết quả 5-Fold Stratified Cross-Validation

| Model | Best params | CV precision 1 | CV recall 1 | CV F1 1 mean ± std | CV ROC-AUC | Fit time (s) | Được chọn? |
|---|---|---|---|---|---|---|---|
| **Baseline Dummy (most_frequent)** | `strategy=most_frequent` | 0.0000 | 0.0000 | **0.0000 ± 0.0000** | 0.5000 | 0.04s | Không |
| **Decision Tree** | `class_weight=balanced`, `criterion=gini`, `max_depth=3`, `min_samples_leaf=1` | 0.5162 | 0.7759 | **0.6194 ± 0.0182** | 0.8151 | 0.33s | Không |
| **Logistic Regression** | `C=0.1`, `class_weight=balanced` | 0.5195 | 0.7973 | **0.6289 ± 0.0213** | 0.8454 | 0.54s |  **CHỌN (Champion)** |
| **Random Forest** | `n_estimators=400`, `min_samples_leaf=3`, `max_features=sqrt`, `max_depth=8`, `class_weight=balanced` | 0.5277 | 0.7886 | **0.6321 ± 0.0139** | 0.8472 | 3.43s | Không |

---

## 2. Quyết định lựa chọn và Biện luận (Model Selection Rationale)

### Mô hình được chọn làm Champion: `Logistic Regression`

> Khoảng cách CV F1 giữa 'random_forest' (0.6321) và 'logistic_regression' (0.6289) là 0.0031 (< 0.005). Theo quy tắc ưu tiên tại Plan 03, áp dụng tiêu chí phụ: 'logistic_regression' có Recall lớp 1 cao hơn (0.7973 so với 0.7886), giúp hạn chế bỏ sót khách hàng rời bỏ (False Negatives). Vì vậy chọn 'logistic_regression' làm Champion.

### Phân tích chi tiết giữa các họ mô hình:

1. **Baseline Dummy (`most_frequent`):**
   - Dự đoán toàn bộ nhãn 0 (No). Precision, Recall và F1 lớp 1 đều bằng 0,0000; Accuracy đạt 73.46% tương ứng với tỷ lệ đa số tự nhiên.
   - Đóng vai trò mốc so sánh tối thiểu (Sanity check) để khẳng định các mô hình học máy thực sự học được tri thức hữu ích từ dữ liệu.

2. **Cây quyết định (Decision Tree):**
   - Cấu hình tốt nhất: `class_weight=balanced`, `criterion=gini`, `max_depth=3`, `min_samples_leaf=1`.
   - Đạt mean CV F1: **0.6194 ± 0.0182**, ROC-AUC: **0.8151**.
   - Ưu điểm: Tốc độ huấn luyện rất nhanh, cây có khả năng giải thích trực quan cao; tuy nhiên độ nhạy và F1 còn hạn chế do cây đơn lẻ dễ nhạy cảm với phương sai phân chia.

3. **Hồi quy Logistic (Logistic Regression):**
   - Cấu hình tốt nhất: `C=0.1`, `class_weight=balanced`.
   - Đạt mean CV F1: **0.6289 ± 0.0213**, ROC-AUC: **0.8454**, Recall: **0.7973**.
   - Nhận xét: Với `class_weight='balanced'`, mô hình phân biệt tốt lớp thiểu số, đạt độ nhạy rất cao và ROC-AUC vượt trội. Dễ dàng diễn giải hệ số Log-Odds đối với nghiệp vụ viễn thông.

4. **Rừng ngẫu nhiên (Random Forest):**
   - Cấu hình tốt nhất: `n_estimators=400`, `min_samples_leaf=3`, `max_features=sqrt`, `max_depth=8`, `class_weight=balanced`.
   - Đạt mean CV F1: **0.6321 ± 0.0139**, ROC-AUC: **0.8472**, Recall: **0.7886**.
   - Nhận xét: Tổ hợp nhiều cây giúp giảm phương sai, cho ROC-AUC rất cao và F1 vượt trội.

---

## 3. Khóa cấu hình và Cam kết kiểm thử độc lập

- Mô hình `logistic_regression` đã được fit lại trên toàn bộ 5.634 mẫu của tập phát triển với siêu tham số đã khóa: `C=0.1`, `class_weight=balanced`.
- Ngưỡng phân loại được cố định ở `threshold = 0.5`.
- Pipeline hoàn chỉnh bao gồm tiền xử lý (`preprocessor`) và bộ phân loại (`classifier`) được lưu thành artifact độc lập.
- **Cam kết:** Không điều chỉnh lại siêu tham số dựa trên kết quả của tập kiểm thử độc lập (Test Set). Bất kỳ bảng đánh giá Test Set nào sau đây chỉ mang tính chất kiểm chứng khách quan.
