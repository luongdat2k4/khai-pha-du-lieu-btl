# Báo cáo kiểm kê và đánh giá chất lượng dữ liệu (Data Audit Report)

- **Tệp dữ liệu nguồn:** `telco-customer-churn-on-icp4d/data/Telco-Customer-Churn.csv`
- **Mã băm SHA-256:** `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`
- **Kích thước dữ liệu gốc:** 7,043 dòng, 21 cột

---

## 1. Phân bố nhãn mục tiêu (Target Distribution)

| Nhãn Churn | Số lượng mẫu | Tỷ lệ (%) |
|---|---|---|
| `No` | 5,174 | 73.46% |
| `Yes` | 1,869 | 26.54% |

> [!NOTE]
> Tỷ lệ rời bỏ là khoảng 26.54% (1.869 mẫu Yes / 5.174 mẫu No). Dữ liệu có sự mất cân bằng giữa hai lớp (tỷ lệ xấp xỉ 1:2.77). Do đó, chỉ số Accuracy thông thường không phản ánh toàn diện hiệu quả mô hình; cần sử dụng thêm Precision, Recall, F1-score và ROC-AUC.

---

## 2. Kiểm kê các cột và kiểu dữ liệu (Schema & Missingness)

| Cột | Dtype | Giá trị thiếu (NaN) | Chuỗi rỗng / Blank | Số giá trị phân biệt | Ví dụ giá trị |
|---|---|---|---|---|---|
| `customerID` | `str` | 0 | 0 | 7043 | 7590-VHVEG, 5575-GNVDE, 3668-QPYBK |
| `gender` | `str` | 0 | 0 | 2 | Female, Male |
| `SeniorCitizen` | `int64` | 0 | 0 | 2 | 0, 1 |
| `Partner` | `str` | 0 | 0 | 2 | Yes, No |
| `Dependents` | `str` | 0 | 0 | 2 | No, Yes |
| `tenure` | `int64` | 0 | 0 | 73 | 1, 34, 2 |
| `PhoneService` | `str` | 0 | 0 | 2 | No, Yes |
| `MultipleLines` | `str` | 0 | 0 | 3 | No phone service, No, Yes |
| `InternetService` | `str` | 0 | 0 | 3 | DSL, Fiber optic, No |
| `OnlineSecurity` | `str` | 0 | 0 | 3 | No, Yes, No internet service |
| `OnlineBackup` | `str` | 0 | 0 | 3 | Yes, No, No internet service |
| `DeviceProtection` | `str` | 0 | 0 | 3 | No, Yes, No internet service |
| `TechSupport` | `str` | 0 | 0 | 3 | No, Yes, No internet service |
| `StreamingTV` | `str` | 0 | 0 | 3 | No, Yes, No internet service |
| `StreamingMovies` | `str` | 0 | 0 | 3 | No, Yes, No internet service |
| `Contract` | `str` | 0 | 0 | 3 | Month-to-month, One year, Two year |
| `PaperlessBilling` | `str` | 0 | 0 | 2 | Yes, No |
| `PaymentMethod` | `str` | 0 | 0 | 4 | Electronic check, Mailed check, Bank transfer (automatic) |
| `MonthlyCharges` | `float64` | 0 | 0 | 1585 | 29.85, 56.95, 53.85 |
| `TotalCharges` | `str` | 0 | 11 | 6531 | 29.85, 1889.5, 108.15 |
| `Churn` | `str` | 0 | 0 | 2 | No, Yes |

---

## 3. Phân tích chi tiết giá trị trống ở TotalCharges

- **Số lượng bản ghi trống:** 11 dòng.
- **Đặc điểm `tenure`:** {0: 11}
- **Đặc điểm nhãn `Churn`:** {'No': 11}

> [!IMPORTANT]
> Toàn bộ 11 dòng có `TotalCharges` trống đều có `tenure = 0` và `Churn = No`. Đây là các khách hàng mới gia nhập hệ thống trong kỳ tính cước hiện tại, chưa hoàn thành chu kỳ thanh toán tháng đầu tiên.
> **Quy tắc làm sạch:** Chuyển chuỗi trống thành `NaN`. Trong pipeline huấn luyện, sử dụng `SimpleImputer(strategy='median')` được fit trên từng training fold để điền giá trị thiếu một cách khách quan, bảo toàn toàn bộ 7.043 mẫu dữ liệu.

---

## 4. Kiểm tra trùng lặp và xung đột dữ liệu

- **Trùng lặp toàn bộ dòng (Exact duplicate rows):** 0 dòng.
- **Trùng lặp mã khách hàng (`customerID`):** 0 dòng. Mỗi mã khách hàng là duy nhất.
- **Trùng lặp 19 đặc trưng với mã ID khác nhau:** 40 dòng. Đây là các khách hàng độc lập có cùng hồ sơ gói cước; không được tự động xóa bỏ.
- **Xung đột nghiệp vụ điện thoại (`PhoneService='No'` & `MultipleLines='Yes'`):** 0 trường hợp.
- **Xung đột nghiệp vụ Internet (`InternetService='No'` & gói phụ `Yes/No`):** 0 trường hợp.

> [!TIP]
> Bộ dữ liệu gốc có độ nhất quán nghiệp vụ cao, không có lỗi xung đột logic chéo giữa các dịch vụ. Tất cả các giá trị 'No internet service' và 'No phone service' được giữ nguyên như các trạng thái hợp lệ.

---

## 5. Thống kê mô tả các thuộc tính số và ngoại lai

| Thuộc tính | Min | Max | Trung bình | Độ lệch chuẩn | Trung vị (Median) | Q1 | Q3 | IQR | Số ngoại lai (IQR) |
|---|---|---|---|---|---|---|---|---|---|
| `tenure` | 0.0 | 72.0 | 32.37 | 24.56 | 29.0 | 9.0 | 55.0 | 46.00 | 0 |
| `MonthlyCharges` | 18.25 | 118.75 | 64.76 | 30.09 | 70.35 | 35.5 | 89.85 | 54.35 | 0 |
| `TotalCharges` | 18.8 | 8684.8 | 2283.3 | 2266.77 | 1397.475 | 401.45 | 3794.7375 | 3393.29 | 0 |

> [!NOTE]
> - `tenure` dao động từ 0 đến 72 tháng, không có giá trị âm hay ngoại lai kỹ thuật.
> - `MonthlyCharges` dao động từ 18.25 đến 118.75 USD, phân bố hợp lý theo các gói cước.
> - `TotalCharges` có đuôi kéo dài tới 8.684.80 USD nhưng phản ánh đúng tổng chi phí tích lũy theo thời gian của khách hàng lâu năm (không có giá trị âm).

---

## 6. Kết luận và quy tắc làm sạch chuẩn hóa

1. **Bảo toàn số mẫu:** Giữ nguyên toàn bộ 7.043 bản ghi gốc.
2. **Xử lý chuỗi và khoảng trắng:** Trim toàn bộ khoảng trắng ở các cột danh mục.
3. **Chuyển kiểu `TotalCharges`:** Ép kiểu từ chuỗi sang `float64`, chuỗi trống thành `np.nan`.
4. **Chuẩn hóa `SeniorCitizen`:** Giữ dạng chuỗi `'0'` / `'1'` để mã hóa One-Hot nhất quán.
5. **Mã hóa nhãn `Churn`:** Chuyển `No` $\rightarrow 0$, `Yes` $\rightarrow 1$.
6. **Không rò rỉ thông tin:** Toàn bộ phép điền thiếu (`SimpleImputer`) và chuẩn hóa (`StandardScaler`) chỉ fit trên từng fold huấn luyện.
