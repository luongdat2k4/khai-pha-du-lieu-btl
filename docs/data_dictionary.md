# Từ điển dữ liệu (Data Dictionary) — Telco Customer Churn

Tài liệu này đặc tả chi tiết 21 thuộc tính trong bộ dữ liệu Telco Customer Churn, bao gồm vai trò, kiểu dữ liệu, miền giá trị hợp lệ, quy tắc tiền xử lý và các ràng buộc toàn vẹn dữ liệu theo thiết kế hệ thống.

---

## 1. Tổng quan bộ dữ liệu

- **Nguồn:** IBM Business Analytics Telco Customer Churn dataset.
- **Tệp dữ liệu:** `telco-customer-churn-on-icp4d/data/Telco-Customer-Churn.csv`.
- **Quy mô gốc:** 7.043 bản ghi và 21 cột.
- **Phân loại cột:**
  - 01 cột định danh: `customerID` (loại khỏi ma trận đặc trưng $X$).
  - 01 cột nhãn mục tiêu: `Churn` (biến đổi thành nhị phân 0/1).
  - 19 cột đặc trưng đầu vào: 3 biến số liên tục/rời rạc, 16 biến định tính (danh mục/nhị phân).

---

## 2. Bảng đặc tả 21 thuộc tính

| STT | Tên thuộc tính | Nhóm | Kiểu dữ liệu | Miền giá trị hợp lệ | Vai trò | Quy tắc xử lý thiếu / ngoại lệ |
|---|---|---|---|---|---|---|
| 1 | `customerID` | Định danh | Chuỗi (string) | Chuỗi định dạng `\d{4}-[A-Z]{5}` (VD: `7590-VHVEG`) | Định danh bản ghi | Không dùng trong $X$. Dùng để quản lý, truy vết và liên kết kết quả dự đoán với khách hàng. |
| 2 | `gender` | Nhân khẩu học | Chuỗi danh mục | `Female`, `Male` | Đặc trưng $X$ | Chuẩn hóa trim khoảng trắng; mã hóa One-Hot. |
| 3 | `SeniorCitizen` | Nhân khẩu học | Nhị phân (int/str) | `0`, `1` | Đặc trưng $X$ | Biến nhị phân chỉ khách hàng cao tuổi. Giữ nguyên 0/1, đưa vào nhánh danh mục/nhị phân One-Hot. |
| 4 | `Partner` | Nhân khẩu học | Chuỗi danh mục | `Yes`, `No` | Đặc trưng $X$ | Có bạn đời hay không. Chuẩn hóa trim; mã hóa One-Hot. |
| 5 | `Dependents` | Nhân khẩu học | Chuỗi danh mục | `Yes`, `No` | Đặc trưng $X$ | Có người phụ thuộc kinh tế hay không. Chuẩn hóa trim; mã hóa One-Hot. |
| 6 | `tenure` | Thời gian dùng | Số nguyên (integer) | Số nguyên $\ge 0$ (tháng, quan sát thực tế 0 – 72) | Đặc trưng $X$ | Giữ nguyên dạng số; nếu thiếu (trong dữ liệu inference), điền bằng `median` từ training fold. |
| 7 | `PhoneService` | Dịch vụ thoại | Chuỗi danh mục | `Yes`, `No` | Đặc trưng $X$ | Sử dụng dịch vụ điện thoại cố định. Chuẩn hóa trim; mã hóa One-Hot. |
| 8 | `MultipleLines` | Dịch vụ thoại | Chuỗi danh mục | `Yes`, `No`, `No phone service` | Đặc trưng $X$ | Sử dụng nhiều đường dây điện thoại. Phụ thuộc `PhoneService`. Giữ nguyên 3 giá trị, không coi `No phone service` là thiếu; mã hóa One-Hot. |
| 9 | `InternetService` | Dịch vụ Internet | Chuỗi danh mục | `DSL`, `Fiber optic`, `No` | Đặc trưng $X$ | Loại hình kết nối mạng Internet. Mã hóa One-Hot. |
| 10 | `OnlineSecurity` | Dịch vụ GTGT | Chuỗi danh mục | `Yes`, `No`, `No internet service` | Đặc trưng $X$ | Gói dịch vụ bảo mật trực tuyến. Giữ nguyên 3 giá trị; mã hóa One-Hot. |
| 11 | `OnlineBackup` | Dịch vụ GTGT | Chuỗi danh mục | `Yes`, `No`, `No internet service` | Đặc trưng $X$ | Gói dịch vụ sao lưu dữ liệu đám mây. Giữ nguyên 3 giá trị; mã hóa One-Hot. |
| 12 | `DeviceProtection` | Dịch vụ GTGT | Chuỗi danh mục | `Yes`, `No`, `No internet service` | Đặc trưng $X$ | Gói dịch vụ bảo hiểm/bảo vệ thiết bị. Giữ nguyên 3 giá trị; mã hóa One-Hot. |
| 13 | `TechSupport` | Dịch vụ GTGT | Chuỗi danh mục | `Yes`, `No`, `No internet service` | Đặc trưng $X$ | Gói dịch vụ hỗ trợ kỹ thuật ưu tiên. Giữ nguyên 3 giá trị; mã hóa One-Hot. |
| 14 | `StreamingTV` | Dịch vụ GTGT | Chuỗi danh mục | `Yes`, `No`, `No internet service` | Đặc trưng $X$ | Gói dịch vụ truyền hình trực tuyến qua Internet. Giữ nguyên 3 giá trị; mã hóa One-Hot. |
| 15 | `StreamingMovies` | Dịch vụ GTGT | Chuỗi danh mục | `Yes`, `No`, `No internet service` | Đặc trưng $X$ | Gói dịch vụ xem phim trực tuyến qua Internet. Giữ nguyên 3 giá trị; mã hóa One-Hot. |
| 16 | `Contract` | Hợp đồng | Chuỗi danh mục | `Month-to-month`, `One year`, `Two year` | Đặc trưng $X$ | Thời hạn hợp đồng cung cấp dịch vụ viễn thông. Mã hóa One-Hot. |
| 17 | `PaperlessBilling` | Thanh toán | Chuỗi danh mục | `Yes`, `No` | Đặc trưng $X$ | Nhận hóa đơn điện tử thay cho hóa đơn giấy. Mã hóa One-Hot. |
| 18 | `PaymentMethod` | Thanh toán | Chuỗi danh mục | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` | Đặc trưng $X$ | Phương thức thanh toán cước phí. Mã hóa One-Hot. |
| 19 | `MonthlyCharges` | Chi phí | Số thực (float) | Số thực $\ge 0$ (quan sát 18.25 – 118.75) | Đặc trưng $X$ | Phí dịch vụ hàng tháng. Dấu chấm thập phân. Chuẩn hóa bằng `StandardScaler` cho LR, passthrough cho DT/RF. Điền thiếu bằng `median`. |
| 20 | `TotalCharges` | Chi phí | Số thực (float) | Số thực $\ge 0$ hoặc thiếu (chuỗi trống) | Đặc trưng $X$ | Tổng phí lũy kế khách hàng đã thanh toán. Cần ép kiểu từ chuỗi sang số; 11 bản ghi có khoảng trắng ` ` được biến đổi thành `np.nan` và điền thiếu bằng `median` từ training fold. |
| 21 | `Churn` | Nhãn mục tiêu | Chuỗi / Số nguyên | `Yes`, `No` $\rightarrow$ `1`, `0` | Nhãn $y$ | Nhãn rời bỏ: `Yes` = 1 (khách rời bỏ), `No` = 0 (khách tiếp tục sử dụng). Tuyệt đối không đưa vào $X$. Bản ghi thiếu nhãn không đưa vào huấn luyện có giám sát. |

---

## 3. Ràng buộc toàn vẹn và kiểm tra logic nghiệp vụ

1. **Ràng buộc dịch vụ điện thoại:**
   - Nếu `PhoneService == 'No'`, thì bắt buộc `MultipleLines == 'No phone service'`.
   - Nếu `MultipleLines in ['Yes', 'No']`, thì bắt buộc `PhoneService == 'Yes'`.
2. **Ràng buộc dịch vụ Internet:**
   - Nếu `InternetService == 'No'`, thì tất cả 6 dịch vụ phụ:
     `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
     đều bắt buộc nhận giá trị `'No internet service'`.
   - Ngược lại, nếu `InternetService in ['DSL', 'Fiber optic']`, thì 6 dịch vụ phụ trên chỉ nhận giá trị `'Yes'` hoặc `'No'`.
3. **Ràng buộc thời gian sử dụng và cước phí:**
   - `tenure` phải là số nguyên $\ge 0$.
   - Nếu `tenure == 0` (khách hàng mới đăng ký trong tháng hiện tại), cước phí `TotalCharges` có thể chưa phát sinh (bằng 0 hoặc thiếu).
   - `MonthlyCharges` và `TotalCharges` phải là số thực không âm ($\ge 0$).
4. **Ràng buộc nhãn:**
   - Nhãn `Churn` ở tập huấn luyện chỉ nhận hai giá trị chuẩn hóa là 0 và 1 (hoặc `No` và `Yes`). Bản ghi thiếu nhãn bị loại khỏi tập huấn luyện.

---

## 4. Chính sách xử lý giá trị đặc biệt

- **Chuỗi trống và khoảng trắng:** Được tự động trim đầu cuối. Chuỗi trống ở cột số như `TotalCharges` chuyển thành `NaN` và được điền bằng `SimpleImputer(strategy='median')`.
- **Giá trị ngoại lai (Outliers):** Phí hàng tháng và tổng phí có đuôi dài nhưng là giao dịch hợp lệ của khách hàng doanh nghiệp hoặc dùng lâu năm, không tự ý xóa bỏ bản ghi.
- **Danh mục chưa gặp (Unseen categories):** Khi ứng dụng inference gặp danh mục mới lạ, OneHotEncoder được cấu hình `handle_unknown='ignore'` để tránh văng ngoại lệ, đồng thời lớp tiền kiểm tra schema phát cảnh báo.
