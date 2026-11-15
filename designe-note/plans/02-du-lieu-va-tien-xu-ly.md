# Plan 02 — Dữ liệu, khám phá và tiền xử lý

Người chính: TV1. Người phối hợp: TV2/TV3 kiểm tra pipeline; TV4/TV5 dùng chung schema. Phụ thuộc: Plan 01. Đầu ra dự kiến: `reports/data_audit.md`, `docs/data_dictionary.md`, `data/splits/split_manifest.csv`, notebook EDA và mã pipeline trong `src/telco_churn/`.

## 1. Kiểm kê dữ liệu ban đầu

Nguồn: `telco-customer-churn-on-icp4d/data/Telco-Customer-Churn.csv`. Giữ nguyên CSV gốc, tính SHA-256 và ghi vào audit. Không sửa trực tiếp file nguồn.

Đã kiểm tra khi lập plan: 7.043 dòng, 21 cột; 5.174 No, 1.869 Yes; 11 `TotalCharges` trống/khoảng trắng; không trùng `customerID`. Các kiểm tra khác bên dưới vẫn cần thực hiện, không coi như đã đạt.

### Data dictionary phải bao phủ đủ 21 cột

| Nhóm | Cột | Kiểu/miền đầu vào và xử lý |
|---|---|---|
| Định danh | `customerID` | Chuỗi, giữ để truy vết, loại khỏi X |
| Nhãn | `Churn` | Yes/No; đổi 1/0; không điền thiếu nhãn |
| Khách hàng | `gender` | Female/Male; danh mục |
| Khách hàng | `SeniorCitizen` | 0/1; biến nhị phân dù biểu diễn số |
| Khách hàng | `Partner`, `Dependents` | Yes/No |
| Thời gian sử dụng | `tenure` | Số nguyên không âm, đơn vị tháng |
| Điện thoại | `PhoneService` | Yes/No |
| Điện thoại | `MultipleLines` | Yes/No/No phone service |
| Internet | `InternetService` | DSL/Fiber optic/No |
| Dịch vụ phụ | `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Yes/No/No internet service |
| Hợp đồng | `Contract` | Month-to-month/One year/Two year; dùng one-hot |
| Hóa đơn | `PaperlessBilling` | Yes/No |
| Thanh toán | `PaymentMethod` | Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic) |
| Chi phí | `MonthlyCharges`, `TotalCharges` | Số thực không âm; dấu thập phân chấm; TotalCharges cho phép thiếu để pipeline điền |

Miền liệt kê phải được đối chiếu bằng `unique()` với CSV. Các giới hạn trên quan sát được như max tenure là thống kê tham khảo, không tự biến thành giới hạn nghiệp vụ cho khách mới. Giá trị rất cao nhưng hợp lệ nên được cảnh báo thay vì tự động xóa.

## 2. Audit và quy tắc làm sạch

1. Đọc CSV và ghi shape, tên cột, dtype, số thiếu, chuỗi rỗng, số giá trị khác nhau, min/max số, phân bố nhãn.
2. Trim khoảng trắng ở chuỗi; chuẩn hóa Yes/No theo ánh xạ định trước. Chỉ chấp nhận các biến thể được liệt kê, không sửa đoán giá trị lạ.
3. Chuyển `TotalCharges` sang số; chuỗi trống thành NaN. Giá trị không rỗng nhưng không chuyển được phải vào báo cáo lỗi, không âm thầm biến toàn bộ thành thiếu.
4. Kiểm tra 11 dòng thiếu có đặc điểm gì, đặc biệt `tenure=0`; chưa kết luận tất cả đều như vậy khi chưa xác minh. Phương án chính vẫn là median từ training fold, giữ đủ mẫu.
5. Kiểm tra trùng toàn dòng, trùng ID, trùng 19 đặc trưng và nhãn mâu thuẫn. Bản sao cùng ID/hồ sơ phải xử lý trước split hoặc cùng nhóm để tránh xuất hiện ở hai phía. Hồ sơ giống nhau nhưng ID khác có thể là khách khác; không tự xóa chỉ vì giống đặc trưng.
6. Kiểm tra bất nhất: PhoneService=No nhưng MultipleLines=Yes; InternetService=No nhưng dịch vụ internet bổ sung=Yes. Gắn cờ và ghi cách xử lý; không sửa hàng loạt chỉ dựa vào phỏng đoán.
7. Nhãn thiếu/sai: cách ly khỏi tập huấn luyện. Dòng có giá trị âm hoặc sai định dạng: ghi lý do; không đưa vào học cho tới khi giải quyết.
8. Ngoại lai: mô tả bằng boxplot/IQR nhưng không xóa tự động vì có thể là khách chi phí cao hợp lệ. Nếu thử clipping thì chỉ học ngưỡng trên training fold và coi là một thí nghiệm.

Các thao tác định trước như trim, kiểm tra schema, chuyển kiểu và loại bản sao có thể thực hiện trước split. Mọi thao tác cần học thống kê, dùng liên hệ với nhãn hoặc quyết định chọn đặc trưng phải nằm sau split, trong quá trình phát triển.

## 3. Chia dữ liệu và khóa tập kiểm thử

1. Tách `customerID`, `X` gồm 19 cột và `y` 1/0.
2. Chia `test_size=0.2`, `stratify=y`, `random_state=42`.
3. Nếu giữ đủ 7.043 dòng, kích thước dự kiến là 5.634 phát triển và 1.409 test. Nếu audit loại mẫu thì tính lại và giải thích, không ép cho khớp số cũ.
4. Lưu manifest tối thiểu gồm `customerID`, `split`, seed, hash dữ liệu nguồn và phiên bản quy tắc làm sạch.
5. Kiểm tra giao ID hai tập bằng rỗng; tỷ lệ lớp gần nhau. Không mở biểu đồ Churn theo thuộc tính trên test để chọn mô hình.
6. Trên 80% phát triển, tạo `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` và dùng cùng chỉ mục fold cho mọi mô hình.

Không lưu bảng đã scale toàn bộ rồi mới chia; không mã hóa hay oversample toàn dữ liệu trước CV. Audit cấu trúc toàn file và việc dùng nhãn để phân tầng không phải tuning; cần tách chúng khỏi EDA phục vụ lựa chọn mô hình.

## 4. EDA trên phần phát triển

| Câu hỏi | Hình/bảng | Kết luận cần ghi |
|---|---|---|
| Nhãn mất cân bằng thế nào? | Countplot + tỷ lệ | Tại sao accuracy chưa đủ |
| Thâm niên liên quan churn thế nào? | Histogram/boxplot theo Churn | Mô tả liên hệ, tránh kết luận nhân quả |
| Loại hợp đồng liên quan churn thế nào? | Churn rate và số mẫu theo Contract | Không chỉ nhìn phần trăm ở nhóm rất nhỏ |
| Chi phí có khác biệt giữa hai lớp? | Phân bố MonthlyCharges/TotalCharges | Kiểm tra lệch và ngoại lai |
| Dịch vụ/thanh toán có khác biệt? | Tỷ lệ theo InternetService, TechSupport, PaymentMethod | Lập giả thuyết để kiểm tra, không khẳng định sẵn |
| Biến số có tương quan? | Ma trận tương quan biến số | Lưu ý tenure và TotalCharges có thể dư thừa |
| Chất lượng sau làm sạch? | Bảng thiếu, lỗi và số mẫu trước/sau | Mỗi thay đổi có lý do |

Mỗi hình cần tiêu đề, nhãn trục, số mẫu/tập dữ liệu và 2–4 câu nhận xét. Lưu hình vào `reports/figures/eda/`. Không dùng biểu đồ để “chứng minh” một yếu tố gây churn.

## 5. Thiết kế pipeline dùng lại

### Nhánh số

- Các biến số liên tục/đếm: `tenure`, `MonthlyCharges`, `TotalCharges`.
- `SimpleImputer(strategy='median')` fit trong training fold.
- LR: thêm `StandardScaler`; DT/RF: passthrough sau imputer.

### Nhánh danh mục và nhị phân

- 16 biến còn lại, bao gồm `SeniorCitizen` sau khi chuẩn hóa về miền 0/1 nhất quán.
- Imputer `most_frequent`, sau đó one-hot. Thiếu toàn cột phải bị từ chối ở validation; không coi là thiếu ô có thể điền.
- Giữ nguyên `No internet service` và `No phone service`, không mặc định coi là dữ liệu thiếu.
- Encoder có khả năng xử lý danh mục chưa gặp khi transform. Lớp validation ứng dụng vẫn chặn danh mục ngoài schema hoặc yêu cầu xử lý rõ; cấu hình encoder là lớp dự phòng, không lý do để bỏ kiểm tra đầu vào.

Ghép bằng `ColumnTransformer`, rồi đưa vào `Pipeline(preprocessor, classifier)`. Chuyển kiểu định trước phải dùng cùng hàm ở training và inference, hoặc transformer được đóng gói trong pipeline. Không viết lại một bản tiền xử lý khác trong Streamlit.

### Mất cân bằng và đặc trưng mới

- Chính: thử `class_weight=None` và `balanced` trong CV, giữ phân bố test nguyên bản.
- Không bắt buộc SMOTE trong bản đầu vì có biến one-hot và dễ sinh mẫu sai nghĩa. Nếu mở rộng phải nghiên cứu phương pháp phù hợp dữ liệu hỗn hợp và resample bên trong training fold.
- Bản đầu dùng 19 biến gốc. Các biến như số dịch vụ bổ sung chỉ thử khi mô hình chính đã hoàn tất; tạo biến từ đầu vào có sẵn, không từ Churn hay thông tin tương lai.
- Không bắt buộc PCA hoặc lựa chọn đặc trưng vì số cột gốc nhỏ và cần diễn giải. Nếu thử loại biến thì quyết định bằng CV, không bằng test.

## 6. Bàn giao và nghiệm thu

- `schema.py`: tên cột, miền danh mục, kiểu dữ liệu và kiểm tra nhất quán.
- `data.py`: đọc/chuẩn hóa định trước, audit và split.
- `preprocessing.py`: hàm tạo preprocessor theo loại mô hình.
- Manifest chia tập, notebook EDA và báo cáo audit có hash nguồn.
- Mẫu CSV hợp lệ, mẫu thiếu TotalCharges, mẫu lỗi schema để TV4/TV5 tích hợp.

Điều kiện đạt: giữ đúng mapping nhãn; ID/Churn không lọt vào X; fit preprocessor theo fold; transform cùng một hồ sơ cho kết quả nhất quán ở luồng đơn và CSV; mô tả rõ số mẫu trước/sau và mọi giá trị được thay thế.
