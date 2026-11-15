# Đặc tả bài toán — Dự đoán khách hàng rời bỏ dịch vụ viễn thông

**Dự án:** Khai phá dữ liệu khách hàng viễn thông (IBM Telco Customer Churn)  
**Tài liệu:** Đặc tả bài toán và phạm vi kỹ thuật (Mốc M0)  
**Trạng thái:** Đã chốt theo [Plan 01](../designe-note/plans/01-bai-toan-va-co-so-ly-thuyet.md) và [note.md](../designe-note/note.md)

---

## 1. Bối cảnh và Động lực nghiệp vụ

Trong thị trường viễn thông bão hòa và có tính cạnh tranh khốc liệt, chi phí để thu hút một khách hàng mới (Customer Acquisition Cost - CAC) thường cao hơn gấp 5 đến 7 lần chi phí duy trì một khách hàng hiện hữu (Customer Retention Cost - CRC). Sự rời bỏ của khách hàng (Customer Churn) không chỉ làm giảm trực tiếp dòng tiền định kỳ (Monthly Recurring Revenue - MRR) mà còn kéo theo chi phí vận hành và chiêu thị nhằm bù đắp lượng người dùng sụt giảm.

Mục tiêu cốt lõi của đề tài là áp dụng các phương pháp và quy trình **Khai phá dữ liệu (Data Mining / Knowledge Discovery in Databases - KDD)** nhằm:
1. Phát hiện sớm các khách hàng có xác suất rời bỏ cao từ các đặc trưng hợp đồng, dịch vụ và nhân khẩu học.
2. Xếp hạng nguy cơ để bộ phận chăm sóc khách hàng (Customer Support / CRM) chủ động phân bổ nguồn lực chăm sóc và ưu đãi giữ chân đúng đối tượng.
3. Cung cấp cơ sở định lượng, loại bỏ cảm tính chủ quan trong việc đánh giá hành vi khách hàng.

---

## 2. Phát biểu bài toán khai phá dữ liệu

### 2.1. Phân loại hình thức bài toán
Theo phân loại bài toán khai phá dữ liệu trong giáo trình *Khai phá dữ liệu và Kho dữ liệu* (Chương 1, Mục 1.2 & 1.3):
- Đề tài thuộc nhóm **Khai phá dữ liệu dạng tiên đoán (Predictive Data Mining)**, cụ thể là **Bài toán phân loại có giám sát (Supervised Classification)** với nhãn nhị phân ($y \in \{0, 1\}$).
- Không phải bài toán học không giám sát (phân cụm), không phải khai phá luật kết hợp hay bài toán hồi quy (mặc dù có sử dụng mô hình Logistic Regression làm bộ phân lớp).

### 2.2. Đối tượng phân tích
- Mỗi quan sát (bản ghi) trong tập dữ liệu tương ứng với **một hồ sơ khách hàng cá nhân** đang sử dụng hoặc vừa chấm dứt dịch vụ tại công ty viễn thông IBM Telco.
- Dữ liệu nguồn: [Telco-Customer-Churn.csv](../telco-customer-churn-on-icp4d/data/Telco-Customer-Churn.csv), gồm $N = 7.043$ hồ sơ với 21 cột thuộc tính ban đầu.

### 2.3. Không gian nhãn mục tiêu ($y$)
- Cột mục tiêu gốc: `Churn` (nhận hai giá trị chuỗi: `'Yes'` hoặc `'No'`).
- Ánh xạ nhãn nhị phân chuẩn hóa:
  $$y = \begin{cases} 1, & \text{nếu } \text{Churn} = \text{'Yes'} \quad (\text{Khách hàng rời bỏ dịch vụ - Lớp dương / Positive class}) \\ 0, & \text{nếu } \text{Churn} = \text{'No'} \quad (\text{Khách hàng tiếp tục ở lại - Lớp âm / Negative class}) \end{cases}$$
- **Quy ước bất biến:** Lớp dương luôn luôn là $y = 1$ (`Churn = Yes`). Toàn bộ các chỉ số đánh giá ($\text{Precision}$, $\text{Recall}$, $F_1$, ngưỡng quyết định) đều được tính toán và tối ưu hóa dựa trên việc nhận diện lớp dương này.

### 2.4. Không gian thuộc tính đầu vào ($x \in \mathcal{X} \subset \mathbb{R}^{d}$)
Sau khi loại bỏ định danh `customerID` và nhãn mục tiêu `Churn`, vector đầu vào $x$ gồm đúng **19 thuộc tính**, được phân thành 4 nhóm nghiệp vụ chính:

1. **Nhóm nhân khẩu học và cá nhân (Demographics - 4 thuộc tính):**
   - `gender` (Danh mục: `'Female'`, `'Male'`): Giới tính khách hàng.
   - `SeniorCitizen` (Nhị phân số: `0`, `1`): Khách hàng có phải là người cao tuổi (từ 65 tuổi trở lên) hay không.
   - `Partner` (Danh mục: `'Yes'`, `'No'`): Khách hàng có bạn đời/vợ/chồng hay không.
   - `Dependents` (Danh mục: `'Yes'`, `'No'`): Khách hàng có người phụ thuộc (con cái, người thân) hay không.

2. **Nhóm thời gian gắn bó (Tenure - 1 thuộc tính):**
   - `tenure` (Số nguyên liên tục, $[0, 72]$): Số tháng khách hàng đã ký hợp đồng và duy trì dịch vụ với nhà mạng.

3. **Nhóm dịch vụ viễn thông và giá trị gia tăng (Services - 9 thuộc tính):**
   - `PhoneService` (Danh mục: `'Yes'`, `'No'`): Có đăng ký dịch vụ thoại cố định hay không.
   - `MultipleLines` (Danh mục: `'No phone service'`, `'No'`, `'Yes'`): Có đăng ký nhiều đường dây thoại không.
   - `InternetService` (Danh mục: `'DSL'`, `'Fiber optic'`, `'No'`): Loại kết nối Internet băng rộng.
   - `OnlineSecurity` (Danh mục: `'No internet service'`, `'No'`, `'Yes'`): Dịch vụ bảo mật trực tuyến.
   - `OnlineBackup` (Danh mục: `'No internet service'`, `'No'`, `'Yes'`): Dịch vụ sao lưu dữ liệu đám mây.
   - `DeviceProtection` (Danh mục: `'No internet service'`, `'No'`, `'Yes'`): Gói bảo hiểm và bảo vệ thiết bị.
   - `TechSupport` (Danh mục: `'No internet service'`, `'No'`, `'Yes'`): Dịch vụ hỗ trợ kỹ thuật ưu tiên.
   - `StreamingTV` (Danh mục: `'No internet service'`, `'No'`, `'Yes'`): Dịch vụ truyền hình trực tuyến qua Internet.
   - `StreamingMovies` (Danh mục: `'No internet service'`, `'No'`, `'Yes'`): Dịch vụ phim trực tuyến qua Internet.

4. **Nhóm hợp đồng, phương thức thanh toán và chi phí (Account & Charges - 5 thuộc tính):**
   - `Contract` (Danh mục: `'Month-to-month'`, `'One year'`, `'Two year'`): Loại kỳ hạn hợp đồng.
   - `PaperlessBilling` (Danh mục: `'Yes'`, `'No'`): Khách hàng có nhận hóa đơn điện tử không dùng giấy hay không.
   - `PaymentMethod` (Danh mục: 4 giá trị): Hình thức thanh toán (`'Electronic check'`, `'Mailed check'`, `'Bank transfer (automatic)'`, `'Credit card (automatic)'`).
   - `MonthlyCharges` (Số thực liên tục, $[18.25, 118.75]$): Số tiền cước phải trả trong tháng gần nhất (USD).
   - `TotalCharges` (Số thực liên tục, $[18.80, 8684.80]$): Tổng số tiền cước tích lũy từ khi bắt đầu dịch vụ đến nay (có 11 trường hợp chuỗi trống đại diện cho khách hàng mới có `tenure = 0`).

---

## 3. Đặc tả đầu ra và người dùng hệ thống

### 3.1. Đầu ra của mô hình dự đoán
Với mỗi hồ sơ đầu vào $x_i$, hệ thống xuất ra bộ giá trị chuẩn hóa:
1. **Điểm nguy cơ (Risk Score / Estimated Posterior Probability):**
   $$\hat{p}_i = P(y = 1 \mid x_i) \in [0, 1]$$
   Đại diện cho xác suất ước lượng của mô hình rằng khách hàng thuộc lớp rời bỏ (`Churn = Yes`).
2. **Nhãn phân loại dự đoán ($\hat{y}_i$):**
   $$\hat{y}_i = \mathbb{I}(\hat{p}_i \ge \theta) \in \{0, 1\}$$
   Với $\theta = 0.5$ là ngưỡng quyết định mặc định (Default Decision Threshold) ở mốc M0.
3. **Phân nhóm mức độ ưu tiên chăm sóc (Priority Tier):**
   - **Cao (High Risk):** $\hat{p}_i \ge 0.70$ — Cần liên hệ can thiệp khẩn cấp.
   - **Trung bình (Medium Risk):** $0.40 \le \hat{p}_i < 0.70$ — Cần đưa vào danh sách theo dõi và gửi thông điệp giữ chân định kỳ.
   - **Thấp (Low Risk):** $\hat{p}_i < 0.40$ — Nhóm khách hàng ổn định, duy trì chính sách tiêu chuẩn.
4. **Metadata phiên bản mô hình (Model Version & Provenance):** Tên mô hình, mã phiên bản (`model_version`), ngày huấn luyện và hash của artifact được dùng để tạo ra dự đoán.

### 3.2. Đối tượng sử dụng (Stakeholders)
- **Nhân viên / Quản lý Chăm sóc Khách hàng (Customer Retention Team):** Người dùng trực tiếp giao diện Streamlit để tra cứu từng khách hàng cụ thể hoặc tải tệp danh sách khách hàng theo lô (batch upload) nhằm lọc ra danh sách có nguy cơ rời bỏ cao nhất.
- **Giảng viên và Hội đồng chấm Đề tài (Evaluators):** Xem xét tính đúng đắn về mặt khoa học, khả năng tái lập thực nghiệm, mức độ tuân thủ quy trình KDD, sự hiểu biết về bản chất thuật toán và kiến trúc triển khai thực tế.

### 3.3. Quyết định nghiệp vụ được hỗ trợ và Ranh giới trách nhiệm
- **Phạm vi hỗ trợ:** Mô hình đóng vai trò là **hệ thống hỗ trợ ra quyết định (Decision Support System - DSS)**. Mô hình xếp hạng nguy cơ và đề xuất các đối tượng cần chú ý.
- **Ranh giới trách nhiệm con người:** Nhân viên nghiệp vụ là người đưa ra quyết định cuối cùng về việc có thực hiện cuộc gọi, tặng voucher hay chuyển đổi gói cước hay không.
- **Cảnh báo về cách diễn giải xác suất:** Điểm nguy cơ $\hat{p} = 0.80$ là điểm số đầu ra của mô hình đối với lớp rời bỏ trong không gian đặc trưng tĩnh; con số này **không đồng nghĩa** với việc "80% khả năng khách hàng này chắc chắn sẽ cắt dịch vụ trong vòng 30 ngày tới". Để có thể diễn giải thành xác suất theo thời gian thực, dữ liệu cần có dấu thời gian (time-series/survival data) và mô hình phải trải qua quy trình hiệu chuẩn xác suất (Probability Calibration).

---

## 4. Các quyết định kỹ thuật bắt buộc chốt tại Mốc M0

Nhằm đảm bảo tính trung thực khoa học, chặt chẽ trong thực nghiệm và khả năng bảo vệ thành công trước hội đồng, nhóm thống nhất 6 nguyên tắc bất biến:

1. **Bản chất dữ liệu tĩnh:** Sử dụng bộ dữ liệu hiện có như một ảnh chụp tĩnh (cross-sectional snapshot) để giải bài toán phân loại nhãn đã có. Không tuyên bố sai lệch rằng đây là hệ thống dự báo chuỗi thời gian thực tế.
2. **Loại bỏ hoàn toàn định danh khỏi quá trình học:** Cột `customerID` là mã định danh ngẫu nhiên (chuỗi ký tự dạng `7590-VHVEG`), không mang giá trị quy nạp tri thức và có nguy cơ gây quá khớp giả tạo nếu mã hóa. Cột này **bắt buộc bị loại bỏ** khỏi vector đặc trưng huấn luyện và dự đoán, nhưng vẫn được lưu trữ trong cơ sở dữ liệu để phục vụ việc truy vết, đối chiếu.
3. **Chống rò rỉ nhãn (Data Leakage Prevention):** Tuyệt đối không để thuộc tính `Churn` xuất hiện trong không gian đầu vào của bất kỳ bước tiền xử lý hay mô hình nào khi dự đoán.
4. **Phạm vi ba mô hình yêu cầu và mốc đối chứng:** Triển khai đầy đủ và công bằng ba mô hình: **Decision Tree**, **Logistic Regression** và **Random Forest**. Bổ sung thêm mô hình ngây thơ (`DummyClassifier`) làm mốc so sánh cơ sở (Baseline) để chứng minh tính hiệu quả của các mô hình học máy.
5. **Cố định ngưỡng quyết định cơ bản $\theta = 0.5$:** Toàn bộ việc so sánh, lựa chọn mô hình và báo cáo chỉ số kiểm thử chính thức sử dụng ngưỡng xác suất $\theta = 0.5$. Thử nghiệm quét ngưỡng (threshold tuning) chỉ được thực hiện như một phân tích mở rộng độc lập, không dùng để thay đổi kết quả so sánh cốt lõi.
6. **Không đưa ra các ước tính tài chính thiếu căn cứ:** Không tự ý bịa đặt doanh thu giữ chân, hệ số ROI hay hiệu quả chiến dịch tài chính trong báo cáo khoa học khi bộ dữ liệu chưa cung cấp thông tin chi phí can thiệp và phản ứng thực tế của khách hàng.

---

## 5. Ánh xạ quy trình KDD trong Dự án

Quy trình phát hiện tri thức trong cơ sở dữ liệu (KDD Process - Fayyad et al., 1996) được áp dụng làm xương sống cho toàn bộ cấu trúc dự án:

```
[Raw Data: CSV] 
       │ 
       ▼ (1. Selection)
[Target Data: 19 Features + Label] 
       │ 
       ▼ (2. Preprocessing & Cleaning)
[Clean Data: Numeric Imputation, Type Casting] 
       │ 
       ▼ (3. Transformation)
[Transformed Data: One-Hot Encoding, StandardScaler] 
       │ 
       ▼ (4. Data Mining)
[Trained Models: Decision Tree, Logistic Regression, Random Forest via 5-Fold CV] 
       │ 
       ▼ (5. Evaluation & Interpretation)
[Evaluation: Confusion Matrix, Precision, Recall, F1, ROC-AUC, Feature Importance] 
       │ 
       ▼ (6. Knowledge Deployment)
[Knowledge Use: Streamlit Dashboard, SQLite Registry, Active Model Serving]
```

### Bảng đối chiếu chi tiết 8 bước KDD và Sản phẩm bàn giao

| STT | Bước KDD theo Giáo trình | Nhiệm vụ cụ thể trong Đề tài | Bằng chứng & Tệp kết quả dự kiến |
|:---:|---|---|---|
| **1** | **Hiểu lĩnh vực và xác định mục tiêu** | Phân tích bài toán rời bỏ mạng viễn thông, sự bất đối xứng chi phí giữa bỏ sót (FN) và cảnh báo nhầm (FP), ranh giới bài toán phân loại tĩnh. | `docs/problem-definition.md`<br>`designe-note/note.md` |
| **2** | **Tạo tập dữ liệu đích (Target Data)** | Lựa chọn tệp `Telco-Customer-Churn.csv`, loại bỏ `customerID` khỏi tập đặc trưng, cô lập nhãn `Churn`, tách 19 biến đầu vào. | `docs/data_dictionary.md`<br>`scripts/audit_data.py`<br>`telco-customer-churn-on-icp4d/data/Telco-Customer-Churn.csv` |
| **3** | **Làm sạch và Tiền xử lý (Cleaning & Preprocessing)** | Ép kiểu `TotalCharges` về số thực, phát hiện và xử lý 11 giá trị trống bằng trung vị (median) của tập huấn luyện, kiểm tra trùng lặp bản ghi. | `reports/data_audit.md`<br>`src/telco_churn/data.py`<br>`src/telco_churn/preprocessing.py` |
| **4** | **Chuyển đổi dữ liệu (Transformation)** | Phân chia 80% train / 20% test phân tầng (Stratified); mã hóa One-Hot cho biến danh mục; chuẩn hóa z-score (StandardScaler) cho các biến số trong Logistic Regression. | `data/splits/split_manifest.csv`<br>`src/telco_churn/preprocessing.py`<br>`scripts/make_splits.py` |
| **5** | **Lựa chọn chức năng và thuật toán** | Lựa chọn phân loại có giám sát nhị phân; thiết lập 3 thuật toán: Cây quyết định (Decision Tree), Hồi quy Logistic (Logistic Regression), Rừng ngẫu nhiên (Random Forest). | `docs/theory.md`<br>`src/telco_churn/training.py`<br>`configs/experiment.json` |
| **6** | **Khai phá dữ liệu (Data Mining)** | Thiết lập 5-Fold Stratified Cross-Validation trên tập phát triển; quét lưới siêu tham số (GridSearchCV / RandomizedSearchCV); ghi log quá trình huấn luyện. | `reports/experiments/`<br>`artifacts/models/<version>/`<br>`scripts/train.py` |
| **7** | **Đánh giá và Biểu diễn tri thức** | Đánh giá qua ma trận nhầm lẫn, Precision, Recall, F1-score (lớp Churn), ROC-AUC; trực quan hóa ROC Curve, PR Curve, tầm quan trọng của thuộc tính. | `reports/model_selection.md`<br>`reports/final_evaluation.md`<br>`scripts/evaluate.py`<br>`reports/figures/` |
| **8** | **Sử dụng và Vận hành tri thức** | Xây dựng giao diện tương tác Streamlit cho phép dự đoán đơn lẻ, dự đoán theo lô CSV; lưu lịch sử dự đoán và mô phỏng cập nhật dữ liệu với SQLite. | `app.py`<br>`ui/*.py`<br>`src/telco_churn/inference.py`<br>`src/telco_churn/storage.py` |

---

## 6. Giới hạn và Phạm vi không thuộc đề tài

Để đảm bảo tính khả thi và tập trung vào chất lượng học thuật của môn học:
- **Cơ sở dữ liệu SQLite trong ứng dụng:** Đóng vai trò là cơ sở dữ liệu giao dịch ứng dụng (Application Datastore) lưu trữ lịch sử dự đoán và nhãn phản hồi; không phải là kho dữ liệu (Data Warehouse) đa chiều hay hệ thống OLAP phức tạp.
- **Quy trình huấn luyện lại (Retraining):** Được thiết kế theo cơ chế mô phỏng (Simulation / Dry-run) và quy trình chuẩn (Gatekeeper, Champion/Candidate comparison, Rollback), minh họa khả năng tiến hóa của hệ thống phát hiện tri thức khi có nhãn thực tế mới.
- **Không áp dụng luật kết hợp (Association Rules) hay Phân cụm (Clustering):** Vì mục tiêu chính là phân loại nhãn Churn; các thuật toán này không phục vụ trực tiếp mục tiêu tối ưu hóa F1-score đã đề ra.