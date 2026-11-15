# BÁO CÁO BÀI TẬP LỚN: KHAI PHÁ DỮ LIỆU VÀ KHO DỮ LIỆU
## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ PHƯƠNG PHÁP KHAI PHÁ DỮ LIỆU

**Đề tài:** Dự đoán khách hàng rời bỏ dịch vụ viễn thông (Customer Churn Prediction)  
**Nhóm sinh viên thực hiện:** Nhóm bài tập lớn  
**Phân công biên soạn chương:** Thành viên TV2 (Thuật toán), TV1 (Dữ liệu & Tiền xử lý), TV3 (Đánh giá & Thực nghiệm)

---

### 2.1. TỔNG QUAN VỀ KHAI PHÁ DỮ LIỆU VÀ QUY TRÌNH KDD

#### 2.1.1. Khái niệm Khai phá dữ liệu và Bài toán Phân loại
Theo giáo trình *Khai phá dữ liệu và Kho dữ liệu*, **Khai phá dữ liệu (Data Mining)** là quá trình trích xuất tự động các mẫu tri thức tiềm ẩn, mới mẻ, có giá trị và có thể hiểu được từ các tập dữ liệu lớn. Trong cây phân loại nhiệm vụ khai phá dữ liệu:
- **Khai phá dữ liệu dạng mô tả (Descriptive Data Mining):** Tập trung phát hiện các mẫu thể hiện tính chất hiện hữu của dữ liệu (ví dụ: Phân cụm - Clustering, Khai phá luật kết hợp - Association Rules).
- **Khai phá dữ liệu dạng tiên đoán (Predictive Data Mining):** Sử dụng các biến đã biết trong quá khứ để xây dựng mô hình dự đoán giá trị của các biến chưa biết hoặc biến tương lai. Bài toán phân loại khách hàng rời bỏ dịch vụ viễn thông thuộc nhóm bài toán tiên đoán này.

Cụ thể, **Bài toán phân loại có giám sát (Supervised Classification)** được định nghĩa như sau:
Cho tập dữ liệu huấn luyện $\mathcal{D} = \{(x_1, y_1), (x_2, y_2), \dots, (x_N, y_N)\}$, trong đó mỗi bản ghi $x_i = (x_{i1}, x_{i2}, \dots, x_{id}) \in \mathcal{X}$ là một vector thuộc tính đại diện cho hồ sơ khách hàng, và $y_i \in \mathcal{Y} = \{0, 1\}$ là thuộc tính nhãn mục tiêu rời rạc:
$$y_i = \begin{cases} 1, & \text{nếu khách hàng rời bỏ dịch vụ (Churn = Yes)} \\ 0, & \text{nếu khách hàng tiếp tục sử dụng (Churn = No)} \end{cases}$$
Mục tiêu là tìm ra một ánh xạ hàm giả định $f: \mathcal{X} \to \mathcal{Y}$ (hoặc phân phối xác suất hậu nghiệm $P(y=1 \mid x)$) có khả năng tổng quát hóa tốt nhất trên tập dữ liệu chưa từng quan sát trước đó.

#### 2.1.2. Nguyên lý Quy nạp và Nguy cơ Quá khớp
Quá trình xây dựng bộ phân lớp từ dữ liệu mẫu tuân theo **nguyên lý suy diễn quy nạp (Inductive Inference)**: đúc kết quy luật tổng quát từ các trường hợp riêng lẻ đã quan sát.  
Tuy nhiên, giáo trình nhấn mạnh rằng suy diễn quy nạp luôn tiềm ẩn rủi ro tri thức thu được chỉ đúng trên tập dữ liệu quan sát cục bộ mà sai khác trên tổng thể (hiện tượng "thiên nga đen"). Trong học máy, hiện tượng này tương ứng với **Quá khớp (Overfitting)**: mô hình ghi nhớ toàn bộ đặc tính ngẫu nhiên và nhiễu của tập dữ liệu huấn luyện, đạt độ chính xác gần như hoàn hảo trên tập huấn luyện nhưng hoàn toàn thất bại khi áp dụng vào dữ liệu kiểm thử độc lập. Do đó, quy trình thực nghiệm đòi hỏi việc phân tách dữ liệu nghiêm ngặt và áp dụng các kỹ thuật điều chuẩn (Regularization).

#### 2.1.3. Ánh xạ 8 bước của Quy trình KDD trong Đề tài
Dự án được tổ chức và triển khai tuân thủ tuyệt đối quy trình 8 bước Phát hiện tri thức trong Cơ sở dữ liệu (Knowledge Discovery in Databases - KDD):

1. **Hiểu lĩnh vực ứng dụng và xác định mục tiêu:** Phân tích bản chất kinh doanh viễn thông; xác định chi phí mất khách hàng cao hơn chi phí chăm sóc giữ chân; định nghĩa rõ lớp dương mục tiêu là $y = 1$ (`Churn = Yes`).
2. **Tạo tập dữ liệu đích:** Chọn lọc tệp dữ liệu nguồn IBM Telco gồm 7.043 bản ghi; loại bỏ cột định danh `customerID` để ngăn chặn việc ghi nhớ giả tạo; cô lập 19 thuộc tính đầu vào và 1 biến nhãn.
3. **Tiền xử lý và làm sạch dữ liệu:** Kiểm tra dữ liệu khuyết thiếu; chuyển đổi kiểu dữ liệu cột `TotalCharges` từ chuỗi ký tự sang số thực; phát hiện 11 giá trị trống và thay thế bằng trung vị (Median) tính toán từ tập huấn luyện; rà soát tính nhất quán và trùng lặp bản ghi.
4. **Chuyển đổi dữ liệu:** Phân chia tập dữ liệu theo tỷ lệ 80% phát triển và 20% kiểm thử độc lập bằng phương pháp phân tầng (Stratified); thực hiện mã hóa One-Hot cho các biến định danh/phân loại; chuẩn hóa z-score cho các biến số liên tục phục vụ mô hình tuyến tính.
5. **Lựa chọn chức năng và thuật toán khai phá:** Chọn chức năng phân loại nhị phân; thiết lập ba kiến trúc mô hình đại diện cho các trường phái học máy khác nhau: Cây quyết định (Decision Tree - họ thuật toán Hunt/CART), Hồi quy Logistic (Logistic Regression - mô hình tuyến tính) và Rừng ngẫu nhiên (Random Forest - mô hình tập hợp nhiều cây).
6. **Khai phá dữ liệu:** Thực thi huấn luyện mô hình bằng phương pháp xác nhận chéo phân tầng 5 nếp gấp (5-Fold Stratified Cross-Validation) trên tập 80% phát triển; tối ưu hóa không gian siêu tham số bằng lưới tìm kiếm (Grid Search).
7. **Đánh giá và biểu diễn tri thức:** Đánh giá chất lượng mô hình trên ma trận nhầm lẫn (Confusion Matrix), điểm F1 trên lớp rời bỏ, diện tích dưới đường cong ROC (ROC-AUC) và PR (PR-AUC); trích xuất các thuộc tính có mức độ đóng góp quan trọng nhất đối với hành vi rời bỏ.
8. **Sử dụng và vận hành tri thức:** Đóng gói toàn bộ quy trình tiền xử lý và mô hình thành pipeline; tích hợp vào ứng dụng Streamlit cung cấp giao diện dự đoán theo thời gian thực và theo lô; lưu trữ lịch sử dự đoán trên cơ sở dữ liệu SQLite và thiết lập cơ chế giám sát huấn luyện lại khi có nhãn thực tế mới.

---

### 2.2. MÔ HÌNH CÂY QUYẾT ĐỊNH (DECISION TREE)

Cây quyết định là mô hình phân loại được trình bày chi tiết và có vị trí trọng tâm trong giáo trình khai phá dữ liệu (Chương 4, Mục 4.3).

#### 2.2.1. Cấu trúc Cây và Thuật toán Hunt
Cây quyết định tổ chức tri thức dưới dạng một cấu trúc hình cây phân cấp bao gồm:
- **Nút gốc (Root node):** Điểm khởi đầu chứa toàn bộ không gian dữ liệu huấn luyện.
- **Các nút trong (Internal nodes):** Mỗi nút ứng với một phép kiểm tra trên một thuộc tính $x_j$.
- **Các nhánh (Branches):** Đại diện cho các kết quả của phép kiểm tra (phân nhánh theo giá trị danh mục hoặc theo ngưỡng giá trị liên tục $x_j \le \theta$).
- **Các nút lá (Leaf nodes):** Điểm kết thúc gán nhãn phân loại hoặc phân phối xác suất lớp.

Khung thuật toán cốt lõi là **thuật toán đệ quy của Hunt**:
- Tại mỗi bước, xét tập dữ liệu $\mathcal{D}_t$ tại nút $t$. Nếu tất cả các bản ghi trong $\mathcal{D}_t$ cùng thuộc một lớp $y_k$, nút $t$ trở thành nút lá mang nhãn $y_k$.
- Nếu $\mathcal{D}_t$ chứa hỗn tạp nhiều lớp, thuật toán áp dụng chiến lược tham lam để tìm thuộc tính và phép phân chia sao cho tối đa hóa mức độ thuần nhất (độ giảm hỗn tạp) của các nút con sinh ra.

#### 2.2.2. Các Tiêu chuẩn Đo lường Độ hỗn tạp

1. **Chỉ số GINI (Gini Impurity):**
   Chỉ số Gini đo lường xác suất một phần tử bị gán nhãn sai nếu được gán nhãn ngẫu nhiên theo phân phối lớp tại nút:
   $$\text{Gini}(t) = 1 - \sum_{k=1}^C [p(k \mid t)]^2$$
   Trong đó $p(k \mid t)$ là tỷ lệ mẫu thuộc lớp $k$ tại nút $t$.  
   - Khi nút thuần nhất tuyệt đối (tất cả các mẫu cùng một lớp): $\text{Gini}(t) = 1 - 1^2 = 0$.
   - Đối với bài toán 2 lớp ($C=2$), khi phân phối cân bằng 50/50: $\text{Gini}(t) = 1 - (0.5^2 + 0.5^2) = 0.5$ (hỗn tạp tối đa).

2. **Chỉ số Entropy và Độ lợi Thông tin (Information Gain):**
   Đo lường lượng thông tin trung bình chứa trong phân phối xác suất của nút:
   $$\text{Entropy}(t) = - \sum_{k=1}^C p(k \mid t) \log_2 p(k \mid t)$$
   Độ lợi thông tin (Information Gain) khi phân chia nút cha $p$ thành $k$ nút con:
   $$\text{Gain}(p, \text{split}) = \text{Entropy}(p) - \sum_{i=1}^k \frac{n_i}{n_p} \text{Entropy}(c_i)$$
   Độ giảm chỉ số Gini ($\Delta \text{Gini}$) được tính toán hoàn toàn tương tự:
   $$\Delta \text{Gini}(p, \text{split}) = \text{Gini}(p) - \sum_{i=1}^k \frac{n_i}{n_p} \text{Gini}(c_i)$$

#### 2.2.3. Minh họa Tính toán Thủ công Phép phân nhánh theo Gini
Để kiểm chứng tính chính xác của công thức, nhóm thực hiện một ví dụ số học minh họa:
- Xét nút cha $p$ gồm $N = 10$ bản ghi, trong đó có $5 \text{ mẫu Churn=Yes}$ và $5 \text{ mẫu Churn=No}$.
  $$\text{Gini}(p) = 1 - \left[\left(\frac{5}{10}\right)^2 + \left(\frac{5}{10}\right)^2\right] = 1 - (0.25 + 0.25) = 0.500$$
- Giả sử một phép kiểm tra thuộc tính nhị phân chia nút cha thành 2 nút con $c_1$ và $c_2$:
  - Nút con $c_1$ gồm 5 mẫu: $4 \text{ Yes}$ và $1 \text{ No}$.
    $$\text{Gini}(c_1) = 1 - \left[\left(\frac{4}{5}\right)^2 + \left(\frac{1}{5}\right)^2\right] = 1 - (0.64 + 0.04) = 0.320$$
  - Nút con $c_2$ gồm 5 mẫu: $1 \text{ Yes}$ và $4 \text{ No}$.
    $$\text{Gini}(c_2) = 1 - \left[\left(\frac{1}{5}\right)^2 + \left(\frac{4}{5}\right)^2\right] = 1 - (0.04 + 0.64) = 0.320$$
- Tổng chỉ số Gini có trọng số của phép phân chia:
  $$\text{Gini}_{\text{split}} = \frac{5}{10} \times 0.320 + \frac{5}{10} \times 0.320 = 0.320$$
- Độ giảm Gini (Gini Gain):
  $$\Delta \text{Gini} = \text{Gini}(p) - \text{Gini}_{\text{split}} = 0.500 - 0.320 = 0.180$$
Phép phân tách làm giảm độ hỗn tạp đi 0.180 đơn vị, chứng minh chất lượng phân tách vượt trội so với nút ban đầu.

#### 2.2.4. Phân định Thuật toán CART và Thư viện scikit-learn
- Trong giáo trình có đề cập đến các thuật toán ID3, C4.5 và CART.
- Thuật toán **ID3** phân chia đa nhánh trên biến định lượng rời rạc bằng Information Gain; **C4.5** cải tiến bằng Gain Ratio và hỗ trợ thuộc tính liên tục.
- Thư viện `scikit-learn` sử dụng thuật toán **CART (Classification and Regression Trees)** tối ưu hóa. Cây sinh ra trong scikit-learn **luôn luôn là cây nhị phân**. Kể cả khi người dùng chọn tham số `criterion="entropy"`, thuật toán vẫn thực hiện phân nhánh nhị phân bằng tiêu chuẩn Entropy chứ **không biến thành C4.5 hay ID3**. Nhóm ghi nhận chuẩn xác đặc điểm kiến trúc này.

#### 2.2.5. Kiểm soát Quá khớp trên Cây quyết định
Để ngăn chặn cây phát triển quá sâu dẫn đến học thuộc lòng dữ liệu, nhóm sử dụng hai cơ chế kiểm soát:
1. **Dừng sớm (Pre-pruning):** Thiết lập các siêu tham số `max_depth` (độ sâu tối đa), `min_samples_split` (số mẫu tối thiểu để phân tách) và `min_samples_leaf` (số mẫu tối thiểu tại một nút lá).
2. **Tỉa cành sau huấn luyện (Post-pruning):** Áp dụng kỹ thuật tỉa cành theo độ phức tạp chi phí (Minimal Cost-Complexity Pruning) thông qua tham số $\alpha$ (`ccp_alpha`), tối thiểu hóa hàm:
   $$R_\alpha(T) = R(T) + \alpha |T|$$
   giúp loại bỏ các nhánh cây con có đóng góp nhỏ vào khả năng phân lớp tổng thể.

---

### 2.3. MÔ HÌNH HỒI QUY LOGISTIC (LOGISTIC REGRESSION)

Mô hình Hồi quy Logistic là đại diện tiêu biểu cho phương pháp phân loại tham số tuyến tính, được bổ sung vào đề tài nhằm làm mốc đối chứng khoa học với Cây quyết định.

#### 2.3.1. Bản chất Phân loại và Hàm Sigmoid
Mặc dù chứa thuật ngữ "Regression", mô hình này thực chất là một bộ phân lớp xác suất. Mô hình giả định xác suất hậu nghiệm của lớp rời bỏ tuân theo hàm phi tuyến Sigmoid của một tổ hợp tuyến tính các thuộc tính đầu vào:
$$P(y = 1 \mid x) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
$$z = w^T x + b = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b$$
Hàm Sigmoid có miền giá trị chặt chẽ trong khoảng $(0, 1)$, cho phép mô hình ánh xạ bất kỳ giá trị thực nào của $z$ thành một xác suất có ý nghĩa toán học.

#### 2.3.2. Hàm mất mát Log-loss và Điều chuẩn L2
Để tìm bộ trọng số tối ưu $(w, b)$, mô hình tối thiểu hóa hàm mất mát Cross-Entropy nhị phân có thành phần phạt điều chuẩn $L_2$ (Ridge Regularization):
$$J(w, b) = - \frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right] + \frac{1}{2C} \|w\|_2^2$$
Trong đó:
- $\hat{p}_i = \sigma(w^T x_i + b)$.
- $C > 0$ là siêu tham số kiểm soát nghịch đảo cường độ điều chuẩn. Giá trị $C$ nhỏ tăng cường phạt các trọng số có độ lớn bất thường, giúp mô hình đơn giản hơn và chống quá khớp hiệu quả.

#### 2.3.3. Yêu cầu Chuẩn hóa Dữ liệu và Diễn giải Hệ số
- **Sự cần thiết của StandardScaler:** Do thuật toán tối ưu hóa dựa trên Gradient Descent và hàm mục tiêu chứa số hạng phạt độ lớn trọng số $\|w\|_2^2$, nếu các thuộc tính không có cùng thang đo (ví dụ: `TotalCharges` có thể lên tới 8.000 trong khi `SeniorCitizen` chỉ là 0 hoặc 1), mô hình sẽ cập nhật trọng số bị lệch và phạt không công bằng. Do đó, tất cả các thuộc tính số liên tục bắt buộc phải được quy chuẩn hóa về trung bình 0 và phương sai 1 thông qua `StandardScaler`.
- **Ý nghĩa của hệ số $w_j$:** Mô hình thể hiện mối quan hệ tuyến tính giữa các thuộc tính với hàm Logit (log-odds):
  $$\ln\left(\frac{P(y=1\mid x)}{1 - P(y=1\mid x)}\right) = w^T x + b$$
  Một hệ số $w_j > 0$ cho biết khi đặc trưng $x_j$ tăng lên 1 đơn vị (và các yếu tố khác không đổi), tỷ số chênh rời bỏ sẽ nhân thêm một lượng $e^{w_j}$. Cần lưu ý rằng đây chỉ là **mối tương quan thống kê có điều kiện (conditional association)**, hoàn toàn không suy diễn thành quan hệ nhân quả trong kinh doanh.

---

### 2.4. MÔ HÌNH RỪNG NGẪU NHIÊN (RANDOM FOREST)

Rừng ngẫu nhiên đại diện cho trường phái học máy kết hợp (Ensemble Learning), được xây dựng từ tập hợp nhiều cây quyết định nhằm khắc phục triệt để nhược điểm phương sai cao của một cây đơn lẻ.

#### 2.4.1. Nguyên lý Đóng bao (Bagging) và Không gian Thuộc tính con
Random Forest kết hợp hai kỹ thuật ngẫu nhiên hóa chính:
1. **Bootstrap Aggregating (Bagging):** Từ tập huấn luyện ban đầu kích thước $N$, thuật toán rút mẫu ngẫu nhiên có hoàn lại để tạo ra $B$ tập dữ liệu huấn luyện con độc lập. Trung bình mỗi tập con chứa khoảng 63.2% các mẫu dữ liệu duy nhất, 36.8% mẫu còn lại tạo thành tập ngoài bao (Out-Of-Bag).
2. **Chọn ngẫu nhiên không gian đặc trưng (Feature Subspace Sampling):** Tại mỗi nút phân nhánh của mỗi cây trong rừng, thuật toán không tìm kiếm trên toàn bộ $d$ đặc trưng mà chỉ chọn ngẫu nhiên một tập con gồm $m = \lfloor\sqrt{d}\rfloor$ đặc trưng để tìm phép phân chia tốt nhất.

#### 2.4.2. Cơ chế Giảm Phương sai Toán học
Xét tập hợp $B$ cây quyết định độc lập, mỗi cây có phương sai $\sigma^2$. Khi kết hợp trung bình dự đoán của $B$ cây:
$$\text{Var}\left(\frac{1}{B} \sum_{b=1}^B T_b(x)\right) = \frac{\sigma^2}{B} \xrightarrow{B \to \infty} 0$$
Trong thực tế, các cây trong rừng có tương quan dương với hệ số $\rho > 0$, khi đó phương sai của toàn bộ rừng là:
$$\text{Var}(\text{Forest}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
Nhờ kỹ thuật chọn ngẫu nhiên tập con đặc trưng tại mỗi nút, các cây trong rừng trở nên đa dạng và giảm mạnh độ tương quan $\rho$. Khi số lượng cây $B$ đủ lớn, số hạng thứ hai tiến về 0, giúp mô hình tổng thể **giảm mạnh phương sai mà vẫn bảo toàn độ chệch thấp**, mang lại năng lực tổng quát hóa vượt trội trên dữ liệu mới.

#### 2.4.3. Siêu tham số Chính và Sự đánh đổi
- Các tham số cấu hình trọng tâm bao gồm: `n_estimators` (số lượng cây trong rừng, thường từ 100 đến 300), `max_depth` (chiều sâu tối đa của mỗi cây), `min_samples_leaf` (mẫu tối thiểu tại lá) và `max_features` (tỷ lệ thuộc tính ngẫu nhiên tại mỗi nút).
- **Đánh đổi kỹ thuật:** Random Forest mang lại độ chính xác và khả năng chống quá khớp cao hơn Cây quyết định đơn lẻ, nhưng chi phí tính toán huấn luyện lớn hơn và mất đi khả năng giải thích trực quan từng nhánh luật (mô hình hộp đen).

---

### 2.5. PHƯƠNG PHÁP LUẬN ĐÁNH GIÁ VÀ THIẾT KẾ THỰC NGHIỆM

#### 2.5.1. Ma trận Nhầm lẫn và Bối cảnh Nghiệp vụ Rời mạng
Ma trận nhầm lẫn (Confusion Matrix) theo chuẩn `scikit-learn` trong đề tài được thiết lập như sau:

$$\begin{array}{c|c|c}
& \textbf{Dự đoán: No (0)} & \textbf{Dự đoán: Yes (1)} \\
\hline
\textbf{Thực tế: No (0)} & \text{TN (True Negative)} & \text{FP (False Positive)} \\
\hline
\textbf{Thực tế: Yes (1)} & \text{FN (False Negative)} & \text{TP (True Positive)} \\
\end{array}$$

Trong bài toán viễn thông:
- $\text{TP}$: Bắt đúng khách hàng thực sự rời mạng $\to$ Giữ chân kịp thời, bảo vệ doanh thu.
- $\text{FN}$: Bỏ sót khách hàng rời mạng $\to$ **Tổn thất nghiêm trọng nhất**, mất khách hàng vào tay đối thủ.
- $\text{FP}$: Cảnh báo nhầm khách hàng trung thành $\to$ Tốn chi phí ưu đãi/chăm sóc không cần thiết.
- $\text{TN}$: Nhận diện đúng khách hàng ở lại $\to$ Không tốn chi phí can thiệp.

#### 2.5.2. Các Chỉ số Đánh giá Hiệu năng

- **Độ chính xác (Accuracy):**
  $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
- **Độ chuẩn xác (Precision):**
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
- **Độ thu hồi / Độ nhạy (Recall):**
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
- **Điểm F1-score (Trung bình điều hòa trên lớp Churn):**
  $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### 2.5.3. Lý giải Khoa học: Tại sao KHÔNG chọn mô hình dựa trên Accuracy?
Trong bộ dữ liệu IBM Telco Churn:
- Số khách không rời bỏ (`No`): 5.174 khách ($73.46\%$).
- Số khách rời bỏ (`Yes`): 1.869 khách ($26.54\%$).
- Tỷ lệ mất cân bằng lớp là xấp xỉ $3 : 1$.

Nếu một mô hình ngây thơ (Dummy Classifier) dự đoán nhãn toàn bộ khách hàng là `No` ($\hat{y}_i = 0$ với mọi $i$):
- $\text{TN} = 5.174, \text{FN} = 1.869, \text{TP} = 0, \text{FP} = 0$.
- Độ chính xác đạt được:
  $$\text{Accuracy} = \frac{5.174}{7.043} \approx 73.46\%$$
- Tuy nhiên: $\text{Recall} = 0.0\%$, $\text{F1-score} = 0.0$. Mô hình này bỏ sót 100% khách hàng rời mạng, hoàn toàn vô dụng cho doanh nghiệp.  
Vì vậy, **$F_1$-score trên lớp dương (`Churn = Yes`) được chọn làm tiêu chuẩn tối thượng** để lựa chọn mô hình.

#### 2.5.4. Các Chỉ số Đánh giá Xếp hạng Xác suất: ROC-AUC và PR-AUC
- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):** Biểu diễn mối quan hệ giữa $\text{TPR}$ (Recall) và $\text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}}$ trên toàn bộ dải ngưỡng xác suất từ 0 đến 1. Điểm số ROC-AUC đo lường năng lực phân tách tổng thể và xếp hạng mức độ nguy cơ của mô hình độc lập với ngưỡng phân loại $\theta = 0.5$.
- **PR-AUC (Precision-Recall Area Under Curve):** Tập trung đánh giá chất lượng phân loại riêng trên lớp thiểu số, không bị thổi phồng bởi số lượng lớn các trường hợp âm tính đúng ($\text{TN}$).

#### 2.5.5. Thiết kế Xác nhận chéo và Nguyên tắc Chống rò rỉ Dữ liệu
Để đảm bảo tính khách quan và khả năng tái lập của kết quả:
1. **Phân chia 80/20 phân tầng (Stratified Holdout Split):** Toàn bộ 7.043 mẫu được tách thành 5.634 mẫu phát triển (80%) và 1.409 mẫu kiểm thử độc lập (20%). Cả hai tập đều duy trì chính xác tỷ lệ 26.54% lớp Churn.
2. **Xác nhận chéo 5 nếp gấp (5-Fold Stratified Cross-Validation):** Tiến hành trên tập 80% phát triển. Báo cáo trung bình và độ lệch chuẩn ($\mu \pm \sigma$) của các chỉ số để lựa chọn mô hình có hiệu năng cao và ổn định nhất.
3. **Bảo vệ tính toàn vẹn của tập Kiểm thử (Test Set Isolation):** Tập test 20% bị khóa hoàn toàn trong toàn bộ giai đoạn thử nghiệm và tinh chỉnh tham số. Quyết định chọn mô hình chiến thắng hoàn toàn dựa trên kết quả xác nhận chéo. Tập test chỉ được sử dụng đúng một lần duy nhất ở cuối quy trình để báo cáo hiệu năng thực tế.
4. **Đóng gói Pipeline chống rò rỉ:** Mọi phép tính toán tiền xử lý (tính trung vị điền thiếu, chuẩn hóa thang đo, mã hóa danh mục) đều được thực hiện nghiêm ngặt bên trong từng nếp gấp huấn luyện thông qua `sklearn.pipeline.Pipeline`, tuyệt đối không tính toán thống kê trên toàn bộ tập dữ liệu trước khi phân chia.