# Cơ sở lý thuyết và Phương pháp phân loại dữ liệu

**Dự án:** Khai phá dữ liệu khách hàng viễn thông (IBM Telco Customer Churn)  
**Tài liệu:** Tổng hợp cơ sở lý thuyết, công thức toán học và thiết kế đánh giá  
**Căn cứ:** Giáo trình *Khai phá dữ liệu và Kho dữ liệu* (Học viện Công nghệ Bưu chính Viễn thông) và các tài liệu chuyên ngành học máy chuẩn quốc tế

---

## 1. Bản đồ lý thuyết và Đối chiếu với Giáo trình

Để đảm bảo tính chuẩn mực học thuật khi báo cáo và bảo vệ trước giảng viên, tài liệu này phân định rạch ròi giữa **kiến thức trọng tâm có trong giáo trình** và **các kỹ thuật hiện đại mở rộng** phục vụ yêu cầu thực tế của đề tài:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TỔNG QUAN CƠ SỞ LÝ THUYẾT                       │
├───────────────────────────────────┬────────────────────────────────────┤
│   NỘI DUNG TRONG GIÁO TRÌNH       │   PHẦN MỞ RỘNG THEO YÊU CẦU ĐỀ TÀI │
│   (DMDWv1.2.docx.md)              │   (Scikit-learn, ML Best Practices)│
├───────────────────────────────────┼────────────────────────────────────┤
│ • Chương 1 (Mục 1.1–1.3, tr.8–13):│ • Mô hình Logistic Regression      │
│   Khái niệm KPDL, phân loại,      │   (Cơ chế Sigmoid, Log-loss, L2    │
│   nguyên lý quy nạp.              │   regularization, giải thích hệ số)│
│ • Chương 1 (Mục 1.4, tr.14–16):   │ • Mô hình Random Forest            │
│   8 bước quy trình KDD tổng thể.  │   (Ensemble Bagging, Feature       │
│ • Chương 4 (Mục 4.1, tr.119–131): │   Subspace Sampling, out-of-bag)   │
│   Tiền xử lý: đối tượng/thuộc tính│ • Pipeline tự động hóa scikit-learn│
│   làm sạch, điền thiếu, mã hóa,   │   (Chống rò rỉ dữ liệu giữa folds) │
│   chuẩn hóa, lấy mẫu.             │ • Độ đo ROC-AUC và PR-AUC          │
│ • Chương 4 (Mục 4.3, tr.145–162): │   (Đánh giá khả năng xếp hạng phân │
│   Thuật toán Hunt, Cây quyết định,│   tách xác suất lớp dương)         │
│   Gini, Entropy, Gain, dừng sớm.  │ • Vận hành & Cập nhật mô hình      │
│ • Chương 4 (Đánh giá, tr.162–166):│   (Champion / Candidate framework, │
│   Confusion Matrix, nhược điểm    │   retraining gate, rollback)       │
│   Accuracy, Precision, Recall, F1,│                                    │
│   Holdout, K-Fold Cross-Validation│                                    │
└───────────────────────────────────┴────────────────────────────────────┘
```

---

## 2. Các khái niệm nền tảng trong Khai phá dữ liệu

### 2.1. Khai phá dữ liệu và Nguyên lý quy nạp
Theo giáo trình (Chương 1, Mục 1.3), khai phá dữ liệu phân loại dựa trên **nguyên lý quy nạp (Inductive Learning)**:
- Từ một tập hữu hạn các mẫu quan sát cụ thể có nhãn trước $\mathcal{D}_{\text{train}} = \{(x_i, y_i)\}_{i=1}^N$, thuật toán học tìm kiếm một hàm tổng quát hóa (bộ phân lớp):
  $$f: \mathcal{X} \to \mathcal{Y}$$
  sao cho $f(x)$ dự đoán chính xác nhãn $y$ của các bản ghi chưa từng thấy trong tương lai ($\mathcal{D}_{\text{unseen}}$).
- **Rủi ro của nguyên lý quy nạp (Cạm bẫy quá khớp / Thiên nga đen):** Giáo trình lấy ví dụ về việc quan sát thiên nga trắng tại châu Âu dẫn đến quy nạp sai rằng "mọi thiên nga đều màu trắng". Tương tự, nếu tập mẫu quan sát không mang tính đại diện, hoặc mô hình học quá mức các chi tiết nhiễu ngẫu nhiên của tập huấn luyện, mô hình sẽ có sai số kiểm thử rất lớn (Overfitting). Do đó, việc phân tách độc lập giữa tập huấn luyện và tập kiểm thử là nguyên tắc khoa học bắt buộc.

### 2.2. Quy trình Phát hiện tri thức trong Cơ sở dữ liệu (KDD)
Khai phá dữ liệu không chỉ đơn thuần là việc gọi một hàm huấn luyện thuật toán học máy, mà là một bước cốt lõi nằm trong quy trình KDD 8 giai đoạn khép kín (Fayyad et al., 1996; Giáo trình Chương 1, Mục 1.4):
1. **Học từ lĩnh vực ứng dụng (Application Domain Understanding):** Nắm rõ bài toán kinh doanh viễn thông, sự khác biệt giữa giữ chân khách hàng và tìm khách mới.
2. **Tạo tập dữ liệu đích (Target Dataset Creation):** Chọn lọc các bảng dữ liệu, lọc bỏ định danh (`customerID`), định nghĩa chính xác lớp dương mục tiêu.
3. **Tiền xử lý và Làm sạch dữ liệu (Data Cleaning & Preprocessing):** Chiếm tới 60–70% công sức toàn bộ quy trình; xử lý giá trị thiếu, ép đúng kiểu dữ liệu, khử trùng lặp.
4. **Chuyển đổi dữ liệu (Data Transformation):** Mã hóa biến danh mục (One-Hot Encoding), chuẩn hóa thuộc tính số liên tục, phân chia tập dữ liệu.
5. **Lựa chọn chức năng và thuật toán khai phá (Algorithm Selection):** Chọn hàm mục tiêu phân loại nhị phân và các mô hình tiêu biểu (Cây quyết định, Logistic Regression, Random Forest).
6. **Khai phá dữ liệu (Data Mining):** Tiến hành huấn luyện mô hình, tối ưu hóa siêu tham số bằng xác nhận chéo phân tầng 5 nếp gấp.
7. **Đánh giá và Biểu diễn tri thức (Pattern Evaluation & Knowledge Representation):** Sử dụng các độ đo khoa học (Confusion Matrix, F1-score, ROC-AUC) để phân tích chất lượng mẫu tri thức phát hiện được.
8. **Sử dụng và Tích hợp tri thức (Knowledge Deployment):** Đưa tri thức vào ứng dụng phục vụ người dùng cuối (giao diện Streamlit, cơ sở dữ liệu SQLite theo dõi dự đoán).

---

## 3. Lý thuyết chi tiết của Ba mô hình phân loại

### 3.1. Cây quyết định (Decision Tree) — Mô hình gắn trực tiếp với Giáo trình

#### 3.1.1. Cấu trúc và Thuật toán Hunt tổng quát
Cây quyết định là mô hình biểu diễn phi tham số dạng đồ thị có hướng không chu trình (DAG), gồm:
- **Nút gốc (Root Node):** Nút trên cùng đại diện cho toàn bộ tập dữ liệu ban đầu.
- **Nút trong (Internal Nodes / Decision Nodes):** Chứa điều kiện kiểm tra trên một thuộc tính đầu vào cụ thể.
- **Các nhánh (Branches):** Đại diện cho các kết quả khả dĩ của phép kiểm tra.
- **Nút lá (Leaf Nodes / Terminal Nodes):** Chứa phân phối xác suất các lớp hoặc gán nhãn dự đoán chung cuộc.

Thuật toán nền tảng để xây dựng cây quyết định là **thuật toán Hunt** (Giáo trình Chương 4, Mục 4.3). Cho tập dữ liệu $\mathcal{D}_t$ tại nút $t$:
1. Nếu $\mathcal{D}_t$ chỉ chứa các bản ghi thuộc cùng một lớp $y_k$, thì $t$ trở thành nút lá mang nhãn $y_k$.
2. Nếu $\mathcal{D}_t$ là tập rỗng, $t$ trở thành nút lá mang nhãn của lớp chiếm đa số ở nút cha.
3. Nếu $\mathcal{D}_t$ chứa các bản ghi thuộc nhiều lớp khác nhau, áp dụng chiến lược tham lam (Greedy Strategy) để chọn thuộc tính và điều kiện kiểm tra phân tách $\mathcal{D}_t$ thành các tập con con $\{\mathcal{D}_{t, 1}, \mathcal{D}_{t, 2}, \dots, \mathcal{D}_{t, k}\}$, sau đó gọi đệ quy thuật toán trên từng tập con.

#### 3.1.2. Các độ đo độ hỗn tạp (Impurity Measures)
Mục tiêu của phép phân nhánh là làm cho các nút con có độ thuần nhất (Purity) cao nhất có thể, tức độ hỗn tạp (Impurity) giảm nhiều nhất.

1. **Chỉ số GINI (Gini Impurity):**
   Được sử dụng trong thuật toán CART (Classification and Regression Trees):
   $$\text{Gini}(t) = 1 - \sum_{k=1}^C [p(k \mid t)]^2$$
   Trong đó $p(k \mid t)$ là tỷ lệ bản ghi thuộc lớp $k$ tại nút $t$, và $C$ là số lớp ($C = 2$ trong bài toán nhị phân).
   - Khi nút $t$ hoàn toàn thuần nhất (100% bản ghi thuộc một lớp): $\text{Gini}(t) = 1 - (1^2 + 0^2) = 0$ (Độ hỗn tạp cực tiểu).
   - Khi nút $t$ phân bố đều 50/50 giữa 2 lớp: $\text{Gini}(t) = 1 - (0.5^2 + 0.5^2) = 0.5$ (Độ hỗn tạp cực đại cho bài toán nhị phân).

2. **Độ đo Entropy:**
   Được sử dụng trong thuật toán ID3 và C4.5:
   $$\text{Entropy}(t) = - \sum_{k=1}^C p(k \mid t) \log_2 p(k \mid t)$$
   với quy ước toán học chuẩn $0 \log_2 0 = 0$.
   - Khi nút $t$ thuần nhất: $\text{Entropy}(t) = 0$.
   - Khi nút $t$ phân bố 50/50: $\text{Entropy}(t) = - (0.5 \log_2 0.5 + 0.5 \log_2 0.5) = 1.0$.

3. **Độ lợi thông tin (Information Gain) và Độ giảm Gini (Gini Gain):**
   Khi một nút cha $p$ chứa $n_p$ bản ghi được phân chia thành $k$ nút con $\{c_1, c_2, \dots, c_k\}$ với số bản ghi tương ứng là $\{n_1, n_2, \dots, n_k\}$:
   $$\text{Gain}_{\text{Entropy}}(p, \text{split}) = \text{Entropy}(p) - \sum_{i=1}^k \frac{n_i}{n_p} \text{Entropy}(c_i)$$
   $$\Delta \text{Gini}(p, \text{split}) = \text{Gini}(p) - \sum_{i=1}^k \frac{n_i}{n_p} \text{Gini}(c_i)$$
   Thuật toán sẽ duyệt qua mọi thuộc tính và mọi điểm chia hợp lệ để chọn phép chia có $\Delta \text{Gini}$ (hoặc $\text{Gain}$) lớn nhất.

> [!IMPORTANT]
> **Đính chính lỗi biên dịch/OCR trong tài liệu tham khảo:**  
> Trong bản tài liệu chuyển đổi OCR (trang 156 của giáo trình), có câu văn diễn đạt nhầm lẫn: *"phân nhánh với số lượng bản ghi ở lớp 0 là 5, lớp 1 là 5 được gọi là phân nhánh đồng nhất; phân nhánh 9 và 1 được gọi là không đồng nhất"*.  
> **Khẳng định khoa học chính xác:** Tỷ lệ $5/5$ là trạng thái **hỗn tạp tối đa (Maximum Impurity)**, hoàn toàn không đồng nhất ($\text{Gini} = 0.5$, $\text{Entropy} = 1.0$). Tỷ lệ $9/1$ là trạng thái **thuần nhất cao (High Purity)**, mức độ hỗn tạp rất thấp ($\text{Gini} = 0.18$, $\text{Entropy} \approx 0.469$). Trong báo cáo và bảo vệ, nhóm sử dụng định nghĩa toán học chuẩn tắc này.

#### 3.1.3. Ví dụ tính tay kiểm chứng chỉ số Gini
Để minh chứng tường minh cơ chế hoạt động của thuật toán, xét một ví dụ nhỏ giả định gồm $N = 10$ khách hàng:
- Nút cha $p$ chứa: $5 \text{ Yes}$ ($y = 1$) và $5 \text{ No}$ ($y = 0$).
- Tần suất các lớp: $p(\text{Yes} \mid p) = \frac{5}{10} = 0.5$, $p(\text{No} \mid p) = \frac{5}{10} = 0.5$.
- Chỉ số Gini tại nút cha:
  $$\text{Gini}(p) = 1 - \left( 0.5^2 + 0.5^2 \right) = 1 - (0.25 + 0.25) = 0.500$$

Giả sử một thuộc tính nhị phân (ví dụ: `Contract == 'Month-to-month'`) chia nút cha thành 2 nút con $c_1$ và $c_2$:
- **Nút con trái $c_1$** (gồm 5 bản ghi): $4 \text{ Yes}$ và $1 \text{ No}$ ($n_1 = 5$).
  $$p(\text{Yes} \mid c_1) = \frac{4}{5} = 0.8; \quad p(\text{No} \mid c_1) = \frac{1}{5} = 0.2$$
  $$\text{Gini}(c_1) = 1 - \left( 0.8^2 + 0.2^2 \right) = 1 - (0.64 + 0.04) = 1 - 0.68 = 0.320$$
- **Nút con phải $c_2$** (gồm 5 bản ghi): $1 \text{ Yes}$ và $4 \text{ No}$ ($n_2 = 5$).
  $$p(\text{Yes} \mid c_2) = \frac{1}{5} = 0.2; \quad p(\text{No} \mid c_2) = \frac{4}{5} = 0.8$$
  $$\text{Gini}(c_2) = 1 - \left( 0.2^2 + 0.8^2 \right) = 1 - (0.04 + 0.64) = 1 - 0.68 = 0.320$$

Chỉ số Gini có trọng số sau khi phân chia:
$$\text{Gini}_{\text{split}} = \frac{n_1}{n_p} \text{Gini}(c_1) + \frac{n_2}{n_p} \text{Gini}(c_2) = \frac{5}{10} \times 0.320 + \frac{5}{10} \times 0.320 = 0.160 + 0.160 = 0.320$$

Độ giảm Gini (Gini Gain):
$$\Delta \text{Gini} = \text{Gini}(p) - \text{Gini}_{\text{split}} = 0.500 - 0.320 = 0.180$$
Vì $\Delta \text{Gini} = 0.180 > 0$, phép phân chia này đã cải thiện đáng kể độ thuần nhất và được ưu tiên lựa chọn.

#### 3.1.4. Phân biệt rõ CART với ID3 và C4.5
- **ID3 (Quinlan, 1986):** Xây dựng cây đa nhánh trên thuộc tính rời rạc bằng tiêu chí Information Gain. Nhược điểm: thiên vị các thuộc tính có nhiều giá trị phân biệt (như mã ID).
- **C4.5 (Quinlan, 1993):** Cải tiến ID3 bằng tiêu chí Gain Ratio ($\text{Gain} / \text{SplitInfo}$), hỗ trợ thuộc tính liên tục và xử lý dữ liệu thiếu.
- **CART (Breiman et al., 1984) trong thư viện scikit-learn:** `DecisionTreeClassifier` trong scikit-learn **luôn luôn xây dựng cây nhị phân (Binary Tree)**, tức mỗi nút trong luôn chỉ có đúng 2 nhánh con. Kể cả khi thiết lập `criterion='entropy'`, scikit-learn vẫn chỉ áp dụng công thức Entropy để chọn phép chia đôi nhị phân tốt nhất chứ **không trở thành thuật toán C4.5 hay ID3**. Nhóm ghi rõ điều này trong báo cáo để bảo vệ chuẩn xác về mặt kỹ thuật.

#### 3.1.5. Cơ chế kiểm soát quá khớp (Overfitting Prevention)
Nếu phát triển không giới hạn đến khi mọi nút lá đều có $\text{Gini} = 0$, cây sẽ bị quá khớp nghiêm trọng. Có hai hướng giải quyết:
1. **Dừng sớm (Pre-pruning / Early Stopping):**
   - `max_depth`: Khống chế chiều sâu tối đa của cây.
   - `min_samples_split`: Số lượng mẫu tối thiểu cần có tại một nút để cho phép phân tách tiếp.
   - `min_samples_leaf`: Số lượng mẫu tối thiểu bắt buộc phải có ở một nút lá.
2. **Tỉa cành sau huấn luyện (Post-pruning / Cost-Complexity Pruning):**
   - Sử dụng tham số `ccp_alpha` ($\alpha \ge 0$) để tối thiểu hóa hàm mục tiêu chi phí - độ phức tạp:
     $$R_\alpha(T) = R(T) + \alpha |T|$$
     trong đó $R(T)$ là tổng sai số phân loại trên các lá, và $|T|$ là số lượng nút lá của cây.

---

### 3.2. Hồi quy Logistic (Logistic Regression) — Mô hình tuyến tính làm mốc đối chứng

#### 3.2.1. Bản chất phân loại và Hàm kích hoạt Sigmoid
Mặc dù tên gọi mang từ "Regression" (xuất phát từ việc mô hình hóa hồi quy cho biến đổi log-odds), Logistic Regression là một mô hình phân loại xác suất tuyến tính:
- Với vector đầu vào $x \in \mathbb{R}^d$, mô hình tính tổ hợp tuyến tính:
  $$z = w^T x + b = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b$$
- Sau đó đưa qua hàm phi tuyến **Sigmoid (Standard Logistic Function)** để nén giá trị $z \in (-\infty, +\infty)$ về khoảng xác suất hợp lệ $[0, 1]$:
  $$P(y = 1 \mid x) = \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-(w^T x + b)}}$$
- Xác suất khách hàng không rời bỏ:
  $$P(y = 0 \mid x) = 1 - P(y = 1 \mid x) = \frac{1}{1 + e^{z}}$$

#### 3.2.2. Tỷ số chênh (Odds) và Log-Odds
Tỷ số chênh (Odds) rời bỏ được định nghĩa:
$$\text{Odds} = \frac{P(y = 1 \mid x)}{1 - P(y = 1 \mid x)} = e^{w^T x + b}$$
Lấy logarit tự nhiên hai vế (hàm Logit):
$$\ln(\text{Odds}) = \ln\left(\frac{P(y=1\mid x)}{1 - P(y=1\mid x)}\right) = w^T x + b$$
Như vậy, Logistic Regression giả định rằng **log-odds của lớp rời bỏ có mối quan hệ tuyến tính với các thuộc tính đầu vào**.

#### 3.2.3. Tối ưu hóa hàm mất mát và Điều chuẩn L2
Mô hình ước lượng bộ trọng số $(w, b)$ bằng phương pháp hợp lý cực đại (Maximum Likelihood Estimation - MLE), tương đương với việc tối thiểu hóa hàm mất mát Log-loss (Binary Cross-Entropy) có số hạng điều chuẩn $L_2$ (Ridge):
$$J(w, b) = - \frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right] + \frac{1}{2C} \|w\|_2^2$$
Trong đó:
- $\hat{p}_i = \sigma(w^T x_i + b)$.
- $C > 0$ là siêu tham số kiểm soát nghịch đảo cường độ điều chuẩn (`Inverse of regularization strength`):
  * $C$ nhỏ $\to$ phạt trọng số mạnh hơn $\to$ giảm phương sai, chống quá khớp.
  * $C$ lớn $\to$ phạt trọng số yếu hơn $\to$ mô hình bám sát dữ liệu huấn luyện hơn.

#### 3.2.4. Sự cần thiết của Tiền xử lý (Scaling & One-Hot Encoding)
- Logistic Regression cực kỳ nhạy cảm với thang đo của biến số. Nếu biến `TotalCharges` có giá trị hàng nghìn trong khi `tenure` chỉ từ 0 đến 72, gradient descent sẽ cập nhật không ổn định và số hạng điều chuẩn $L_2$ sẽ phạt bất công đối với các biến có biên độ nhỏ. Do đó, các biến số liên tục bắt buộc phải được chuẩn hóa qua `StandardScaler`:
  $$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$
- Các biến danh mục bắt buộc phải được chuyển đổi thành biến giả (Dummy/One-Hot Indicator) để đưa vào tổ hợp tuyến tính.

#### 3.2.5. Diễn giải hệ số (Coefficients Interpretation)
- Một hệ số $w_j > 0$ thể hiện: khi giữ nguyên tất cả các biến số khác, sự gia tăng của đặc trưng $x_j$ làm tăng log-odds rời bỏ (tương ứng tăng tỷ số chênh rời bỏ gấp $e^{w_j}$ lần).
- **Cảnh báo khoa học:** Hệ số mô hình chỉ biểu thị mối liên hệ thống kê có điều kiện (Conditional Association), tuyệt đối **không chứng minh mối quan hệ nhân quả (Causality)**. Hiện tượng đa cộng tuyến (Multicollinearity) giữa các biến tương quan cao (như `tenure`, `MonthlyCharges` và `TotalCharges`) có thể làm đảo lộn dấu hoặc độ lớn của hệ số.

---

### 3.3. Rừng ngẫu nhiên (Random Forest) — Mô hình tập hợp (Ensemble Learning)

#### 3.3.1. Nguyên lý Đóng bao (Bagging) và Không gian thuộc tính con (Feature Subspace)
Random Forest (Breiman, 2001) là thuật toán học kết hợp (Ensemble Learning) dựa trên nguyên lý Đóng bao ngẫu nhiên (Bootstrap Aggregating - Bagging):
1. **Lấy mẫu Bootstrap:** Cho tập huấn luyện $\mathcal{D}$ gồm $N$ bản ghi, thuật toán sinh ra $B$ tập con huấn luyện $\{\mathcal{D}_1, \mathcal{D}_2, \dots, \mathcal{D}_B\}$, mỗi tập con được tạo bằng cách chọn ngẫu nhiên $N$ bản ghi **có hoàn lại** (with replacement) từ $\mathcal{D}$. Theo lý thuyết xác suất, mỗi mẫu bootstrap chứa xấp xỉ $63.2\%$ số bản ghi duy nhất của $\mathcal{D}$, khoảng $36.8\%$ số bản ghi còn lại không được chọn gọi là mẫu ngoài bao (Out-Of-Bag - OOB).
2. **Ngẫu nhiên hóa không gian đặc trưng (Feature Subspace Sampling):** Trên mỗi tập bootstrap, huấn luyện một cây quyết định sâu độc lập. Điểm đặc thù của Random Forest là tại mỗi nút phân tách của cây, thuật toán không duyệt qua toàn bộ $d$ đặc trưng mà chỉ lấy ngẫu nhiên một tập con gồm $m$ đặc trưng (thường chọn $m = \lfloor\sqrt{d}\rfloor$).

#### 3.3.2. Cơ chế giảm phương sai (Variance Reduction)
Một cây quyết định đơn lẻ có độ sâu lớn thường có độ chệch thấp (Low Bias) nhưng phương sai rất cao (High Variance) — chỉ một biến động nhỏ của dữ liệu huấn luyện cũng làm thay đổi hoàn toàn cấu trúc cây.  
Nếu có $B$ biến ngẫu nhiên độc lập cùng phân phối với phương sai $\sigma^2$, phương sai của trung bình $B$ biến là $\frac{\sigma^2}{B} \to 0$ khi $B \to \infty$.  
Nếu các cây có tương quan với nhau với hệ số tương quan $\rho$, phương sai của tập hợp là:
$$\text{Var}(\text{Ensemble}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
Kỹ thuật lấy mẫu bootstrap kết hợp với việc chọn ngẫu nhiên không gian đặc trưng tại mỗi nút giúp **làm giảm mạnh hệ số tương quan $\rho$ giữa các cây**, nhờ đó phương sai của toàn bộ rừng giảm xuống rõ rệt mà vẫn duy trì được độ chệch thấp.

#### 3.3.3. Dự đoán tổng hợp
- Với bài toán phân loại, dự đoán xác suất lớp rời bỏ của rừng là trung bình cộng xác suất dự đoán từ $B$ cây thành viên:
  $$\hat{p}(y = 1 \mid x) = \frac{1}{B} \sum_{b=1}^B \hat{p}_b(y = 1 \mid x)$$
- Nhãn phân loại chung cuộc được xác định theo cơ chế bỏ phiếu đa số (Majority Voting) hoặc so sánh xác suất trung bình với ngưỡng $\theta = 0.5$.

#### 3.3.4. Các siêu tham số cốt lõi
- `n_estimators`: Số lượng cây trong rừng ($B$, thường chọn 100 đến 300; tăng $B$ không gây quá khớp nhưng tăng chi phí tính toán).
- `max_depth`: Chiều sâu tối đa của mỗi cây.
- `min_samples_leaf`: Số lượng mẫu tối thiểu tại nút lá nhằm làm mượt các dự đoán biên.
- `max_features`: Số lượng đặc trưng ngẫu nhiên xét tại mỗi nút (`'sqrt'`, `'log2'`).

#### 3.3.5. Đánh đổi giữa Độ chính xác và Khả năng giải thích
Mặc dù Random Forest thường có độ chính xác và khả năng tổng quát hóa vượt trội hơn một cây đơn lẻ, mô hình này mang bản chất "hộp đen" (Black-box Model). Việc diễn giải quyết định cho từng khách hàng cụ thể không thể trực quan như việc lần theo một chuỗi luật if-else trên một cây duy nhất.

---

## 4. Lý thuyết Đánh giá Mô hình và Đo lường Nghiệp vụ

### 4.1. Ma trận nhầm lẫn (Confusion Matrix)

Quy ước ma trận nhầm lẫn chuẩn hóa theo định dạng quốc tế và thư viện `scikit-learn` trong bài toán phân loại nhị phân ($y \in \{0, 1\}$):

```
                     Nhãn Dự đoán (Predicted)
                     Class 0 (No)        Class 1 (Yes)
                 ┌───────────────────┬───────────────────┐
  Class 0 (No)   │   TN              │   FP              │
Thực tế          │  (True Negative)  │  (False Positive) │
(Actual)         ├───────────────────┼───────────────────┤
  Class 1 (Yes)  │   FN              │   TP              │
                 │  (False Negative) │  (True Positive)  │
                 └───────────────────┴───────────────────┘
```

> [!NOTE]
> **Đối chiếu quy ước với Giáo trình:**  
> Giáo trình (trang 162) sắp xếp thứ tự nhãn ngược lại: hàng 1 là Class Yes, hàng 2 là Class No; dẫn đến các ô vị trí là:
> $$\begin{bmatrix} a (\text{TP}) & b (\text{FN}) \\ c (\text{FP}) & d (\text{TN}) \end{bmatrix}$$
> Trong mã nguồn Python và báo cáo kỹ thuật, nhóm áp dụng chuẩn cấu trúc mảng 2 chiều của `scikit-learn`: `[[TN, FP], [FN, TP]]` và luôn chú thích rõ ràng tên hàng và cột để đảm bảo tính nhất quán tuyệt đối.

### 4.2. Ý nghĩa Nghiệp vụ trong bài toán Rời bỏ Viễn thông

| Thành phần | Thuật ngữ Kỹ thuật | Ý nghĩa Nghiệp vụ Viễn thông | Tác động Tài chính & Vận hành |
|:---:|:---:|---|---|
| **TP** | **True Positive** | Khách hàng thực sự rời mạng và mô hình **phát hiện chính xác**. | Doanh nghiệp nắm bắt cơ hội can thiệp kịp thời bằng các ưu đãi giữ chân; bảo vệ doanh thu định kỳ. |
| **FN** | **False Negative** | Khách hàng rời mạng nhưng mô hình **bỏ sót** (dự đoán nhầm là ở lại). | **Tổn thất nghiêm trọng nhất.** Khách hàng âm thầm chuyển sang đối thủ cạnh tranh; mất vĩnh viễn giá trị vòng đời khách hàng (LTV). |
| **FP** | **False Positive** | Khách hàng trung thành nhưng mô hình **báo động nhầm** là rời mạng. | Lãng phí chi phí chăm sóc (gọi điện, tặng quà không cần thiết); nguy cơ làm phiền khách hàng đang hài lòng. |
| **TN** | **True Negative** | Khách hàng tiếp tục gắn bó và mô hình **xác định đúng**. | Hệ thống vận hành bình thường; không tốn chi phí can thiệp dư thừa. |

**Bất đối xứng chi phí (Cost Asymmetry):** Trong thực tế viễn thông, tổn thất do bỏ sót một khách hàng rời mạng ($FN$) thường lớn hơn rất nhiều so với chi phí gửi thông điệp giữ chân nhầm cho một khách hàng ở lại ($FP$). Vì vậy, việc chỉ theo dõi tỷ lệ phân loại đúng tổng thể là không đầy đủ.

---

### 4.3. Các độ đo Đánh giá hiệu năng

1. **Độ chính xác tổng thể (Accuracy):**
   $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} = \frac{\text{TP} + \text{TN}}{N}$$

2. **Độ chuẩn xác trên lớp rời bỏ (Precision):**
   Đo lường mức độ đáng tin cậy khi mô hình phát cảnh báo rời mạng:
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

3. **Độ nhạy / Thu hồi trên lớp rời bỏ (Recall / Sensitivity / True Positive Rate):**
   Đo lường tỷ lệ bắt trúng khách hàng rời mạng trên thực tế:
   $$\text{Recall} = \text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

4. **Điểm F1 (F1-score trên lớp dương $y = 1$):**
   Trung bình điều hòa (Harmonic Mean) giữa Precision và Recall:
   $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \text{TP}}{2 \text{TP} + \text{FP} + \text{FN}}$$
   Điểm $F_1$ chỉ đạt giá trị cao khi cả Precision và Recall cùng đạt kết quả tốt; phạt nặng trường hợp một trong hai chỉ số bị đánh đổi quá mức.

---

### 4.4. Phân tích Chuyên sâu: Vì sao KHÔNG chọn mô hình theo Accuracy?

Đây là câu hỏi cốt lõi bảo vệ trước hội đồng. Trong tập dữ liệu IBM Telco Churn gồm $N = 7.043$ hồ sơ:
- Số khách không rời bỏ (`Churn = No`): $5.174$ khách ($73.46\%$).
- Số khách rời bỏ (`Churn = Yes`): $1.869$ khách ($26.54\%$).
- Tỷ lệ mất cân bằng lớp xấp xỉ $1 : 2.77$.

**Kịch bản Mô hình ngây thơ (Dummy Classifier):**
Giả sử một bộ phân lớp tầm thường luôn dự đoán nhãn cho tất cả mọi khách hàng là `No` ($\hat{y}_i = 0, \forall i$):
- $\text{TN} = 5.174, \quad \text{FN} = 1.869, \quad \text{TP} = 0, \quad \text{FP} = 0$.
- Độ chính xác:
  $$\text{Accuracy} = \frac{0 + 5.174}{7.043} = 73.46\%$$
- Các chỉ số trên lớp rời bỏ:
  $$\text{Precision} = 0, \quad \text{Recall} = \frac{0}{1.869} = 0.00\%, \quad F_1 = 0.00$$

> [!CAUTION]
> Một mô hình có Accuracy lên tới $73.46\%$ nhưng có $F_1 = 0.0$ và bỏ sót $100\%$ khách hàng rời bỏ là một mô hình hoàn toàn **vô giá trị đối với nghiệp vụ kinh doanh**. Do đó, nhóm thống nhất lấy **$F_1$-score của lớp dương (`Churn = Yes`) làm chỉ số quyết định tiên quyết** để so sánh và lựa chọn mô hình.

---

### 4.5. Độ đo Xếp hạng Xác suất: ROC-AUC và PR-AUC

1. **Đường cong ROC (Receiver Operating Characteristic) và Chỉ số ROC-AUC:**
   - Đồ thị biểu diễn mối quan hệ đánh đổi giữa Tỷ lệ bắt đúng ($\text{TPR} = \text{Recall}$) và Tỷ lệ báo động nhầm trên lớp âm ($\text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}}$) khi quét ngưỡng phân loại $\theta$ từ $0.0$ đến $1.0$.
   - **ROC-AUC (Area Under the ROC Curve):** Diện tích dưới đường cong ROC ($[0.5, 1.0]$). Thể hiện xác suất mà mô hình gán điểm nguy cơ $\hat{p}$ cho một khách hàng rời mạng thực tế cao hơn điểm nguy cơ gán cho một khách hàng ở lại được chọn ngẫu nhiên.
   - **Đặc tính kỹ thuật:** ROC-AUC độc lập với việc lựa chọn một ngưỡng cắt cụ thể và đo lường khả năng xếp hạng phân tách tổng thể của mô hình.

2. **Đường cong Precision-Recall (PR Curve) và PR-AUC (Average Precision - AP):**
   - Đồ thị biểu diễn Precision theo Recall. Khác với ROC, đường cong PR tập trung trực tiếp vào lớp thiểu số (lớp rời mạng) mà không bị chi phối bởi số lượng lớn của lớp âm ($\text{TN}$).
   - Chỉ số Average Precision (AP) là thước đo bổ trợ lý tưởng trong các bài toán dữ liệu mất cân bằng.

---

## 5. Thiết kế Thực nghiệm và Giao thức Chống rò rỉ dữ liệu

Quy trình đánh giá thực nghiệm được thiết kế nghiêm ngặt theo 2 tầng bảo vệ:

```
Tập Dữ liệu Đích (7.043 mẫu)
       │
       ├────────────────────────────────────────┬────────────────────────────────────────┐
       ▼                                        ▼                                        ▼
Tập Phát triển (Development Set) - 80%     (Khóa kín)     Tập Kiểm thử Độc lập (Test Set) - 20%
       │                                                                                 │
       ▼                                                                                 │
Phân tầng 5 Nếp gấp (5-Fold Stratified CV)                                               │
       ├── Fold 1: Train 80% (64%) ── Validate 20% (16%)                                 │
       ├── Fold 2: Train 80% (64%) ── Validate 20% (16%)                                 │
       ├── Fold 3: Train 80% (64%) ── Validate 20% (16%)                                 │
       ├── Fold 4: Train 80% (64%) ── Validate 20% (16%)                                 │
       └── Fold 5: Train 80% (64%) ── Validate 20% (16%)                                 │
                 │                                                                       │
                 ▼                                                                       │
  Tính Mean ± Std của F1 & ROC-AUC                                                       │
                 │                                                                       │
                 ▼                                                                       │
  CHỌN MÔ HÌNH VÔ ĐỊCH (CHAMPION)                                                        │
                 │                                                                       │
                 ▼                                                                       │
Huấn luyện lại trên toàn bộ 80% Train ───────────────────────────────────► Đánh giá 1 lần duy nhất trên Test
```

### 5.1. Phân chia Train/Test 80/20 có phân tầng (Stratified Holdout)
- Sử dụng hàm băm hoặc seed ngẫu nhiên cố định để phân chia $5.634$ mẫu ($80\%$) cho tập phát triển và $1.409$ mẫu ($20\%$) cho tập kiểm thử độc lập.
- Cơ chế phân tầng (Stratified) đảm bảo tỷ lệ lớp rời bỏ trên cả hai tập đều duy trì chính xác ở mức $26.54\%$.

### 5.2. Xác nhận chéo 5 nếp gấp có phân tầng (5-Fold Stratified CV)
- Trên tập phát triển 80%, chia dữ liệu thành 5 phần con độc lập.
- Lần lượt dùng 4 phần để huấn luyện và 1 phần để kiểm tra thẩm định (Validation).
- Báo cáo kết quả dưới dạng giá trị trung bình kèm độ lệch chuẩn: $\mu \pm \sigma$. Mô hình được chọn phải vừa có F1 trung bình cao, vừa có độ lệch chuẩn nhỏ để chứng minh tính ổn định khi gặp phân phối dữ liệu biến thiên.

### 5.3. Nguyên tắc Chống rò rỉ dữ liệu (Data Leakage Protection)
1. **Tuyệt đối không dùng tập Test để chọn mô hình hoặc chỉnh tham số:** Tập Test $20\%$ bị niêm phong hoàn toàn trong suốt quá trình thử nghiệm mô hình và dò tìm siêu tham số. Chỉ sau khi quyết định chọn mô hình được ghi nhận chính thức trong biên bản (`reports/model_selection.md`), mô hình chiến thắng mới được đánh giá một lần duy nhất trên tập Test để ghi nhận năng lực tổng quát hóa thực tế.
2. **Quy tắc đóng gói Pipeline:** Mọi thao tác tính toán thống kê (như tính trung vị điền thiếu của `TotalCharges`, thống kê trung bình/độ lệch chuẩn trong `StandardScaler`, danh mục trong `OneHotEncoder`) **bắt buộc phải được tính toán duy nhất trên tập Train của từng nếp gấp (Fold)** thông qua `sklearn.pipeline.Pipeline`, sau đó mới áp dụng phép biến đổi (`transform`) lên tập Validation hoặc Test. Tuyệt đối không thực hiện tiền xử lý trên toàn bộ tập dữ liệu trước khi chia nếp gấp.