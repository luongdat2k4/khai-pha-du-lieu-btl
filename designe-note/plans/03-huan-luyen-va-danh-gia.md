# Plan 03 — Huấn luyện, so sánh và đánh giá mô hình

Người chính: TV2 làm Decision Tree và Logistic Regression; TV3 làm Random Forest, bộ đánh giá và bảng tổng hợp. TV1 rà soát rò rỉ dữ liệu. Đầu vào: schema, split manifest và pipeline từ Plan 02. Đầu ra: log thực nghiệm, bảng CV, quyết định chọn mô hình, báo cáo test và gói mô hình dùng cho ứng dụng.

## 1. Khóa giao thức trước khi chạy

- Lớp dương: `1 = Churn Yes`; mapping phải xuất hiện trong metadata.
- Cùng 80% phát triển, 20% test và cùng 5 fold phân tầng cho mọi mô hình.
- Tiêu chí chính: trung bình F1 lớp 1 ở ngưỡng 0,5 trên validation folds.
- Chỉ số phụ: precision, recall lớp 1, ROC-AUC, accuracy, thời gian fit; có thể bổ sung AP và Brier score.
- ROC-AUC/AP dùng điểm lớp 1; F1/precision/recall dùng nhãn ở ngưỡng đã công bố.
- Khi mẫu không có dự đoán dương, xử lý chia 0 nhất quán và báo hiện tượng, không bỏ fold khỏi bảng.
- Dùng `random_state=42` ở nơi có ngẫu nhiên; ghi Python, thư viện, cấu hình CPU và thời gian chạy.

Tập test chỉ dùng sau khi đã khóa mô hình, tham số, tiền xử lý và ngưỡng. CV được dùng để lựa chọn nên điểm CV tốt nhất có thể lạc quan; không gọi đó là ước lượng độc lập cuối cùng. Test là phần kiểm chứng độc lập của bài tập này.

## 2. Baseline và chạy thông luồng

1. Chạy DummyClassifier `most_frequent` qua đúng bộ đánh giá. Với lớp đa số No, dự kiến F1 lớp 1 bằng 0; accuracy phải tính theo split thực tế.
2. Chạy một cấu hình đơn giản của DT, LR và RF để phát hiện lỗi schema, chuyển kiểu, solver và scorer.
3. Ghi số mẫu, số cột sau one-hot, thời gian và warning. Không bỏ qua cảnh báo không hội tụ của LR.
4. Hoàn thiện giao thức đo trước khi tuning. Notebook chỉ dùng phân tích; kết quả chính sinh từ mã có thể chạy lại.

## 3. Không gian tham số đề xuất

Đây là cấu hình dự kiến; kiểm tra API theo phiên bản scikit-learn thực tế khi triển khai và khóa môi trường. Không cần tìm kiếm hàng nghìn cấu hình trên bộ dữ liệu nhỏ.

| Mô hình | Cấu hình cố định ban đầu | Tham số tìm kiếm |
|---|---|---|
| Logistic Regression | penalty L2, solver lbfgs, max_iter 2.000; có scaling | `C: [0.01, 0.1, 1, 10, 100]`; `class_weight: [None, balanced]` |
| Decision Tree | random_state 42; không scaling | `criterion: [gini, entropy]`; `max_depth: [3, 5, 8, None]`; `min_samples_leaf: [1, 5, 15]`; `class_weight: [None, balanced]` |
| Random Forest | random_state 42; không scaling | `n_estimators: [200, 400]`; `max_depth: [8, 16, None]`; `min_samples_leaf: [1, 3, 8]`; `max_features: [sqrt, 0.5]`; `class_weight: [None, balanced]` |

- LR: GridSearch 10 cấu hình × 5 fold = 50 lượt fit, chưa tính refit.
- DT: GridSearch 48 cấu hình × 5 fold = 240 lượt fit.
- RF: không gian 72 cấu hình; dùng RandomizedSearch lấy 24 cấu hình với seed cố định = 120 lượt fit.
- Tổng tìm kiếm chính khoảng 410 lượt fit, cộng baseline, chạy sơ bộ và refit. Đo thời gian trên máy thật trước khi mở rộng.
- Chỉ song song ở một tầng để tránh quá tải: nếu search dùng nhiều tiến trình thì RF trong mỗi lượt giới hạn số luồng phù hợp.
- `min_samples_split` hoặc `ccp_alpha` chỉ thêm ở vòng mở rộng có lý do, không mở toàn bộ tổ hợp cùng lúc.

## 4. Quy trình CV và lựa chọn

1. Tạo pipeline riêng cho mỗi họ mô hình nhưng dùng cùng schema và fold.
2. Search đa metric; `refit` theo scorer F1 lớp 1. Lưu kết quả từng cấu hình, điểm từng fold, mean/std và tham số.
3. Với mỗi họ mô hình, lấy cấu hình có mean CV F1 cao nhất. Với các cấu hình bằng nhau trong cùng họ, ưu tiên mô hình đơn giản hơn theo quy tắc được ghi trước.
4. Chọn họ mô hình có mean CV F1 cao nhất. Nếu chênh lệch tuyệt đối dưới 0,005, coi là gần nhau theo chính sách nhóm: ưu tiên recall lớp 1, tiếp đến mô hình dễ giải thích/chạy nhẹ hơn. Đây là quy tắc thực hành, không phải kiểm định thống kê.
5. Ghi quyết định vào `reports/model_selection.md` trước khi mở test, gồm cả lý do khi chọn cấu hình không có F1 tuyệt đối cao nhất theo quy tắc gần nhau.
6. Fit lại pipeline được chọn trên toàn bộ 80% phát triển; giữ nguyên threshold 0,5.

Không chọn mô hình có F1 test cao nhất rồi quay lại ghi rằng đã chọn bằng CV. Nếu cần bảng test cả ba mô hình để trình bày, chỉ sinh sau khi khóa quyết định từ CV, ghi rõ bảng đó mang tính mô tả và không dùng đổi lựa chọn.

### Bảng CV cần có

| Model | Best params | CV precision 1 | CV recall 1 | CV F1 1 mean ± std | CV ROC-AUC | Fit time | Được chọn? |
|---|---|---|---|---|---|---|---|
| Dummy | Điền khi chạy | | | | | | |
| Decision Tree | Điền khi chạy | | | | | | |
| Logistic Regression | Điền khi chạy | | | | | | |
| Random Forest | Điền khi chạy | | | | | | |

Không điền số giả. Ghi số chữ số thập phân thống nhất, ví dụ 4 chữ số cho bảng và 2 chữ số cho phần trăm.

## 5. Đánh giá cuối trên test

Đầu ra bắt buộc:

1. Classification report theo hai lớp và các metric ưu tiên cho lớp 1.
2. Confusion matrix số đếm, hàng thực tế/cột dự đoán với tên lớp rõ ràng; có thể thêm bản chuẩn hóa theo hàng.
3. ROC curve và ROC-AUC từ xác suất lớp 1. PR curve/AP là phần bổ sung hữu ích.
4. Bảng kết quả test kèm threshold, phiên bản, hash dữ liệu và split.
5. Diễn giải FN/FP trong bối cảnh chăm sóc khách hàng; chỉ nhận xét chi phí định tính khi chưa có số tiền thực tế.
6. Đo inference một hồ sơ và một batch nhỏ trên máy demo; ghi cách đo, không khẳng định SLA doanh nghiệp.

Nếu test xấu hơn CV, báo cáo chênh lệch và phân tích nguyên nhân khả dĩ. Chỉ sửa và chạy lại khi có lỗi thực thi/đánh giá xác định được; ghi lịch sử sửa. Nếu thay đổi thiết kế sau khi đã nhìn test, đánh dấu test đã ảnh hưởng phát triển và cần dữ liệu độc lập mới để xác nhận.

## 6. Diễn giải mô hình và tri thức rút ra

- DT: hình cây giới hạn độ sâu hiển thị 3–4, giải thích một đường từ gốc đến lá. Nếu dùng cây rút gọn riêng để minh họa, nói rõ nó không phải mô hình triển khai.
- LR: hệ số sau preprocessing, tên biến one-hot và chiều liên hệ; tránh diễn giải hệ số là tác động nhân quả.
- RF: ưu tiên permutation importance trên validation thuộc phần phát triển; đọc thêm feature importance của cây như thông tin bổ trợ có thiên lệch.
- Với one-hot, tổng hợp/giải thích theo biến gốc khi hợp lý để người xem không nhầm mỗi dummy là một thuộc tính độc lập về nghiệp vụ.
- Chọn vài ca TP/FN/FP trên validation để phân tích lỗi trước khi khóa mô hình. Phân tích trên test sau cùng chỉ để báo cáo, không tuning tiếp.
- Phát biểu dạng “nhóm hợp đồng X có tỷ lệ churn cao hơn trong tập phát triển” kèm số liệu; không suy thành “đổi hợp đồng chắc chắn làm khách ở lại”.

## 7. Điểm nguy cơ, ngưỡng và phần mở rộng

Bản bắt buộc: dùng xác suất mô hình làm điểm xếp hạng, nhãn ở 0,5. UI ghi “điểm nguy cơ do mô hình ước lượng”. Mức ưu tiên chỉ là chính sách hiển thị, xem Plan 05.

Nếu có thời gian tối ưu threshold/calibration, tạo một thí nghiệm riêng: dùng vòng CV bên ngoài và chỉ tuning/calibration trên dữ liệu huấn luyện bên trong từng vòng, hoặc dành validation riêng trong phần phát triển. Chốt toàn bộ bằng dữ liệu phát triển rồi mới đánh giá test. Không tối ưu threshold và công bố F1 trên cùng dữ liệu đó như đánh giá độc lập.

Không bật slider threshold rồi báo kết quả đó là mô hình gốc. Nếu demo slider, gắn nhãn “mô phỏng chính sách”; ngưỡng dùng khi lưu dự đoán phải được lưu cùng phiên bản.

## 8. Gói mô hình và nghiệm thu

`artifacts/models/<model_version>/` gồm:

- `pipeline.joblib`: toàn bộ preprocessing + classifier, chỉ tải artifact do nhóm tạo.
- `metadata.json`: loại mô hình, tham số, mapping nhãn, feature schema, threshold, seed, hash CSV/manifest, phiên bản thư viện, thời điểm train và commit nếu có.
- `cv_summary.json`, `test_metrics.json` và tham chiếu log tìm kiếm.
- Không đưa dữ liệu test làm mẫu vào cơ sở dữ liệu huấn luyện lại.

Điều kiện đạt: người khác chạy lại được và thu được metric khớp trong sai số số học hợp lý; không có ID/nhãn trong features; cả ba mô hình dùng cùng giao thức; quyết định chọn mô hình có trước đánh giá test; artifact dự đoán được qua hợp đồng Plan 04.
