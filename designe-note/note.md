Nhóm chọn bài toán phân loại, với đề tài Dự đoán khách hàng rời bỏ dịch vụ viễn thông.
Bài toán :
- Dùng thông tin khách hàng, dịch vụ, hợp đồng và chi phí để dự đoán khách có rời bỏ hay không, kèm điểm nguy cơ.
- Mục tiêu là giúp doanh nghiệp chọn khách hàng cần ưu tiên chăm sóc.
Dữ liệu và cách làm : 
- Dùng IBM Telco Customer Churn, gồm 7.043 khách hàng, 21 cột. Mỗi dòng là một hồ sơ, có nhãn Churn cho biết khách đã rời bỏ hay chưa. link dataset : E:\năm 4\ki 2 năm 4\Khai phá dữ liệu\hoc-lai\telco-customer-churn-on-icp4d\data\Telco-Customer-Churn.csv.
- Xử lý dữ liệu thiếu, trùng và sai kiểu, bỏ mã khách hàng khỏi đầu vào, mã hóa và chuẩn hóa các cột cần thiết.
- Thử 3 mô hình gồm Decision Tree, Logistic Regression và Random Forest.
So sánh kết quả:
- Chia dữ liệu 80/20, dùng xác nhận chéo 5 fold trên phần phát triển.
- So sánh Precision, Recall, F1 và ROC-AUC, ưu tiên F1 của nhóm rời bỏ để chọn mô hình, rồi đánh giá trên tập kiểm thử.
Ứng dụng:
- Dùng Python, scikit-learn và Streamlit. Cho phép nhập một khách hoặc tải CSV rồi xem kết quả dự đoán.
- Lưu hồ sơ mới, cập nhật nhãn thực tế, tiền xử lý và huấn luyện lại hàng tháng khi đủ dữ liệu. So sánh kết quả trước khi cập nhật mô hình.
Phân công 5 người :
- Dữ liệu và tiền xử lý.
- Decision Tree và Logistic Regression.
- Random Forest và so sánh kết quả.
- Giao diện ứng dụng.
- Lưu dữ liệu mới, huấn luyện lại và ghép các phần để demo.