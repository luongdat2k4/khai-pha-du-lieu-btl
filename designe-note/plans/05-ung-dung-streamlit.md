# Plan 05 — Ứng dụng Streamlit

Người chính: TV4. Phối hợp: TV1 cung cấp validation; TV5 cung cấp inference, storage và registry; TV2/TV3 cung cấp model bundle và báo cáo. Mục tiêu: demo trọn luồng người dùng bằng mô hình đã huấn luyện, có thông báo rõ và kết quả có thể truy vết.

## 1. Các màn hình và thứ tự làm

| Ưu tiên | Màn hình | Chức năng |
|---|---|---|
| P0 | Tổng quan | Giới thiệu bài toán, dữ liệu, phiên bản mô hình đang chạy và giới hạn điểm nguy cơ |
| P0 | Dự đoán một khách | Form đủ 19 đầu vào, ID tùy chọn, kiểm tra và hiển thị kết quả |
| P0 | Dự đoán CSV | Tải mẫu, upload, kiểm tra, dự đoán, xếp hạng, tải kết quả |
| P0 | Lịch sử và nhãn thực tế | Xem hồ sơ đã lưu, cập nhật nhãn có nguồn và thời điểm |
| P1 | Kết quả mô hình | Bảng CV, model được chọn, confusion matrix/ROC từ báo cáo thật |
| P1 | Quản lý mô hình | Xem điều kiện retrain, chạy demo dry-run, xem so sánh và phiên bản |
| P2 | Phân tích mở rộng | Bộ lọc, biểu đồ score, mô phỏng threshold, giải thích nâng cao |

P0 và P1 thuộc bản nộp dự kiến; P2 chỉ sau khi luồng chính và thực nghiệm đã ổn định.

## 2. Dự đoán một khách

### Thiết kế form

Chia thành bốn nhóm dễ nhập: thông tin khách hàng; điện thoại/internet; hợp đồng/thanh toán; thời gian sử dụng/chi phí. Dùng radio/selectbox cho danh mục, input số cho số. Nhãn hiển thị tiếng Việt nhưng ánh xạ về giá trị schema gốc.

- `SeniorCitizen` hiển thị “Khách cao tuổi” và chuyển về 0/1; không tự đặt ngưỡng tuổi vì CSV chỉ có cờ.
- `PhoneService=No` thì MultipleLines phải là No phone service.
- `InternetService=No` thì sáu dịch vụ internet phụ phải là No internet service.
- Khi thay đổi dịch vụ chính, cập nhật phụ thuộc nhất quán; kiểm tra lại bằng validator trước inference.
- `TotalCharges` có lựa chọn “chưa có thông tin”, truyền thiếu vào pipeline; không gán 0 thay cho thiếu nếu người dùng không khai báo 0.
- ID tùy chọn khi dự đoán, được cấp ID nội bộ nếu lưu; ID không vào pipeline.

### Kết quả

Hiển thị nhãn, điểm nguy cơ, mức ưu tiên, ngưỡng và model version. Kèm giải thích ngắn về hỗ trợ ưu tiên chăm sóc; không khẳng định khách chắc chắn rời bỏ.

Chính sách hiển thị đề xuất: Thấp nếu score <0,3; Trung bình nếu 0,3 ≤ score <0,6; Cao nếu score ≥0,6. Nhãn dự đoán vẫn theo 0,5. Hai cách phân loại có mục đích khác nhau nên UI phải ghi rõ; các mốc 0,3/0,6 là minh họa nghiệp vụ, không phải ngưỡng đã được dữ liệu xác nhận.

Nút “Lưu hồ sơ và kết quả” thực hiện rõ ràng; trạng thái lưu thành công phải có ID. Không tự tạo nhãn thực tế từ kết quả dự đoán.

## 3. Dự đoán CSV

1. Có CSV mẫu UTF-8 với 19 cột đầu vào và `customerID` tùy chọn; không có Churn.
2. Upload, hiển thị số dòng/cột và preview; giới hạn demo đề xuất 10 MB và 10.000 dòng, đưa vào config.
3. Kiểm tra tên cột, cột trùng, file rỗng, lỗi encoding/định dạng, sai số và danh mục ngoài miền.
4. Thiếu cột bắt buộc thì chặn batch, xuất hướng dẫn sửa. Có cột Churn thì hướng sang cập nhật nhãn thay vì dùng làm feature.
5. Nếu chỉ một số dòng lỗi, hiển thị bảng lỗi và CSV lỗi; mặc định chặn dự đoán toàn batch. Có thể cho chọn “chỉ dự đoán dòng hợp lệ” sau khi người dùng thấy số dòng bị loại.
6. Dự đoán các dòng hợp lệ một lần, giữ input_row_id; không trộn thứ tự khi ghép đầu vào và đầu ra.
7. Hiển thị tổng số, số nhãn Yes, phân bố mức ưu tiên và bảng có thể sắp xếp score giảm dần.
8. Cho tải kết quả CSV kèm ID, score, label, threshold, version; định dạng UTF-8 phù hợp đọc tiếng Việt.
9. Lưu batch nếu người dùng chọn; chống lưu lặp do rerun hoặc bấm hai lần.

Các cột thừa khác được liệt kê và chỉ bỏ sau thông báo rõ. Không tự map cột sai tên gần giống vì có thể đổi nghĩa dữ liệu.

## 4. Lịch sử và cập nhật nhãn

- Lọc theo customerID, ngày, batch, version và trạng thái có/chưa có nhãn thật.
- Xem snapshot đầu vào và dự đoán gốc, không thay bằng hồ sơ đã chỉnh sửa sau này.
- Cập nhật `actual_churn` bằng form riêng: chọn Yes/No, ngày quan sát, nguồn và ghi chú.
- Phân biệt “chưa biết” với No. Không gán No tự động cho các khách chưa quay lại cập nhật.
- Cho nhập CSV nhãn theo snapshot_id; xác thực khóa và báo dòng không tìm thấy/trùng/mâu thuẫn.
- Lưu lịch sử sửa nhãn và lý do; không ghi đè xóa dấu vết.
- Tạo hồ sơ mới khi thay đổi đặc trưng để dự đoán lại; không sửa snapshot quá khứ.

## 5. Kết quả và quản lý mô hình

Màn kết quả đọc artifact từ lần thực nghiệm đã lưu; không fit hoặc tính lại test mỗi lần mở trang. Hiển thị CV và test ở hai bảng có nhãn rõ. Không trộn mean CV với điểm test trong cùng cột.

Màn quản lý cho biết ngày retrain gần nhất, số nhãn đủ điều kiện, lý do chưa được train, lịch sử candidate/champion và quyết định. Hành động kích hoạt chỉ dùng mô hình đã qua quy trình Plan 06, không có nút thay thế mù quáng.

Nếu dữ liệu demo là mô phỏng, gắn nhãn ngay trên màn hình. Dữ liệu test gốc không được upload rồi lưu vào kho dữ liệu mới để tạo cảm giác hệ thống đang học thêm.

## 6. Tích hợp và tối ưu vừa đủ

- Dùng cơ chế cache phù hợp để tải bundle một lần theo version; không giữ mô hình cũ khi active version thay đổi.
- Dùng form/session state để tránh inference/lưu DB lại mỗi rerun.
- Khi retrain, chạy luồng quản lý tách khỏi tác vụ dự đoán; mô hình đang active vẫn phục vụ đến khi candidate đạt.
- Biểu đồ chọn lọc, không đưa log kỹ thuật dài vào giao diện nhân viên. Thông tin phiên bản và giới hạn liên quan quyết định vẫn phải hiển thị.
- Giữ bản demo cục bộ chạy tốt trước khi nghĩ đến deploy.

## 7. Nghiệm thu

- Nhập một khách và upload chính khách đó cho cùng kết quả trong sai số số học.
- Người dùng biết rõ giá trị nào bị thiếu/sai và cách sửa.
- CSV kết quả giữ đúng số dòng hợp lệ và liên kết được với dòng nguồn.
- Nhãn/điểm không thay đổi vì bấm lưu hoặc rerun.
- Lưu được, đọc lại được, cập nhật nhãn thật được và truy vết version được.
- Khi thiếu model/DB lỗi không báo thành công giả; mô hình mới active được tải đúng.
