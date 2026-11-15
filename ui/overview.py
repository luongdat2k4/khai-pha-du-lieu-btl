"""Màn hình Tổng quan dự án: Giới thiệu bài toán, dữ liệu, mô hình active và chính sách rủi ro."""

import os
import streamlit as st
from telco_churn.registry import get_active_model_info


def render_overview():
    st.title("📊 Hệ Thống Dự Đoán Khách Hàng Rời Bỏ Dịch Vụ Viễn Thông")
    st.markdown("### *Telco Customer Churn Prediction & Retention Management Platform*")

    st.markdown("---")

    # Badges row
    col1, col2, col3, col4 = st.columns(4)

    active_info = {}
    try:
        active_info = get_active_model_info()
        active_ver = active_info.get("active_version", "N/A")
        active_type = active_info.get("model_type", "N/A")
    except Exception:
        active_ver = "Chưa nạp"
        active_type = "N/A"

    with col1:
        st.metric("Phiên bản Active", active_ver)
    with col2:
        st.metric("Kiến trúc Mô hình", active_type.replace("_", " ").title())
    with col3:
        st.metric("Bộ dữ liệu gốc", "7.043 bản ghi")
    with col4:
        st.metric("Số đặc trưng", "19 đầu vào")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("1. Mục tiêu và Ý nghĩa Nghiệp vụ")
        st.markdown("""
        Trong ngành viễn thông, chi phí thu hút một khách hàng mới thường cao gấp 5–7 lần so với chi phí giữ chân khách hàng hiện hữu. 
        Mục tiêu của hệ thống:
        - **Dự báo nguy cơ rời bỏ (Churn Risk)** của từng thuê bao dựa trên hành vi sử dụng, gói cước và hợp đồng.
        - **Phân nhóm ưu tiên chăm sóc** (Thấp / Trung bình / Cao) giúp đội ngũ CSKH và Retention tối ưu hóa ngân sách và tài nguyên can thiệp.
        - **Cung cấp công cụ vận hành hoàn chỉnh**: Dự đoán đơn lẻ trực quan, xử lý tập tin CSV hàng loạt, lưu vết snapshot dữ liệu và hỗ trợ cập nhật nhãn thực tế phục vụ chu trình kiểm chuẩn mô hình.
        """)

        st.subheader("2. Chính sách Phân nhóm Nguy cơ & Hành động CSKH")
        st.markdown("""
        | Mức ưu tiên | Khoảng điểm Churn Score | Hành động can thiệp đề xuất |
        |---|---|---|
        | 🟢 **Thấp (Low)** | `Score < 0.40` | Duy trì chăm sóc chuẩn; gửi khảo sát hài lòng định kỳ; không spam ưu đãi. |
        | 🟡 **Trung bình (Medium)** | `0.40 ≤ Score < 0.70` | Gửi email thông báo ưu đãi gói data; tư vấn nâng cấp gói cước phù hợp; kiểm tra chất lượng kết nối. |
        | 🔴 **Cao (High)** | `Score ≥ 0.70` | **Ưu tiên cao nhất**: Nhân viên chăm sóc gọi điện trực tiếp; áp dụng chính sách giảm cước hoặc tặng cước tháng. |
        """)
        st.caption("Lưu ý: Ngưỡng quyết định nhãn rời bỏ chuẩn là `Threshold = 0.50`. Phân nhóm 3 mức trên nhằm phục vụ chiến lược can thiệp theo ngân sách.")

    with col_right:
        st.subheader("3. 19 Thuộc tính Đầu vào")
        with st.expander("👤 Thông tin nhân khẩu học (4)", expanded=True):
            st.markdown("- `gender`: Giới tính (Female / Male)\n- `SeniorCitizen`: Khách cao tuổi (0 / 1)\n- `Partner`: Có bạn đời/vợ chồng (Yes / No)\n- `Dependents`: Có người phụ thuộc (Yes / No)")

        with st.expander("📞 Dịch vụ Thoại & Internet (8)", expanded=True):
            st.markdown("""
            - `PhoneService`: Dịch vụ điện thoại (Yes / No)
            - `MultipleLines`: Nhiều đường dây
            - `InternetService`: Dịch vụ Internet (DSL / Fiber optic / No)
            - `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`: 6 dịch vụ GTGT đi kèm
            """)

        with st.expander("💳 Hợp đồng & Thanh toán (4)", expanded=True):
            st.markdown("- `Contract`: Loại hợp đồng (Month-to-month / One year / Two year)\n- `PaperlessBilling`: Hóa đơn điện tử (Yes / No)\n- `PaymentMethod`: Phương thức thanh toán")

        with st.expander("💰 Thời gian & Chi phí (3)", expanded=True):
            st.markdown("- `tenure`: Số tháng gắn bó (0 – 72 tháng)\n- `MonthlyCharges`: Cước phí hàng tháng ($)\n- `TotalCharges`: Tổng cước tích lũy ($)")
