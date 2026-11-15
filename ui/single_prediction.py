"""Màn hình Dự đoán một khách hàng: Form 19 trường, xử lý phụ thuộc, dự đoán và lưu snapshot."""

import os
import uuid
import streamlit as st
import pandas as pd
import numpy as np

from telco_churn.inference import load_model_bundle, predict_records
from telco_churn.storage import save_prediction_batch
from telco_churn.schema import (
    CATEGORICAL_DOMAINS,
    INTERNET_ADDON_SERVICES,
    validate_records,
)


def render_single_prediction():
    st.title("👤 Dự Đoán Nguy Cơ Rời Bỏ Cho Một Khách Hàng")
    st.markdown("Nhập thông tin 19 thuộc tính dịch vụ và hợp đồng để dự đoán xác suất khách hàng rời bỏ dịch vụ.")

    # Try loading bundle
    try:
        bundle = load_model_bundle()
    except Exception as e:
        st.error(f"❌ Không thể nạp mô hình: {e}. Vui lòng kiểm tra thư mục artifacts.")
        return

    with st.form("single_prediction_form"):
        st.subheader("1. Định danh & Nhân khẩu học")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            customer_id = st.text_input("Mã khách hàng (Tùy chọn)", placeholder="VD: 7590-VHVEG")
        with col2:
            gender = st.selectbox("Giới tính", options=["Female", "Male"])
        with col3:
            senior_val = st.selectbox("Khách hàng cao tuổi", options=["Không (0)", "Có (1)"])
            senior_citizen = 1 if "1" in senior_val else 0
        with col4:
            partner = st.selectbox("Có người phối ngẫu", options=["No", "Yes"])

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            dependents = st.selectbox("Có người phụ thuộc", options=["No", "Yes"])
        with col6:
            tenure = st.number_input("Thời gian sử dụng (Tháng)", min_value=0, max_value=72, value=12, step=1)
        with col7:
            monthly_charges = st.number_input("Cước hàng tháng ($)", min_value=18.0, max_value=130.0, value=70.35, step=0.05)
        with col8:
            has_total_charges = st.checkbox("Có thông tin Tổng cước", value=True)
            if has_total_charges:
                total_charges = st.number_input("Tổng cước lũy kế ($)", min_value=0.0, max_value=10000.0, value=840.0, step=1.0)
            else:
                total_charges = np.nan

        st.subheader("2. Dịch vụ Viễn thông & Internet")
        col_phone, col_internet = st.columns(2)
        with col_phone:
            phone_service = st.selectbox("Dịch vụ điện thoại (PhoneService)", options=["Yes", "No"])
            if phone_service == "No":
                multiple_lines = "No phone service"
                st.info("ℹ️ PhoneService='No' ➜ MultipleLines tự động chọn: **No phone service**")
            else:
                multiple_lines = st.selectbox("Nhiều đường dây (MultipleLines)", options=["No", "Yes"])

        with col_internet:
            internet_service = st.selectbox("Dịch vụ Internet (InternetService)", options=["DSL", "Fiber optic", "No"])

        st.markdown("**Các dịch vụ gia tăng Internet (Add-on Services):**")
        col_a1, col_a2, col_a3 = st.columns(3)
        col_a4, col_a5, col_a6 = st.columns(3)

        if internet_service == "No":
            st.info("ℹ️ InternetService='No' ➜ Tất cả 6 dịch vụ phụ tự động chọn: **No internet service**")
            online_sec = "No internet service"
            online_bk = "No internet service"
            dev_prot = "No internet service"
            tech_sup = "No internet service"
            stream_tv = "No internet service"
            stream_mov = "No internet service"
        else:
            with col_a1:
                online_sec = st.selectbox("Bảo mật trực tuyến (OnlineSecurity)", options=["No", "Yes"])
            with col_a2:
                online_bk = st.selectbox("Sao lưu trực tuyến (OnlineBackup)", options=["No", "Yes"])
            with col_a3:
                dev_prot = st.selectbox("Bảo vệ thiết bị (DeviceProtection)", options=["No", "Yes"])
            with col_a4:
                tech_sup = st.selectbox("Hỗ trợ kỹ thuật (TechSupport)", options=["No", "Yes"])
            with col_a5:
                stream_tv = st.selectbox("Truyền hình TV (StreamingTV)", options=["No", "Yes"])
            with col_a6:
                stream_mov = st.selectbox("Phim trực tuyến (StreamingMovies)", options=["No", "Yes"])

        st.subheader("3. Hợp đồng & Phương thức Thanh toán")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            contract = st.selectbox("Loại hợp đồng (Contract)", options=["Month-to-month", "One year", "Two year"])
        with col_c2:
            paperless = st.selectbox("Hóa đơn điện tử (PaperlessBilling)", options=["Yes", "No"])
        with col_c3:
            payment_method = st.selectbox(
                "Phương thức thanh toán (PaymentMethod)",
                options=[
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        submit_predict = st.form_submit_button("🔮 Thực hiện Dự đoán", use_container_width=True)

    if submit_predict:
        # Prepare single record DataFrame
        record_dict = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_bk,
            "DeviceProtection": dev_prot,
            "TechSupport": tech_sup,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_mov,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
        if customer_id and customer_id.strip():
            record_dict["customerID"] = customer_id.strip()

        input_df = pd.DataFrame([record_dict])

        # Validate input
        valid_df, issues = validate_records(input_df, mode="inference")
        if issues:
            st.error("❌ Phát hiện lỗi trong thông tin nhập vào:")
            for iss in issues:
                st.write(f"- Cột `{iss.get('column')}`: {iss.get('message')}")
            return

        # Perform inference
        results_df = predict_records(valid_df, bundle)
        res = results_df.iloc[0]

        # Store in session state for saving
        st.session_state["last_single_input"] = valid_df
        st.session_state["last_single_output"] = results_df
        st.session_state["single_saved"] = False

        st.markdown("---")
        st.subheader("🎯 Kết Quả Dự Đoán")

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)

        score = float(res["churn_score"])
        priority = res["priority_level"]
        label = res["churn_label"]

        with res_col1:
            st.metric("Điểm nguy cơ (Score)", f"{score:.2%}")
        with res_col2:
            st.metric("Dự đoán Churn", f"{label} (Threshold={res['threshold']})")
        with res_col3:
            if priority == "Cao":
                st.error(f"🔴 Mức ưu tiên: **{priority}**")
            elif priority == "Trung bình":
                st.warning(f"🟡 Mức ưu tiên: **{priority}**")
            else:
                st.success(f"🟢 Mức ưu tiên: **{priority}**")
        with res_col4:
            st.metric("Phiên bản Model", res["model_version"])

        # Interpretation & Action box
        if priority == "Cao":
            st.error(
                f"⚠️ **Khách hàng có nguy cơ rời bỏ CAO ({score:.2%})!**\n\n"
                "Khuyến nghị: Chuyển ngay hồ sơ cho chuyên viên chăm sóc VIP; liên hệ trong vòng 24h; "
                "đề xuất chính sách chiết khấu cước hoặc tặng gói dịch vụ bổ trợ."
            )
        elif priority == "Trung bình":
            st.warning(
                f"⚡ **Khách hàng có nguy cơ rời bỏ TRUNG BÌNH ({score:.2%})!**\n\n"
                "Khuyến nghị: Gửi email khảo sát trải nghiệm dịch vụ; gợi ý nâng cấp gói cước tiết kiệm hơn."
            )
        else:
            st.success(
                f"✅ **Khách hàng có nguy cơ rời bỏ THẤP ({score:.2%}).**\n\n"
                "Khuyến nghị: Tiếp tục duy trì quy trình chăm sóc tiêu chuẩn."
            )

    # Save to SQLite section
    if "last_single_input" in st.session_state and not st.session_state.get("single_saved", False):
        st.markdown("---")
        save_col1, save_col2 = st.columns([2, 1])
        with save_col1:
            st.markdown("Bạn có muốn lưu hồ sơ và kết quả dự đoán này vào cơ sở dữ liệu để theo dõi?")
        with save_col2:
            if st.button("💾 Lưu Hồ Sơ & Kết Quả Vào Database", type="primary", use_container_width=True):
                save_key = f"single_{st.session_state['last_single_output'].iloc[0]['predicted_at']}_{uuid.uuid4().hex[:6]}"
                try:
                    save_res = save_prediction_batch(
                        inputs=st.session_state["last_single_input"],
                        outputs=st.session_state["last_single_output"],
                        source="single_form",
                        idempotency_key=save_key,
                    )
                    st.session_state["single_saved"] = True
                    st.success(f"✅ Đã lưu thành công! Mã Snapshot: `{save_res['snapshot_ids'][0]}`, Mã Dự đoán: `{save_res['prediction_ids'][0]}`")
                except Exception as e:
                    st.error(f"❌ Lỗi khi lưu vào database: {e}")
