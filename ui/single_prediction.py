"""Màn hình Dự đoán một khách hàng: Form 19 trường theo 4 nhóm, xử lý phụ thuộc, dự đoán và lưu snapshot."""

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
    st.markdown(
        "Nhập thông tin 19 thuộc tính dịch vụ và hợp đồng để ước tính điểm nguy cơ rời bỏ (Churn Risk) "
        "và phân nhóm mức độ ưu tiên can thiệp chăm sóc khách hàng."
    )

    # Try loading bundle
    try:
        bundle = load_model_bundle()
    except Exception as e:
        st.error(f"❌ Không thể nạp mô hình: {e}. Vui lòng kiểm tra thư mục artifacts.")
        return

    with st.form("single_prediction_form"):
        # Nhóm 1: Thông tin khách hàng
        st.subheader("1. Thông Tin Khách Hàng & Nhân Khẩu Học")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            customer_id = st.text_input(
                "Mã khách hàng (Tùy chọn)",
                placeholder="VD: 7590-VHVEG",
                help="Mã này dùng để đối chiếu hồ sơ, không đưa vào vector đặc trưng của mô hình.",
            )
        with c2:
            gender = st.selectbox("Giới tính (gender)", options=["Female", "Male"])
        with c3:
            senior_val = st.selectbox(
                "Khách cao tuổi (SeniorCitizen)",
                options=["Không (0)", "Có (1)"],
                help="Dữ liệu gốc lưu cờ nhị phân 0/1; hệ thống không tự đặt ngưỡng tuổi suy diễn.",
            )
            senior_citizen = 1 if "1" in senior_val else 0
        with c4:
            partner = st.selectbox("Có người phối ngẫu (Partner)", options=["No", "Yes"])

        col_dep, _ = st.columns([1, 3])
        with col_dep:
            dependents = st.selectbox("Có người phụ thuộc (Dependents)", options=["No", "Yes"])

        st.markdown("---")

        # Nhóm 2: Dịch vụ Điện thoại & Internet
        st.subheader("2. Dịch Vụ Điện Thoại & Internet")
        col_phone, col_internet = st.columns(2)
        with col_phone:
            phone_service = st.selectbox(
                "Dịch vụ điện thoại (PhoneService)",
                options=["Yes", "No"],
                help="Nếu chọn 'No', trường MultipleLines sẽ tự động ràng buộc là 'No phone service'.",
            )
            if phone_service == "No":
                multiple_lines = "No phone service"
                st.info("ℹ️ PhoneService='No' ➜ MultipleLines tự động: **No phone service**")
            else:
                multiple_lines = st.selectbox(
                    "Nhiều đường dây (MultipleLines)",
                    options=["No", "Yes"],
                )

        with col_internet:
            internet_service = st.selectbox(
                "Dịch vụ Internet (InternetService)",
                options=["DSL", "Fiber optic", "No"],
                help="Nếu chọn 'No', tất cả 6 dịch vụ GTGT phụ thuộc sẽ tự động chọn 'No internet service'.",
            )

        st.markdown("**Các dịch vụ gia tăng Internet (Add-on Services):**")
        col_a1, col_a2, col_a3 = st.columns(3)
        col_a4, col_a5, col_a6 = st.columns(3)

        if internet_service == "No":
            st.info("ℹ️ InternetService='No' ➜ Tất cả 6 dịch vụ phụ tự động nhận giá trị: **No internet service**")
            online_sec = "No internet service"
            online_bk = "No internet service"
            dev_prot = "No internet service"
            tech_sup = "No internet service"
            stream_tv = "No internet service"
            stream_mov = "No internet service"
        else:
            with col_a1:
                online_sec = st.selectbox("Bảo mật mạng (OnlineSecurity)", options=["No", "Yes"])
            with col_a2:
                online_bk = st.selectbox("Sao lưu trực tuyến (OnlineBackup)", options=["No", "Yes"])
            with col_a3:
                dev_prot = st.selectbox("Bảo vệ thiết bị (DeviceProtection)", options=["No", "Yes"])
            with col_a4:
                tech_sup = st.selectbox("Hỗ trợ kỹ thuật (TechSupport)", options=["No", "Yes"])
            with col_a5:
                stream_tv = st.selectbox("Truyền hình trực tuyến (StreamingTV)", options=["No", "Yes"])
            with col_a6:
                stream_mov = st.selectbox("Phim trực tuyến (StreamingMovies)", options=["No", "Yes"])

        st.markdown("---")

        # Nhóm 3: Hợp đồng & Thanh toán
        st.subheader("3. Hợp Đồng & Thanh Toán")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            contract = st.selectbox(
                "Loại hợp đồng (Contract)",
                options=["Month-to-month", "One year", "Two year"],
            )
        with col_c2:
            paperless = st.selectbox(
                "Hóa đơn điện tử (PaperlessBilling)",
                options=["Yes", "No"],
            )
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

        st.markdown("---")

        # Nhóm 4: Thời gian sử dụng & Chi phí
        st.subheader("4. Thời Gian Sử Dụng & Chi Phí")
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            tenure = st.number_input(
                "Thời gian sử dụng (Tháng, 0 – 72)",
                min_value=0,
                max_value=72,
                value=12,
                step=1,
            )
        with col_t2:
            monthly_charges = st.number_input(
                "Cước hàng tháng ($)",
                min_value=18.0,
                max_value=130.0,
                value=70.35,
                step=0.05,
            )
        with col_t3:
            missing_total = st.checkbox(
                "Chưa có thông tin Tổng cước (Khách hàng mới)",
                value=False,
                help="Nếu tích chọn, giá trị TotalCharges sẽ để trống (NaN) để bộ tiền xử lý điền giá trị ước lượng từ train, không tự ý gán 0.",
            )
            if missing_total:
                total_charges = np.nan
                st.caption("ℹ️ Đang truyền giá trị thiếu (missing) vào pipeline.")
            else:
                total_charges = st.number_input(
                    "Tổng cước lũy kế ($)",
                    min_value=0.0,
                    max_value=10000.0,
                    value=840.0,
                    step=1.0,
                )

        st.markdown("---")
        submit_predict = st.form_submit_button("🔮 Thực Hiện Dự Đoán", use_container_width=True)

    if submit_predict:
        # Enforce consistency before validation
        if phone_service == "No":
            multiple_lines = "No phone service"
        if internet_service == "No":
            online_sec = "No internet service"
            online_bk = "No internet service"
            dev_prot = "No internet service"
            tech_sup = "No internet service"
            stream_tv = "No internet service"
            stream_mov = "No internet service"

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

        # Validate input using central schema validator
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
            st.metric(
                label="Điểm nguy cơ (Score)",
                value=f"{score:.2%}",
                delta="Nguy cơ cao" if score >= float(res["threshold"]) else "Nguy cơ thấp",
                delta_color="off",
            )
        with res_col2:
            st.metric(
                label="Dự đoán Churn",
                value=f"{label}",
                delta=f"Ngưỡng chuẩn = {res['threshold']}",
                delta_color="off",
            )
        with res_col3:
            badge_icon = "🔴" if priority == "Cao" else "🟡" if priority == "Trung bình" else "🟢"
            threshold_desc = "≥60%" if priority == "Cao" else "30%–60%" if priority == "Trung bình" else "<30%"
            st.metric(
                label="Mức ưu tiên CSKH",
                value=f"{badge_icon} {priority}",
                delta=f"Khoảng điểm: {threshold_desc}",
                delta_color="off",
            )
        with res_col4:
            st.metric(
                label="Phiên bản Model",
                value=res["model_version"],
                delta="Mô hình Champion",
                delta_color="off",
            )

        # Interpretation & Action box
        st.markdown("#### 💡 Khuyến nghị nghiệp vụ chăm sóc khách hàng:")
        if priority == "Cao":
            st.error(
                f"🔴 **Khách hàng thuộc nhóm ưu tiên CAO (Điểm nguy cơ: {score:.2%})!**\n\n"
                "**Hành động can thiệp đề xuất:**\n"
                "- Chuyển ngay hồ sơ cho chuyên viên chăm sóc VIP; liên hệ trong vòng 24–48h.\n"
                "- Tìm hiểu lý do không hài lòng; đề xuất chính sách chiết khấu cước dài hạn hoặc tặng gói cước data/dịch vụ gia tăng.\n"
                "- *Lưu ý:* Điểm nguy cơ thể hiện mức độ cần can thiệp chăm sóc ưu tiên, không khẳng định khách chắc chắn sẽ rời bỏ."
            )
        elif priority == "Trung bình":
            st.warning(
                f"🟡 **Khách hàng thuộc nhóm ưu tiên TRUNG BÌNH (Điểm nguy cơ: {score:.2%})!**\n\n"
                "**Hành động can thiệp đề xuất:**\n"
                "- Gửi email/SMS khảo sát trải nghiệm dịch vụ viễn thông và đường truyền.\n"
                "- Gợi ý các gói cước tích hợp tiết kiệm hơn phù hợp với thói quen sử dụng.\n"
                "- *Lưu ý:* Điểm nguy cơ thể hiện mức độ cần can thiệp chăm sóc ưu tiên, không khẳng định khách chắc chắn sẽ rời bỏ."
            )
        else:
            st.success(
                f"🟢 **Khách hàng thuộc nhóm ưu tiên THẤP (Điểm nguy cơ: {score:.2%}).**\n\n"
                "**Hành động can thiệp đề xuất:**\n"
                "- Tiếp tục duy trì quy trình chăm sóc khách hàng định kỳ tiêu chuẩn.\n"
                "- Tránh spam tin nhắn ưu đãi không cần thiết làm phiền khách hàng.\n"
                "- *Lưu ý:* Điểm nguy cơ thể hiện mức độ cần can thiệp chăm sóc ưu tiên, không khẳng định khách chắc chắn sẽ rời bỏ."
            )

        st.caption(
            "📌 **Quy ước hệ thống (Plan 05 Section 2):** Nhãn Churn='Yes' khi điểm score ≥ 0.50 (ngưỡng quyết định mô hình). "
            "Mức ưu tiên CSKH (Thấp: <0.30, Trung bình: 0.30–0.60, Cao: ≥0.60) là chính sách phân bổ ngân sách CSKH minh họa, "
            "không phải ngưỡng đã được dữ liệu xác nhận."
        )

    # Save to SQLite section
    if "last_single_input" in st.session_state and not st.session_state.get("single_saved", False):
        st.markdown("---")
        save_col1, save_col2 = st.columns([2, 1])
        with save_col1:
            st.markdown(
                "**Lưu hồ sơ và kết quả vào hệ thống:**\n\n"
                "Thao tác này sẽ lưu snapshot đặc trưng bất biến và dự đoán vào cơ sở dữ liệu SQLite để phục vụ truy vết. "
                "*Lưu ý: Hệ thống không tự tạo nhãn thực tế từ kết quả dự đoán.*"
            )
        with save_col2:
            if st.button("💾 Lưu Hồ Sơ & Kết Quả", type="primary", use_container_width=True):
                save_key = f"single_{st.session_state['last_single_output'].iloc[0]['predicted_at']}_{uuid.uuid4().hex[:6]}"
                try:
                    save_res = save_prediction_batch(
                        inputs=st.session_state["last_single_input"],
                        outputs=st.session_state["last_single_output"],
                        source="single_form",
                        idempotency_key=save_key,
                    )
                    st.session_state["single_saved"] = True
                    st.success(
                        f"✅ **Đã lưu thành công!**\n\n"
                        f"- Mã Snapshot: `{save_res['snapshot_ids'][0]}`\n"
                        f"- Mã Dự đoán: `{save_res['prediction_ids'][0]}`"
                    )
                except Exception as e:
                    st.error(f"❌ Lỗi khi lưu vào database: {e}")
