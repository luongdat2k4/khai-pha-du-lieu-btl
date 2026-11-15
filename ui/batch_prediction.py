"""Màn hình Dự đoán hàng loạt qua CSV: Tải mẫu, upload, thẩm định, dự đoán, xếp hạng và lưu vết."""

import os
import uuid
import streamlit as st
import pandas as pd
import numpy as np

from telco_churn.inference import load_model_bundle, predict_records
from telco_churn.storage import save_prediction_batch
from telco_churn.schema import (
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    ID_COLUMN,
    validate_records,
)


def render_batch_prediction():
    st.title("📁 Dự Đoán Theo Lô Qua Tệp Tin CSV")
    st.markdown("Tải lên danh sách nhiều khách hàng qua tệp CSV để dự đoán nguy cơ rời bỏ đồng loạt, phân loại mức ưu tiên và xuất báo cáo.")

    # Try loading bundle
    try:
        bundle = load_model_bundle()
    except Exception as e:
        st.error(f"❌ Không thể nạp mô hình: {e}. Vui lòng kiểm tra thư mục artifacts.")
        return

    # Section: Download sample templates
    col_sample, col_info = st.columns([1, 2])
    with col_sample:
        sample_path = os.path.join("data", "samples", "valid_sample.csv")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                st.download_button(
                    label="📥 Tải Tệp Tin CSV Mẫu Chuẩn",
                    data=f.read(),
                    file_name="telco_churn_sample_valid.csv",
                    mime="text/csv",
                    help="Tệp mẫu gồm 19 cột đặc trưng và customerID, không chứa cột Churn.",
                )
    with col_info:
        st.caption("ℹ️ Tệp tải lên cần chứa đủ 19 cột đặc trưng theo định dạng chuẩn. Cột `customerID` là tùy chọn. Nếu có cột `Churn`, hệ thống sẽ từ chối để tránh rò rỉ nhãn.")

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Chọn tệp tin CSV (kích thước tối đa 10 MB):",
        type=["csv"],
        help="Hỗ trợ mã hóa UTF-8 hoặc ANSI/ASCII chuẩn.",
    )

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Không thể đọc tệp CSV: {e}")
            return

        st.subheader(f"1. Tiếp nhận Dữ liệu: {len(raw_df):,} dòng, {len(raw_df.columns)} cột")

        # Check for Churn column in prediction input
        if TARGET_COLUMN in raw_df.columns:
            st.warning(
                f"⚠️ Phát hiện cột `{TARGET_COLUMN}` trong tệp dự đoán! "
                "Theo quy tắc chống rò rỉ thông tin (Plan 04/05), luồng dự đoán không được sử dụng nhãn thật. "
                "Cột `Churn` sẽ tự động được loại bỏ khỏi đầu vào mô hình. "
                "Nếu bạn muốn ghi nhận nhãn thực tế, vui lòng sử dụng màn hình 'Lịch sử & Nhãn Thực Tế'."
            )
            raw_df = raw_df.drop(columns=[TARGET_COLUMN])

        # Validate DataFrame
        valid_df, issues = validate_records(raw_df, mode="inference")

        if issues:
            st.error(f"⚠️ Phát hiện {len(issues)} vấn đề về dữ liệu trong tệp:")
            issues_df = pd.DataFrame(issues)
            st.dataframe(issues_df, use_container_width=True)

            if valid_df.empty:
                st.error("❌ Không có dòng nào hợp lệ để tiến hành dự đoán. Vui lòng sửa lỗi và tải lại tệp.")
                return
            else:
                st.warning(f"Có {len(valid_df)}/{len(raw_df)} dòng hợp lệ sẵn sàng dự đoán.")
                allow_partial = st.checkbox("Chỉ tiếp tục dự đoán trên các dòng hợp lệ", value=True)
                if not allow_partial:
                    st.info("Đã tạm dừng dự đoán theo yêu cầu của bạn.")
                    return
        else:
            st.success("✅ Toàn bộ các dòng dữ liệu đều vượt qua kiểm tra tính hợp lệ và nhất quán!")

        with st.expander("👁️ Xem trước 5 dòng dữ liệu đầu vào", expanded=False):
            st.dataframe(valid_df.head(), use_container_width=True)

        # Prediction execution
        if st.button("🚀 Bắt Đầu Dự Đoán Theo Lô", type="primary", use_container_width=True):
            with st.spinner("Đang tính toán dự đoán qua pipeline..."):
                results_df = predict_records(valid_df, bundle)

                # Cache in session state
                st.session_state["batch_inputs"] = valid_df
                st.session_state["batch_results"] = results_df
                st.session_state["batch_saved"] = False
                st.session_state["batch_filename"] = uploaded_file.name

    # Display results if available in session_state
    if "batch_results" in st.session_state:
        results_df = st.session_state["batch_results"]
        inputs_df = st.session_state["batch_inputs"]

        st.markdown("---")
        st.subheader("2. Kết Quả Dự Đoán Theo Lô")

        total_records = len(results_df)
        churn_yes_count = int((results_df["churn_prediction"] == 1).sum())
        churn_rate = churn_yes_count / total_records if total_records > 0 else 0.0

        pri_counts = results_df["priority_level"].value_counts().to_dict()
        high_risk_count = pri_counts.get("Cao", 0)
        med_risk_count = pri_counts.get("Trung bình", 0)
        low_risk_count = pri_counts.get("Thấp", 0)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Tổng số bản ghi", f"{total_records:,}")
        with c2:
            st.metric("Dự báo Rời bỏ (Yes)", f"{churn_yes_count:,} ({churn_rate:.1%})")
        with c3:
            st.metric("Ưu tiên Cao (🔴)", f"{high_risk_count:,}")
        with c4:
            st.metric("Ưu tiên TB / Thấp", f"{med_risk_count:,} / {low_risk_count:,}")

        # Combined table for display (ordered by score desc)
        display_df = results_df.copy()
        display_df = display_df.sort_values(by="churn_score", ascending=False)

        st.markdown("#### Bảng Chi Tiết Dự Đoán (Sắp xếp theo Điểm Nguy Cơ giảm dần)")
        st.dataframe(
            display_df[[
                "input_row_id", "customerID", "churn_score", "churn_label",
                "priority_level", "threshold", "model_version", "predicted_at"
            ]],
            use_container_width=True,
        )

        # Download result CSV button
        col_dl, col_db = st.columns([1, 1])
        with col_dl:
            csv_data = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Tải Kết Quả Dự Đoán (CSV UTF-8)",
                data=csv_data,
                file_name=f"churn_predictions_{st.session_state.get('batch_filename', 'batch')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Save to SQLite database
        with col_db:
            if not st.session_state.get("batch_saved", False):
                if st.button("💾 Lưu Lô Dự Đoán Vào Database", type="secondary", use_container_width=True):
                    idempotency_key = f"batch_{st.session_state.get('batch_filename', 'upload')}_{total_records}_{results_df.iloc[0]['predicted_at']}"
                    try:
                        save_res = save_prediction_batch(
                            inputs=inputs_df,
                            outputs=results_df,
                            source="csv_upload",
                            idempotency_key=idempotency_key,
                        )
                        st.session_state["batch_saved"] = True
                        st.success(f"✅ Đã lưu thành công lô gồm {save_res['count']} hồ sơ! Mã lô: `{save_res['batch_id']}`")
                    except Exception as e:
                        st.error(f"❌ Lỗi khi lưu vào database: {e}")
            else:
                st.info("ℹ️ Lô dự đoán này đã được lưu vào cơ sở dữ liệu.")
