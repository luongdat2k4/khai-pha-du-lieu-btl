"""Màn hình Dự đoán hàng loạt qua CSV theo Plan 05 Section 3:
Tải mẫu, upload, thẩm định, lọc cột thừa, phát hiện rò rỉ nhãn, tải CSV lỗi, xếp hạng và lưu vết.
"""

import os
import io
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

# Demo limits conforming to Plan 05 Section 3
MAX_FILE_SIZE_MB = 10
MAX_BATCH_ROWS = 10000


def render_batch_prediction():
    st.title("📁 Dự Đoán Theo Lô Qua Tệp Tin CSV")
    st.markdown(
        "Tải lên tệp CSV chứa danh sách khách hàng để dự đoán đồng loạt nguy cơ rời bỏ, "
        "phân loại mức độ ưu tiên can thiệp và xuất báo cáo kết quả."
    )

    # Try loading bundle
    try:
        bundle = load_model_bundle()
    except Exception as e:
        st.error(f"❌ Không thể nạp mô hình: {e}. Vui lòng kiểm tra thư mục artifacts.")
        return

    # Section 1: Download sample templates
    st.subheader("1. Tệp Tin CSV Mẫu")
    col_sample1, col_sample2, col_info = st.columns([1, 1, 2])
    with col_sample1:
        sample_path = os.path.join("data", "samples", "valid_sample.csv")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                st.download_button(
                    label="📥 Tải Mẫu CSV Chuẩn (Hợp lệ)",
                    data=f.read(),
                    file_name="telco_churn_sample_valid.csv",
                    mime="text/csv",
                    help="Gồm đủ 19 cột đặc trưng và customerID, không chứa nhãn Churn.",
                    use_container_width=True,
                )
    with col_sample2:
        err_sample_path = os.path.join("data", "samples", "invalid_schema_sample.csv")
        if os.path.exists(err_sample_path):
            with open(err_sample_path, "rb") as f:
                st.download_button(
                    label="🧪 Tải Mẫu CSV Thử Nghiệm Lỗi",
                    data=f.read(),
                    file_name="telco_churn_sample_errors.csv",
                    mime="text/csv",
                    help="Mẫu chứa các lỗi về miền giá trị và tính bất nhất để kiểm thử validator.",
                    use_container_width=True,
                )
    with col_info:
        st.info(
            f"ℹ️ **Quy định tệp tải lên (Plan 05 Section 3):**\n"
            f"- Định dạng: CSV (UTF-8).\n"
            f"- Giới hạn: Tối đa **{MAX_FILE_SIZE_MB} MB** và **{MAX_BATCH_ROWS:,} dòng**.\n"
            f"- Cột bắt buộc: Đủ 19 đặc trưng dịch vụ/hợp đồng. `customerID` là tùy chọn."
        )

    st.markdown("---")

    # Section 2: Upload CSV
    st.subheader("2. Tải Lên Tệp Tin CSV")
    uploaded_file = st.file_uploader(
        "Chọn tệp tin CSV từ máy tính:",
        type=["csv"],
        help=f"Giới hạn dung lượng tối đa {MAX_FILE_SIZE_MB} MB.",
    )

    if uploaded_file is not None:
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"❌ Dung lượng tệp ({file_size_mb:.2f} MB) vượt quá giới hạn cho phép ({MAX_FILE_SIZE_MB} MB)!")
            return

        # Attempt reading CSV
        try:
            raw_bytes = uploaded_file.getvalue()
            # Try utf-8-sig to handle possible Excel BOM
            raw_df = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8-sig")
        except UnicodeDecodeError:
            try:
                raw_df = pd.read_csv(io.BytesIO(raw_bytes), encoding="latin1")
                st.warning("⚠️ Tệp tin sử dụng mã hóa ANSI/Latin-1. Hệ thống đã tự động chuyển đổi sang UTF-8.")
            except Exception as e:
                st.error(f"❌ Lỗi định dạng encoding tệp CSV: {e}")
                return
        except Exception as e:
            st.error(f"❌ Không thể đọc tệp CSV (file có thể bị rỗng hoặc lỗi cấu trúc): {e}")
            return

        if raw_df.empty:
            st.error("❌ Tệp CSV rỗng (không có dòng dữ liệu nào)!")
            return

        if len(raw_df) > MAX_BATCH_ROWS:
            st.error(f"❌ Số dòng dữ liệu ({len(raw_df):,}) vượt quá giới hạn demo ({MAX_BATCH_ROWS:,} dòng)!")
            return

        # Check duplicate column names
        if len(raw_df.columns) != len(set(raw_df.columns)):
            st.error("❌ Tệp CSV chứa các cột bị trùng tên. Vui lòng kiểm tra lại tiêu đề các cột.")
            return

        st.success(f"📁 Đã tiếp nhận tệp **{uploaded_file.name}**: {len(raw_df):,} dòng, {len(raw_df.columns)} cột.")

        # Check for Churn column in input (anti-leakage policy)
        proceed_after_churn = True
        if TARGET_COLUMN in raw_df.columns:
            st.warning(
                f"⚠️ **Phát hiện cột `{TARGET_COLUMN}` trong tệp dữ liệu dự đoán!**\n\n"
                f"Theo nguyên tắc chống rò rỉ thông tin (Plan 04/05), luồng dự đoán tuyệt đối không sử dụng nhãn thật làm đặc trưng. "
                f"Nếu bạn muốn ghi nhận nhãn thực tế phục vụ đánh giá mô hình, vui lòng chuyển sang màn hình **'Lịch Sử & Nhãn Thực Tế'**."
            )
            confirm_drop_churn = st.checkbox(
                f"Tôi hiểu và đồng ý loại bỏ cột `{TARGET_COLUMN}` khỏi đầu vào để tiếp tục dự đoán",
                value=False,
            )
            if not confirm_drop_churn:
                st.info("Vui lòng tích xác nhận phía trên để tiếp tục, hoặc chỉnh sửa lại tệp CSV.")
                return
            raw_df = raw_df.drop(columns=[TARGET_COLUMN])

        # Identify unrecognized extra columns
        known_cols = set(ALL_FEATURE_COLUMNS) | {ID_COLUMN, TARGET_COLUMN}
        extra_cols = [c for c in raw_df.columns if c not in known_cols]
        if extra_cols:
            st.info(
                f"ℹ️ **Thông báo về cột thừa (Plan 05 Section 3):** Phát hiện {len(extra_cols)} cột không thuộc 19 đặc trưng chuẩn: "
                f"`{extra_cols}`. Các cột này sẽ được loại bỏ khỏi vector đầu vào của mô hình."
            )
            raw_df = raw_df.drop(columns=extra_cols)

        # Validate DataFrame
        valid_df, issues = validate_records(raw_df, mode="inference")

        has_issues = len(issues) > 0
        allow_predict = False

        if has_issues:
            st.error(f"⚠️ Phát hiện **{len(issues)} vấn đề** về dữ liệu trong tệp CSV:")
            issues_df = pd.DataFrame(issues)
            st.dataframe(issues_df, use_container_width=True)

            # Generate error rows CSV for download
            err_row_indices = sorted(list(set(iss["row_index"] for iss in issues if "row_index" in iss)))
            if err_row_indices and any(r < len(raw_df) for r in err_row_indices):
                err_rows_df = raw_df.iloc[[r for r in err_row_indices if r < len(raw_df)]].copy()
                err_csv = err_rows_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Tải Danh Sách Các Dòng Dữ Liệu Bị Lỗi (CSV)",
                    data=err_csv,
                    file_name=f"error_rows_{uploaded_file.name}",
                    mime="text/csv",
                    help="Tải các dòng dữ liệu không đạt chuẩn để nhân viên rà soát và chỉnh sửa.",
                )

            if valid_df.empty:
                st.error("❌ Không có dòng dữ liệu nào hợp lệ để tiến hành dự đoán. Vui lòng sửa lỗi và tải lại tệp.")
                return
            else:
                st.warning(
                    f"Có **{len(valid_df)}/{len(raw_df)} dòng hợp lệ** ({len(err_row_indices)} dòng không hợp lệ bị loại bỏ). "
                    "Theo chính sách Plan 05 Section 3, mặc định hệ thống CHẶN dự đoán toàn lô khi phát hiện lỗi."
                )
                allow_partial = st.checkbox(
                    "Tôi muốn chỉ tiếp tục dự đoán trên các dòng hợp lệ đã qua kiểm chuẩn",
                    value=False,
                )
                if allow_partial:
                    allow_predict = True
                else:
                    st.info("Đã chặn dự đoán để bảo đảm tính toàn vẹn dữ liệu. Tích chọn ô trên nếu bạn muốn tiếp tục.")
        else:
            st.success("✅ Toàn bộ các dòng dữ liệu đều vượt qua kiểm tra tính hợp lệ và nhất quán!")
            allow_predict = True

        with st.expander("👁️ Xem trước 5 dòng dữ liệu đầu vào đã chuẩn hóa", expanded=False):
            st.dataframe(valid_df.head(), use_container_width=True)

        # Prediction execution button
        if allow_predict:
            if st.button("🚀 Bắt Đầu Dự Đoán Cho Lô Dữ Liệu", type="primary", use_container_width=True):
                with st.spinner(f"Đang tính toán dự đoán qua pipeline cho {len(valid_df):,} bản ghi..."):
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
        st.subheader("3. Kết Quả Dự Đoán Theo Lô")

        total_records = len(results_df)
        churn_yes_count = int((results_df["churn_prediction"] == 1).sum())
        churn_rate = churn_yes_count / total_records if total_records > 0 else 0.0

        pri_counts = results_df["priority_level"].value_counts().to_dict()
        high_risk_count = pri_counts.get("Cao", 0)
        med_risk_count = pri_counts.get("Trung bình", 0)
        low_risk_count = pri_counts.get("Thấp", 0)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Tổng số bản ghi dự đoán", f"{total_records:,}")
        with c2:
            st.metric("Dự báo Churn = Yes", f"{churn_yes_count:,} ({churn_rate:.1%})")
        with c3:
            st.metric("Ưu tiên Cao (🔴 ≥60%)", f"{high_risk_count:,} ({high_risk_count/total_records:.1%})")
        with c4:
            st.metric("Ưu tiên TB / Thấp", f"{med_risk_count:,} / {low_risk_count:,}")

        st.caption(
            "📌 **Quy ước hệ thống (Plan 05 Section 2):** Nhãn Churn='Yes' khi điểm score ≥ 0.50 (ngưỡng quyết định mô hình). "
            "Mức ưu tiên CSKH (Thấp: <0.30, Trung bình: 0.30–0.60, Cao: ≥0.60) là chính sách phân bổ ngân sách CSKH minh họa."
        )

        # Combined table for display (sorted by score desc)
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
            csv_data = display_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 Tải Kết Quả Dự Đoán (CSV UTF-8)",
                data=csv_data,
                file_name=f"churn_predictions_{st.session_state.get('batch_filename', 'batch')}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Tệp tải xuống chứa đầy đủ ID, điểm số nguy cơ, nhãn dự đoán, mức ưu tiên và phiên bản mô hình.",
            )

        # Save to SQLite database
        with col_db:
            if not st.session_state.get("batch_saved", False):
                if st.button("💾 Lưu Lô Dự Đoán Vào Cơ Sở Dữ Liệu", type="secondary", use_container_width=True):
                    idempotency_key = f"batch_{st.session_state.get('batch_filename', 'upload')}_{total_records}_{results_df.iloc[0]['predicted_at']}"
                    try:
                        save_res = save_prediction_batch(
                            inputs=inputs_df,
                            outputs=results_df,
                            source="csv_upload",
                            idempotency_key=idempotency_key,
                        )
                        st.session_state["batch_saved"] = True
                        st.success(
                            f"✅ **Đã lưu thành công lô gồm {save_res['count']} hồ sơ!**\n\n"
                            f"Mã Lô (Batch ID): `{save_res['batch_id']}`"
                        )
                    except Exception as e:
                        st.error(f"❌ Lỗi khi lưu vào database: {e}")
            else:
                st.info("ℹ️ Lô dự đoán này đã được lưu vào cơ sở dữ liệu (chống lưu trùng lặp thành công).")
