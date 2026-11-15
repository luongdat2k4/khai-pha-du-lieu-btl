"""Màn hình Lịch sử & Cập nhật nhãn thực tế theo Plan 05 Section 4:
Lọc đa tiêu chí (ID, ngày, batch, version, trạng thái nhãn), xem snapshot gốc, cập nhật nhãn đơn lẻ và theo lô qua CSV.
"""

import io
import json
from datetime import datetime, timezone, date
import streamlit as st
import pandas as pd

from telco_churn.storage import (
    get_prediction_history,
    record_actual_label,
    record_actual_labels_batch,
    DEFAULT_DB_PATH,
)


def render_history():
    st.title("📜 Lịch Sử Dự Đoán & Cập Nhật Nhãn Thực Tế")
    st.markdown(
        "Xem lại các hồ sơ và dự đoán đã lưu trong cơ sở dữ liệu SQLite, truy vết snapshot đầu vào bất biến, "
        "và ghi nhận nhãn thực tế (*Actual Churn*) có lưu vết kiểm toán phục vụ đánh giá và chu trình huấn luyện lại."
    )

    # 1. Bộ lọc tìm kiếm
    st.subheader("1. Bộ Lọc Tìm Kiếm Hồ Sơ")
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        cust_filter = st.text_input("Tìm theo Mã Khách Hàng", placeholder="VD: 7590-VHVEG hoặc CUST_")
    with f_col2:
        model_filter = st.text_input("Lọc theo Phiên Bản Model", placeholder="VD: model_001")
    with f_col3:
        batch_filter = st.text_input("Lọc theo Mã Lô (Batch ID)", placeholder="VD: batch_2026...")

    f_col4, f_col5, f_col6 = st.columns(3)
    with f_col4:
        label_status = st.selectbox(
            "Trạng thái Nhãn Thực Tế",
            options=["Tất cả", "Chưa có nhãn", "Đã có nhãn"],
            help="Hệ thống phân biệt rõ 'chưa có nhãn' với nhãn 'No' (vẫn ở lại). Tuyệt đối không tự động gán No cho hồ sơ chưa cập nhật.",
        )
    with f_col5:
        use_date_filter = st.checkbox("Lọc theo khoảng ngày dự đoán", value=False)
        start_iso = None
        end_iso = None
        if use_date_filter:
            date_range = st.date_input("Chọn khoảng ngày:", value=(date(2026, 1, 1), date.today()))
            if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                start_iso = f"{date_range[0].isoformat()}T00:00:00"
                end_iso = f"{date_range[1].isoformat()}T23:59:59"
    with f_col6:
        limit_val = st.number_input("Số lượng bản ghi tối đa", min_value=10, max_value=1000, value=50, step=10)

    has_label_val = None
    if label_status == "Đã có nhãn":
        has_label_val = True
    elif label_status == "Chưa có nhãn":
        has_label_val = False

    # Fetch history
    try:
        history_df = get_prediction_history(
            limit=int(limit_val),
            customer_id=cust_filter.strip() if cust_filter.strip() else None,
            model_version=model_filter.strip() if model_filter.strip() else None,
            batch_id=batch_filter.strip() if batch_filter.strip() else None,
            start_date=start_iso,
            end_date=end_iso,
            has_label=has_label_val,
        )
    except Exception as e:
        st.error(f"❌ Không thể truy vấn lịch sử từ database: {e}")
        return

    st.markdown("---")

    # 2. Danh sách bản ghi
    st.subheader(f"2. Danh Sách Bản Ghi Dự Đoán ({len(history_df)} kết quả)")

    if history_df.empty:
        st.info("ℹ️ Chưa có bản ghi nào phù hợp với bộ lọc trong cơ sở dữ liệu.")
    else:
        disp_cols = [
            "prediction_id",
            "batch_id",
            "snapshot_id",
            "external_customer_id",
            "model_version",
            "churn_score",
            "predicted_label",
            "actual_churn",
            "label_observed_at",
            "label_source",
            "predicted_at",
        ]
        available_cols = [c for c in disp_cols if c in history_df.columns]
        st.dataframe(history_df[available_cols], use_container_width=True)

        # Allow downloading history as CSV
        hist_csv = history_df[available_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Tải Danh Sách Lịch Sử (CSV UTF-8)",
            data=hist_csv,
            file_name="telco_prediction_history.csv",
            mime="text/csv",
        )

        # Inspect specific snapshot details
        st.markdown("#### 🔍 Xem Chi Tiết Snapshot Đầu Vào Bất Biến")
        selected_snap_id = st.selectbox(
            "Chọn Snapshot ID để kiểm tra thông tin 19 đặc trưng gốc:",
            options=history_df["snapshot_id"].tolist(),
            format_func=lambda x: f"{x} (Khách: {history_df.loc[history_df['snapshot_id']==x, 'external_customer_id'].values[0] or 'N/A'})",
        )

        if selected_snap_id:
            row_match = history_df[history_df["snapshot_id"] == selected_snap_id].iloc[0]
            with st.expander(f"📦 Chi tiết Snapshot `{selected_snap_id}`", expanded=True):
                s_c1, s_c2 = st.columns(2)
                with s_c1:
                    st.write(f"- **Mã Dự đoán:** `{row_match.get('prediction_id')}`")
                    st.write(f"- **Mã Lô (Batch ID):** `{row_match.get('batch_id')}`")
                    st.write(f"- **Mô hình:** `{row_match.get('model_version')}`")
                    st.write(f"- **Điểm Churn Score:** `{row_match.get('churn_score'):.2%}` (Dự đoán: `{row_match.get('predicted_label')}`)")
                    actual_txt = "Chưa có nhãn" if pd.isna(row_match.get("actual_churn")) else ("Yes (1)" if row_match.get("actual_churn") == 1 else "No (0)")
                    st.write(f"- **Nhãn thực tế hiện tại:** `{actual_txt}`")
                with s_c2:
                    st.write(f"- **Thời điểm dự đoán:** `{row_match.get('predicted_at')}`")
                    st.write(f"- **Nguồn snapshot:** `{row_match.get('snapshot_source')}`")
                    st.write(f"- **Dữ liệu mô phỏng (Synthetic):** `{'Có' if row_match.get('is_synthetic') else 'Không'}`")
                    st.write(f"- **Ghi chú nhãn:** `{row_match.get('label_note') or 'Không có'}`")

                # Parse features JSON
                features_str = row_match.get("features_json")
                if features_str:
                    try:
                        feat_dict = json.loads(features_str)
                        st.markdown("**19 Đặc trưng đầu vào lúc dự đoán (Immutable):**")
                        feat_df = pd.DataFrame([feat_dict])
                        st.dataframe(feat_df, use_container_width=True)
                    except Exception as e:
                        st.caption(f"Lỗi đọc JSON: {e}")

    st.markdown("---")

    # 3. Cập nhật nhãn thực tế
    st.subheader("3. Cập Nhật Nhãn Thực Tế (Actual Churn)")
    st.markdown(
        "> [!IMPORTANT]\n"
        "> **Nguyên tắc bảo toàn lịch sử (Plan 06 Section 1):** Nhãn thực tế được lưu vào bảng riêng `actual_label_events` "
        "> và liên kết qua `snapshot_id`. Thao tác cập nhật **không bao giờ ghi đè** hay làm biến dạng kết quả dự đoán ban đầu. "
        "> Nếu một khách hàng thay đổi trạng thái hoặc sửa nhãn, hệ thống ghi nhận một sự kiện kiểm toán mới có liên kết hoàn tác."
    )

    tab_single, tab_batch = st.tabs(["📝 Cập nhật một hồ sơ", "📁 Cập nhật hàng loạt qua CSV"])

    with tab_single:
        with st.form("record_actual_label_form"):
            l_col1, l_col2, l_col3 = st.columns(3)
            with l_col1:
                snapshot_id_input = st.text_input("Mã Snapshot ID*", placeholder="snap_xxxx")
            with l_col2:
                actual_churn_select = st.selectbox(
                    "Kết quả thực tế (Actual Churn)*",
                    options=["Yes (Đã rời bỏ - 1)", "No (Vẫn ở lại - 0)"],
                )
                churn_val = 1 if "Yes" in actual_churn_select else 0
            with l_col3:
                observed_date = st.date_input("Ngày quan sát thực tế*")

            l_col4, l_col5 = st.columns(2)
            with l_col4:
                source_input = st.selectbox(
                    "Nguồn thông tin xác nhận*",
                    options=[
                        "CSKH / Tổng đài xác nhận",
                        "Hệ thống thanh toán / Hủy hợp đồng",
                        "Khảo sát khách hàng định kỳ",
                        "Kiểm toán dữ liệu vận hành",
                        "Nhập thủ công demo",
                    ],
                )
            with l_col5:
                note_input = st.text_input("Ghi chú bổ sung", placeholder="VD: Khách chuyển nhà mạng vì giá cước...")

            submit_label = st.form_submit_button("📝 Ghi Nhận Nhãn Thực Tế Cho Snapshot", use_container_width=True)

        if submit_label:
            if not snapshot_id_input or not snapshot_id_input.strip():
                st.error("❌ Vui lòng nhập Mã Snapshot ID.")
            else:
                snap_id = snapshot_id_input.strip()
                observed_iso = f"{observed_date.isoformat()}T00:00:00Z"
                try:
                    res = record_actual_label(
                        snapshot_id=snap_id,
                        label=churn_val,
                        observed_at=observed_iso,
                        source=source_input,
                        note=note_input if note_input else None,
                    )
                    st.success(
                        f"✅ Đã ghi nhận nhãn thành công! Mã sự kiện: `{res['event_id']}` | "
                        f"Nhãn: `{'Yes (1)' if churn_val == 1 else 'No (0)'}`"
                    )
                    if res.get("supersedes_event_id"):
                        st.info(f"ℹ️ Sự kiện này ghi đè (supersedes) sự kiện trước đó: `{res['supersedes_event_id']}` (Lịch sử cũ vẫn được bảo lưu đầy đủ).")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Lỗi khi ghi nhận nhãn: {e}")

    with tab_batch:
        st.markdown(
            "Tải lên tệp CSV chứa danh sách nhãn thực tế thu thập từ CRM hoặc hệ thống thanh toán. "
            "Tệp CSV bắt buộc gồm hai cột: `snapshot_id` và `actual_churn` (chấp nhận 0/1 hoặc No/Yes). "
            "Các cột tùy chọn: `observed_at`, `source`, `note`."
        )

        # Download batch label template button
        sample_label_path = os.path.join("data", "samples", "sample_actual_labels.csv")
        if os.path.exists(sample_label_path):
            with open(sample_label_path, "rb") as f:
                st.download_button(
                    label="📥 Tải Tệp Tin Mẫu Cập Nhật Nhãn CSV",
                    data=f.read(),
                    file_name="sample_actual_labels.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        labels_file = st.file_uploader(
            "Chọn tệp CSV cập nhật nhãn thực tế:",
            type=["csv"],
            key="batch_actual_labels_uploader",
        )

        if labels_file is not None:
            try:
                raw_bytes = labels_file.getvalue()
                labels_df = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8-sig")
            except Exception as e:
                st.error(f"❌ Không thể đọc tệp CSV: {e}")
                labels_df = None

            if labels_df is not None and not labels_df.empty:
                st.write(f"Đã đọc **{len(labels_df)} dòng** từ tệp.")
                st.dataframe(labels_df.head(), use_container_width=True)

                if st.button("🚀 Thực Hiện Cập Nhật Nhãn Hàng Loạt", type="primary", use_container_width=True):
                    with st.spinner("Đang xác thực và ghi nhận nhãn vào hệ thống..."):
                        batch_res = record_actual_labels_batch(
                            records=labels_df,
                            default_source=f"CSV: {labels_file.name}",
                        )

                    st.markdown("#### Kết quả cập nhật:")
                    c_tot, c_succ, c_fail = st.columns(3)
                    with c_tot:
                        st.metric("Tổng số dòng trong tệp", batch_res["total"])
                    with c_succ:
                        st.metric("Cập nhật thành công", batch_res["success_count"])
                    with c_fail:
                        st.metric("Bị lỗi / Từ chối", batch_res["failed_count"])

                    if batch_res["errors"]:
                        st.warning("⚠️ Chi tiết các dòng gặp lỗi (không hợp lệ, không tìm thấy snapshot, hoặc trùng lặp):")
                        st.dataframe(pd.DataFrame(batch_res["errors"]), use_container_width=True)

                    if batch_res["success_count"] > 0:
                        st.success(f"✅ Đã ghi nhận thành công {batch_res['success_count']} sự kiện nhãn thực tế có lưu vết kiểm toán!")
                        if st.button("🔄 Tải lại trang để xem dữ liệu mới"):
                            st.rerun()
