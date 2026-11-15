"""Màn hình Lịch sử & Cập nhật nhãn thực tế: Truy vết dự đoán, kiểm tra snapshot, ghi nhận actual churn."""

from datetime import datetime, timezone
import streamlit as st
import pandas as pd

from telco_churn.storage import (
    get_prediction_history,
    record_actual_label,
    DEFAULT_DB_PATH,
)


def render_history():
    st.title("📜 Lịch Sử Dự Đoán & Cập Nhật Nhãn Thực Tế")
    st.markdown("Xem lại các hồ sơ và dự đoán đã lưu trong SQLite, đồng thời cập nhật nhãn thực tế (*Actual Churn*) có lưu vết kiểm toán phục vụ đánh giá mô hình.")

    # 1. Filters row
    st.subheader("1. Bộ Lọc Tìm Kiếm")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with f_col1:
        cust_filter = st.text_input("Tìm theo Mã Khách Hàng", placeholder="VD: 7590-VHVEG hoặc CUST_")
    with f_col2:
        model_filter = st.text_input("Lọc theo Phiên Bản Model", placeholder="VD: model_001")
    with f_col3:
        label_status = st.selectbox("Trạng thái Nhãn Thực Tế", options=["Tất cả", "Chưa có nhãn", "Đã có nhãn"])
    with f_col4:
        limit_val = st.number_input("Số lượng tối đa hiển thị", min_value=10, max_value=500, value=50, step=10)

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
            has_label=has_label_val,
        )
    except Exception as e:
        st.error(f"❌ Không thể truy vấn lịch sử từ database: {e}")
        return

    st.subheader(f"2. Danh Sách Bản Ghi ({len(history_df)} kết quả)")

    if history_df.empty:
        st.info("ℹ️ Chưa có bản ghi nào phù hợp với bộ lọc trong cơ sở dữ liệu.")
    else:
        # Format display columns
        disp_cols = [
            "prediction_id",
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

    st.markdown("---")

    # 2. Form to record/update actual label
    st.subheader("3. Ghi Nhận / Cập Nhật Nhãn Thực Tế Cho Snapshot")
    st.markdown("""
    > [!NOTE]
    > Theo nguyên tắc bất biến (Plan 06 Section 1), nhãn thực tế được lưu vào bảng riêng `actual_label_events` 
    > và liên kết qua `snapshot_id`. Thao tác này **không bao giờ ghi đè** hay làm biến dạng kết quả dự đoán ban đầu.
    """)

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
            note_input = st.text_input("Ghi chú bổ sung", placeholder="VD: Khách chuyển sang nhà mạng đối thủ do giá cước...")

        submit_label = st.form_submit_button("📝 Ghi Nhận Nhãn Thực Tế", use_container_width=True)

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
                    st.info(f"ℹ️ Sự kiện này ghi đè (supersedes) sự kiện trước đó: `{res['supersedes_event_id']}` (Lịch sử cũ vẫn được bảo lưu).")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Lỗi khi ghi nhận nhãn: {e}")
