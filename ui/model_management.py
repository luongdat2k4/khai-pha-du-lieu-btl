"""Màn hình Quản lý mô hình & Huấn luyện lại: Kiểm tra gate, chạy dry-run, kích hoạt và rollback."""

import os
import json
import streamlit as st
import pandas as pd

from telco_churn.registry import (
    get_active_model_info,
    set_active_model,
    rollback_model,
    list_available_models,
)
from telco_churn.storage import (
    get_snapshots_with_labels,
    get_model_versions_from_db,
)
from telco_churn.retraining import (
    run_retraining,
    load_retraining_config,
    evaluate_gate_conditions,
)


def render_model_management():
    st.title("⚙️ Quản Lý Mô Hình & Chu Trình Huấn Luyện Lại")
    st.markdown("Theo dõi tình trạng mô hình active, kiểm tra các điều kiện cổng an toàn (*Gate*), chạy thử nghiệm huấn luyện candidate (*Dry-run*) và thực hiện hoàn tác (*Rollback*).")

    # 1. Current Active Model
    st.subheader("1. Mô Hình Đang Phục Vụ (Active Champion)")

    try:
        active_info = get_active_model_info()
        active_ver = active_info.get("active_version", "N/A")
        active_type = active_info.get("model_type", "N/A")
        updated_at = active_info.get("updated_at_utc", "N/A")
        prev_ver = active_info.get("previous_version", "Không có")
    except Exception as e:
        st.error(f"❌ Không thể đọc active_model.json: {e}")
        return

    ac1, ac2, ac3, ac4 = st.columns(4)
    with ac1:
        st.metric("Phiên bản Active", active_ver)
    with ac2:
        st.metric("Kiến trúc", active_type)
    with ac3:
        st.metric("Phiên bản trước", prev_ver or "Gốc")
    with ac4:
        st.metric("Cập nhật lúc", updated_at[:19].replace("T", " ") if updated_at != "N/A" else "N/A")

    st.markdown("---")

    # 2. Retraining Gate Checklist
    st.subheader("2. Kiểm Tra Cổng Điều Kiện Huấn Luyện Lại (Monthly Gate Checklist)")
    st.markdown("""
    > [!NOTE]
    > Theo chính sách Plan 06 Section 4, mô hình chỉ được phép bước vào chu trình huấn luyện lại khi thỏa mãn 
    > đồng thời các tiêu chí cổng an toàn về số lượng và cân bằng dữ liệu nhãn thật.
    """)

    cfg = load_retraining_config()
    gate_cfg = cfg.get("gate", {})

    df_labeled = get_snapshots_with_labels(include_synthetic=False)
    gate_passed, gate_details = evaluate_gate_conditions(df_labeled, gate_cfg)

    g_col1, g_col2, g_col3, g_col4 = st.columns(4)
    with g_col1:
        req_snaps = gate_cfg.get("min_new_snapshots", 200)
        curr_snaps = gate_details["total_eligible_snapshots"]
        st.metric("Dữ liệu nhãn mới", f"{curr_snaps} / {req_snaps}")
    with g_col2:
        req_pos = gate_cfg.get("min_pos_labels", 30)
        curr_pos = gate_details["class_1_count"]
        st.metric("Số mẫu Churn=Yes", f"{curr_pos} / {req_pos}")
    with g_col3:
        req_neg = gate_cfg.get("min_neg_labels", 30)
        curr_neg = gate_details["class_0_count"]
        st.metric("Số mẫu Churn=No", f"{curr_neg} / {req_neg}")
    with g_col4:
        if gate_passed:
            st.success("🟢 CỔNG ĐẠT ĐIỀU KIỆN")
        else:
            st.warning("🟡 CHƯA ĐỦ ĐIỀU KIỆN")

    if not gate_passed:
        with st.expander("⚠️ Chi tiết lý do chưa đạt cổng an toàn:", expanded=True):
            for fail in gate_details["failures"]:
                st.write(f"- {fail}")

    st.markdown("---")

    # 3. Interactive Actions: Dry-run and Demo
    st.subheader("3. Thao Tác Huấn Luyện Lại & Kiểm Thử So Sánh")

    act_col1, act_col2 = st.columns(2)

    with act_col1:
        st.markdown("**A. Chạy Thử Nghiệm Kiểm Tra Cổng (Dry-run)**")
        st.caption("Kiểm tra điều kiện cổng và chỉ số candidate mà không thay đổi mô hình đang phục vụ.")
        if st.button("🧪 Chạy Kiểm Tra An Toàn (Dry-run)", use_container_width=True):
            with st.spinner("Đang chạy kiểm tra gate..."):
                res = run_retraining(dry_run=True)
                if res["decision"] == "SKIP":
                    st.warning(f"⏩ Kết quả: **SKIP** — Lý do: {res['reason']}")
                else:
                    st.info(f"Kết quả: **{res['decision']}** — {res['reason']}")

    with act_col2:
        st.markdown("**B. Chạy Mô Phỏng Chu Trình Cập Nhật (Demo Mode)**")
        st.caption("Cho phép bỏ qua ngưỡng 200 mẫu để thử nghiệm huấn luyện candidate và so sánh trên tập holdout.")
        if st.button("🚀 Chạy Mô Phỏng Candidate & So Sánh (Demo)", type="primary", use_container_width=True):
            with st.spinner("Đang huấn luyện candidate và tính toán so sánh trên tập comparison..."):
                res_demo = run_retraining(
                    dry_run=True,
                    ignore_gate=True,
                    allow_synthetic=True,
                )
                st.success("✅ Đã hoàn tất chu trình mô phỏng!")

                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    st.markdown(f"**Quyết định:** `{res_demo['decision']}`")
                    st.write(res_demo["reason"])
                with d_col2:
                    if res_demo.get("candidate_metrics") and res_demo.get("champion_metrics"):
                        champ = res_demo["champion_metrics"]
                        cand = res_demo["candidate_metrics"]
                        st.write(f"- Champion F1: **{champ['f1']:.4f}**")
                        st.write(f"- Candidate F1: **{cand['f1']:.4f}** (Delta: {res_demo.get('delta_f1'):+.4f})")
                        st.write(f"- Candidate Recall: **{cand['recall']:.4f}** (Delta: {res_demo.get('delta_recall'):+.4f})")

    st.markdown("---")

    # 4. Rollback and Model Versions
    st.subheader("4. Lịch Sử Phiên Bản & Hoàn Tác (Rollback)")

    models_list = list_available_models()
    if models_list:
        rows = []
        for m in models_list:
            rows.append({
                "Phiên bản": m.get("model_version"),
                "Kiến trúc": m.get("model_type"),
                "Trạng thái": "🟢 Active" if m.get("is_active") else "Lưu trữ (Archived)",
                "Ngưỡng (Threshold)": m.get("threshold", 0.5),
                "Thời điểm tạo": m.get("created_at_utc", "N/A")[:19].replace("T", " "),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Rollback action
    rb_col1, rb_col2 = st.columns([2, 1])
    with rb_col1:
        st.markdown("Trong trường hợp mô hình mới phát sinh sai lệch hoặc lỗi nghiệp vụ, quản trị viên có thể hoàn tác ngay lập tức về phiên bản đã kiểm chứng trước đó.")
    with rb_col2:
        if st.button("⏪ Hoàn Tác Về Phiên Bản Trước (Rollback)", type="secondary", use_container_width=True):
            try:
                rb_info = rollback_model()
                st.success(f"✅ Đã hoàn tác thành công về phiên bản: `{rb_info['active_version']}`!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Không thể hoàn tác: {e}")
