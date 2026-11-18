"""Màn hình Quản lý mô hình & Huấn luyện lại theo Plan 05 Section 5:
Kiểm tra gate, ngày retrain gần nhất, chạy dry-run, kích hoạt an toàn và rollback.
"""

import os
import json
import streamlit as st
import pandas as pd

from telco_churn.registry import (
    get_active_model_info,
    set_active_model,
    rollback_model,
    reconcile_active_model,
    list_available_models,
)
from telco_churn.storage import (
    get_snapshots_with_labels,
    get_training_runs,
)
from telco_churn.retraining import (
    run_retraining,
    load_retraining_config,
    evaluate_gate_conditions,
)


def render_model_management():
    st.title("⚙️ Quản Lý Mô Hình & Chu Trình Huấn Luyện Lại")
    st.markdown(
        "Theo dõi tình trạng mô hình active, kiểm tra các điều kiện cổng an toàn (*Gate*), "
        "chạy thử nghiệm huấn luyện candidate (*Dry-run/Demo*), đối chiếu candidate vs champion và thực hiện hoàn tác (*Rollback*)."
    )

    # 1. Current Active Model & Last Retrain Date
    st.subheader("1. Mô Hình Đang Phục Vụ (Active Champion) & Tình Trạng Vận Hành")

    try:
        active_info = get_active_model_info()
        active_ver = active_info.get("active_version", "N/A")
        active_type = active_info.get("model_type", "N/A")
        updated_at = active_info.get("updated_at_utc", "N/A")
        prev_ver = active_info.get("previous_version", "Không có")
    except Exception as e:
        st.error(f"❌ Không thể đọc active_model.json: {e}")
        return

    # Check last retrain run
    last_runs = get_training_runs(limit=1)
    last_retrain_str = "Chưa thực hiện"
    if not last_runs.empty:
        last_retrain_str = last_runs.iloc[0]["started_at"][:19].replace("T", " ")

    ac1, ac2, ac3, ac4 = st.columns(4)
    with ac1:
        st.metric("Phiên bản Active", active_ver, delta="Champion phục vụ", delta_color="off")
    with ac2:
        st.metric("Kiến trúc", active_type.replace("_", " ").title(), delta="L2 Regularized", delta_color="off")
    with ac3:
        st.metric("Phiên bản trước", prev_ver or "Gốc", delta="Sẵn sàng rollback", delta_color="off")
    with ac4:
        st.metric("Ngày Retrain gần nhất", last_retrain_str, delta="Audit Log", delta_color="off")

    st.markdown("---")

    # 2. Retraining Gate Checklist
    st.subheader("2. Kiểm Tra Cổng Điều Kiện Huấn Luyện Lại (Monthly Gate Checklist)")
    st.markdown("""
    > [!IMPORTANT]
    > **Chính sách cổng an toàn (Plan 06 Section 4):** Mô hình chỉ được phép bước vào chu trình huấn luyện lại khi thỏa mãn 
    > đồng thời các tiêu chí cổng an toàn về số lượng hồ sơ mới và sự cân bằng giữa hai lớp nhãn thực tế (*Actual Churn*).
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
        with st.expander("⚠️ Chi tiết các lý do chưa đạt cổng an toàn:", expanded=True):
            for fail in gate_details["failures"]:
                st.write(f"- {fail}")

    st.markdown("---")

    # 3. Interactive Actions: 4 Scenarios conforming to Plan 06 Section 8
    st.subheader("3. Thao Tác Huấn Luyện Lại & 4 Kịch Bản Vận Hành (Plan 06 Section 8)")
    st.markdown("""
    > [!NOTE]
    > **Nguyên tắc an toàn (Plan 05 & Plan 06):** 
    > - Không có nút bấm "thay thế mô hình mù quáng". Mọi thao tác kích hoạt candidate đều phải tuân thủ điều kiện thăng cấp khách quan (F1 gain ≥ 0.01 và Recall drop ≤ 0.02).
    > - **Dữ liệu demo là dữ liệu mô phỏng (Synthetic):** Hệ thống gắn nhãn minh bạch trên màn hình; tuyệt đối không lấy dữ liệu Test Set gốc nạp vào cơ sở dữ liệu để tạo cảm giác hệ thống đang tự học thêm.
    """)

    tab_a, tab_b, tab_c, tab_d = st.tabs([
        "⚠️ Kịch bản A: Chưa đủ Gate (Skip)",
        "🧪 Kịch bản B: Mô phỏng Candidate (Comparison)",
        "🛑 Kịch bản C: Candidate kém (Reject)",
        "🚀 Kịch bản D: Thăng cấp an toàn (Promote & Rollback)",
    ])

    with tab_a:
        st.markdown("**Kịch bản A: Dữ liệu mới chưa đạt cổng an toàn (Gate Insufficient)**")
        st.caption("Kiểm tra điều kiện cổng và quyết định SKIP, bảo lưu nguyên trạng mô hình Champion đang phục vụ.")
        if st.button("🧪 Chạy Kiểm Tra Cổng An Toàn (Kịch bản A)", key="btn_scen_a", use_container_width=True):
            with st.spinner("Đang chạy kiểm tra gate..."):
                res = run_retraining(dry_run=True, ignore_gate=False)
                if res["decision"] == "SKIP":
                    st.warning(f"⏩ **Quyết định: SKIP** (Bảo lưu Champion `{res.get('champion_version')}`)\n\n**Lý do:** {res['reason']}")
                else:
                    st.info(f"Kết quả: **{res['decision']}** — {res['reason']}")

    with tab_b:
        st.markdown("**Kịch bản B: Đủ dữ liệu mô phỏng — So sánh Champion vs Candidate**")
        st.caption("Chạy huấn luyện candidate và tính toán các chỉ số so sánh (F1, Recall, Precision, Latency) trên tập đối chiếu độc lập.")
        if st.button("🚀 Chạy Mô Phỏng Candidate & Đối Chiếu (Kịch bản B)", key="btn_scen_b", type="primary", use_container_width=True):
            with st.spinner("Đang huấn luyện candidate và tính toán so sánh trên tập comparison..."):
                res_demo = run_retraining(
                    dry_run=True,
                    ignore_gate=True,
                    allow_synthetic=True,
                )
                st.success("✅ Đã hoàn tất chu trình mô phỏng đối chiếu!")

                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    st.markdown(f"**Quyết định:** `{res_demo['decision']}`")
                    st.write(f"**Lý do:** {res_demo['reason']}")
                    if res_demo.get("snapshot_path"):
                        st.caption(f"Snapshot File: `{res_demo['snapshot_path']}`")
                        st.caption(f"SHA-256 Hash: `{res_demo.get('snapshot_hash', 'N/A')[:16]}...`")
                with d_col2:
                    if res_demo.get("candidate_metrics") and res_demo.get("champion_metrics"):
                        champ = res_demo["champion_metrics"]
                        cand = res_demo["candidate_metrics"]
                        st.write(f"- Champion F1: **{champ['f1']:.4f}** | Latency: `{champ.get('prediction_time_ms', 0):.2f} ms`")
                        st.write(f"- Candidate F1: **{cand['f1']:.4f}** (Delta: `{res_demo.get('delta_f1'):+.4f}`) | Latency: `{cand.get('prediction_time_ms', 0):.2f} ms`")
                        st.write(f"- Candidate Recall: **{cand['recall']:.4f}** (Delta: `{res_demo.get('delta_recall'):+.4f}`)")

    with tab_c:
        st.markdown("**Kịch bản C: Candidate không đạt tiêu chuẩn thăng cấp (Reject)**")
        st.caption("Minh họa trường hợp Candidate có chỉ số F1 tăng chưa đủ 0.01 hoặc Recall sụt giảm quá 0.02. Hệ thống kiên quyết từ chối kích hoạt.")
        if st.button("🛑 Mô Phỏng Quyết Định Từ Chối (Kịch bản C)", key="btn_scen_c", use_container_width=True):
            with st.spinner("Đang chạy đối soát ngưỡng thăng cấp nghiêm ngặt..."):
                strict_cfg = load_retraining_config()
                strict_cfg["promotion_policy"]["min_f1_gain"] = 0.50  # Unattainable threshold to demo rejection
                res_c = run_retraining(
                    dry_run=True,
                    ignore_gate=True,
                    allow_synthetic=True,
                    config=strict_cfg,
                )
                st.error(f"🛑 **Quyết định: {res_c['decision']}**")
                st.write(f"**Lý do:** {res_c['reason']}")
                if res_c.get("champion_version"):
                    st.info(f"✅ Hệ thống bảo toàn an toàn phiên bản Champion: `{res_c['champion_version']}`.")

    with tab_d:
        st.markdown("**Kịch bản D: Candidate đạt chuẩn ➜ Thăng cấp (Promote), Kích hoạt & Rollback**")
        st.caption("Khi candidate vượt qua gate, cải thiện chỉ số và vượt qua smoke test, hệ thống kích hoạt nguyên tử phiên bản mới.")
        if st.button("🌟 Chạy Thử Thăng Cấp Candidate (Kịch bản D — Dry-run/Inspect)", key="btn_scen_d", use_container_width=True):
            with st.spinner("Đang chạy chu trình thẩm định thăng cấp..."):
                res_d = run_retraining(
                    dry_run=True,
                    ignore_gate=True,
                    allow_synthetic=True,
                )
                st.info(f"**Kết quả thẩm định:** `{res_d['decision']}`")
                st.write(f"**Chi tiết:** {res_d['reason']}")
                st.caption("Ghi chú: Nút này chạy ở chế độ dry-run an toàn để thanh tra trước khi kích hoạt.")

    st.markdown("---")

    # 4. Training Runs Audit History
    st.subheader("4. Lịch Sử Các Đợt Huấn Luyện Lại (Candidate/Champion & Quyết Định)")
    runs_df = get_training_runs(limit=20)
    if not runs_df.empty:
        st.dataframe(
            runs_df[[
                "run_id", "started_at", "status", "decision", "candidate_version",
                "champion_version", "reason"
            ]],
            use_container_width=True,
        )
    else:
        st.info("ℹ️ Chưa có lịch sử đợt retraining nào được ghi nhận trong cơ sở dữ liệu.")

    st.markdown("---")

    # 5. Rollback and Model Versions
    st.subheader("5. Danh Sách Phiên Bản Mô Hình, Đối Soát & Hoàn Tác (Rollback)")

    models_list = list_available_models()
    if models_list:
        rows = []
        for m in models_list:
            rows.append({
                "Phiên bản": m.get("model_version"),
                "Kiến trúc": m.get("model_type"),
                "Trạng thái": "🟢 Active (Đang phục vụ)" if m.get("is_active") else "Lưu trữ (Archived)",
                "Ngưỡng (Threshold)": m.get("threshold", 0.5),
                "Thời điểm tạo": m.get("created_at_utc", "N/A")[:19].replace("T", " "),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Rollback & Reconcile actions
    rb_col1, rb_col2, rb_col3 = st.columns([2, 1, 1])
    with rb_col1:
        st.markdown(
            "Trong trường hợp mô hình mới phát sinh sai lệch hoặc lỗi nghiệp vụ, "
            "quản trị viên có thể hoàn tác ngay lập tức về phiên bản đã kiểm chứng trước đó."
        )
    with rb_col2:
        if st.button("🔄 Đối Soát & Đồng Bộ (Reconcile)", use_container_width=True):
            try:
                rec_info = reconcile_active_model()
                st.success(f"✅ Đã đối soát thành công! Phiên bản Active: `{rec_info['active_version']}`")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Lỗi đối soát: {e}")
    with rb_col3:
        if st.button("⏪ Hoàn Tác Về Trước (Rollback)", type="secondary", use_container_width=True):
            try:
                rb_info = rollback_model()
                st.success(f"✅ Đã hoàn tác thành công về phiên bản: `{rb_info['active_version']}`!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Không thể hoàn tác: {e}")
                st.error(f"❌ Không thể hoàn tác: {e}")
