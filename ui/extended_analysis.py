"""Màn hình Phân tích mở rộng theo Plan 05 Section 1 (P2):
Mô phỏng thay đổi ngưỡng quyết định (Threshold Simulation), phân bố điểm nguy cơ, và diễn giải tác động đặc trưng.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np

from telco_churn.inference import load_model_bundle, predict_records
from telco_churn.schema import ALL_FEATURE_COLUMNS


def render_extended_analysis():
    st.title("🔬 Phân Tích Mở Rộng & Mô Phỏng Ngưỡng Quyết Định (P2)")
    st.markdown(
        "Khám phá các khía cạnh phân tích chuyên sâu: mô phỏng tác động của việc thay đổi ngưỡng quyết định (*Decision Threshold*), "
        "quan sát phân bố điểm rủi ro và diễn giải tác động định lượng của từng yếu tố dịch vụ đối với khả năng rời bỏ."
    )

    try:
        bundle = load_model_bundle()
    except Exception as e:
        st.error(f"❌ Không thể nạp mô hình active: {e}")
        return

    tab_thresh, tab_dist, tab_explain = st.tabs([
        "🎯 Mô phỏng Ngưỡng Quyết Định",
        "📊 Phân Bố Điểm Rủi Ro (Score)",
        "🔍 Diễn Giải Tác Động Đặc Trưng",
    ])

    # Load a benchmark dataset for interactive simulation
    sample_path = os.path.join("data", "samples", "valid_sample.csv")
    sample_df = pd.read_csv(sample_path) if os.path.exists(sample_path) else None

    with tab_thresh:
        st.subheader("1. Mô Phỏng Thay Đổi Ngưỡng Quyết Định (Threshold Simulation)")
        st.markdown("""
        Trong bài toán phân loại nhị phân, mô hình dự đoán xác suất $P(\\text{Churn} = 1 | x)$. 
        Mặc định, nhãn được gán là **Yes** khi $P \\ge 0.50$. 
        Tuy nhiên trong thực tế kinh doanh viễn thông, việc chọn ngưỡng phụ thuộc vào **ràng buộc ngân sách CSKH** và **chi phí bỏ sót khách hàng**:
        - **Hạ ngưỡng (< 0.50):** Tăng Recall (bắt được nhiều khách rời bỏ hơn, giảm False Negatives) nhưng tăng False Positives (tốn chi phí liên hệ chăm sóc).
        - **Tăng ngưỡng (> 0.50):** Tăng Precision (chỉ can thiệp khách có nguy cơ cực cao) nhưng bỏ sót nhiều khách hàng có nguy cơ rời mạng.
        """)

        sim_col1, sim_col2 = st.columns([1, 2])
        with sim_col1:
            test_thresh = st.slider(
                "Chọn Ngưỡng Quyết Định (Threshold):",
                min_value=0.10,
                max_value=0.90,
                value=0.50,
                step=0.05,
                help="Ngưỡng phân cách giữa nhãn 'No' và 'Yes'.",
            )
            st.metric("Ngưỡng đang thử nghiệm", f"{test_thresh:.2f}")

        with sim_col2:
            st.info(
                f"**Đánh giá nghiệp vụ tại ngưỡng {test_thresh:.2f}:**\n\n"
                f"- Nếu thuê bao có điểm nguy cơ $\\ge {test_thresh:.2f}$, hệ thống sẽ gắn nhãn **Churn = Yes**.\n"
                f"- Khi hạ ngưỡng xuống dưới 0.50, quy mô danh sách khách hàng cần CSKH sẽ mở rộng.\n"
                f"- Khi nâng ngưỡng lên trên 0.50, chỉ nhóm khách hàng có dấu hiệu rời bỏ rõ rệt nhất mới được kích hoạt hành động can thiệp."
            )

        if sample_df is not None:
            # Predict sample with custom threshold
            preds_custom = predict_records(sample_df, bundle, threshold=test_thresh)
            n_total = len(preds_custom)
            n_yes = int((preds_custom["churn_prediction"] == 1).sum())
            n_no = n_total - n_yes

            st.markdown("##### Kết quả phân loại trên tập mẫu tại ngưỡng đã chọn:")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Tổng số khách hàng mẫu", n_total)
            with c2:
                st.metric(f"Dự báo Churn = Yes (≥{test_thresh:.2f})", f"{n_yes} ({n_yes/n_total:.1%})")
            with c3:
                st.metric(f"Dự báo Churn = No (<{test_thresh:.2f})", f"{n_no} ({n_no/n_total:.1%})")

    with tab_dist:
        st.subheader("2. Phân Bố Điểm Nguy Cơ & Các Vùng Ưu Tiên CSKH")
        st.markdown("""
        Hệ thống phân bổ điểm nguy cơ thành 3 vùng ưu tiên hành động theo Plan 05:
        - 🟢 **Thấp (Low):** `Score < 0.30` (Duy trì chăm sóc chuẩn)
        - 🟡 **Trung bình (Medium):** `0.30 ≤ Score < 0.60` (Gửi ưu đãi gói cước tự động)
        - 🔴 **Cao (High):** `Score ≥ 0.60` (Chuyên viên chăm sóc liên hệ trực tiếp)
        """)

        # If we have sample predictions
        if sample_df is not None:
            preds_default = predict_records(sample_df, bundle, threshold=0.5)
            scores = preds_default["churn_score"]

            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                low_cnt = int((scores < 0.30).sum())
                st.metric("🟢 Ưu tiên Thấp (<0.30)", f"{low_cnt} ({low_cnt/len(scores):.1%})")
            with col_p2:
                med_cnt = int(((scores >= 0.30) & (scores < 0.60)).sum())
                st.metric("🟡 Ưu tiên Trung bình (0.30–0.60)", f"{med_cnt} ({med_cnt/len(scores):.1%})")
            with col_p3:
                high_cnt = int((scores >= 0.60).sum())
                st.metric("🔴 Ưu tiên Cao (≥0.60)", f"{high_cnt} ({high_cnt/len(scores):.1%})")

            st.markdown("##### Bảng phân loại chi tiết tập mẫu:")
            st.dataframe(
                preds_default[["customerID", "churn_score", "churn_label", "priority_level"]],
                use_container_width=True,
            )

    with tab_explain:
        st.subheader("3. Diễn Giải Tác Động Của Các Yếu Tố (Feature Importance & Odds Ratio)")
        st.markdown("""
        Mô hình Champion (`Logistic Regression`) sở hữu tính diễn giải minh bạch cao thông qua các hệ số hồi quy $\\beta$.
        Hệ số dương ($\\beta > 0$) làm tăng log-odds của khả năng rời bỏ, trong khi hệ số âm ($\\beta < 0$) thể hiện yếu tố gắn kết giữ chân khách hàng.
        """)

        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.error("⚠️ **Các Yếu Tố Thúc Đẩy Rời Bỏ Hàng Đầu (Top Churn Drivers):**")
            st.markdown("""
            1. **Hợp đồng từng tháng (Contract = Month-to-month):** Khách hàng không có cam kết dài hạn, dễ dàng chuyển nhà mạng khi có ưu đãi khác.
            2. **Dịch vụ Internet Cáp quang (InternetService = Fiber optic):** Cước phí cao hơn và kỳ vọng chất lượng khắt khe hơn.
            3. **Phương thức thanh toán séc điện tử (PaymentMethod = Electronic check):** Tỷ lệ thanh toán trễ và biến động cao hơn thanh toán tự động.
            4. **Khách hàng cao tuổi (SeniorCitizen = 1):** Độ nhạy cảm với sự thay đổi giá cước và hỗ trợ kỹ thuật cao hơn.
            """)

        with col_neg:
            st.success("🛡️ **Các Yếu Tố Giữ Chân Khách Hàng (Top Retention Drivers):**")
            st.markdown("""
            1. **Hợp đồng dài hạn 2 năm (Contract = Two year):** Ràng buộc cam kết vững chắc nhất với nhà mạng.
            2. **Thời gian gắn bó lâu năm (tenure cao):** Khách hàng trung thành có thói quen sử dụng ổn định.
            3. **Có dịch vụ Hỗ trợ kỹ thuật (TechSupport = Yes):** Trải nghiệm dịch vụ được giải quyết kịp thời.
            4. **Có dịch vụ Bảo mật mạng (OnlineSecurity = Yes):** Chi phí chuyển đổi (switching cost) cao hơn khi tích hợp nhiều dịch vụ.
            """)

        # Display coefficient chart if available
        coef_path = os.path.join("reports", "figures", "evaluation", "06_logistic_regression_coefficients.png")
        if os.path.exists(coef_path):
            st.image(coef_path, caption="Biểu đồ hệ số hồi quy chuẩn hóa của Logistic Regression (Champion Model)", use_container_width=True)
