"""Màn hình Kết quả & Đánh giá mô hình: Bảng kiểm định chéo CV, kiểm thử Test, và trực quan hóa hiệu năng."""

import os
import json
import streamlit as st
import pandas as pd


def render_model_results():
    st.title("📈 Kết Quả & Đánh Giá Mô Hình Khai Phá")
    st.markdown("Xem lại toàn bộ kết quả thực nghiệm 5-Fold Stratified Cross-Validation, quyết định chọn Champion Model, và số liệu kiểm thử trên tập Test 20% độc lập.")

    # 1. Cross-validation Summary Table
    st.subheader("1. Bảng Tổng Hợp Kiểm Định Chéo 5-Fold (Tập Phát Triển 80% — 5.634 mẫu)")
    st.markdown("""
    > [!IMPORTANT]
    > **Giao thức nghiên cứu chuẩn:** Quá trình tìm kiếm siêu tham số và so sánh 3 họ mô hình (Decision Tree, Logistic Regression, Random Forest) 
    > được thực hiện hoàn toàn trên 80% tập phát triển (Development Set) qua 5-Fold Stratified CV trước khi mở tập kiểm thử độc lập.
    """)

    cv_summary_path = os.path.join("artifacts", "models", "model_001", "cv_summary.json")
    if os.path.exists(cv_summary_path):
        with open(cv_summary_path, "r", encoding="utf-8") as f:
            cv_data = json.load(f)

        display_names = {
            "dummy": "Baseline Dummy (most_frequent)",
            "decision_tree": "Decision Tree",
            "logistic_regression": "Logistic Regression",
            "random_forest": "Random Forest",
        }

        table_rows = []
        for k, name in display_names.items():
            if k in cv_data:
                m = cv_data[k]
                params_str = ", ".join(f"{pk}={pv}" for pk, pv in m.get("best_params", {}).items())
                table_rows.append({
                    "Mô hình": name,
                    "Siêu tham số tối ưu": params_str if params_str else "N/A",
                    "CV F1 (Lớp 1)": f"{m.get('mean_f1', 0.0):.4f} ± {m.get('std_f1', 0.0):.4f}",
                    "CV Recall (Lớp 1)": f"{m.get('mean_recall', 0.0):.4f}",
                    "CV Precision (Lớp 1)": f"{m.get('mean_precision', 0.0):.4f}",
                    "CV ROC-AUC": f"{m.get('mean_roc_auc', 0.0):.4f}",
                    "CV Accuracy": f"{m.get('mean_accuracy', 0.0):.4f}",
                    "Thời gian Fit (s)": f"{m.get('mean_fit_time', 0.0):.2f}s",
                    "Vai trò": "🏆 Champion" if k == "logistic_regression" else "So sánh",
                })

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    else:
        st.info("Chưa tìm thấy cv_summary.json trong artifacts.")

    # 2. Champion Rationale
    with st.expander("💡 Biện Luận Lựa Chọn Champion Model (Plan 03 Section 4)", expanded=True):
        st.markdown("""
        - **Mô hình được chọn:** `Logistic Regression` (với `C=10.0`, `class_weight='balanced'`, `penalty='l2'`).
        - **Biện luận:**
          1. **F1-Score cao nhất:** Đạt mean CV F1 = **0.6334**, vượt trội so với Random Forest (0.6186) và Decision Tree (0.6033).
          2. **Độ nhạy (Recall) vượt trội:** Nhờ trọng số `class_weight='balanced'`, Logistic Regression đạt Recall lớp Churn lên tới **0.8012** (hạn chế tối đa bỏ sót khách hàng rời bỏ - False Negatives).
          3. **Tính diễn giải và ổn định cao:** Hệ số hồi quy có thể trực tiếp chuyển dịch thành hệ số rủi ro cho từng dịch vụ viễn thông, tốc độ suy luận dưới 5ms, cực kỳ phù hợp cho môi trường ứng dụng thực tế.
        """)

    st.markdown("---")

    # 3. Test Set Evaluation Metrics
    st.subheader("2. Đánh Giá Cuối Cùng Trên Tập Kiểm Thử Độc Lập (Test Set 20% — 1.409 mẫu)")

    test_metrics_path = os.path.join("artifacts", "models", "model_001", "test_metrics.json")
    if os.path.exists(test_metrics_path):
        with open(test_metrics_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        # Champion test metrics
        champ_test = test_data.get("champion_test_metrics", {})
        tc1, tc2, tc3, tc4, tc5 = st.columns(5)
        with tc1:
            st.metric("Test Accuracy", f"{champ_test.get('accuracy', 0.0):.2%}")
        with tc2:
            st.metric("Test F1 (Lớp 1)", f"{champ_test.get('f1', 0.0):.4f}")
        with tc3:
            st.metric("Test Recall (Lớp 1)", f"{champ_test.get('recall', 0.0):.2%}")
        with tc4:
            st.metric("Test Precision", f"{champ_test.get('precision', 0.0):.2%}")
        with tc5:
            st.metric("Test ROC-AUC", f"{champ_test.get('roc_auc', 0.0):.4f}")

    # 4. Evaluation Plots
    st.markdown("---")
    st.subheader("3. Trực Quan Hóa Hiệu Năng & Khả Năng Phân Tách")

    eval_fig_dir = os.path.join("reports", "figures", "evaluation")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Ma trận nhầm lẫn (Confusion Matrix)",
        "Đường cong ROC & PR",
        "So sánh CV vs Test",
        "Hệ số & Trọng số đặc trưng",
    ])

    with tab1:
        cm_path = os.path.join(eval_fig_dir, "01_confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Ma trận nhầm lẫn (Confusion Matrix) của 4 mô hình trên tập Test 20%", use_container_width=True)

    with tab2:
        col_roc, col_pr = st.columns(2)
        with col_roc:
            roc_path = os.path.join(eval_fig_dir, "02_roc_curves.png")
            if os.path.exists(roc_path):
                st.image(roc_path, caption="Đường cong ROC (Receiver Operating Characteristic)", use_container_width=True)
        with col_pr:
            pr_path = os.path.join(eval_fig_dir, "03_pr_curves.png")
            if os.path.exists(pr_path):
                st.image(pr_path, caption="Đường cong Precision-Recall (PR Curve)", use_container_width=True)

    with tab3:
        cv_test_path = os.path.join(eval_fig_dir, "04_cv_vs_test_comparison.png")
        if os.path.exists(cv_test_path):
            st.image(cv_test_path, caption="So sánh điểm F1 giữa 5-Fold CV và Test Set độc lập", use_container_width=True)

    with tab4:
        lr_coef_path = os.path.join(eval_fig_dir, "06_logistic_regression_coefficients.png")
        rf_imp_path = os.path.join(eval_fig_dir, "07_random_forest_importance.png")
        dt_struct_path = os.path.join(eval_fig_dir, "05_decision_tree_structure.png")

        if os.path.exists(lr_coef_path):
            st.image(lr_coef_path, caption="Hệ số hồi quy hàng đầu của Logistic Regression (Champion)", use_container_width=True)
        if os.path.exists(rf_imp_path):
            st.image(rf_imp_path, caption="Tầm quan trọng của đặc trưng trong Random Forest (MDI)", use_container_width=True)
