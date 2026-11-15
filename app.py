"""Streamlit Web Application for Telco Customer Churn Prediction conforming to Plan 04 and Plan 05."""

import sys
import os
import streamlit as st

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from ui.overview import render_overview
from ui.single_prediction import render_single_prediction
from ui.batch_prediction import render_batch_prediction
from ui.history import render_history
from ui.model_results import render_model_results
from ui.model_management import render_model_management

# Page configuration
st.set_page_config(
    page_title="Telco Churn Prediction Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px 15px;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    st.sidebar.title("📡 Telco Retention AI")
    st.sidebar.markdown("**Hệ thống Dự đoán Rời bỏ Dịch vụ Viễn thông**")
    st.sidebar.caption("Đề tài Khai phá Dữ liệu — Nhóm 5 thành viên")

    menu_options = [
        "📊 Tổng quan dự án",
        "👤 Dự đoán một khách hàng",
        "📁 Dự đoán theo lô (CSV)",
        "📜 Lịch sử & Nhãn thực tế",
        "📈 Kết quả & Đánh giá mô hình",
        "⚙️ Quản lý & Retraining",
    ]

    choice = st.sidebar.radio(
        "Điều hướng chức năng:",
        options=menu_options,
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Thông tin hệ thống")
    st.sidebar.info(
        "**Dữ liệu:** IBM Telco Churn (7.043 mẫu)\n\n"
        "**Mô hình Active:** Logistic Regression (Champion)\n\n"
        "**Bộ nhớ đệm:** SQLite Runtime DB\n\n"
        "**Ngưỡng quyết định:** 0.50"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("© 2026 Bài tập lớn Khai phá dữ liệu. Giữ quyền bảo lưu.")

    # Page Routing
    if choice == "📊 Tổng quan dự án":
        render_overview()
    elif choice == "👤 Dự đoán một khách hàng":
        render_single_prediction()
    elif choice == "📁 Dự đoán theo lô (CSV)":
        render_batch_prediction()
    elif choice == "📜 Lịch sử & Nhãn thực tế":
        render_history()
    elif choice == "📈 Kết quả & Đánh giá mô hình":
        render_model_results()
    elif choice == "⚙️ Quản lý & Retraining":
        render_model_management()


if __name__ == "__main__":
    main()
