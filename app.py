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
from ui.extended_analysis import render_extended_analysis

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
    /* Main container layout */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    /* Modern Theme-Adaptive Metric Cards */
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color) !important;
        padding: 16px 20px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(128, 128, 128, 0.22) !important;
        border-left: 5px solid #1E88E5 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.16) !important;
        border-color: rgba(30, 136, 229, 0.5) !important;
    }

    /* Metric Label */
    div[data-testid="stMetricLabel"] {
        color: var(--text-color) !important;
        opacity: 0.88 !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.2px !important;
        margin-bottom: 6px !important;
    }

    div[data-testid="stMetricLabel"] * {
        color: var(--text-color) !important;
        opacity: 0.88 !important;
    }

    /* Metric Value */
    div[data-testid="stMetricValue"] {
        color: var(--text-color) !important;
        font-size: 1.7rem !important;
        font-weight: 700 !important;
        line-height: 1.25 !important;
    }

    div[data-testid="stMetricValue"] * {
        color: var(--text-color) !important;
    }

    /* Metric Delta */
    div[data-testid="stMetricDelta"] {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        margin-top: 4px !important;
    }

    /* Primary and Secondary Buttons */
    button[kind="primary"] {
        background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 2px 6px rgba(30, 136, 229, 0.3) !important;
        transition: all 0.2s ease !important;
    }

    button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(30, 136, 229, 0.45) !important;
    }

    button[kind="secondary"] {
        border-radius: 8px !important;
        border: 1px solid rgba(128, 128, 128, 0.3) !important;
        transition: all 0.2s ease !important;
    }

    button[kind="secondary"]:hover {
        border-color: #1E88E5 !important;
        color: #1E88E5 !important;
    }

    /* Alert and Message Callouts */
    div[data-testid="stAlert"] {
        border-radius: 10px !important;
        border: 1px solid rgba(128, 128, 128, 0.15) !important;
    }

    /* Clean Card Expander */
    div[data-testid="stExpander"] {
        border-radius: 10px !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        overflow: hidden !important;
    }

    /* Sidebar Navigation Enhancement */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.15);
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
        "🔬 Phân tích mở rộng (P2)",
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
        "**Ngưỡng quyết định:** 0.50\n\n"
        "**Mức ưu tiên:** Thấp (<0.3) | TB (0.3–0.6) | Cao (≥0.6)"
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
    elif choice == "🔬 Phân tích mở rộng (P2)":
        render_extended_analysis()


if __name__ == "__main__":
    main()
