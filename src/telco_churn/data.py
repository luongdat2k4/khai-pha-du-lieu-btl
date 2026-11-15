"""Data loading, auditing, pre-split cleaning, and stratified splitting."""

import hashlib
import os
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

from .schema import (
    ID_COLUMN,
    TARGET_COLUMN,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    ALL_FEATURE_COLUMNS,
    INTERNET_ADDON_SERVICES,
    validate_records,
)


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data into a pandas DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")
    return pd.read_csv(filepath)


def audit_dataset(df_raw: pd.DataFrame, source_file_path: Optional[str] = None) -> Dict[str, Any]:
    """Perform a thorough audit of the raw dataset.

    Returns a structured dictionary with all audit statistics.
    """
    total_rows, total_cols = df_raw.shape
    file_hash = compute_file_sha256(source_file_path) if source_file_path and os.path.exists(source_file_path) else "N/A"

    # Column dtypes and null counts
    column_stats = []
    for col in df_raw.columns:
        series = df_raw[col]
        # Count explicit nulls
        null_count = int(series.isna().sum())
        # Count whitespace strings
        blank_count = int(series.apply(lambda x: 1 if isinstance(x, str) and x.strip() == "" else 0).sum())
        n_unique = int(series.nunique(dropna=False))
        sample_vals = series.dropna().unique()[:5].tolist()
        column_stats.append({
            "column": col,
            "dtype": str(series.dtype),
            "null_count": null_count,
            "blank_count": blank_count,
            "n_unique": n_unique,
            "sample_values": sample_vals,
        })

    # Target distribution
    target_dist = {}
    if TARGET_COLUMN in df_raw.columns:
        counts = df_raw[TARGET_COLUMN].value_counts(dropna=False).to_dict()
        total = len(df_raw)
        for k, v in counts.items():
            target_dist[str(k)] = {
                "count": int(v),
                "percentage": round(v / total * 100, 2),
            }

    # TotalCharges whitespace analysis
    tc_blank_mask = df_raw["TotalCharges"].apply(lambda x: isinstance(x, str) and x.strip() == "")
    tc_blank_rows = df_raw[tc_blank_mask]
    tc_blank_count = int(tc_blank_mask.sum())
    tc_blank_tenure_dist = tc_blank_rows["tenure"].value_counts().to_dict() if tc_blank_count > 0 else {}
    tc_blank_churn_dist = tc_blank_rows[TARGET_COLUMN].value_counts().to_dict() if tc_blank_count > 0 and TARGET_COLUMN in df_raw.columns else {}

    # Duplicate checks
    total_duplicate_rows = int(df_raw.duplicated().sum())
    id_duplicates = int(df_raw.duplicated(subset=[ID_COLUMN]).sum()) if ID_COLUMN in df_raw.columns else 0
    feature_duplicates = int(df_raw.duplicated(subset=ALL_FEATURE_COLUMNS).sum()) if set(ALL_FEATURE_COLUMNS).issubset(df_raw.columns) else 0

    # Inconsistency checks
    # 1. PhoneService = No vs MultipleLines
    phone_conflict_mask = (df_raw["PhoneService"] == "No") & (df_raw["MultipleLines"] == "Yes")
    phone_conflicts = int(phone_conflict_mask.sum())

    # 2. InternetService = No vs Internet Add-ons
    internet_conflicts = 0
    internet_no_mask = df_raw["InternetService"] == "No"
    for addon in INTERNET_ADDON_SERVICES:
        if addon in df_raw.columns:
            conflict = internet_no_mask & (df_raw[addon].isin(["Yes", "No"]))
            internet_conflicts += int(conflict.sum())

    # Numeric summary (converting TotalCharges for stats)
    numeric_summaries = {}
    df_num_check = df_raw.copy()
    df_num_check["TotalCharges_num"] = pd.to_numeric(df_num_check["TotalCharges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce")

    for col, src_col in [("tenure", "tenure"), ("MonthlyCharges", "MonthlyCharges"), ("TotalCharges", "TotalCharges_num")]:
        if src_col in df_num_check.columns:
            s = df_num_check[src_col].dropna()
            q1 = float(s.quantile(0.25))
            q3 = float(s.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers_count = int(((s < lower_bound) | (s > upper_bound)).sum())

            numeric_summaries[col] = {
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": round(float(s.mean()), 2),
                "std": round(float(s.std()), 2),
                "median": float(s.median()),
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "outliers_count": outliers_count,
            }

    return {
        "file_hash": file_hash,
        "total_rows": total_rows,
        "total_cols": total_cols,
        "column_stats": column_stats,
        "target_dist": target_dist,
        "tc_blank_count": tc_blank_count,
        "tc_blank_tenure_dist": tc_blank_tenure_dist,
        "tc_blank_churn_dist": tc_blank_churn_dist,
        "total_duplicate_rows": total_duplicate_rows,
        "id_duplicates": id_duplicates,
        "feature_duplicates": feature_duplicates,
        "phone_conflicts": phone_conflicts,
        "internet_conflicts": internet_conflicts,
        "numeric_summaries": numeric_summaries,
    }


def generate_audit_markdown(audit: Dict[str, Any], output_path: str) -> None:
    """Generate Markdown audit report from audit dictionary."""
    lines = [
        "# Báo cáo kiểm kê và đánh giá chất lượng dữ liệu (Data Audit Report)",
        "",
        f"- **Tệp dữ liệu nguồn:** `telco-customer-churn-on-icp4d/data/Telco-Customer-Churn.csv`",
        f"- **Mã băm SHA-256:** `{audit['file_hash']}`",
        f"- **Kích thước dữ liệu gốc:** {audit['total_rows']:,} dòng, {audit['total_cols']} cột",
        "",
        "---",
        "",
        "## 1. Phân bố nhãn mục tiêu (Target Distribution)",
        "",
        "| Nhãn Churn | Số lượng mẫu | Tỷ lệ (%) |",
        "|---|---|---|",
    ]
    for k, v in audit["target_dist"].items():
        lines.append(f"| `{k}` | {v['count']:,} | {v['percentage']}% |")

    lines.extend([
        "",
        f"> [!NOTE]",
        f"> Tỷ lệ rời bỏ là khoảng 26.54% (1.869 mẫu Yes / 5.174 mẫu No). Dữ liệu có sự mất cân bằng giữa hai lớp (tỷ lệ xấp xỉ 1:2.77). Do đó, chỉ số Accuracy thông thường không phản ánh toàn diện hiệu quả mô hình; cần sử dụng thêm Precision, Recall, F1-score và ROC-AUC.",
        "",
        "---",
        "",
        "## 2. Kiểm kê các cột và kiểu dữ liệu (Schema & Missingness)",
        "",
        "| Cột | Dtype | Giá trị thiếu (NaN) | Chuỗi rỗng / Blank | Số giá trị phân biệt | Ví dụ giá trị |",
        "|---|---|---|---|---|---|",
    ])
    for col_info in audit["column_stats"]:
        samples = ", ".join([str(x) for x in col_info["sample_values"][:3]])
        lines.append(
            f"| `{col_info['column']}` | `{col_info['dtype']}` | {col_info['null_count']} | {col_info['blank_count']} | {col_info['n_unique']} | {samples} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Phân tích chi tiết giá trị trống ở TotalCharges",
        "",
        f"- **Số lượng bản ghi trống:** {audit['tc_blank_count']} dòng.",
        f"- **Đặc điểm `tenure`:** {audit['tc_blank_tenure_dist']}",
        f"- **Đặc điểm nhãn `Churn`:** {audit['tc_blank_churn_dist']}",
        "",
        "> [!IMPORTANT]",
        "> Toàn bộ 11 dòng có `TotalCharges` trống đều có `tenure = 0` và `Churn = No`. Đây là các khách hàng mới gia nhập hệ thống trong kỳ tính cước hiện tại, chưa hoàn thành chu kỳ thanh toán tháng đầu tiên.",
        "> **Quy tắc làm sạch:** Chuyển chuỗi trống thành `NaN`. Trong pipeline huấn luyện, sử dụng `SimpleImputer(strategy='median')` được fit trên từng training fold để điền giá trị thiếu một cách khách quan, bảo toàn toàn bộ 7.043 mẫu dữ liệu.",
        "",
        "---",
        "",
        "## 4. Kiểm tra trùng lặp và xung đột dữ liệu",
        "",
        f"- **Trùng lặp toàn bộ dòng (Exact duplicate rows):** {audit['total_duplicate_rows']} dòng.",
        f"- **Trùng lặp mã khách hàng (`customerID`):** {audit['id_duplicates']} dòng. Mỗi mã khách hàng là duy nhất.",
        f"- **Trùng lặp 19 đặc trưng với mã ID khác nhau:** {audit['feature_duplicates']} dòng. Đây là các khách hàng độc lập có cùng hồ sơ gói cước; không được tự động xóa bỏ.",
        f"- **Xung đột nghiệp vụ điện thoại (`PhoneService='No'` & `MultipleLines='Yes'`):** {audit['phone_conflicts']} trường hợp.",
        f"- **Xung đột nghiệp vụ Internet (`InternetService='No'` & gói phụ `Yes/No`):** {audit['internet_conflicts']} trường hợp.",
        "",
        "> [!TIP]",
        "> Bộ dữ liệu gốc có độ nhất quán nghiệp vụ cao, không có lỗi xung đột logic chéo giữa các dịch vụ. Tất cả các giá trị 'No internet service' và 'No phone service' được giữ nguyên như các trạng thái hợp lệ.",
        "",
        "---",
        "",
        "## 5. Thống kê mô tả các thuộc tính số và ngoại lai",
        "",
        "| Thuộc tính | Min | Max | Trung bình | Độ lệch chuẩn | Trung vị (Median) | Q1 | Q3 | IQR | Số ngoại lai (IQR) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])
    for col, s in audit["numeric_summaries"].items():
        lines.append(
            f"| `{col}` | {s['min']} | {s['max']} | {s['mean']} | {s['std']} | {s['median']} | {s['q1']} | {s['q3']} | {s['iqr']:.2f} | {s['outliers_count']} |"
        )

    lines.extend([
        "",
        "> [!NOTE]",
        "> - `tenure` dao động từ 0 đến 72 tháng, không có giá trị âm hay ngoại lai kỹ thuật.",
        "> - `MonthlyCharges` dao động từ 18.25 đến 118.75 USD, phân bố hợp lý theo các gói cước.",
        "> - `TotalCharges` có đuôi kéo dài tới 8.684.80 USD nhưng phản ánh đúng tổng chi phí tích lũy theo thời gian của khách hàng lâu năm (không có giá trị âm).",
        "",
        "---",
        "",
        "## 6. Kết luận và quy tắc làm sạch chuẩn hóa",
        "",
        "1. **Bảo toàn số mẫu:** Giữ nguyên toàn bộ 7.043 bản ghi gốc.",
        "2. **Xử lý chuỗi và khoảng trắng:** Trim toàn bộ khoảng trắng ở các cột danh mục.",
        "3. **Chuyển kiểu `TotalCharges`:** Ép kiểu từ chuỗi sang `float64`, chuỗi trống thành `np.nan`.",
        "4. **Chuẩn hóa `SeniorCitizen`:** Giữ dạng chuỗi `'0'` / `'1'` để mã hóa One-Hot nhất quán.",
        "5. **Mã hóa nhãn `Churn`:** Chuyển `No` $\\rightarrow 0$, `Yes` $\\rightarrow 1$.",
        "6. **Không rò rỉ thông tin:** Toàn bộ phép điền thiếu (`SimpleImputer`) và chuẩn hóa (`StandardScaler`) chỉ fit trên từng fold huấn luyện.",
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def create_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    n_splits: int = 5,
    output_manifest_path: Optional[str] = None,
    source_file_path: Optional[str] = None,
    cleaning_version: str = "v1",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform stratified split into development (80%) and test (20%) sets.

    Creates 5-fold CV indices on the development set and generates split manifest.

    Returns:
        dev_df: Development subset (80%, 5,634 rows).
        test_df: Test subset (20%, 1,409 rows).
        manifest_df: Manifest tracking split membership and fold assignments.
    """
    if ID_COLUMN not in df.columns:
        raise ValueError(f"Required identifier column '{ID_COLUMN}' not found in DataFrame")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Required target column '{TARGET_COLUMN}' not found in DataFrame")

    file_hash = compute_file_sha256(source_file_path) if source_file_path and os.path.exists(source_file_path) else "N/A"

    # Stratified train/test split
    dev_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET_COLUMN],
        random_state=random_state,
        shuffle=True,
    )

    # Verification: Ensure disjoint customerIDs
    dev_ids = set(dev_df[ID_COLUMN])
    test_ids = set(test_df[ID_COLUMN])
    overlap = dev_ids.intersection(test_ids)
    if overlap:
        raise RuntimeError(f"Data leak detected! {len(overlap)} IDs exist in both dev and test sets")

    # Generate 5-fold Stratified CV splits on development set
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    dev_folds = np.zeros(len(dev_df), dtype=int)

    for fold_idx, (_, val_idx) in enumerate(skf.split(dev_df, dev_df[TARGET_COLUMN])):
        dev_folds[val_idx] = fold_idx

    # Build manifest DataFrame
    dev_records = []
    for (idx, row), fold in zip(dev_df.iterrows(), dev_folds):
        dev_records.append({
            "customerID": row[ID_COLUMN],
            "split": "dev",
            "fold": fold,
            "Churn": row[TARGET_COLUMN],
            "seed": random_state,
            "data_hash": file_hash,
            "cleaning_version": cleaning_version,
        })

    test_records = []
    for idx, row in test_df.iterrows():
        test_records.append({
            "customerID": row[ID_COLUMN],
            "split": "test",
            "fold": -1,  # -1 indicates held-out test
            "Churn": row[TARGET_COLUMN],
            "seed": random_state,
            "data_hash": file_hash,
            "cleaning_version": cleaning_version,
        })

    manifest_df = pd.DataFrame(dev_records + test_records)

    # Save manifest if path provided
    if output_manifest_path:
        os.makedirs(os.path.dirname(output_manifest_path), exist_ok=True)
        manifest_df.to_csv(output_manifest_path, index=False)

    return dev_df.copy(), test_df.copy(), manifest_df
