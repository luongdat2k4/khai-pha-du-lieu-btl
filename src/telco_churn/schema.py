"""Schema definitions and input validation for Telco Customer Churn."""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

# Column definitions
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

NUMERIC_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
ALL_COLUMNS = [ID_COLUMN] + ALL_FEATURE_COLUMNS + [TARGET_COLUMN]

# Categorical allowed domains
CATEGORICAL_DOMAINS: Dict[str, List[str]] = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["0", "1", 0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

INTERNET_ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

TARGET_MAPPING: Dict[str, int] = {
    "No": 0,
    "Yes": 1,
}

REVERSE_TARGET_MAPPING: Dict[int, str] = {
    0: "No",
    1: "Yes",
}


def check_business_consistency(row: pd.Series) -> List[str]:
    """Check logical consistency rules across services for a single record."""
    issues = []

    # Phone service consistency
    phone = str(row.get("PhoneService", "")).strip()
    multiple = str(row.get("MultipleLines", "")).strip()
    if phone == "No" and multiple == "Yes":
        issues.append("PhoneService is 'No' but MultipleLines is 'Yes'")

    # Internet service consistency
    internet = str(row.get("InternetService", "")).strip()
    if internet == "No":
        for svc in INTERNET_ADDON_SERVICES:
            val = str(row.get(svc, "")).strip()
            if val in ["Yes", "No"]:
                issues.append(f"InternetService is 'No' but {svc} is '{val}' instead of 'No internet service'")

    # Numeric bounds
    tenure = row.get("tenure")
    if pd.notna(tenure):
        try:
            if float(tenure) < 0:
                issues.append(f"tenure ({tenure}) cannot be negative")
        except (ValueError, TypeError):
            issues.append(f"tenure ({tenure}) is not a valid number")

    monthly = row.get("MonthlyCharges")
    if pd.notna(monthly):
        try:
            if float(monthly) < 0:
                issues.append(f"MonthlyCharges ({monthly}) cannot be negative")
        except (ValueError, TypeError):
            issues.append(f"MonthlyCharges ({monthly}) is not a valid number")

    total = row.get("TotalCharges")
    if pd.notna(total) and str(total).strip() != "":
        try:
            if float(total) < 0:
                issues.append(f"TotalCharges ({total}) cannot be negative")
        except (ValueError, TypeError):
            issues.append(f"TotalCharges ({total}) is not a valid number")

    return issues


def validate_records(
    df: pd.DataFrame,
    mode: str = "inference"
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Validate DataFrame against Telco Churn schema.

    Args:
        df: Input raw DataFrame.
        mode: 'training' or 'inference'.

    Returns:
        valid_df: Standardized valid records.
        errors: List of error dictionaries with 'row', 'column', 'code', 'message'.
    """
    errors: List[Dict[str, Any]] = []
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if df.empty:
        return pd.DataFrame(), errors

    # Check required columns
    required_cols = list(ALL_FEATURE_COLUMNS)
    if mode == "training":
        required_cols.append(TARGET_COLUMN)

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        for c in missing_cols:
            errors.append({
                "row": None,
                "column": c,
                "code": "MISSING_COLUMN",
                "message": f"Required column '{c}' is missing"
            })
        return pd.DataFrame(), errors

    processed_df = df.copy()
    invalid_row_indices = set()

    # Standardize string fields: trim whitespace
    for col in processed_df.columns:
        if processed_df[col].dtype == object:
            processed_df[col] = processed_df[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )

    # Standardize SeniorCitizen to string "0" or "1" for consistent categorical encoding
    if "SeniorCitizen" in processed_df.columns:
        def _clean_sc(val):
            if pd.isna(val):
                return val
            s = str(val).strip()
            if s in ["0", "0.0"]:
                return "0"
            if s in ["1", "1.0"]:
                return "1"
            return s
        processed_df["SeniorCitizen"] = processed_df["SeniorCitizen"].apply(_clean_sc)

    # Convert TotalCharges: convert empty/space to NaN, validate non-convertible
    if "TotalCharges" in processed_df.columns:
        def _convert_total_charges(idx, val):
            if pd.isna(val) or val == "" or (isinstance(val, str) and val.strip() == ""):
                return np.nan
            try:
                num = float(val)
                if num < 0:
                    errors.append({
                        "row": idx,
                        "column": "TotalCharges",
                        "code": "NEGATIVE_VALUE",
                        "message": f"TotalCharges cannot be negative: {val}"
                    })
                    invalid_row_indices.add(idx)
                    return np.nan
                return num
            except (ValueError, TypeError):
                errors.append({
                    "row": idx,
                    "column": "TotalCharges",
                    "code": "INVALID_NUMERIC",
                    "message": f"Cannot convert TotalCharges to float: '{val}'"
                })
                invalid_row_indices.add(idx)
                return np.nan

        converted_tc = [
            _convert_total_charges(idx, val)
            for idx, val in zip(processed_df.index, processed_df["TotalCharges"])
        ]
        processed_df["TotalCharges"] = converted_tc

    # Convert tenure and MonthlyCharges to numeric
    for col in ["tenure", "MonthlyCharges"]:
        if col in processed_df.columns:
            for idx, val in zip(processed_df.index, processed_df[col]):
                if pd.isna(val) or val == "":
                    continue
                try:
                    num = float(val)
                    if num < 0:
                        errors.append({
                            "row": idx,
                            "column": col,
                            "code": "NEGATIVE_VALUE",
                            "message": f"{col} cannot be negative: {val}"
                        })
                        invalid_row_indices.add(idx)
                except (ValueError, TypeError):
                    errors.append({
                        "row": idx,
                        "column": col,
                        "code": "INVALID_NUMERIC",
                        "message": f"Cannot convert {col} to numeric: '{val}'"
                    })
                    invalid_row_indices.add(idx)

    # Validate categorical domains
    for col in CATEGORICAL_COLUMNS:
        if col not in processed_df.columns:
            continue
        valid_domain = [str(d) for d in CATEGORICAL_DOMAINS[col]]
        for idx, val in zip(processed_df.index, processed_df[col]):
            if pd.isna(val) or val == "":
                continue
            str_val = str(val)
            if str_val not in valid_domain:
                errors.append({
                    "row": idx,
                    "column": col,
                    "code": "INVALID_CATEGORY",
                    "message": f"Value '{str_val}' in {col} is not in allowed domain: {valid_domain}"
                })
                invalid_row_indices.add(idx)

    # Validate target column if in training mode
    if mode == "training" and TARGET_COLUMN in processed_df.columns:
        for idx, val in zip(processed_df.index, processed_df[TARGET_COLUMN]):
            if pd.isna(val) or val == "":
                errors.append({
                    "row": idx,
                    "column": TARGET_COLUMN,
                    "code": "MISSING_TARGET",
                    "message": "Target Churn is missing"
                })
                invalid_row_indices.add(idx)
            else:
                str_val = str(val)
                if str_val not in ["Yes", "No", "1", "0", 1, 0]:
                    errors.append({
                        "row": idx,
                        "column": TARGET_COLUMN,
                        "code": "INVALID_TARGET",
                        "message": f"Target Churn value '{str_val}' must be Yes/No or 1/0"
                    })
                    invalid_row_indices.add(idx)

    # Check business consistency
    for idx, row in processed_df.iterrows():
        issues = check_business_consistency(row)
        for issue in issues:
            errors.append({
                "row": idx,
                "column": "BUSINESS_RULE",
                "code": "INCONSISTENT_SERVICE",
                "message": issue
            })
            invalid_row_indices.add(idx)

    # Filter out invalid rows
    valid_df = processed_df.drop(index=list(invalid_row_indices)).copy()

    # Standardize target in training mode to 0/1
    if mode == "training" and TARGET_COLUMN in valid_df.columns:
        valid_df[TARGET_COLUMN] = valid_df[TARGET_COLUMN].apply(
            lambda x: TARGET_MAPPING[x] if str(x) in TARGET_MAPPING else int(x)
        )

    return valid_df, errors
