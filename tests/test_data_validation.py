"""Unit tests for data validation, schema enforcement, and preprocessing pipelines."""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from telco_churn.schema import (
    validate_records,
    check_business_consistency,
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from telco_churn.preprocessing import build_preprocessor


def test_valid_sample_validation():
    sample_path = os.path.join("data", "samples", "valid_sample.csv")
    df = pd.read_csv(sample_path)
    valid_df, errors = validate_records(df, mode="inference")
    assert len(errors) == 0, f"Expected 0 errors on valid sample, got: {errors}"
    assert len(valid_df) == len(df)


def test_missing_column_error():
    sample_path = os.path.join("data", "samples", "valid_sample.csv")
    df = pd.read_csv(sample_path)
    df_missing = df.drop(columns=["MonthlyCharges"])
    valid_df, errors = validate_records(df_missing, mode="inference")
    assert len(valid_df) == 0
    assert any(e["code"] == "MISSING_COLUMN" and e["column"] == "MonthlyCharges" for e in errors)


def test_missing_totalcharges_handling():
    sample_path = os.path.join("data", "samples", "missing_totalcharges_sample.csv")
    df = pd.read_csv(sample_path)
    valid_df, errors = validate_records(df, mode="inference")
    # Missing/blank TotalCharges should convert to np.nan without being rejected
    assert len(valid_df) == len(df)
    assert valid_df["TotalCharges"].isna().all()


def test_invalid_schema_sample():
    sample_path = os.path.join("data", "samples", "invalid_schema_sample.csv")
    df = pd.read_csv(sample_path)
    valid_df, errors = validate_records(df, mode="inference")
    # All rows in invalid_schema_sample.csv have deliberate errors
    assert len(valid_df) < len(df)
    codes = [e["code"] for e in errors]
    assert "INVALID_CATEGORY" in codes
    assert "NEGATIVE_VALUE" in codes
    assert "INCONSISTENT_SERVICE" in codes


def test_business_consistency():
    # Phone conflict
    row_phone_conflict = pd.Series({
        "PhoneService": "No",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "tenure": 10,
        "MonthlyCharges": 30.0,
        "TotalCharges": 300.0,
    })
    issues = check_business_consistency(row_phone_conflict)
    assert any("MultipleLines is 'Yes'" in msg for msg in issues)

    # Internet conflict
    row_internet_conflict = pd.Series({
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "No",
        "TechSupport": "Yes",
        "tenure": 5,
        "MonthlyCharges": 20.0,
        "TotalCharges": 100.0,
    })
    issues = check_business_consistency(row_internet_conflict)
    assert any("TechSupport" in msg for msg in issues)


def test_target_mapping_in_training_mode():
    sample_path = os.path.join("data", "samples", "valid_sample.csv")
    df = pd.read_csv(sample_path)
    df["Churn"] = ["No", "No", "Yes", "No", "No", "Yes"]
    valid_df, errors = validate_records(df, mode="training")
    assert len(errors) == 0
    assert set(valid_df["Churn"].unique()).issubset({0, 1})
    assert list(valid_df["Churn"]) == [0, 0, 1, 0, 0, 1]


def test_preprocessor_structure():
    # Logistic Regression preprocessor should have scaler
    lr_prep = build_preprocessor("logistic")
    num_step_names = [name for name, _ in lr_prep.transformers[0][1].steps]
    assert "scaler" in num_step_names
    assert "imputer" in num_step_names

    # Tree preprocessor should not have scaler
    dt_prep = build_preprocessor("decision_tree")
    dt_num_step_names = [name for name, _ in dt_prep.transformers[0][1].steps]
    assert "scaler" not in dt_num_step_names
    assert "imputer" in dt_num_step_names

    # Invalid model kind
    with pytest.raises(ValueError):
        build_preprocessor("neural_network")
