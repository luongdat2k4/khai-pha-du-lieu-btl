"""Unit tests for model inference, schema compliance, priority tiers, and bundle loading."""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from telco_churn.inference import (
    load_model_bundle,
    predict_records,
    compute_priority_level,
    benchmark_inference_speed,
)
from telco_churn.schema import ALL_FEATURE_COLUMNS, ID_COLUMN


@pytest.fixture(scope="module")
def sample_valid_df():
    sample_path = os.path.join("data", "samples", "valid_sample.csv")
    return pd.read_csv(sample_path)


def test_priority_level_policy():
    assert compute_priority_level(0.15) == "Thấp"
    assert compute_priority_level(0.29) == "Thấp"
    assert compute_priority_level(0.30) == "Trung bình"
    assert compute_priority_level(0.55) == "Trung bình"
    assert compute_priority_level(0.60) == "Cao"
    assert compute_priority_level(0.95) == "Cao"


def test_load_active_model_bundle():
    bundle = load_model_bundle()
    assert "pipeline" in bundle
    assert "metadata" in bundle
    assert "class_1_idx" in bundle
    assert bundle["version"] is not None

    meta = bundle["metadata"]
    assert "model_type" in meta
    assert "best_params" in meta
    assert meta["threshold"] == 0.5
    assert meta["class_mapping"] == {"No": 0, "Yes": 1}

    pipeline = bundle["pipeline"]
    classifier = pipeline.named_steps["classifier"]
    assert hasattr(classifier, "classes_")
    assert 1 in classifier.classes_


def test_single_and_batch_prediction_schema(sample_valid_df):
    bundle = load_model_bundle()

    # Predict batch
    res_batch = predict_records(sample_valid_df, bundle)

    expected_cols = [
        "input_row_id", "customerID", "churn_prediction", "churn_label",
        "churn_score", "priority_level", "threshold", "model_version", "predicted_at"
    ]
    assert list(res_batch.columns) == expected_cols
    assert len(res_batch) == len(sample_valid_df)

    # Validate value constraints
    for _, row in res_batch.iterrows():
        assert row["churn_prediction"] in [0, 1]
        assert row["churn_label"] in ["No", "Yes"]
        assert 0.0 <= row["churn_score"] <= 1.0
        assert row["priority_level"] in ["Thấp", "Trung bình", "Cao"]
        assert row["threshold"] == 0.5

    # Single prediction
    single_row = sample_valid_df.iloc[[0]]
    res_single = predict_records(single_row, bundle)
    assert len(res_single) == 1
    assert res_single["customerID"].iloc[0] == sample_valid_df["customerID"].iloc[0]
    assert res_single["churn_score"].iloc[0] == res_batch["churn_score"].iloc[0]


def test_prediction_without_customer_id(sample_valid_df):
    bundle = load_model_bundle()
    df_no_id = sample_valid_df.drop(columns=[ID_COLUMN])
    res = predict_records(df_no_id, bundle)
    assert len(res) == len(sample_valid_df)
    assert res["customerID"].str.startswith("GEN_").all()


def test_missing_feature_column_raises(sample_valid_df):
    bundle = load_model_bundle()
    df_broken = sample_valid_df.drop(columns=["Contract"])
    with pytest.raises(ValueError, match="Missing required feature columns"):
        predict_records(df_broken, bundle)


def test_threshold_monotonicity(sample_valid_df):
    bundle = load_model_bundle()
    res_low_thresh = predict_records(sample_valid_df, bundle, threshold=0.1)
    res_high_thresh = predict_records(sample_valid_df, bundle, threshold=0.9)

    pos_low = (res_low_thresh["churn_prediction"] == 1).sum()
    pos_high = (res_high_thresh["churn_prediction"] == 1).sum()
    assert pos_low >= pos_high, "Lower threshold should yield at least as many churn predictions"
