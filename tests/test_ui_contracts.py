"""Acceptance and integration tests for Plan 05 Streamlit UI contracts and business criteria."""

import os
import sys
import tempfile
import pandas as pd
import numpy as np
import pytest

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from telco_churn.inference import load_model_bundle, predict_records, compute_priority_level
from telco_churn.schema import validate_records, ALL_FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN
from telco_churn.storage import (
    init_db,
    save_prediction_batch,
    record_actual_label,
    record_actual_labels_batch,
    get_prediction_history,
)


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_ui_churn.db"
    db_str = str(db_file)
    init_db(db_str)
    return db_str


@pytest.fixture(scope="module")
def sample_valid_df():
    sample_path = os.path.join("data", "samples", "valid_sample.csv")
    return pd.read_csv(sample_path)


@pytest.fixture(scope="module")
def sample_invalid_df():
    sample_path = os.path.join("data", "samples", "invalid_schema_sample.csv")
    return pd.read_csv(sample_path)


def test_acceptance_criteria_1_single_and_batch_consistency(sample_valid_df):
    """Nghiệm thu 1: Nhập một khách và upload chính khách đó cho cùng kết quả trong sai số số học."""
    bundle = load_model_bundle()

    # Predict single customer (row 0)
    single_record = sample_valid_df.iloc[[0]].copy()
    single_result = predict_records(single_record, bundle)

    # Predict batch of all rows
    batch_result = predict_records(sample_valid_df, bundle)

    # Compare row 0 in batch vs single
    batch_first_row = batch_result.iloc[0]
    single_res = single_result.iloc[0]

    assert single_res["customerID"] == batch_first_row["customerID"]
    assert np.isclose(single_res["churn_score"], batch_first_row["churn_score"], atol=1e-5)
    assert single_res["churn_prediction"] == batch_first_row["churn_prediction"]
    assert single_res["churn_label"] == batch_first_row["churn_label"]
    assert single_res["priority_level"] == batch_first_row["priority_level"]
    assert single_res["threshold"] == batch_first_row["threshold"]
    assert single_res["model_version"] == batch_first_row["model_version"]


def test_acceptance_criteria_2_validation_error_clarity(sample_invalid_df):
    """Nghiệm thu 2: Người dùng biết rõ giá trị nào bị thiếu/sai và cách sửa."""
    valid_df, issues = validate_records(sample_invalid_df, mode="inference")

    assert len(issues) > 0
    # Check that issues contain detailed information
    for iss in issues:
        assert "row_index" in iss or "row" in iss or "column" in iss
        assert "message" in iss
        assert len(iss["message"]) > 0


def test_acceptance_criteria_3_csv_result_alignment(sample_valid_df):
    """Nghiệm thu 3: CSV kết quả giữ đúng số dòng hợp lệ và liên kết được với dòng nguồn."""
    bundle = load_model_bundle()
    results = predict_records(sample_valid_df, bundle)

    assert len(results) == len(sample_valid_df)
    assert results["input_row_id"].tolist() == list(range(len(sample_valid_df)))
    for idx in range(len(sample_valid_df)):
        assert results.iloc[idx]["customerID"] == sample_valid_df.iloc[idx]["customerID"]


def test_acceptance_criteria_4_idempotent_no_duplicate_rerun(temp_db, sample_valid_df):
    """Nghiệm thu 4: Nhãn/điểm không thay đổi vì bấm lưu hoặc rerun; không sinh bản ghi trùng."""
    bundle = load_model_bundle()
    preds = predict_records(sample_valid_df, bundle)

    idem_key = "test_key_rerun_12345"

    # First click
    res1 = save_prediction_batch(
        inputs=sample_valid_df,
        outputs=preds,
        source="csv_upload",
        idempotency_key=idem_key,
        db_path=temp_db,
    )
    assert res1["status"] == "created"

    # Streamlit rerun / second click
    res2 = save_prediction_batch(
        inputs=sample_valid_df,
        outputs=preds,
        source="csv_upload",
        idempotency_key=idem_key,
        db_path=temp_db,
    )
    assert res2["status"] == "existing"
    assert res2["batch_id"] == res1["batch_id"]

    history = get_prediction_history(limit=500, db_path=temp_db)
    assert len(history) == len(sample_valid_df)


def test_acceptance_criteria_5_traceability_and_label_updates(temp_db, sample_valid_df):
    """Nghiệm thu 5: Lưu được, đọc lại được, cập nhật nhãn thật được và truy vết version được."""
    bundle = load_model_bundle()
    preds = predict_records(sample_valid_df.iloc[:2], bundle)

    res = save_prediction_batch(
        inputs=sample_valid_df.iloc[:2],
        outputs=preds,
        source="trace_test",
        db_path=temp_db,
    )
    snap_id = res["snapshot_ids"][0]

    # Verify initial state: no label
    h_init = get_prediction_history(has_label=False, db_path=temp_db)
    assert len(h_init) == 2
    assert pd.isna(h_init.loc[h_init["snapshot_id"] == snap_id, "actual_churn"].values[0])

    # Record actual label
    evt1 = record_actual_label(
        snapshot_id=snap_id,
        label=1,
        observed_at="2026-11-15T10:00:00Z",
        source="call_center",
        db_path=temp_db,
    )
    assert evt1["actual_churn"] == 1

    # Verify query finds the labeled snapshot
    h_labeled = get_prediction_history(has_label=True, db_path=temp_db)
    assert len(h_labeled) == 1
    assert h_labeled["snapshot_id"].iloc[0] == snap_id
    assert h_labeled["actual_churn"].iloc[0] == 1
    assert h_labeled["model_version"].iloc[0] == bundle["version"]

    # Verify original prediction score and features were not modified
    assert np.isclose(h_labeled["churn_score"].iloc[0], preds.iloc[0]["churn_score"], atol=1e-5)
    assert h_labeled["predicted_label"].iloc[0] == preds.iloc[0]["churn_label"]


def test_acceptance_criteria_6_batch_label_upload_validation(temp_db, sample_valid_df):
    """Nghiệm thu 6: Cho nhập CSV nhãn theo snapshot_id; xác thực khóa và báo dòng không tìm thấy/trùng."""
    bundle = load_model_bundle()
    preds = predict_records(sample_valid_df.iloc[:2], bundle)

    save_res = save_prediction_batch(
        inputs=sample_valid_df.iloc[:2],
        outputs=preds,
        source="test_batch_labels",
        db_path=temp_db,
    )
    valid_snap = save_res["snapshot_ids"][0]

    df_test_labels = pd.DataFrame([
        {"snapshot_id": valid_snap, "actual_churn": "Yes", "source": "billing"},
        {"snapshot_id": "non_existent_snap", "actual_churn": "No"},  # Non-existent
        {"snapshot_id": valid_snap, "actual_churn": "No"},  # Duplicate within batch
    ])

    batch_res = record_actual_labels_batch(df_test_labels, db_path=temp_db)
    assert batch_res["total"] == 3
    assert batch_res["success_count"] == 1
    assert batch_res["failed_count"] == 2
    assert len(batch_res["errors"]) == 2
    reasons = [e["reason"] for e in batch_res["errors"]]
    assert any("does not exist" in r for r in reasons)
    assert any("trùng lặp" in r for r in reasons)
