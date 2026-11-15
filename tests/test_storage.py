"""Unit tests for SQLite storage, transactions, idempotency, and audit trails."""

import os
import sys
import tempfile
import sqlite3
import pytest
import pandas as pd

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from telco_churn.storage import (
    init_db,
    save_prediction_batch,
    record_actual_label,
    get_prediction_history,
    get_snapshots_with_labels,
    get_db_connection,
)
from telco_churn.inference import load_model_bundle, predict_records


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_churn.db"
    db_str = str(db_file)
    init_db(db_str)
    return db_str


@pytest.fixture(scope="module")
def sample_valid_df():
    sample_path = os.path.join("data", "samples", "valid_sample.csv")
    return pd.read_csv(sample_path)


def test_db_initialization_tables(temp_db):
    conn = get_db_connection(temp_db)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {r[0] for r in cur.fetchall()}
    conn.close()

    expected_tables = {
        "schema_migrations",
        "customers",
        "customer_snapshots",
        "prediction_batches",
        "predictions",
        "actual_label_events",
        "model_versions",
        "training_runs",
        "training_membership",
    }
    assert expected_tables.issubset(tables)


def test_save_prediction_batch(temp_db, sample_valid_df):
    bundle = load_model_bundle()
    preds = predict_records(sample_valid_df, bundle)

    res = save_prediction_batch(
        inputs=sample_valid_df,
        outputs=preds,
        source="unit_test",
        idempotency_key="batch_test_001",
        db_path=temp_db,
    )

    assert res["status"] == "created"
    assert res["count"] == len(sample_valid_df)
    assert len(res["snapshot_ids"]) == len(sample_valid_df)
    assert len(res["prediction_ids"]) == len(sample_valid_df)

    history = get_prediction_history(limit=50, db_path=temp_db)
    assert len(history) == len(sample_valid_df)
    assert "churn_score" in history.columns
    assert "predicted_label" in history.columns


def test_idempotency_prevents_duplicate_records(temp_db, sample_valid_df):
    bundle = load_model_bundle()
    preds = predict_records(sample_valid_df, bundle)

    key = "idempotent_test_key_123"

    # First save
    res1 = save_prediction_batch(
        inputs=sample_valid_df,
        outputs=preds,
        source="batch_upload",
        idempotency_key=key,
        db_path=temp_db,
    )
    assert res1["status"] == "created"

    # Second save with identical idempotency key (simulating Streamlit rerun)
    res2 = save_prediction_batch(
        inputs=sample_valid_df,
        outputs=preds,
        source="batch_upload",
        idempotency_key=key,
        db_path=temp_db,
    )
    assert res2["status"] == "existing"
    assert res2["batch_id"] == res1["batch_id"]

    # Verify no duplicate entries in database
    history = get_prediction_history(limit=100, db_path=temp_db)
    assert len(history) == len(sample_valid_df)


def test_record_actual_label_and_audit_history(temp_db, sample_valid_df):
    bundle = load_model_bundle()
    preds = predict_records(sample_valid_df.iloc[[0]], bundle)

    save_res = save_prediction_batch(
        inputs=sample_valid_df.iloc[[0]],
        outputs=preds,
        source="single_form",
        db_path=temp_db,
    )
    snapshot_id = save_res["snapshot_ids"][0]

    # Record first actual label
    evt1 = record_actual_label(
        snapshot_id=snapshot_id,
        label="Yes",
        observed_at="2026-11-15T00:00:00Z",
        source="crm_agent",
        note="Customer cancelled over phone",
        db_path=temp_db,
    )
    assert evt1["actual_churn"] == 1
    assert evt1["supersedes_event_id"] is None

    # Check history reflects label
    hist1 = get_prediction_history(has_label=True, db_path=temp_db)
    assert len(hist1) == 1
    assert hist1["actual_churn"].iloc[0] == 1
    assert hist1["label_source"].iloc[0] == "crm_agent"

    # Record label correction (audit trail)
    evt2 = record_actual_label(
        snapshot_id=snapshot_id,
        label="No",
        observed_at="2026-11-15T01:00:00Z",
        source="retention_audit",
        note="Retention offer accepted; customer stayed",
        db_path=temp_db,
    )
    assert evt2["actual_churn"] == 0
    assert evt2["supersedes_event_id"] == evt1["event_id"]

    # History should now reflect latest active label 0
    hist2 = get_prediction_history(has_label=True, db_path=temp_db)
    assert len(hist2) == 1
    assert hist2["actual_churn"].iloc[0] == 0
    assert hist2["label_source"].iloc[0] == "retention_audit"


def test_record_actual_label_invalid_snapshot(temp_db):
    with pytest.raises(ValueError, match="does not exist in customer_snapshots"):
        record_actual_label(
            snapshot_id="non_existent_id",
            label="Yes",
            observed_at="2026-11-15T00:00:00Z",
            source="crm",
            db_path=temp_db,
        )
