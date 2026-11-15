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
    record_actual_labels_batch,
    get_prediction_history,
    get_snapshots_with_labels,
    record_training_membership,
    get_training_memberships,
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


def test_record_actual_labels_batch(temp_db, sample_valid_df):
    bundle = load_model_bundle()
    preds = predict_records(sample_valid_df.iloc[:3], bundle)

    save_res = save_prediction_batch(
        inputs=sample_valid_df.iloc[:3],
        outputs=preds,
        source="batch_test",
        db_path=temp_db,
    )
    snaps = save_res["snapshot_ids"]

    # Valid batch update
    batch_df = pd.DataFrame([
        {"snapshot_id": snaps[0], "actual_churn": "Yes", "source": "call_center"},
        {"snapshot_id": snaps[1], "actual_churn": 0, "source": "billing_system"},
        {"snapshot_id": "invalid_snap_123", "actual_churn": "No"},  # Should be recorded as error
    ])

    batch_res = record_actual_labels_batch(batch_df, default_source="bulk_import", db_path=temp_db)
    assert batch_res["total"] == 3
    assert batch_res["success_count"] == 2
    assert batch_res["failed_count"] == 1
    assert len(batch_res["errors"]) == 1
    assert "does not exist" in batch_res["errors"][0]["reason"]

    # Verify history
    hist = get_prediction_history(has_label=True, db_path=temp_db)
    assert len(hist) == 2


def test_record_actual_label_earlier_than_snapshot_rejected(temp_db, sample_valid_df):
    bundle = load_model_bundle()
    preds = predict_records(sample_valid_df.iloc[[0]], bundle)
    save_res = save_prediction_batch(
        inputs=sample_valid_df.iloc[[0]],
        outputs=preds,
        source="single_test",
        db_path=temp_db,
    )
    snap_id = save_res["snapshot_ids"][0]

    # Attempt to record a label with observation date 10 days before snapshot
    with pytest.raises(ValueError, match="không được sớm hơn ngày tạo snapshot"):
        record_actual_label(
            snapshot_id=snap_id,
            label="Yes",
            observed_at="2020-01-01T00:00:00Z",
            source="crm_agent",
            db_path=temp_db,
        )


def test_training_membership_and_exclude_trained(temp_db, sample_valid_df):
    bundle = load_model_bundle()
    preds = predict_records(sample_valid_df.iloc[:4], bundle)
    save_res = save_prediction_batch(
        inputs=sample_valid_df.iloc[:4],
        outputs=preds,
        source="unit_test",
        db_path=temp_db,
    )
    snaps = save_res["snapshot_ids"]
    events = []
    for s in snaps:
        evt = record_actual_label(snapshot_id=s, label=1, observed_at="2026-11-15T00:00:00Z", source="test", db_path=temp_db)
        events.append(evt)

    # Initial query: all 4 have labels
    labeled_df = get_snapshots_with_labels(exclude_trained=True, db_path=temp_db)
    assert len(labeled_df) == 4

    # Record training membership for run_001
    run_id = "test_run_001"
    conn = get_db_connection(temp_db)
    conn.execute(
        "INSERT INTO training_runs (run_id, as_of, status, started_at) VALUES (?, ?, ?, ?)",
        (run_id, "2026-11-15T00:00:00Z", "COMPLETED", "2026-11-15T00:00:00Z"),
    )
    conn.commit()
    conn.close()

    mems = [
        {"snapshot_id": snaps[0], "partition": "train", "label_event_id": events[0]["event_id"]},
        {"snapshot_id": snaps[1], "partition": "train", "label_event_id": events[1]["event_id"]},
        {"snapshot_id": snaps[2], "partition": "comparison", "label_event_id": events[2]["event_id"]},
    ]
    record_training_membership(run_id, mems, db_path=temp_db)

    # Verify memberships recorded
    mem_df = get_training_memberships(run_id, db_path=temp_db)
    assert len(mem_df) == 3

    # Now query with exclude_trained=True: only snaps[3] remains
    remaining_df = get_snapshots_with_labels(exclude_trained=True, db_path=temp_db)
    assert len(remaining_df) == 1
    assert remaining_df["snapshot_id"].iloc[0] == snaps[3]


