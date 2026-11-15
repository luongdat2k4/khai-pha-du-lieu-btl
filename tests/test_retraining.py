"""Unit tests for retraining gate conditions, candidate comparison, promotion, and rollback."""

import os
import sys
import tempfile
import json
import shutil
import pytest
import pandas as pd

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from telco_churn.storage import (
    init_db,
    save_prediction_batch,
    record_actual_label,
    get_training_runs,
    get_db_connection,
)
from telco_churn.inference import load_model_bundle, predict_records
from telco_churn.registry import (
    get_active_model_info,
    set_active_model,
    rollback_model,
    reconcile_active_model,
    validate_artifact_bundle,
)
from telco_churn.retraining import (
    run_retraining,
    evaluate_gate_conditions,
    split_by_customer_group,
    RetrainingLock,
    load_retraining_config,
)


@pytest.fixture
def temp_env(tmp_path):
    db_file = tmp_path / "test_churn.db"
    db_str = str(db_file)
    init_db(db_str)
    return {
        "db_path": db_str,
        "tmp_dir": str(tmp_path),
    }


@pytest.fixture
def temp_artifacts(tmp_path):
    art_dir = tmp_path / "artifacts"
    shutil.copytree("artifacts", str(art_dir))
    return str(art_dir)


@pytest.fixture(scope="module")
def sample_valid_df():
    sample_path = os.path.join("data", "samples", "valid_sample.csv")
    return pd.read_csv(sample_path)


def test_validate_artifact_bundle():
    # Should validate current model_001
    assert validate_artifact_bundle("model_001", artifacts_dir="artifacts") is True


def test_gate_insufficient_data(temp_env):
    res = run_retraining(
        dry_run=True,
        db_path=temp_env["db_path"],
        artifacts_dir="artifacts",
    )
    assert res["gate_passed"] is False
    assert res["decision"] == "SKIP"
    assert "Chưa đủ dữ liệu mới" in res["reason"]


def test_retraining_lifecycle_forced_gate(temp_env, sample_valid_df):
    bundle = load_model_bundle()
    preds = predict_records(sample_valid_df, bundle)

    save_res = save_prediction_batch(
        inputs=sample_valid_df,
        outputs=preds,
        source="retrain_test",
        is_synthetic=True,
        db_path=temp_env["db_path"],
    )

    # Label all snapshots: mix of 1 and 0
    for idx, snap_id in enumerate(save_res["snapshot_ids"]):
        lbl = 1 if idx % 2 == 0 else 0
        record_actual_label(
            snapshot_id=snap_id,
            label=lbl,
            observed_at="2026-11-15T00:00:00Z",
            source="manual_test",
            db_path=temp_env["db_path"],
        )

    # Run with ignore_gate=True and allow_synthetic=True, dry_run=True
    res = run_retraining(
        dry_run=True,
        ignore_gate=True,
        allow_synthetic=True,
        db_path=temp_env["db_path"],
        artifacts_dir="artifacts",
    )

    assert res["decision"] in ["PROMOTE", "REJECT"]
    assert "candidate_metrics" in res
    assert "champion_metrics" in res
    assert res["dry_run"] is True

    # Active model should remain unchanged in dry_run mode
    active_info = get_active_model_info("artifacts")
    assert active_info["active_version"] == "model_001"


def test_rollback_model_error_when_no_prior(temp_env):
    # If no prior version, rollback should raise ValueError or handle cleanly
    active_path = os.path.join("artifacts", "active_model.json")
    with open(active_path, "r", encoding="utf-8") as f:
        curr = json.load(f)

    if curr.get("previous_version") is None:
        with pytest.raises(ValueError, match="No previous model version"):
            rollback_model(artifacts_dir="artifacts", db_path=temp_env["db_path"])


def test_scenario_a_gate_insufficient_skip(temp_env):
    """Scenario A: Insufficient data (<200) skips retraining and preserves Champion (Plan 06 Section 8)."""
    res = run_retraining(
        dry_run=True,
        ignore_gate=False,
        db_path=temp_env["db_path"],
        artifacts_dir="artifacts",
    )
    assert res["gate_passed"] is False
    assert res["decision"] == "SKIP"
    assert "Chưa đủ dữ liệu mới" in res["reason"]

    runs = get_training_runs(limit=1, db_path=temp_env["db_path"])
    assert not runs.empty
    assert runs.iloc[0]["decision"] == "SKIP"


def test_scenario_b_synthetic_candidate_comparison(temp_env, sample_valid_df):
    """Scenario B: Synthetic data evaluated on identical comparison set (Plan 06 Section 8)."""
    bundle = load_model_bundle()
    preds = predict_records(sample_valid_df, bundle)

    save_res = save_prediction_batch(
        inputs=sample_valid_df,
        outputs=preds,
        source="synthetic_test",
        is_synthetic=True,
        db_path=temp_env["db_path"],
    )

    for idx, snap_id in enumerate(save_res["snapshot_ids"]):
        record_actual_label(
            snapshot_id=snap_id,
            label=1 if idx % 2 == 0 else 0,
            observed_at="2026-11-15T00:00:00Z",
            source="synthetic_source",
            db_path=temp_env["db_path"],
        )

    res = run_retraining(
        dry_run=True,
        ignore_gate=True,
        allow_synthetic=True,
        db_path=temp_env["db_path"],
        artifacts_dir="artifacts",
    )

    assert res["decision"] in ["PROMOTE", "REJECT"]
    assert res["candidate_metrics"] is not None
    assert res["champion_metrics"] is not None
    assert "prediction_time_ms" in res["candidate_metrics"]
    assert "prediction_time_ms" in res["champion_metrics"]
    assert res["snapshot_path"] is not None
    assert res["snapshot_hash"] is not None
    assert res["dry_run"] is True


def test_scenario_c_candidate_rejection(temp_env, sample_valid_df):
    """Scenario C: Inferior or insufficient gain candidate rejected, champion preserved (Plan 06 Section 8)."""
    bundle = load_model_bundle()
    preds = predict_records(sample_valid_df, bundle)

    save_res = save_prediction_batch(
        inputs=sample_valid_df,
        outputs=preds,
        source="test_c",
        is_synthetic=True,
        db_path=temp_env["db_path"],
    )

    for idx, snap_id in enumerate(save_res["snapshot_ids"]):
        record_actual_label(
            snapshot_id=snap_id,
            label=1 if idx % 2 == 0 else 0,
            observed_at="2026-11-15T00:00:00Z",
            source="test_c",
            db_path=temp_env["db_path"],
        )

    cfg = load_retraining_config()
    # Require 0.99 F1 gain (impossible) to force rejection
    cfg["promotion_policy"]["min_f1_gain"] = 0.99

    res = run_retraining(
        dry_run=True,
        ignore_gate=True,
        allow_synthetic=True,
        config=cfg,
        db_path=temp_env["db_path"],
        artifacts_dir="artifacts",
    )

    assert res["decision"] == "REJECT"
    assert "Candidate không đạt điều kiện thăng cấp" in res["reason"]
    active = get_active_model_info("artifacts")
    assert active["active_version"] == "model_001"


def test_scenario_d_candidate_promotion_and_rollback(temp_env, temp_artifacts, sample_valid_df):
    """Scenario D: Candidate promoted, active pointer updated atomically, rollback verified (Plan 06 Section 8)."""
    bundle = load_model_bundle(artifacts_dir=temp_artifacts)
    preds = predict_records(sample_valid_df, bundle)

    save_res = save_prediction_batch(
        inputs=sample_valid_df,
        outputs=preds,
        source="test_d",
        is_synthetic=True,
        db_path=temp_env["db_path"],
    )

    for idx, snap_id in enumerate(save_res["snapshot_ids"]):
        record_actual_label(
            snapshot_id=snap_id,
            label=1 if idx % 2 == 0 else 0,
            observed_at="2026-11-15T00:00:00Z",
            source="test_d",
            db_path=temp_env["db_path"],
        )

    cfg = load_retraining_config()
    # Set permissive criteria so candidate is promoted
    cfg["promotion_policy"]["min_f1_gain"] = -1.0
    cfg["promotion_policy"]["max_recall_drop"] = 1.0

    res = run_retraining(
        dry_run=False,
        ignore_gate=True,
        allow_synthetic=True,
        config=cfg,
        db_path=temp_env["db_path"],
        artifacts_dir=temp_artifacts,
    )

    assert res["decision"] == "PROMOTE"
    new_ver = res["candidate_version"]
    assert new_ver is not None
    assert new_ver != "model_001"

    # Verify active pointer updated
    active = get_active_model_info(temp_artifacts)
    assert active["active_version"] == new_ver
    assert active["previous_version"] == "model_001"

    # Verify candidate bundle is valid
    assert validate_artifact_bundle(new_ver, artifacts_dir=temp_artifacts) is True

    # Test Rollback
    rb_res = rollback_model(artifacts_dir=temp_artifacts, db_path=temp_env["db_path"])
    assert rb_res["active_version"] == "model_001"
    active_after_rb = get_active_model_info(temp_artifacts)
    assert active_after_rb["active_version"] == "model_001"


def test_customer_group_split_prevents_leakage():
    """Verify customer group splitting ensures no customer is in both train and comparison (Plan 06 Section 5)."""
    data = []
    # Create 10 customers, each having 3 snapshots
    for cid in range(10):
        for rep in range(3):
            data.append({
                "snapshot_id": f"snap_{cid}_{rep}",
                "customer_id": f"CUST_{cid:03d}",
                "actual_churn": 1 if cid % 2 == 0 else 0,
            })
    df = pd.DataFrame(data)

    train_df, comp_df = split_by_customer_group(
        df,
        group_col="customer_id",
        target_col="actual_churn",
        comp_ratio=0.3,
        random_state=42,
    )

    train_custs = set(train_df["customer_id"])
    comp_custs = set(comp_df["customer_id"])

    assert len(train_custs.intersection(comp_custs)) == 0
    assert len(train_df) + len(comp_df) == len(df)


def test_retraining_concurrency_lock(temp_env):
    """Verify RetrainingLock prevents simultaneous retraining processes (Plan 06 Section 7)."""
    with RetrainingLock(temp_env["tmp_dir"]):
        # Attempt to acquire another lock on same directory
        with pytest.raises(RuntimeError, match="Một tác vụ huấn luyện lại đang được thực thi"):
            with RetrainingLock(temp_env["tmp_dir"]):
                pass


def test_schedule_gate_condition(sample_valid_df):
    """Verify schedule check enforces 1st of month (Plan 06 Section 4)."""
    sample_with_label = sample_valid_df.copy()
    sample_with_label["actual_churn"] = 1

    # Date is 15th of the month -> should fail schedule check
    passed_15, details_15 = evaluate_gate_conditions(
        df_labeled=sample_with_label,
        gate_config={"min_new_snapshots": 1, "min_pos_labels": 1, "min_neg_labels": 0},
        check_schedule=True,
        as_of="2026-11-15T02:00:00Z",
    )
    assert any("Yêu cầu ngày 01 hàng tháng" in f for f in details_15["failures"])

    # Date is 1st of the month -> schedule check passes
    passed_1, details_1 = evaluate_gate_conditions(
        df_labeled=sample_with_label,
        gate_config={"min_new_snapshots": 1, "min_pos_labels": 1, "min_neg_labels": 0},
        check_schedule=True,
        as_of="2026-12-01T02:00:00Z",
    )
    assert not any("Yêu cầu ngày 01 hàng tháng" in f for f in details_1["failures"])


def test_reconcile_active_model(temp_env, temp_artifacts):
    """Verify active model reconciliation between filesystem pointer and SQLite (Plan 06 Section 7)."""
    reconciled = reconcile_active_model(artifacts_dir=temp_artifacts, db_path=temp_env["db_path"])
    assert reconciled["active_version"] == "model_001"

    # Simulate desync in SQLite
    conn = get_db_connection(temp_env["db_path"])
    conn.execute("UPDATE model_versions SET status = 'archived' WHERE model_version = 'model_001'")
    conn.commit()

    # Reconcile should fix it
    reconcile_active_model(artifacts_dir=temp_artifacts, db_path=temp_env["db_path"])
    cur = conn.cursor()
    cur.execute("SELECT status FROM model_versions WHERE model_version = 'model_001'")
    row = cur.fetchone()
    conn.close()
    assert row["status"] == "active"

