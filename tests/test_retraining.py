"""Unit tests for retraining gate conditions, candidate comparison, promotion, and rollback."""

import os
import sys
import tempfile
import json
import pytest
import pandas as pd

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from telco_churn.storage import (
    init_db,
    save_prediction_batch,
    record_actual_label,
)
from telco_churn.inference import load_model_bundle, predict_records
from telco_churn.registry import (
    get_active_model_info,
    set_active_model,
    rollback_model,
    validate_artifact_bundle,
)
from telco_churn.retraining import (
    run_retraining,
    evaluate_gate_conditions,
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
