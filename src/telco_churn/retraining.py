"""Continuous training, monthly gates, candidate comparison, and safe promotion."""

import os
import json
import uuid
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)
import joblib

from .schema import (
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    validate_records,
)
from .data import load_raw_data, compute_file_sha256
from .preprocessing import build_preprocessor
from .training import build_base_classifier, get_environment_metadata
from .inference import load_model_bundle, predict_records
from .storage import (
    get_db_connection,
    get_snapshots_with_labels,
    DEFAULT_DB_PATH,
)
from .registry import (
    get_active_model_info,
    set_active_model,
    register_model_version,
    validate_artifact_bundle,
)

DEFAULT_CONFIG_PATH = os.path.join("configs", "retraining.json")


def load_retraining_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load retraining configuration from JSON file."""
    path = config_path or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        # Default fallback config conforming to Plan 06
        return {
            "gate": {
                "min_new_snapshots": 200,
                "min_pos_labels": 30,
                "min_neg_labels": 30,
                "require_both_classes": True,
                "comparison_ratio": 0.25,
                "allow_synthetic": False,
            },
            "promotion_policy": {
                "min_f1_gain": 0.01,
                "max_recall_drop": 0.02,
                "threshold": 0.5,
            },
            "training": {
                "random_state": 42,
                "use_base_dev_data": True,
                "manifest_path": "data/splits/split_manifest.csv",
                "raw_data_path": "telco-customer-churn-on-icp4d/data/Telco-Customer-Churn.csv",
            },
            "paths": {
                "db_path": DEFAULT_DB_PATH,
                "artifacts_dir": "artifacts",
                "reports_dir": "reports",
            },
        }

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_gate_conditions(
    df_labeled: pd.DataFrame,
    gate_config: Dict[str, Any],
    manifest_path: str = "data/splits/split_manifest.csv",
) -> Tuple[bool, Dict[str, Any]]:
    """Verify all 6 gate conditions specified in Plan 06 Section 4."""
    total_count = len(df_labeled)
    min_snapshots = gate_config.get("min_new_snapshots", 200)
    min_pos = gate_config.get("min_pos_labels", 30)
    min_neg = gate_config.get("min_neg_labels", 30)

    details = {
        "total_eligible_snapshots": total_count,
        "min_snapshots_required": min_snapshots,
        "class_1_count": 0,
        "class_0_count": 0,
        "min_pos_required": min_pos,
        "min_neg_required": min_neg,
        "schema_valid": False,
        "no_test_leakage": True,
        "failures": [],
    }

    if total_count < min_snapshots:
        details["failures"].append(
            f"Chưa đủ dữ liệu mới ({total_count}/{min_snapshots} snapshots)."
        )

    if total_count > 0 and "actual_churn" in df_labeled.columns:
        c1 = int((df_labeled["actual_churn"] == 1).sum())
        c0 = int((df_labeled["actual_churn"] == 0).sum())
        details["class_1_count"] = c1
        details["class_0_count"] = c0

        if c1 < min_pos or c0 < min_neg:
            details["failures"].append(
                f"Chưa đủ cân bằng lớp (Yes={c1}/{min_pos}, No={c0}/{min_neg})."
            )

        # Check required columns
        missing_features = [col for col in ALL_FEATURE_COLUMNS if col not in df_labeled.columns]
        if missing_features:
            details["failures"].append(f"Thiếu các cột đặc trưng: {missing_features}")
        else:
            details["schema_valid"] = True

        # Check test set leakage
        if os.path.exists(manifest_path) and "customerID" in df_labeled.columns:
            manifest = pd.read_csv(manifest_path)
            test_ids = set(manifest[manifest["split"] == "test"]["customerID"].astype(str))
            overlap = set(df_labeled["customerID"].astype(str)).intersection(test_ids)
            if overlap:
                details["no_test_leakage"] = False
                details["failures"].append(
                    f"Rò rỉ dữ liệu: {len(overlap)} khách hàng trùng với tập Test gốc 20% đã khóa."
                )

    passed = len(details["failures"]) == 0
    return passed, details


def train_candidate_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str,
    params: Dict[str, Any],
    random_state: int = 42,
) -> Any:
    """Build and fit candidate pipeline with champion family and parameters."""
    from sklearn.pipeline import Pipeline

    preprocessor = build_preprocessor(model_type)
    classifier = build_base_classifier(model_type, random_state=random_state, **params)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def compute_metrics_on_comparison(
    pipeline: Any,
    X_comp: pd.DataFrame,
    y_comp: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate pipeline on holdout comparison set."""
    probs = pipeline.predict_proba(X_comp)[:, 1]
    preds = (probs >= threshold).astype(int)

    f1 = float(f1_score(y_comp, preds, pos_label=1, zero_division=0))
    prec = float(precision_score(y_comp, preds, pos_label=1, zero_division=0))
    rec = float(recall_score(y_comp, preds, pos_label=1, zero_division=0))
    acc = float(accuracy_score(y_comp, preds))
    try:
        roc = float(roc_auc_score(y_comp, probs))
    except Exception:
        roc = 0.5

    cm = confusion_matrix(y_comp, preds).tolist()

    return {
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "roc_auc": round(roc, 4),
        "accuracy": round(acc, 4),
        "confusion_matrix": cm,
        "sample_count": len(y_comp),
        "class_1_count": int((y_comp == 1).sum()),
        "class_0_count": int((y_comp == 0).sum()),
    }


def get_next_model_version(artifacts_dir: str = "artifacts") -> str:
    """Determine the next sequential model version identifier."""
    models_dir = os.path.join(artifacts_dir, "models")
    if not os.path.exists(models_dir):
        return "model_001"

    existing_nums = []
    for d in os.listdir(models_dir):
        if d.startswith("model_"):
            try:
                num = int(d.replace("model_", ""))
                existing_nums.append(num)
            except ValueError:
                continue

    next_num = max(existing_nums, default=0) + 1
    return f"model_{next_num:03d}"


def run_retraining(
    as_of: Optional[str] = None,
    dry_run: bool = False,
    config: Optional[Union[str, Dict[str, Any]]] = None,
    ignore_gate: bool = False,
    allow_synthetic: Optional[bool] = None,
    db_path: Optional[str] = None,
    artifacts_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the full retraining lifecycle conforming to Plan 04 Section 3 and Plan 06.

    Returns:
        Summary dict containing run_id, gate result, candidate metrics, decision, and rationale.
    """
    start_time = datetime.now(timezone.utc)
    run_id = f"retrain_{start_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    as_of_ts = as_of or start_time.isoformat()

    # Load configuration
    cfg = load_retraining_config(config) if isinstance(config, (str, type(None))) else config
    gate_cfg = cfg.get("gate", {})
    promo_cfg = cfg.get("promotion_policy", {})
    train_cfg = cfg.get("training", {})
    paths_cfg = cfg.get("paths", {})

    target_db = db_path or paths_cfg.get("db_path", DEFAULT_DB_PATH)
    target_artifacts = artifacts_dir or paths_cfg.get("artifacts_dir", "artifacts")
    manifest_path = train_cfg.get("manifest_path", "data/splits/split_manifest.csv")
    raw_data_path = train_cfg.get("raw_data_path", "telco-customer-churn-on-icp4d/data/Telco-Customer-Churn.csv")

    effective_allow_synth = allow_synthetic if allow_synthetic is not None else gate_cfg.get("allow_synthetic", False)

    # 1. Retrieve eligible labeled snapshots from SQLite
    df_labeled = get_snapshots_with_labels(
        as_of=as_of_ts,
        include_synthetic=effective_allow_synth,
        db_path=target_db,
    )

    # 2. Evaluate gate conditions
    gate_passed, gate_details = evaluate_gate_conditions(df_labeled, gate_cfg, manifest_path)

    # If gate failed and not in forced/demo mode
    if not gate_passed and not ignore_gate:
        reason = "; ".join(gate_details["failures"])
        # Log to DB
        _record_training_run_db(
            run_id=run_id,
            as_of=as_of_ts,
            config_json=json.dumps(cfg),
            status="SKIPPED",
            decision="SKIP",
            reason=reason,
            started_at=start_time.isoformat(),
            finished_at=datetime.now(timezone.utc).isoformat(),
            db_path=target_db,
        )
        return {
            "run_id": run_id,
            "as_of": as_of_ts,
            "gate_passed": False,
            "gate_details": gate_details,
            "decision": "SKIP",
            "reason": reason,
            "candidate_metrics": None,
            "champion_metrics": None,
            "dry_run": dry_run,
        }

    # 3. Partition eligible new data into Train and Comparison holdout
    comp_ratio = float(gate_cfg.get("comparison_ratio", 0.25))
    if len(df_labeled) >= 4 and len(df_labeled["actual_churn"].unique()) > 1:
        new_train_df, new_comp_df = train_test_split(
            df_labeled,
            test_size=comp_ratio,
            stratify=df_labeled["actual_churn"],
            random_state=train_cfg.get("random_state", 42),
        )
    else:
        # Minimal partition for testing
        new_train_df = df_labeled
        new_comp_df = df_labeled

    # 4. Load Champion model bundle
    champion_info = get_active_model_info(target_artifacts)
    champion_version = champion_info.get("active_version")
    champ_bundle = load_model_bundle(champion_version, artifacts_dir=target_artifacts)
    champ_pipeline = champ_bundle["pipeline"]
    champ_meta = champ_bundle["metadata"]
    model_type = champ_meta.get("model_type", "logistic_regression")
    best_params = champ_meta.get("best_params", {})

    # 5. Build candidate training dataset: Base dev set (80%) + new train partition
    X_train_parts = []
    y_train_parts = []

    if train_cfg.get("use_base_dev_data", True) and os.path.exists(manifest_path) and os.path.exists(raw_data_path):
        raw_df = load_raw_data(raw_data_path)
        valid_raw, _ = validate_records(raw_df, mode="training")
        manifest = pd.read_csv(manifest_path)
        dev_manifest = manifest[manifest["split"] == "dev"][["customerID", "fold"]]
        dev_merged = valid_raw.merge(dev_manifest, on="customerID")
        X_train_parts.append(dev_merged[ALL_FEATURE_COLUMNS])
        y_train_parts.append(dev_merged[TARGET_COLUMN])

    if not new_train_df.empty:
        X_train_parts.append(new_train_df[ALL_FEATURE_COLUMNS])
        y_train_parts.append(new_train_df["actual_churn"])

    if X_train_parts:
        X_candidate_train = pd.concat(X_train_parts, ignore_index=True)
        y_candidate_train = pd.concat(y_train_parts, ignore_index=True)
    else:
        raise ValueError("No training data available to fit candidate model")

    # 6. Fit candidate pipeline
    candidate_pipeline = train_candidate_pipeline(
        X_train=X_candidate_train,
        y_train=y_candidate_train,
        model_type=model_type,
        params=best_params,
        random_state=train_cfg.get("random_state", 42),
    )

    # 7. Evaluate BOTH champion and candidate on the EXACT SAME comparison set
    X_comp = new_comp_df[ALL_FEATURE_COLUMNS]
    y_comp = new_comp_df["actual_churn"]

    threshold = float(promo_cfg.get("threshold", 0.5))
    champ_metrics = compute_metrics_on_comparison(champ_pipeline, X_comp, y_comp, threshold=threshold)
    cand_metrics = compute_metrics_on_comparison(candidate_pipeline, X_comp, y_comp, threshold=threshold)

    # 8. Compare & make decision based on Plan 06 Section 6
    min_f1_gain = float(promo_cfg.get("min_f1_gain", 0.01))
    max_recall_drop = float(promo_cfg.get("max_recall_drop", 0.02))

    delta_f1 = cand_metrics["f1"] - champ_metrics["f1"]
    delta_recall = cand_metrics["recall"] - champ_metrics["recall"]

    is_promoted = (delta_f1 >= min_f1_gain) and (delta_recall >= -max_recall_drop)

    if is_promoted:
        decision = "PROMOTE"
        reason = (
            f"Candidate đạt điều kiện thăng cấp: F1 tăng {delta_f1:+.4f} (>= +{min_f1_gain:.4f}) "
            f"và Recall thay đổi {delta_recall:+.4f} (>= -{max_recall_drop:.4f}) trên tập so sánh độc lập."
        )
    else:
        decision = "REJECT"
        reason = (
            f"Candidate không đạt điều kiện thăng cấp: F1 delta={delta_f1:+.4f} (yêu cầu >= +{min_f1_gain:.4f}), "
            f"Recall delta={delta_recall:+.4f} (yêu cầu >= -{max_recall_drop:.4f}). Tiếp tục giữ Champion {champion_version}."
        )

    # 9. Packaging and activation if promoted and not dry_run
    next_ver = get_next_model_version(target_artifacts)
    candidate_version = None

    if decision == "PROMOTE":
        candidate_version = next_ver
        if not dry_run:
            cand_dir = os.path.join(target_artifacts, "models", next_ver)
            os.makedirs(cand_dir, exist_ok=True)

            # Save pipeline
            joblib.dump(candidate_pipeline, os.path.join(cand_dir, "pipeline.joblib"))

            # Save metadata
            cand_meta = {
                "model_version": next_ver,
                "model_type": model_type,
                "best_params": best_params,
                "class_mapping": {"No": 0, "Yes": 1},
                "threshold": threshold,
                "primary_metric": "f1",
                "train_samples": len(X_candidate_train),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "champion_version_superseded": champion_version,
                "comparison_metrics": cand_metrics,
                "environment": get_environment_metadata(),
            }
            with open(os.path.join(cand_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(cand_meta, f, indent=2, ensure_ascii=False)

            with open(os.path.join(cand_dir, "comparison_metrics.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "champion_metrics": champ_metrics,
                    "candidate_metrics": cand_metrics,
                    "delta_f1": delta_f1,
                    "delta_recall": delta_recall,
                    "decision": decision,
                    "reason": reason,
                }, f, indent=2)

            # Register in DB
            register_model_version(
                version=next_ver,
                run_id=run_id,
                artifact_path=cand_dir,
                metrics=cand_metrics,
                status="candidate",
                db_path=target_db,
            )

            # Activate atomic pointer
            set_active_model(next_ver, artifacts_dir=target_artifacts, db_path=target_db)

    # 10. Record run in SQLite
    finished_time = datetime.now(timezone.utc)
    _record_training_run_db(
        run_id=run_id,
        as_of=as_of_ts,
        config_json=json.dumps(cfg),
        champion_version=champion_version,
        candidate_version=candidate_version,
        status="COMPLETED",
        decision=decision,
        reason=reason,
        started_at=start_time.isoformat(),
        finished_at=finished_time.isoformat(),
        db_path=target_db,
    )

    return {
        "run_id": run_id,
        "as_of": as_of_ts,
        "gate_passed": gate_passed,
        "gate_details": gate_details,
        "champion_version": champion_version,
        "candidate_version": candidate_version,
        "decision": decision,
        "reason": reason,
        "champion_metrics": champ_metrics,
        "candidate_metrics": cand_metrics,
        "delta_f1": round(delta_f1, 4),
        "delta_recall": round(delta_recall, 4),
        "dry_run": dry_run,
    }


def _record_training_run_db(
    run_id: str,
    as_of: str,
    config_json: str,
    status: str,
    decision: str,
    reason: str,
    started_at: str,
    finished_at: str,
    champion_version: Optional[str] = None,
    candidate_version: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    """Helper to persist a training run log in SQLite."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO training_runs (
            run_id, as_of, config_json, champion_version, candidate_version,
            status, decision, reason, started_at, finished_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            as_of,
            config_json,
            champion_version,
            candidate_version,
            status,
            decision,
            reason,
            started_at,
            finished_at,
        ),
    )
    conn.commit()
    conn.close()
