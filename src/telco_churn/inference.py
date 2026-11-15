"""Inference engine and model loading according to Plan 04 contracts."""

import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import joblib

from .schema import ALL_FEATURE_COLUMNS, ID_COLUMN


def get_active_model_version(artifacts_dir: str = "artifacts") -> str:
    """Retrieve the currently active model version from active_model.json."""
    active_path = os.path.join(artifacts_dir, "active_model.json")
    if not os.path.exists(active_path):
        raise FileNotFoundError(f"Active model descriptor not found at: {active_path}")

    with open(active_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    active_version = data.get("active_version")
    if not active_version:
        raise ValueError("Invalid active_model.json: missing 'active_version' key")

    return active_version


def load_model_bundle(
    version: Optional[str] = None,
    artifacts_dir: str = "artifacts",
) -> Dict[str, Any]:
    """Load model pipeline and metadata conforming to Plan 04 specifications.

    Args:
        version: Specific model version (e.g. 'model_001'). If None, reads from active_model.json.
        artifacts_dir: Base directory containing models/ and active_model.json.

    Returns:
        bundle: Dictionary containing 'pipeline', 'metadata', 'class_1_idx', and 'version'.
    """
    if version is None:
        version = get_active_model_version(artifacts_dir)

    model_dir = os.path.join(artifacts_dir, "models", version)
    pipeline_path = os.path.join(model_dir, "pipeline.joblib")
    metadata_path = os.path.join(model_dir, "metadata.json")

    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline artifact not found at: {pipeline_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata artifact not found at: {metadata_path}")

    pipeline = joblib.load(pipeline_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Locate class 1 index safely from classifier.classes_
    classifier = pipeline.named_steps.get("classifier")
    if classifier is None:
        raise ValueError("Loaded pipeline does not contain a 'classifier' step")

    if not hasattr(classifier, "classes_"):
        raise ValueError("Loaded classifier has not been fitted (missing classes_ attribute)")

    classes = list(classifier.classes_)
    if 1 in classes:
        class_1_idx = classes.index(1)
    else:
        raise ValueError(f"Classifier classes {classes} do not include positive class '1'")

    return {
        "version": version,
        "model_dir": model_dir,
        "pipeline": pipeline,
        "metadata": metadata,
        "class_1_idx": class_1_idx,
    }


def compute_priority_level(score: float, low_threshold: float = 0.30, high_threshold: float = 0.60) -> str:
    """Determine customer intervention priority level based on estimated churn risk.

    Policy (Conforming to Plan 05 Section 2):
        - Score < 0.30: 'Thấp' (Low risk - standard retention)
        - 0.30 <= Score < 0.60: 'Trung bình' (Medium risk - automated email / loyalty check)
        - Score >= 0.60: 'Cao' (High risk - urgent personal outreach / promo offer)
    """
    if score >= high_threshold:
        return "Cao"
    elif score >= low_threshold:
        return "Trung bình"
    else:
        return "Thấp"


def predict_records(
    df: pd.DataFrame,
    bundle: Dict[str, Any],
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Generate structured churn predictions for a batch of input records.

    Contract conforms strictly to Plan 04 Section 3.

    Args:
        df: Input DataFrame containing 19 features (customerID is optional).
        bundle: Loaded model bundle from load_model_bundle().
        threshold: Decision threshold for class 1 (defaults to metadata threshold or 0.5).

    Returns:
        DataFrame with standardized prediction columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if df.empty:
        return pd.DataFrame(columns=[
            "input_row_id", "customerID", "churn_prediction", "churn_label",
            "churn_score", "priority_level", "threshold", "model_version", "predicted_at",
        ])

    metadata = bundle["metadata"]
    pipeline = bundle["pipeline"]
    class_1_idx = bundle["class_1_idx"]
    version = bundle.get("version", metadata.get("model_version", "unknown"))

    if threshold is None:
        threshold = float(metadata.get("threshold", 0.5))

    # Verify required feature columns
    missing_cols = [c for c in ALL_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns for prediction: {missing_cols}")

    # Extract features
    X = df[ALL_FEATURE_COLUMNS].copy()

    # Obtain class probabilities
    probs = pipeline.predict_proba(X)
    churn_scores = probs[:, class_1_idx]

    # Apply decision threshold
    predictions = (churn_scores >= threshold).astype(int)
    labels = np.where(predictions == 1, "Yes", "No")

    # Resolve customerID
    if ID_COLUMN in df.columns:
        customer_ids = df[ID_COLUMN].astype(str).tolist()
    else:
        customer_ids = [f"GEN_{i:05d}" for i in range(len(df))]

    predicted_at = datetime.now(timezone.utc).isoformat()

    results = []
    for idx, (cid, pred, lbl, score) in enumerate(zip(customer_ids, predictions, labels, churn_scores)):
        rounded_score = round(float(score), 4)
        priority = compute_priority_level(rounded_score)
        results.append({
            "input_row_id": int(idx),
            "customerID": cid,
            "churn_prediction": int(pred),
            "churn_label": str(lbl),
            "churn_score": rounded_score,
            "priority_level": priority,
            "threshold": float(threshold),
            "model_version": str(version),
            "predicted_at": predicted_at,
        })

    return pd.DataFrame(results)


def benchmark_inference_speed(
    bundle: Dict[str, Any],
    sample_df: pd.DataFrame,
    n_single_runs: int = 50,
) -> Dict[str, float]:
    """Measure single-record and batch inference latencies on the local host."""
    single_record = sample_df.iloc[[0]].copy()

    # Warm-up run
    predict_records(single_record, bundle)

    # Benchmark single record
    single_times = []
    for _ in range(n_single_runs):
        t0 = time.perf_counter()
        predict_records(single_record, bundle)
        single_times.append((time.perf_counter() - t0) * 1000.0)  # ms

    # Benchmark batch of size len(sample_df)
    t0 = time.perf_counter()
    predict_records(sample_df, bundle)
    batch_time_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "single_record_latency_ms_mean": float(np.mean(single_times)),
        "single_record_latency_ms_std": float(np.std(single_times)),
        "batch_size": len(sample_df),
        "batch_latency_ms": float(batch_time_ms),
        "batch_throughput_records_per_sec": float(len(sample_df) / (batch_time_ms / 1000.0)),
    }
