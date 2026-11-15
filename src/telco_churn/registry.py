"""Model registry, artifact verification, active model pointer management, and rollback."""

import os
import json
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import joblib

from .storage import (
    upsert_model_version,
    update_model_status,
    get_model_versions_from_db,
    DEFAULT_DB_PATH,
)


def validate_artifact_bundle(
    version_or_path: str,
    artifacts_dir: str = "artifacts",
) -> bool:
    """Verify that an artifact directory contains complete, valid pipeline and metadata files.

    Checks:
        1. pipeline.joblib and metadata.json exist.
        2. pipeline can be loaded and contains a classifier step with classes_ == [0, 1].
        3. metadata.json is valid JSON with required fields.
    """
    if os.path.isdir(version_or_path):
        model_dir = version_or_path
    else:
        model_dir = os.path.join(artifacts_dir, "models", version_or_path)

    pipeline_path = os.path.join(model_dir, "pipeline.joblib")
    metadata_path = os.path.join(model_dir, "metadata.json")

    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Missing pipeline artifact at: {pipeline_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing metadata artifact at: {metadata_path}")

    # Validate metadata JSON
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    required_meta_keys = ["model_version", "model_type", "threshold", "class_mapping"]
    for k in required_meta_keys:
        if k not in meta:
            raise ValueError(f"Metadata in {metadata_path} is missing required key '{k}'")

    # Validate pipeline
    pipeline = joblib.load(pipeline_path)
    if not hasattr(pipeline, "named_steps") or "classifier" not in pipeline.named_steps:
        raise ValueError(f"Pipeline in {pipeline_path} does not contain 'classifier' step")

    clf = pipeline.named_steps["classifier"]
    if not hasattr(clf, "classes_"):
        raise ValueError(f"Classifier in {pipeline_path} is not fitted (no classes_ attribute)")

    classes = list(clf.classes_)
    if 1 not in classes:
        raise ValueError(f"Classifier in {pipeline_path} classes {classes} do not include class 1")

    return True


def get_active_model_info(artifacts_dir: str = "artifacts") -> Dict[str, Any]:
    """Read the active model pointer and metadata."""
    active_path = os.path.join(artifacts_dir, "active_model.json")
    if not os.path.exists(active_path):
        raise FileNotFoundError(f"Active model pointer not found at: {active_path}")

    with open(active_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    # Attach model_dir and check bundle
    version = info.get("active_version")
    if version:
        meta_path = os.path.join(artifacts_dir, "models", version, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as mf:
                info["metadata"] = json.load(mf)

    return info


def set_active_model(
    version: str,
    artifacts_dir: str = "artifacts",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Atomically set the active model version after verification.

    Conforms to Plan 04 Section 3 & Plan 06 Section 7.
    """
    # 1. Validate candidate bundle first
    model_dir = os.path.join(artifacts_dir, "models", version)
    validate_artifact_bundle(model_dir)

    # 2. Read metadata to get model_type
    with open(os.path.join(model_dir, "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 3. Read previous active info for rollback tracking
    active_path = os.path.join(artifacts_dir, "active_model.json")
    prev_version = None
    if os.path.exists(active_path):
        try:
            with open(active_path, "r", encoding="utf-8") as f:
                prev_info = json.load(f)
                prev_version = prev_info.get("active_version")
        except Exception:
            pass

    # 4. Atomic write of active_model.json via temp file
    now_utc = datetime.now(timezone.utc).isoformat()
    new_active_content = {
        "active_version": version,
        "model_type": meta.get("model_type", "unknown"),
        "previous_version": prev_version,
        "updated_at_utc": now_utc,
    }

    temp_fd, temp_path = tempfile.mkstemp(dir=artifacts_dir, prefix="active_", suffix=".tmp")
    with os.fdopen(temp_fd, "w", encoding="utf-8") as tf:
        json.dump(new_active_content, tf, indent=2)

    # Atomic replace
    shutil.move(temp_path, active_path)

    # 5. Synchronize DB model statuses if DB exists
    effective_db = db_path or DEFAULT_DB_PATH
    if os.path.exists(effective_db):
        try:
            if prev_version and prev_version != version:
                update_model_status(prev_version, "archived", effective_db)
            upsert_model_version(
                model_version=version,
                run_id=meta.get("run_id"),
                artifact_path=model_dir,
                metrics=meta.get("test_metrics"),
                status="active",
                schema_version="1.0",
                db_path=effective_db,
            )
        except Exception:
            # DB sync error should not corrupt the active pointer
            pass

    return new_active_content


def rollback_model(
    artifacts_dir: str = "artifacts",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Roll back to the previous active model version."""
    active_path = os.path.join(artifacts_dir, "active_model.json")
    if not os.path.exists(active_path):
        raise FileNotFoundError("active_model.json not found to rollback")

    with open(active_path, "r", encoding="utf-8") as f:
        curr_info = json.load(f)

    prev_version = curr_info.get("previous_version")
    if not prev_version:
        # Fallback: scan models directory for any version other than current
        models_dir = os.path.join(artifacts_dir, "models")
        available = sorted([
            d for d in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, d)) and d != curr_info.get("active_version")
        ])
        if available:
            prev_version = available[-1]
        else:
            raise ValueError("No previous model version available for rollback")

    # Set previous version as active
    return set_active_model(prev_version, artifacts_dir=artifacts_dir, db_path=db_path)


def register_model_version(
    version: str,
    run_id: Optional[str],
    artifact_path: str,
    metrics: Optional[Dict[str, Any]] = None,
    schema_version: str = "1.0",
    status: str = "candidate",
    db_path: Optional[str] = None,
) -> None:
    """Register a new model version into registry and SQLite."""
    validate_artifact_bundle(artifact_path)
    effective_db = db_path or DEFAULT_DB_PATH
    if os.path.exists(os.path.dirname(os.path.abspath(effective_db))):
        upsert_model_version(
            model_version=version,
            run_id=run_id,
            artifact_path=artifact_path,
            metrics=metrics,
            status=status,
            schema_version=schema_version,
            db_path=effective_db,
        )


def list_available_models(
    artifacts_dir: str = "artifacts",
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List all registered or discovered models in the artifacts directory."""
    models_dir = os.path.join(artifacts_dir, "models")
    if not os.path.exists(models_dir):
        return []

    active_info = {}
    try:
        active_info = get_active_model_info(artifacts_dir)
    except Exception:
        pass
    active_ver = active_info.get("active_version")

    versions = []
    for d in sorted(os.listdir(models_dir)):
        meta_path = os.path.join(models_dir, d, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["is_active"] = (d == active_ver)
                meta["directory"] = os.path.join(models_dir, d)
                versions.append(meta)
            except Exception:
                continue

    return versions
