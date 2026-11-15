"""SQLite storage, transaction management, schema migrations, and audit logging."""

import os
import json
import sqlite3
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd

from .schema import ALL_FEATURE_COLUMNS, ID_COLUMN, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS

DEFAULT_DB_PATH = os.path.join("data", "runtime", "churn.db")
SCHEMA_VERSION = 1


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Create a SQLite connection with foreign keys enabled and row factory set."""
    path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize SQLite database tables according to Plan 04 & Plan 06 schema."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS customers (
        internal_customer_id TEXT PRIMARY KEY,
        external_customer_id TEXT,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_customers_external_id ON customers(external_customer_id);

    CREATE TABLE IF NOT EXISTS customer_snapshots (
        snapshot_id TEXT PRIMARY KEY,
        customer_id TEXT NOT NULL,
        observed_at TEXT NOT NULL,
        features_json TEXT NOT NULL,
        schema_version TEXT NOT NULL DEFAULT '1.0',
        source TEXT NOT NULL,
        is_synthetic INTEGER NOT NULL DEFAULT 0,
        payload_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers(internal_customer_id)
    );
    CREATE INDEX IF NOT EXISTS idx_snapshots_customer_id ON customer_snapshots(customer_id);
    CREATE INDEX IF NOT EXISTS idx_snapshots_payload_hash ON customer_snapshots(payload_hash);
    CREATE INDEX IF NOT EXISTS idx_snapshots_observed_at ON customer_snapshots(observed_at);

    CREATE TABLE IF NOT EXISTS prediction_batches (
        batch_id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        idempotency_key TEXT UNIQUE,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_batches_idempotency ON prediction_batches(idempotency_key);

    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id TEXT PRIMARY KEY,
        batch_id TEXT NOT NULL,
        snapshot_id TEXT NOT NULL,
        model_version TEXT NOT NULL,
        churn_score REAL NOT NULL,
        predicted_label TEXT NOT NULL,
        threshold REAL NOT NULL,
        predicted_at TEXT NOT NULL,
        FOREIGN KEY (batch_id) REFERENCES prediction_batches(batch_id),
        FOREIGN KEY (snapshot_id) REFERENCES customer_snapshots(snapshot_id)
    );
    CREATE INDEX IF NOT EXISTS idx_predictions_snapshot_id ON predictions(snapshot_id);
    CREATE INDEX IF NOT EXISTS idx_predictions_batch_id ON predictions(batch_id);
    CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version);

    CREATE TABLE IF NOT EXISTS actual_label_events (
        event_id TEXT PRIMARY KEY,
        snapshot_id TEXT NOT NULL,
        actual_churn INTEGER NOT NULL,
        label_observed_at TEXT NOT NULL,
        label_source TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        supersedes_event_id TEXT,
        note TEXT,
        FOREIGN KEY (snapshot_id) REFERENCES customer_snapshots(snapshot_id),
        FOREIGN KEY (supersedes_event_id) REFERENCES actual_label_events(event_id)
    );
    CREATE INDEX IF NOT EXISTS idx_actual_labels_snapshot ON actual_label_events(snapshot_id);

    CREATE TABLE IF NOT EXISTS model_versions (
        model_version TEXT PRIMARY KEY,
        run_id TEXT,
        artifact_path TEXT NOT NULL,
        schema_version TEXT NOT NULL DEFAULT '1.0',
        metrics_json TEXT,
        status TEXT NOT NULL DEFAULT 'candidate', -- candidate / active / archived / rejected
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS training_runs (
        run_id TEXT PRIMARY KEY,
        as_of TEXT NOT NULL,
        snapshot_path TEXT,
        snapshot_hash TEXT,
        config_json TEXT,
        champion_version TEXT,
        candidate_version TEXT,
        status TEXT NOT NULL, -- COMPLETED, SKIPPED, FAILED
        decision TEXT,         -- PROMOTE, REJECT, SKIP
        reason TEXT,
        started_at TEXT NOT NULL,
        finished_at TEXT
    );

    CREATE TABLE IF NOT EXISTS training_membership (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        snapshot_id TEXT NOT NULL,
        partition TEXT NOT NULL, -- 'train', 'comparison'
        label_event_id TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES training_runs(run_id),
        FOREIGN KEY (snapshot_id) REFERENCES customer_snapshots(snapshot_id),
        FOREIGN KEY (label_event_id) REFERENCES actual_label_events(event_id)
    );
    CREATE INDEX IF NOT EXISTS idx_membership_run ON training_membership(run_id);
    CREATE INDEX IF NOT EXISTS idx_membership_snapshot ON training_membership(snapshot_id);
    """)

    # Record migration version
    cur.execute(
        "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
        (SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def _normalize_feature_val(col: str, v: Any) -> Any:
    """Convert numpy scalars and nulls to standard Python JSON-serializable types, keeping categoricals as str."""
    if v is None or pd.isna(v):
        return None
    if col == "SeniorCitizen":
        if isinstance(v, (int, float, np.integer, np.floating)):
            return str(int(v))
        return str(v).strip()
    if col in NUMERIC_COLUMNS:
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return float(v)
        try:
            return float(v)
        except Exception:
            return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return str(v).strip()


def compute_features_hash(features_dict: Dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash of record features."""
    serialized = json.dumps(features_dict, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def save_prediction_batch(
    inputs: Union[pd.DataFrame, List[Dict[str, Any]]],
    outputs: Union[pd.DataFrame, List[Dict[str, Any]]],
    source: str,
    idempotency_key: Optional[str] = None,
    is_synthetic: bool = False,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Atomically persist prediction batch, immutable customer snapshots, and predictions.

    Conforms to Plan 04 Section 3 contract:
    save_prediction_batch(inputs, outputs, source) -> batch_id, snapshot_ids, prediction_ids; transaction.
    """
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    try:
        # Check idempotency
        if idempotency_key:
            cur.execute(
                "SELECT batch_id FROM prediction_batches WHERE idempotency_key = ?",
                (idempotency_key,),
            )
            row = cur.fetchone()
            if row:
                existing_batch_id = row["batch_id"]
                cur.execute(
                    "SELECT snapshot_id, prediction_id FROM predictions WHERE batch_id = ?",
                    (existing_batch_id,),
                )
                items = cur.fetchall()
                conn.close()
                return {
                    "batch_id": existing_batch_id,
                    "idempotency_key": idempotency_key,
                    "status": "existing",
                    "count": len(items),
                    "snapshot_ids": [r["snapshot_id"] for r in items],
                    "prediction_ids": [r["prediction_id"] for r in items],
                }

        batch_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        created_at_utc = datetime.now(timezone.utc).isoformat()

        # Insert prediction_batch
        cur.execute(
            """
            INSERT INTO prediction_batches (batch_id, source, idempotency_key, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (batch_id, source, idempotency_key, created_at_utc),
        )

        inputs_df = pd.DataFrame(inputs) if not isinstance(inputs, pd.DataFrame) else inputs
        outputs_df = pd.DataFrame(outputs) if not isinstance(outputs, pd.DataFrame) else outputs

        if len(inputs_df) != len(outputs_df):
            raise ValueError(f"Inputs count ({len(inputs_df)}) does not match outputs count ({len(outputs_df)})")

        snapshot_ids = []
        prediction_ids = []

        for idx in range(len(inputs_df)):
            in_row = inputs_df.iloc[idx]
            out_row = outputs_df.iloc[idx]

            # Resolve customer identity
            ext_cust_id = None
            if ID_COLUMN in in_row and pd.notna(in_row[ID_COLUMN]) and str(in_row[ID_COLUMN]).strip() != "":
                ext_cust_id = str(in_row[ID_COLUMN]).strip()
            elif "customerID" in out_row and pd.notna(out_row["customerID"]):
                val = str(out_row["customerID"]).strip()
                if not val.startswith("GEN_"):
                    ext_cust_id = val

            internal_cust_id = None
            if ext_cust_id:
                cur.execute(
                    "SELECT internal_customer_id FROM customers WHERE external_customer_id = ?",
                    (ext_cust_id,),
                )
                cust_row = cur.fetchone()
                if cust_row:
                    internal_cust_id = cust_row["internal_customer_id"]

            if not internal_cust_id:
                internal_cust_id = f"CUST_{uuid.uuid4().hex[:12]}"
                cur.execute(
                    "INSERT INTO customers (internal_customer_id, external_customer_id, created_at) VALUES (?, ?, ?)",
                    (internal_cust_id, ext_cust_id, created_at_utc),
                )

            # Features JSON and hash
            feature_vals = {}
            for col in ALL_FEATURE_COLUMNS:
                if col in in_row:
                    feature_vals[col] = _normalize_feature_val(col, in_row[col])
                else:
                    feature_vals[col] = None

            features_json_str = json.dumps(feature_vals, ensure_ascii=False, default=str)
            payload_hash = compute_features_hash(feature_vals)

            snapshot_id = f"snap_{uuid.uuid4().hex[:16]}"
            observed_at = str(out_row.get("predicted_at", created_at_utc))

            cur.execute(
                """
                INSERT INTO customer_snapshots (
                    snapshot_id, customer_id, observed_at, features_json,
                    schema_version, source, is_synthetic, payload_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    internal_cust_id,
                    observed_at,
                    features_json_str,
                    "1.0",
                    source,
                    1 if is_synthetic else 0,
                    payload_hash,
                    created_at_utc,
                ),
            )
            snapshot_ids.append(snapshot_id)

            prediction_id = f"pred_{uuid.uuid4().hex[:16]}"
            model_ver = str(out_row.get("model_version", "unknown"))
            churn_score = float(out_row.get("churn_score", 0.0))
            pred_label = str(out_row.get("churn_label", "No"))
            threshold = float(out_row.get("threshold", 0.5))
            pred_at = str(out_row.get("predicted_at", created_at_utc))

            cur.execute(
                """
                INSERT INTO predictions (
                    prediction_id, batch_id, snapshot_id, model_version,
                    churn_score, predicted_label, threshold, predicted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    batch_id,
                    snapshot_id,
                    model_ver,
                    churn_score,
                    pred_label,
                    threshold,
                    pred_at,
                ),
            )
            prediction_ids.append(prediction_id)

        conn.commit()
        return {
            "batch_id": batch_id,
            "idempotency_key": idempotency_key,
            "status": "created",
            "count": len(outputs_df),
            "snapshot_ids": snapshot_ids,
            "prediction_ids": prediction_ids,
        }
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def record_actual_label(
    snapshot_id: str,
    label: Union[int, str],
    observed_at: str,
    source: str,
    note: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Record an audit-trailed actual churn label for a customer snapshot.

    Conforms to Plan 04 Section 3 contract:
    record_actual_label(snapshot_id, label, observed_at, source) -> event_id; does not overwrite prediction.
    """
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    try:
        # Check snapshot existence
        cur.execute(
            "SELECT snapshot_id, customer_id, observed_at, created_at FROM customer_snapshots WHERE snapshot_id = ?",
            (snapshot_id,),
        )
        snap = cur.fetchone()
        if not snap:
            raise ValueError(f"Snapshot ID '{snapshot_id}' does not exist in customer_snapshots")

        # Plan 06 Section 3 Item 4: Kiểm tra thời điểm quan sát nhãn không được sớm hơn snapshot
        snap_obs = snap["observed_at"]
        if observed_at and snap_obs:
            try:
                obs_clean = observed_at.replace("Z", "+00:00") if "Z" in observed_at else observed_at
                snap_clean = snap_obs.replace("Z", "+00:00") if "Z" in snap_obs else snap_obs
                dt_obs = datetime.fromisoformat(obs_clean)
                dt_snap = datetime.fromisoformat(snap_clean)
                if dt_obs.date() < dt_snap.date():
                    raise ValueError(
                        f"Thời điểm quan sát nhãn ({observed_at[:10]}) không được sớm hơn ngày tạo snapshot ({snap_obs[:10]})"
                    )
            except ValueError as ve:
                if "không được sớm hơn" in str(ve):
                    raise ve
                if str(observed_at)[:10] < str(snap_obs)[:10]:
                    raise ValueError(
                        f"Thời điểm quan sát nhãn ({observed_at[:10]}) không được sớm hơn ngày tạo snapshot ({snap_obs[:10]})"
                    )

        # Normalize label to 0 or 1
        if isinstance(label, str):
            norm_str = label.strip().lower()
            if norm_str in ["yes", "1", "true"]:
                churn_val = 1
            elif norm_str in ["no", "0", "false"]:
                churn_val = 0
            else:
                raise ValueError(f"Invalid actual label value: '{label}'. Must be 0/1 or 'No'/'Yes'")
        elif isinstance(label, (int, float)):
            churn_val = int(label)
            if churn_val not in [0, 1]:
                raise ValueError(f"Numeric label must be 0 or 1, got {label}")
        else:
            raise ValueError(f"Unsupported label type: {type(label)}")

        # Check existing active label to record superseding history
        cur.execute(
            """
            SELECT event_id FROM actual_label_events
            WHERE snapshot_id = ?
            ORDER BY recorded_at DESC LIMIT 1
            """,
            (snapshot_id,),
        )
        prev_event = cur.fetchone()
        supersedes_id = prev_event["event_id"] if prev_event else None

        event_id = f"evt_{uuid.uuid4().hex[:16]}"
        recorded_at_utc = datetime.now(timezone.utc).isoformat()

        cur.execute(
            """
            INSERT INTO actual_label_events (
                event_id, snapshot_id, actual_churn, label_observed_at,
                label_source, recorded_at, supersedes_event_id, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                snapshot_id,
                churn_val,
                observed_at,
                source,
                recorded_at_utc,
                supersedes_id,
                note,
            ),
        )

        conn.commit()
        return {
            "event_id": event_id,
            "snapshot_id": snapshot_id,
            "actual_churn": churn_val,
            "supersedes_event_id": supersedes_id,
            "recorded_at": recorded_at_utc,
        }
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def get_prediction_history(
    limit: int = 100,
    customer_id: Optional[str] = None,
    model_version: Optional[str] = None,
    batch_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    has_label: Optional[bool] = None,
    db_path: Optional[str] = None,
) -> pd.DataFrame:
    """Retrieve joined prediction records, customer IDs, snapshots, and latest actual labels."""
    conn = get_db_connection(db_path)

    query = """
    WITH latest_labels AS (
        SELECT ale.snapshot_id, ale.event_id, ale.actual_churn, ale.label_observed_at,
               ale.label_source, ale.recorded_at, ale.note,
               ROW_NUMBER() OVER(PARTITION BY ale.snapshot_id ORDER BY ale.recorded_at DESC) as rn
        FROM actual_label_events ale
    )
    SELECT
        p.prediction_id,
        p.batch_id,
        p.snapshot_id,
        c.internal_customer_id,
        c.external_customer_id,
        p.model_version,
        p.churn_score,
        p.predicted_label,
        p.threshold,
        p.predicted_at,
        s.features_json,
        s.is_synthetic,
        s.source as snapshot_source,
        ll.actual_churn,
        ll.label_observed_at,
        ll.label_source,
        ll.recorded_at as label_recorded_at,
        ll.note as label_note
    FROM predictions p
    JOIN customer_snapshots s ON p.snapshot_id = s.snapshot_id
    JOIN customers c ON s.customer_id = c.internal_customer_id
    LEFT JOIN latest_labels ll ON s.snapshot_id = ll.snapshot_id AND ll.rn = 1
    WHERE 1=1
    """
    params: List[Any] = []

    if customer_id:
        query += " AND (c.external_customer_id LIKE ? OR c.internal_customer_id LIKE ?)"
        params.extend([f"%{customer_id}%", f"%{customer_id}%"])

    if model_version:
        query += " AND p.model_version = ?"
        params.append(model_version)

    if batch_id:
        query += " AND p.batch_id LIKE ?"
        params.append(f"%{batch_id}%")

    if start_date:
        query += " AND p.predicted_at >= ?"
        params.append(start_date)

    if end_date:
        query += " AND p.predicted_at <= ?"
        params.append(end_date)

    if has_label is True:
        query += " AND ll.actual_churn IS NOT NULL"
    elif has_label is False:
        query += " AND ll.actual_churn IS NULL"

    query += " ORDER BY p.predicted_at DESC LIMIT ?"
    params.append(limit)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def record_actual_labels_batch(
    records: Union[pd.DataFrame, List[Dict[str, Any]]],
    default_source: str = "CSV Upload",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate and record a batch of actual churn labels conforming to Plan 05 Section 4.

    Validates:
        - Presence of snapshot_id and actual_churn columns
        - Existence of snapshot_id in customer_snapshots
        - Duplicate keys within the batch
        - Valid label values (0/1 or No/Yes)
    """
    df = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records.copy()

    if df.empty:
        return {"total": 0, "success_count": 0, "failed_count": 0, "errors": []}

    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ["snapshot_id", "snapshotid", "id"]:
            col_map[c] = "snapshot_id"
        elif cl in ["actual_churn", "actual_label", "churn", "label"]:
            col_map[c] = "actual_churn"
        elif cl in ["observed_at", "label_observed_at", "date"]:
            col_map[c] = "observed_at"
        elif cl in ["source", "label_source"]:
            col_map[c] = "source"
        elif cl in ["note", "notes", "comment"]:
            col_map[c] = "note"

    df = df.rename(columns=col_map)

    if "snapshot_id" not in df.columns or "actual_churn" not in df.columns:
        raise ValueError("Tệp CSV bắt buộc phải có ít nhất hai cột: 'snapshot_id' và 'actual_churn'")

    errors: List[Dict[str, Any]] = []
    success_events: List[Dict[str, Any]] = []
    seen_snapshots = set()

    now_utc = datetime.now(timezone.utc).isoformat()

    for idx, row in df.iterrows():
        row_num = idx + 1
        snap_id = str(row.get("snapshot_id", "")).strip()
        raw_label = row.get("actual_churn")

        if not snap_id or snap_id.lower() == "nan":
            errors.append({"row": row_num, "snapshot_id": snap_id, "reason": "Thiếu mã snapshot_id"})
            continue

        if snap_id in seen_snapshots:
            errors.append({"row": row_num, "snapshot_id": snap_id, "reason": "Mã snapshot_id bị trùng lặp trong cùng tệp"})
            continue
        seen_snapshots.add(snap_id)

        # Check label value
        try:
            if isinstance(raw_label, (int, float, np.integer, np.floating)) and not pd.isna(raw_label):
                churn_val = int(raw_label)
                if churn_val not in [0, 1]:
                    raise ValueError()
            else:
                lbl_str = str(raw_label).strip().lower()
                if lbl_str in ["yes", "1", "true"]:
                    churn_val = 1
                elif lbl_str in ["no", "0", "false"]:
                    churn_val = 0
                else:
                    raise ValueError()
        except Exception:
            errors.append({
                "row": row_num,
                "snapshot_id": snap_id,
                "reason": f"Giá trị nhãn không hợp lệ: '{raw_label}'. Chỉ chấp nhận 0/1 hoặc No/Yes.",
            })
            continue

        obs_at = row.get("observed_at")
        if not obs_at or pd.isna(obs_at) or str(obs_at).strip() == "":
            obs_at = now_utc
        else:
            obs_at = str(obs_at).strip()

        src = row.get("source")
        if not src or pd.isna(src) or str(src).strip() == "":
            src = default_source
        else:
            src = str(src).strip()

        note_val = row.get("note")
        note = str(note_val).strip() if (note_val and not pd.isna(note_val)) else None

        try:
            evt = record_actual_label(
                snapshot_id=snap_id,
                label=churn_val,
                observed_at=obs_at,
                source=src,
                note=note,
                db_path=db_path,
            )
            success_events.append(evt)
        except Exception as e:
            errors.append({"row": row_num, "snapshot_id": snap_id, "reason": str(e)})

    return {
        "total": len(df),
        "success_count": len(success_events),
        "failed_count": len(errors),
        "success_events": success_events,
        "errors": errors,
    }


def get_training_runs(limit: int = 50, db_path: Optional[str] = None) -> pd.DataFrame:
    """Retrieve history of retraining runs, decisions, candidate/champion versions, and reasons."""
    conn = get_db_connection(db_path)
    query = """
    SELECT
        run_id,
        as_of,
        champion_version,
        candidate_version,
        status,
        decision,
        reason,
        started_at,
        finished_at
    FROM training_runs
    ORDER BY started_at DESC
    LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=[limit])
    conn.close()
    return df


def get_snapshots_with_labels(
    as_of: Optional[str] = None,
    include_synthetic: bool = False,
    exclude_trained: bool = False,
    db_path: Optional[str] = None,
) -> pd.DataFrame:
    """Retrieve all snapshots having a valid actual label up to as_of timestamp.

    If exclude_trained is True, filters out snapshots that have already participated
    in a COMPLETED training run (Plan 06 Section 4 & 7).
    """
    conn = get_db_connection(db_path)

    as_of_val = as_of or datetime.now(timezone.utc).isoformat()

    query = """
    WITH filtered_labels AS (
        SELECT
            ale.event_id,
            ale.snapshot_id,
            ale.actual_churn,
            ale.label_observed_at,
            ale.recorded_at,
            ROW_NUMBER() OVER(PARTITION BY ale.snapshot_id ORDER BY ale.recorded_at DESC) as rn
        FROM actual_label_events ale
        WHERE ale.recorded_at <= ?
    )
    SELECT
        s.snapshot_id,
        s.customer_id,
        c.external_customer_id,
        s.observed_at,
        s.features_json,
        s.payload_hash,
        s.is_synthetic,
        fl.event_id as label_event_id,
        fl.actual_churn,
        fl.label_observed_at
    FROM customer_snapshots s
    JOIN customers c ON s.customer_id = c.internal_customer_id
    JOIN filtered_labels fl ON s.snapshot_id = fl.snapshot_id AND fl.rn = 1
    WHERE s.observed_at <= ?
    """
    params: List[Any] = [as_of_val, as_of_val]

    if not include_synthetic:
        query += " AND s.is_synthetic = 0"

    if exclude_trained:
        query += """
        AND s.snapshot_id NOT IN (
            SELECT tm.snapshot_id
            FROM training_membership tm
            JOIN training_runs tr ON tm.run_id = tr.run_id
            WHERE tr.status = 'COMPLETED'
        )
        """

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty:
        return df

    # Unpack features_json into columns
    unpacked_rows = []
    for _, row in df.iterrows():
        try:
            feats = json.loads(row["features_json"])
        except Exception:
            feats = {}
        feats["snapshot_id"] = row["snapshot_id"]
        feats["customer_id"] = row["customer_id"]
        feats["customerID"] = row["external_customer_id"] or row["customer_id"]
        feats["actual_churn"] = int(row["actual_churn"])
        feats["label_event_id"] = row["label_event_id"]
        feats["payload_hash"] = row["payload_hash"]
        feats["is_synthetic"] = row["is_synthetic"]
        unpacked_rows.append(feats)

    res_df = pd.DataFrame(unpacked_rows)
    if "SeniorCitizen" in res_df.columns:
        res_df["SeniorCitizen"] = res_df["SeniorCitizen"].astype(str).str.replace(r"\.0$", "", regex=True)
    return res_df


def record_training_membership(
    run_id: str,
    memberships: List[Dict[str, str]],
    db_path: Optional[str] = None,
) -> None:
    """Record partition membership (train vs comparison) for snapshots in a retraining run.

    Conforms to Plan 06 Section 2 & Section 5:
    Populates training_membership (run_id, snapshot_id, partition, label_event_id).
    """
    if not memberships:
        return
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO training_membership (run_id, snapshot_id, partition, label_event_id)
        VALUES (?, ?, ?, ?)
        """,
        [(run_id, m["snapshot_id"], m["partition"], m["label_event_id"]) for m in memberships],
    )
    conn.commit()
    conn.close()


def get_training_memberships(
    run_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> pd.DataFrame:
    """Retrieve training membership records."""
    conn = get_db_connection(db_path)
    if run_id:
        df = pd.read_sql_query(
            "SELECT * FROM training_membership WHERE run_id = ?",
            conn,
            params=[run_id],
        )
    else:
        df = pd.read_sql_query("SELECT * FROM training_membership", conn)
    conn.close()
    return df


def get_model_versions_from_db(db_path: Optional[str] = None) -> pd.DataFrame:
    """Retrieve registered model versions."""
    conn = get_db_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM model_versions ORDER BY created_at DESC", conn)
    conn.close()
    return df


def upsert_model_version(
    model_version: str,
    run_id: Optional[str],
    artifact_path: str,
    metrics: Optional[Dict[str, Any]] = None,
    status: str = "candidate",
    schema_version: str = "1.0",
    db_path: Optional[str] = None,
) -> None:
    """Register or update a model version in SQLite."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    metrics_str = json.dumps(metrics, ensure_ascii=False) if metrics else None
    now_utc = datetime.now(timezone.utc).isoformat()

    cur.execute(
        """
        INSERT INTO model_versions (
            model_version, run_id, artifact_path, schema_version, metrics_json, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(model_version) DO UPDATE SET
            run_id = excluded.run_id,
            artifact_path = excluded.artifact_path,
            metrics_json = excluded.metrics_json,
            status = excluded.status
        """,
        (model_version, run_id, artifact_path, schema_version, metrics_str, status, now_utc),
    )
    conn.commit()
    conn.close()


def update_model_status(model_version: str, status: str, db_path: Optional[str] = None) -> None:
    """Update status of a model version (e.g. active, archived, rejected)."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("UPDATE model_versions SET status = ? WHERE model_version = ?", (status, model_version))
    conn.commit()
    conn.close()
