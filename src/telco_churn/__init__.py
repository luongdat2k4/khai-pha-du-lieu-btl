"""Telco Customer Churn Prediction Package conforming to Plan 04 Architecture & Contracts."""

__version__ = "0.1.0"

# 1. Schema & Validation contract
from .schema import (
    ID_COLUMN,
    TARGET_COLUMN,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    ALL_FEATURE_COLUMNS,
    validate_records,
)

# 2. Preprocessor contract
from .preprocessing import build_preprocessor

# 3. Model training & experiment contract
from .training import (
    build_model_pipeline,
    run_grid_search,
    select_champion_model,
    train_experiment,
)

# 4. Evaluation contract
from .evaluation import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves,
)

# 5. Inference contract
from .inference import (
    load_model_bundle,
    predict_records,
    compute_priority_level,
    get_active_model_version,
)

# 6. Storage contract
from .storage import (
    init_db,
    save_prediction_batch,
    record_actual_label,
    get_prediction_history,
    get_snapshots_with_labels,
    get_db_connection,
)

# 7. Registry contract
from .registry import (
    get_active_model_info,
    set_active_model,
    rollback_model,
    register_model_version,
    validate_artifact_bundle,
    list_available_models,
)

# 8. Retraining contract
from .retraining import run_retraining

__all__ = [
    # Schema
    "ID_COLUMN",
    "TARGET_COLUMN",
    "NUMERIC_COLUMNS",
    "CATEGORICAL_COLUMNS",
    "ALL_FEATURE_COLUMNS",
    "validate_records",
    # Preprocessing
    "build_preprocessor",
    # Training
    "build_model_pipeline",
    "run_grid_search",
    "select_champion_model",
    "train_experiment",
    # Evaluation
    "compute_classification_metrics",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_pr_curves",
    # Inference
    "load_model_bundle",
    "predict_records",
    "compute_priority_level",
    "get_active_model_version",
    # Storage
    "init_db",
    "save_prediction_batch",
    "record_actual_label",
    "get_prediction_history",
    "get_snapshots_with_labels",
    "get_db_connection",
    # Registry
    "get_active_model_info",
    "set_active_model",
    "rollback_model",
    "register_model_version",
    "validate_artifact_bundle",
    "list_available_models",
    # Retraining
    "run_retraining",
]
