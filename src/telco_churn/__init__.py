"""Telco Customer Churn Prediction Package."""

__version__ = "0.1.0"

from .schema import (
    ID_COLUMN,
    TARGET_COLUMN,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    ALL_FEATURE_COLUMNS,
    validate_records,
)
from .preprocessing import build_preprocessor
from .training import (
    build_model_pipeline,
    run_grid_search,
    select_champion_model,
)
from .evaluation import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves,
)
from .inference import (
    load_model_bundle,
    predict_records,
)
