"""Model building, cross-validation, hyperparameter tuning, and model selection."""

import os
import time
import platform
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    make_scorer,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)

from .schema import ALL_FEATURE_COLUMNS, TARGET_COLUMN
from .preprocessing import build_preprocessor


# Scorers definition conforming to Plan 03 protocol
SCORERS = {
    "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
    "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
    "recall": make_scorer(recall_score, pos_label=1, zero_division=0),
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
}

# Hyperparameter search grids specified in Plan 03
PARAM_GRIDS = {
    "logistic_regression": {
        "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "classifier__class_weight": [None, "balanced"],
    },
    "decision_tree": {
        "classifier__criterion": ["gini", "entropy"],
        "classifier__max_depth": [3, 5, 8, None],
        "classifier__min_samples_leaf": [1, 5, 15],
        "classifier__class_weight": [None, "balanced"],
    },
    "random_forest": {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [8, 16, None],
        "classifier__min_samples_leaf": [1, 3, 8],
        "classifier__max_features": ["sqrt", 0.5],
        "classifier__class_weight": [None, "balanced"],
    },
}


def build_base_classifier(model_name: str, random_state: int = 42, **kwargs):
    """Instantiate a base classifier according to Plan 03 defaults."""
    name = model_name.lower()
    if name in ["dummy", "baseline"]:
        return DummyClassifier(strategy="most_frequent")
    elif name in ["logistic", "logistic_regression", "lr"]:
        return LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            random_state=random_state,
            **kwargs,
        )
    elif name in ["tree", "decision_tree", "dt"]:
        return DecisionTreeClassifier(
            random_state=random_state,
            **kwargs,
        )
    elif name in ["forest", "random_forest", "rf"]:
        return RandomForestClassifier(
            random_state=random_state,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported model name: '{model_name}'")


def build_model_pipeline(model_name: str, random_state: int = 42, **kwargs) -> Pipeline:
    """Build a complete scikit-learn Pipeline with preprocessor and classifier."""
    preprocessor = build_preprocessor(model_name)
    classifier = build_base_classifier(model_name, random_state=random_state, **kwargs)
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def evaluate_baseline(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    folds: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate DummyClassifier (most_frequent) across the 5 predefined folds."""
    ps = PredefinedSplit(folds)
    n_splits = len(np.unique(folds))

    fold_metrics: List[Dict[str, float]] = []
    total_fit_time = 0.0

    dummy_pipe = build_model_pipeline("dummy")

    for fold_idx, (train_idx, val_idx) in enumerate(ps.split()):
        X_train, y_train = X_dev.iloc[train_idx], y_dev.iloc[train_idx]
        X_val, y_val = X_dev.iloc[val_idx], y_dev.iloc[val_idx]

        t0 = time.time()
        dummy_pipe.fit(X_train, y_train)
        fit_dur = time.time() - t0
        total_fit_time += fit_dur

        y_pred = dummy_pipe.predict(X_val)
        y_prob = dummy_pipe.predict_proba(X_val)[:, 1] if hasattr(dummy_pipe, "predict_proba") else None

        f1 = float(f1_score(y_val, y_pred, pos_label=1, zero_division=0))
        prec = float(precision_score(y_val, y_pred, pos_label=1, zero_division=0))
        rec = float(recall_score(y_val, y_pred, pos_label=1, zero_division=0))
        acc = float(accuracy_score(y_val, y_pred))
        # For DummyClassifier most_frequent, predict_proba is constant, ROC-AUC is 0.5
        roc = 0.5 if y_prob is not None else 0.5

        fold_metrics.append({
            "fold": fold_idx,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "roc_auc": roc,
            "accuracy": acc,
            "fit_time": fit_dur,
        })

    # Fit on entire dev set
    dummy_pipe.fit(X_dev, y_dev)

    mean_f1 = float(np.mean([m["f1"] for m in fold_metrics]))
    std_f1 = float(np.std([m["f1"] for m in fold_metrics]))
    mean_prec = float(np.mean([m["precision"] for m in fold_metrics]))
    std_prec = float(np.std([m["precision"] for m in fold_metrics]))
    mean_rec = float(np.mean([m["recall"] for m in fold_metrics]))
    std_rec = float(np.std([m["recall"] for m in fold_metrics]))
    mean_roc = float(np.mean([m["roc_auc"] for m in fold_metrics]))
    std_roc = float(np.std([m["roc_auc"] for m in fold_metrics]))
    mean_acc = float(np.mean([m["accuracy"] for m in fold_metrics]))
    std_acc = float(np.std([m["accuracy"] for m in fold_metrics]))
    avg_fit_time = total_fit_time / n_splits

    return {
        "model_name": "dummy",
        "best_params": {"strategy": "most_frequent"},
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "mean_precision": mean_prec,
        "std_precision": std_prec,
        "mean_recall": mean_rec,
        "std_recall": std_rec,
        "mean_roc_auc": mean_roc,
        "std_roc_auc": std_roc,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_fit_time": avg_fit_time,
        "total_fit_time": total_fit_time,
        "fold_metrics": fold_metrics,
        "fitted_pipeline": dummy_pipe,
    }


def run_grid_search(
    model_name: str,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    folds: np.ndarray,
    random_state: int = 42,
    rf_n_iter: int = 24,
) -> Dict[str, Any]:
    """Execute hyperparameter tuning using PredefinedSplit and multi-metric scoring."""
    ps = PredefinedSplit(folds)
    pipe = build_model_pipeline(model_name, random_state=random_state)
    param_grid = PARAM_GRIDS[model_name]

    start_time = time.time()

    if model_name == "random_forest":
        # Plan 03: RF has 72 candidates, use RandomizedSearchCV with fixed n_iter=24 and seed=42
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grid,
            n_iter=rf_n_iter,
            scoring=SCORERS,
            refit="f1",
            cv=ps,
            random_state=random_state,
            n_jobs=-1,
            return_train_score=False,
            verbose=0,
        )
    else:
        # Logistic Regression (10 candidates) and Decision Tree (48 candidates) use GridSearchCV
        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=SCORERS,
            refit="f1",
            cv=ps,
            n_jobs=-1,
            return_train_score=False,
            verbose=0,
        )

    search.fit(X_dev, y_dev)
    total_search_time = time.time() - start_time

    cv_results = pd.DataFrame(search.cv_results_)

    # Extract best candidate's stats
    best_idx = search.best_index_
    best_params = search.best_params_
    best_row = cv_results.iloc[best_idx]

    # Clean param names for reporting (remove classifier__ prefix)
    clean_best_params = {
        k.replace("classifier__", ""): (None if v is None else v)
        for k, v in best_params.items()
    }

    # Extract fold scores for best candidate
    fold_metrics = []
    for f in range(len(np.unique(folds))):
        fold_metrics.append({
            "fold": f,
            "f1": float(best_row[f"split{f}_test_f1"]),
            "precision": float(best_row[f"split{f}_test_precision"]),
            "recall": float(best_row[f"split{f}_test_recall"]),
            "roc_auc": float(best_row[f"split{f}_test_roc_auc"]),
            "accuracy": float(best_row[f"split{f}_test_accuracy"]),
        })

    # Convert all configs into structured list for detailed logging
    all_configs_summary = []
    for idx, row in cv_results.iterrows():
        all_configs_summary.append({
            "candidate_index": int(idx),
            "params": {k.replace("classifier__", ""): v for k, v in row["params"].items()},
            "mean_f1": float(row["mean_test_f1"]),
            "std_f1": float(row["std_test_f1"]),
            "mean_precision": float(row["mean_test_precision"]),
            "mean_recall": float(row["mean_test_recall"]),
            "mean_roc_auc": float(row["mean_test_roc_auc"]),
            "mean_accuracy": float(row["mean_test_accuracy"]),
            "mean_fit_time": float(row["mean_fit_time"]),
            "rank_test_f1": int(row["rank_test_f1"]),
        })

    return {
        "model_name": model_name,
        "best_params": clean_best_params,
        "mean_f1": float(best_row["mean_test_f1"]),
        "std_f1": float(best_row["std_test_f1"]),
        "mean_precision": float(best_row["mean_test_precision"]),
        "std_precision": float(best_row["std_test_precision"]),
        "mean_recall": float(best_row["mean_test_recall"]),
        "std_recall": float(best_row["std_test_recall"]),
        "mean_roc_auc": float(best_row["mean_test_roc_auc"]),
        "std_roc_auc": float(best_row["std_test_roc_auc"]),
        "mean_accuracy": float(best_row["mean_test_accuracy"]),
        "std_accuracy": float(best_row["std_test_accuracy"]),
        "mean_fit_time": float(best_row["mean_fit_time"]),
        "total_search_time": total_search_time,
        "n_candidates_evaluated": len(cv_results),
        "fold_metrics": fold_metrics,
        "all_configs": all_configs_summary,
        "fitted_pipeline": search.best_estimator_,
    }


def select_champion_model(
    results_by_model: Dict[str, Dict[str, Any]],
    f1_close_threshold: float = 0.005,
) -> Tuple[str, str]:
    """Select champion model family based on Plan 03 rules.

    Selection Policy (Plan 03, Section 4):
    1. Primary criterion: mean CV F1 score on validation folds.
    2. Close rule: If difference between top candidate and second candidate is < 0.005:
       - prioritize higher class 1 recall.
       - then simpler, faster, and more interpretable model.

    Returns:
        champion_name: Name of selected model.
        rationale: Explanatory text for the selection.
    """
    # Exclude dummy baseline from champion candidates
    candidates = [
        m for m in results_by_model.keys()
        if m not in ["dummy", "baseline"]
    ]

    # Sort candidates by mean_f1 descending
    sorted_candidates = sorted(
        candidates,
        key=lambda m: results_by_model[m]["mean_f1"],
        reverse=True,
    )

    top_model = sorted_candidates[0]
    top_f1 = results_by_model[top_model]["mean_f1"]
    top_recall = results_by_model[top_model]["mean_recall"]

    if len(sorted_candidates) > 1:
        second_model = sorted_candidates[1]
        second_f1 = results_by_model[second_model]["mean_f1"]
        second_recall = results_by_model[second_model]["mean_recall"]
        diff = top_f1 - second_f1

        if diff < f1_close_threshold:
            # Models are within 0.005 absolute F1: apply close rule
            if second_recall > top_recall:
                champion = second_model
                rationale = (
                    f"Khoảng cách CV F1 giữa '{top_model}' ({top_f1:.4f}) và '{second_model}' ({second_f1:.4f}) "
                    f"là {diff:.4f} (< 0.005). Theo quy tắc ưu tiên tại Plan 03, áp dụng tiêu chí phụ: "
                    f"'{second_model}' có Recall lớp 1 cao hơn ({second_recall:.4f} so với {top_recall:.4f}), "
                    f"giúp hạn chế bỏ sót khách hàng rời bỏ (False Negatives). Vì vậy chọn '{second_model}' làm Champion."
                )
                return champion, rationale
            else:
                champion = top_model
                rationale = (
                    f"'{top_model}' đạt mean CV F1 cao nhất ({top_f1:.4f}). "
                    f"Mô hình đứng nhì '{second_model}' có CV F1 là {second_f1:.4f} (chênh lệch {diff:.4f} < 0.005), "
                    f"tuy nhiên '{top_model}' vẫn giữ ưu thế về Recall ({top_recall:.4f} so với {second_recall:.4f}) "
                    f"hoặc khả năng tổng quát. Chọn '{top_model}' làm Champion."
                )
                return champion, rationale

    champion = top_model
    rationale = (
        f"'{champion}' đạt mean CV F1 cao nhất trên 5 fold kiểm định chéo ({top_f1:.4f}), "
        f"vượt trội hơn hẳn các mô hình còn lại (chênh lệch >= {f1_close_threshold:.4f}). "
        f"Được khóa chính thức làm Champion trước khi mở tập test."
    )
    return champion, rationale


def get_environment_metadata() -> Dict[str, Any]:
    """Capture environment, library versions, and system configuration."""
    import sklearn
    import joblib

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
