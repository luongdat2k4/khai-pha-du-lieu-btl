"""Comprehensive evaluation metrics, diagnostic visualizations, and model interpretability."""

import os
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree


# Set uniform plotting aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute binary classification metrics for class 1 (Churn = Yes)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    report = classification_report(
        y_true,
        y_pred,
        target_names=["No Churn (0)", "Churn (1)"],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "threshold": float(threshold),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": [[tn, fp], [fn, tp]],
        },
        "classification_report": report,
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = 0.5

        try:
            metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["average_precision"] = float(np.mean(y_true))

        try:
            metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
        except Exception:
            metrics["brier_score"] = float(np.var(y_true))

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot formatted 2x2 confusion matrix with raw counts and percentages."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Formatted annotation: "Count\n(Percentage)"
    annot = np.empty_like(cm, dtype=object)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        square=True,
        ax=ax,
        linewidths=1.5,
        annot_kws={"size": 13, "weight": "bold"},
        xticklabels=["Dự đoán: Ở lại (0)", "Dự đoán: Rời bỏ (1)"],
        yticklabels=["Thực tế: Ở lại (0)", "Thực tế: Rời bỏ (1)"],
    )

    ax.set_title(f"Ma trận nhầm lẫn (Confusion Matrix) — {model_name}\n(Tập kiểm thử độc lập Test Set: N={len(y_true):,})", fontsize=12, pad=15)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    models_probs: Dict[str, Tuple[np.ndarray, float]],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot ROC curves comparing multiple models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for (name, (probs, auc_score)), color in zip(models_probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, probs)
        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {auc_score:.4f})",
            linewidth=2.2,
            color=color,
        )

    # Diagonal chance line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Ngẫu nhiên (AUC = 0.5000)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Tỷ lệ dương tính giả (False Positive Rate)", fontsize=11)
    ax.set_ylabel("Tỷ lệ dương tính thật (True Positive Rate / Recall)", fontsize=11)
    ax.set_title("Đường cong ROC (Receiver Operating Characteristic) trên tập Test", fontsize=12, pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_pr_curves(
    y_true: np.ndarray,
    models_probs: Dict[str, Tuple[np.ndarray, float]],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Precision-Recall curves comparing multiple models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    baseline_prevalence = float(np.mean(y_true))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for (name, (probs, ap_score)), color in zip(models_probs.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, probs)
        ax.plot(
            recall,
            precision,
            label=f"{name} (AP = {ap_score:.4f})",
            linewidth=2.2,
            color=color,
        )

    # Baseline prevalence line
    ax.axhline(
        y=baseline_prevalence,
        color="k",
        linestyle="--",
        alpha=0.6,
        label=f"Tỷ lệ gốc Churn ({baseline_prevalence:.1%})",
    )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Độ nhạy (Recall lớp 1)", fontsize=11)
    ax.set_ylabel("Độ chính xác (Precision lớp 1)", fontsize=11)
    ax.set_title("Đường cong Precision-Recall trên tập Test", fontsize=12, pad=12)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_model_comparison(
    cv_summary: Dict[str, Dict[str, Any]],
    test_summary: Optional[Dict[str, Dict[str, Any]]] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot side-by-side metric comparison across evaluated models."""
    model_names = [m for m in cv_summary.keys() if m != "dummy"]
    display_names = {
        "decision_tree": "Decision Tree",
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
    }

    metrics = ["F1 (lớp 1)", "Recall (lớp 1)", "Precision (lớp 1)", "ROC-AUC"]
    n_models = len(model_names)
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(n_metrics)
    width = 0.25

    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    for i, model in enumerate(model_names):
        cv = cv_summary[model]
        cv_scores = [
            cv.get("mean_f1", 0.0),
            cv.get("mean_recall", 0.0),
            cv.get("mean_precision", 0.0),
            cv.get("mean_roc_auc", 0.0),
        ]
        disp = display_names.get(model, model)
        rects = ax.bar(
            x + (i - (n_models - 1) / 2) * width,
            cv_scores,
            width,
            label=f"{disp} (CV)",
            color=colors[i % len(colors)],
            alpha=0.85,
        )
        # Add values on top of bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax.set_ylabel("Giá trị chỉ số", fontsize=11)
    ax.set_title("So sánh các chỉ số đánh giá Cross-Validation (5 Folds)", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10, weight="bold")
    ax.set_ylim([0, 1.05])
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_decision_tree_structure(
    dt_pipeline: Any,
    max_depth: int = 3,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Draw decision tree structure up to max_depth."""
    tree_classifier = dt_pipeline.named_steps["classifier"]
    preprocessor = dt_pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    # Clean feature names for readable tree display
    clean_names = [
        f.replace("num__", "").replace("cat__", "").replace(" ", "_")
        for f in feature_names
    ]

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        tree_classifier,
        max_depth=max_depth,
        feature_names=clean_names,
        class_names=["Stay (0)", "Churn (1)"],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
        proportion=True,
    )
    ax.set_title(
        f"Cấu trúc Cây Quyết định (Decision Tree — Độ sâu hiển thị giới hạn: {max_depth})\n"
        "Mỗi nút thể hiện điều kiện phân nhánh, tỷ lệ mẫu và màu sắc biểu thị lớp đa số",
        fontsize=13,
        pad=15,
    )
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_logistic_regression_coefficients(
    lr_pipeline: Any,
    top_n: int = 12,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot top positive and negative coefficients from Logistic Regression."""
    classifier = lr_pipeline.named_steps["classifier"]
    preprocessor = lr_pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    coefs = classifier.coef_[0]
    clean_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]

    df_coef = pd.DataFrame({"feature": clean_names, "coefficient": coefs})
    df_coef["abs_coef"] = df_coef["coefficient"].abs()
    df_coef = df_coef.sort_values(by="coefficient", ascending=True)

    # Top negative (driving retention) and top positive (driving churn)
    top_neg = df_coef.head(top_n // 2)
    top_pos = df_coef.tail(top_n // 2)
    df_plot = pd.concat([top_neg, top_pos]).drop_duplicates()

    colors = ["#2ca02c" if c < 0 else "#d62728" for c in df_plot["coefficient"]]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(df_plot["feature"], df_plot["coefficient"], color=colors, alpha=0.85, edgecolor="#555")

    ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Hệ số hồi quy $\\beta$ (Log-Odds)", fontsize=11)
    ax.set_title(
        "Hệ số Mô hình Hồi quy Logistic (Logistic Regression Coefficients)\n"
        "Đỏ: Làm tăng nguy cơ rời bỏ | Xanh: Gắn bó với dịch vụ (giảm rời bỏ)",
        fontsize=12,
        pad=12,
    )

    for bar in bars:
        width = bar.get_width()
        ha = "left" if width > 0 else "right"
        offset = 0.02 if width > 0 else -0.02
        ax.annotate(
            f"{width:+.3f}",
            xy=(width + offset, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha=ha,
            fontsize=8.5,
            weight="bold",
        )

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_feature_importance(
    pipeline: Any,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    top_n: int = 15,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Random Forest Feature Importance (MDI and Permutation Importance)."""
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    clean_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]

    mdi_importances = classifier.feature_importances_
    df_mdi = pd.DataFrame({"feature": clean_names, "importance": mdi_importances})
    df_mdi = df_mdi.sort_values(by="importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(
        df_mdi["feature"],
        df_mdi["importance"],
        color="#3b7a57",
        alpha=0.85,
        edgecolor="#2d5e43",
    )

    ax.set_xlabel("Mức độ quan trọng (MDI — Mean Decrease in Impurity)", fontsize=11)
    ax.set_title(
        f"Độ quan trọng đặc trưng hàng đầu của Random Forest (Top {top_n})\n"
        "Đóng góp lớn nhất vào quá trình phân tách và giảm độ vẩn đục (Gini/Entropy)",
        fontsize=12,
        pad=12,
    )

    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{width:.4f}",
            xy=(width + 0.002, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha="left",
            fontsize=8.5,
        )

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def extract_error_cases(
    X_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_cases: int = 3,
) -> Dict[str, pd.DataFrame]:
    """Extract sample TP, FN, FP, TN cases for qualitative error analysis."""
    df_analysis = X_df.copy()
    df_analysis["actual_churn"] = y_true.values
    df_analysis["predicted_churn"] = y_pred
    df_analysis["churn_probability"] = np.round(y_prob, 4)

    # True Positives: Actual 1, Pred 1 (High risk correctly identified)
    tp_mask = (df_analysis["actual_churn"] == 1) & (df_analysis["predicted_churn"] == 1)
    tp_samples = df_analysis[tp_mask].sort_values(by="churn_probability", ascending=False).head(n_cases)

    # False Negatives: Actual 1, Pred 0 (Churners missed by model - Critical loss)
    fn_mask = (df_analysis["actual_churn"] == 1) & (df_analysis["predicted_churn"] == 0)
    fn_samples = df_analysis[fn_mask].sort_values(by="churn_probability", ascending=True).head(n_cases)

    # False Positives: Actual 0, Pred 1 (Loyal customers flagged as churners - Unnecessary promotion)
    fp_mask = (df_analysis["actual_churn"] == 0) & (df_analysis["predicted_churn"] == 1)
    fp_samples = df_analysis[fp_mask].sort_values(by="churn_probability", ascending=False).head(n_cases)

    # True Negatives: Actual 0, Pred 0 (Loyal customers correctly identified)
    tn_mask = (df_analysis["actual_churn"] == 0) & (df_analysis["predicted_churn"] == 0)
    tn_samples = df_analysis[tn_mask].sort_values(by="churn_probability", ascending=True).head(n_cases)

    return {
        "true_positives": tp_samples,
        "false_negatives": fn_samples,
        "false_positives": fp_samples,
        "true_negatives": tn_samples,
    }
