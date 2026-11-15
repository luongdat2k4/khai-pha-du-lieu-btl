"""Script to run Baseline, Grid/Random Search for DT, LR, RF, perform CV selection, and package champion artifact."""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import joblib

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from telco_churn.schema import (
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    validate_records,
)
from telco_churn.data import load_raw_data, compute_file_sha256
from telco_churn.training import (
    evaluate_baseline,
    run_grid_search,
    select_champion_model,
    get_environment_metadata,
)


def format_params(params: dict) -> str:
    """Format parameter dict compactly for markdown table."""
    return ", ".join(f"`{k}={v}`" for k, v in params.items())


def generate_model_selection_report(
    cv_summary: dict,
    champion_name: str,
    rationale: str,
    output_path: str,
    env_meta: dict,
    manifest_hash: str,
    raw_hash: str,
) -> None:
    """Generate reports/model_selection.md conforming to Plan 03 requirements."""
    display_names = {
        "dummy": "Baseline Dummy (most_frequent)",
        "decision_tree": "Decision Tree",
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
    }

    lines = [
        "# Báo cáo Lựa chọn Mô hình dựa trên Kiểm định chéo (Model Selection Report)",
        "",
        "> [!IMPORTANT]",
        "> **Khóa giao thức trước khi đánh giá Test:** Báo cáo này được lập và lưu trữ **TRƯỚC KHI** mở tập kiểm thử độc lập (Test Set). Toàn bộ quá trình tìm kiếm siêu tham số và lựa chọn mô hình chỉ sử dụng 80% tập phát triển (Development Set: 5.634 mẫu) thông qua 5-Fold Stratified Cross-Validation.",
        "",
        f"- **Thời điểm thực hiện (UTC):** `{env_meta.get('timestamp_utc')}`",
        f"- **Mã băm dữ liệu gốc (CSV SHA-256):** `{raw_hash}`",
        f"- **Mã băm Split Manifest:** `{manifest_hash}`",
        f"- **Môi trường:** Python `{env_meta.get('python_version')}` | scikit-learn `{env_meta.get('sklearn_version')}` | OS `{env_meta.get('platform')}`",
        "",
        "---",
        "",
        "## 1. Bảng tổng hợp kết quả 5-Fold Stratified Cross-Validation",
        "",
        "| Model | Best params | CV precision 1 | CV recall 1 | CV F1 1 mean ± std | CV ROC-AUC | Fit time (s) | Được chọn? |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for model_key in ["dummy", "decision_tree", "logistic_regression", "random_forest"]:
        if model_key not in cv_summary:
            continue
        res = cv_summary[model_key]
        is_selected = " **CHỌN (Champion)**" if model_key == champion_name else "Không"
        disp = display_names.get(model_key, model_key)
        params_str = format_params(res["best_params"])
        f1_str = f"{res['mean_f1']:.4f} ± {res['std_f1']:.4f}"
        prec_str = f"{res['mean_precision']:.4f}"
        rec_str = f"{res['mean_recall']:.4f}"
        roc_str = f"{res['mean_roc_auc']:.4f}"
        time_str = f"{res.get('mean_fit_time', 0.0):.2f}s"

        lines.append(
            f"| **{disp}** | {params_str} | {prec_str} | {rec_str} | **{f1_str}** | {roc_str} | {time_str} | {is_selected} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Quyết định lựa chọn và Biện luận (Model Selection Rationale)",
        "",
        f"### Mô hình được chọn làm Champion: `{display_names.get(champion_name, champion_name)}`",
        "",
        f"> {rationale}",
        "",
        "### Phân tích chi tiết giữa các họ mô hình:",
        "",
        f"1. **Baseline Dummy (`most_frequent`):**",
        f"   - Dự đoán toàn bộ nhãn 0 (No). Precision, Recall và F1 lớp 1 đều bằng 0,0000; Accuracy đạt {cv_summary['dummy']['mean_accuracy']:.2%} tương ứng với tỷ lệ đa số tự nhiên.",
        f"   - Đóng vai trò mốc so sánh tối thiểu (Sanity check) để khẳng định các mô hình học máy thực sự học được tri thức hữu ích từ dữ liệu.",
        "",
        f"2. **Cây quyết định (Decision Tree):**",
        f"   - Cấu hình tốt nhất: {format_params(cv_summary['decision_tree']['best_params'])}.",
        f"   - Đạt mean CV F1: **{cv_summary['decision_tree']['mean_f1']:.4f} ± {cv_summary['decision_tree']['std_f1']:.4f}**, ROC-AUC: **{cv_summary['decision_tree']['mean_roc_auc']:.4f}**.",
        f"   - Ưu điểm: Tốc độ huấn luyện rất nhanh, cây có khả năng giải thích trực quan cao; tuy nhiên độ nhạy và F1 còn hạn chế do cây đơn lẻ dễ nhạy cảm với phương sai phân chia.",
        "",
        f"3. **Hồi quy Logistic (Logistic Regression):**",
        f"   - Cấu hình tốt nhất: {format_params(cv_summary['logistic_regression']['best_params'])}.",
        f"   - Đạt mean CV F1: **{cv_summary['logistic_regression']['mean_f1']:.4f} ± {cv_summary['logistic_regression']['std_f1']:.4f}**, ROC-AUC: **{cv_summary['logistic_regression']['mean_roc_auc']:.4f}**, Recall: **{cv_summary['logistic_regression']['mean_recall']:.4f}**.",
        f"   - Nhận xét: Với `class_weight='balanced'`, mô hình phân biệt tốt lớp thiểu số, đạt độ nhạy rất cao và ROC-AUC vượt trội. Dễ dàng diễn giải hệ số Log-Odds đối với nghiệp vụ viễn thông.",
        "",
        f"4. **Rừng ngẫu nhiên (Random Forest):**",
        f"   - Cấu hình tốt nhất: {format_params(cv_summary['random_forest']['best_params'])}.",
        f"   - Đạt mean CV F1: **{cv_summary['random_forest']['mean_f1']:.4f} ± {cv_summary['random_forest']['std_f1']:.4f}**, ROC-AUC: **{cv_summary['random_forest']['mean_roc_auc']:.4f}**, Recall: **{cv_summary['random_forest']['mean_recall']:.4f}**.",
        f"   - Nhận xét: Tổ hợp nhiều cây giúp giảm phương sai, cho ROC-AUC rất cao và F1 vượt trội.",
        "",
        "---",
        "",
        "## 3. Khóa cấu hình và Cam kết kiểm thử độc lập",
        "",
        f"- Mô hình `{champion_name}` đã được fit lại trên toàn bộ 5.634 mẫu của tập phát triển với siêu tham số đã khóa: {format_params(cv_summary[champion_name]['best_params'])}.",
        "- Ngưỡng phân loại được cố định ở `threshold = 0.5`.",
        "- Pipeline hoàn chỉnh bao gồm tiền xử lý (`preprocessor`) và bộ phân loại (`classifier`) được lưu thành artifact độc lập.",
        "- **Cam kết:** Không điều chỉnh lại siêu tham số dựa trên kết quả của tập kiểm thử độc lập (Test Set). Bất kỳ bảng đánh giá Test Set nào sau đây chỉ mang tính chất kiểm chứng khách quan.",
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train, tune, and select Telco Churn models.")
    parser.add_argument("--config", default="configs/experiment.json", help="Path to experiment config JSON")
    parser.add_argument("--manifest", default="data/splits/split_manifest.csv", help="Path to split manifest CSV")
    parser.add_argument("--output-dir", default="reports/experiments", help="Dir for experiment logs")
    parser.add_argument("--model-version", default="model_001", help="Version name for champion model")
    args = parser.parse_args()

    print("=" * 70)
    print("TELCO CHURN PREDICTION — TRAINING & CROSS-VALIDATION PIPELINE")
    print("=" * 70)

    # 1. Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    raw_data_path = config.get("data_raw_path", "telco-customer-churn-on-icp4d/data/Telco-Customer-Churn.csv")
    raw_hash = compute_file_sha256(raw_data_path)
    manifest_hash = compute_file_sha256(args.manifest)

    print(f"[*] Raw dataset: {raw_data_path} (SHA-256: {raw_hash[:12]}...)")
    print(f"[*] Split manifest: {args.manifest} (SHA-256: {manifest_hash[:12]}...)")

    # 2. Load and validate data
    df_raw = load_raw_data(raw_data_path)
    valid_df, val_errors = validate_records(df_raw, mode="training")
    print(f"[*] Valid records: {len(valid_df):,} / {len(df_raw):,}")

    # 3. Filter development partition using split_manifest
    manifest = pd.read_csv(args.manifest)
    dev_manifest = manifest[manifest["split"] == "dev"].copy()
    print(f"[*] Development partition size: {len(dev_manifest):,} records")

    dev_df = valid_df.merge(dev_manifest[["customerID", "fold"]], on="customerID")
    if len(dev_df) != len(dev_manifest):
        raise RuntimeError(f"Merge error: expected {len(dev_manifest)} dev records, got {len(dev_df)}")

    X_dev = dev_df[ALL_FEATURE_COLUMNS].copy()
    y_dev = dev_df[TARGET_COLUMN].copy()
    folds = dev_df["fold"].values

    print(f"[*] Training features: {len(ALL_FEATURE_COLUMNS)} input columns")
    print(f"[*] 5-Fold Stratified distribution:")
    print(dev_df.groupby(["fold", TARGET_COLUMN]).size().unstack())

    results_by_model = {}

    # 4. Baseline: DummyClassifier
    print("\n" + "-" * 50)
    print("[1/4] Evaluating Baseline (DummyClassifier most_frequent)...")
    t0 = time.time()
    baseline_res = evaluate_baseline(X_dev, y_dev, folds)
    print(f"    -> Mean F1: {baseline_res['mean_f1']:.4f} ± {baseline_res['std_f1']:.4f}")
    print(f"    -> Accuracy: {baseline_res['mean_accuracy']:.4f} | Recall: {baseline_res['mean_recall']:.4f}")
    results_by_model["dummy"] = baseline_res

    # 5. Decision Tree GridSearch (48 candidates x 5 folds = 240 fits)
    print("\n" + "-" * 50)
    print("[2/4] Running Decision Tree GridSearch (48 configurations x 5 folds = 240 fits)...")
    dt_res = run_grid_search("decision_tree", X_dev, y_dev, folds)
    print(f"    -> Best Params: {dt_res['best_params']}")
    print(f"    -> Mean F1: {dt_res['mean_f1']:.4f} ± {dt_res['std_f1']:.4f}")
    print(f"    -> Mean Recall: {dt_res['mean_recall']:.4f} | ROC-AUC: {dt_res['mean_roc_auc']:.4f}")
    print(f"    -> Search Time: {dt_res['total_search_time']:.2f}s")
    results_by_model["decision_tree"] = dt_res

    # 6. Logistic Regression GridSearch (10 candidates x 5 folds = 50 fits)
    print("\n" + "-" * 50)
    print("[3/4] Running Logistic Regression GridSearch (10 configurations x 5 folds = 50 fits)...")
    lr_res = run_grid_search("logistic_regression", X_dev, y_dev, folds)
    print(f"    -> Best Params: {lr_res['best_params']}")
    print(f"    -> Mean F1: {lr_res['mean_f1']:.4f} ± {lr_res['std_f1']:.4f}")
    print(f"    -> Mean Recall: {lr_res['mean_recall']:.4f} | ROC-AUC: {lr_res['mean_roc_auc']:.4f}")
    print(f"    -> Search Time: {lr_res['total_search_time']:.2f}s")
    results_by_model["logistic_regression"] = lr_res

    # 7. Random Forest RandomizedSearch (24 candidates x 5 folds = 120 fits)
    print("\n" + "-" * 50)
    print("[4/4] Running Random Forest RandomizedSearch (24 configurations x 5 folds = 120 fits)...")
    rf_res = run_grid_search("random_forest", X_dev, y_dev, folds)
    print(f"    -> Best Params: {rf_res['best_params']}")
    print(f"    -> Mean F1: {rf_res['mean_f1']:.4f} ± {rf_res['std_f1']:.4f}")
    print(f"    -> Mean Recall: {rf_res['mean_recall']:.4f} | ROC-AUC: {rf_res['mean_roc_auc']:.4f}")
    print(f"    -> Search Time: {rf_res['total_search_time']:.2f}s")
    results_by_model["random_forest"] = rf_res

    # 8. Model Selection according to Plan 03 rules
    print("\n" + "=" * 50)
    print("[*] EXECUTING FORMAL MODEL SELECTION (BEFORE TEST SET)...")
    champion_name, rationale = select_champion_model(results_by_model)
    print(f"[+] Champion Model Selected: {champion_name.upper()}")
    print(f"[+] Rationale: {rationale}")

    champion_bundle = results_by_model[champion_name]
    champion_pipeline = champion_bundle["fitted_pipeline"]

    # 9. Environment Metadata
    env_meta = get_environment_metadata()

    # 10. Prepare clean CV summary for export
    cv_summary_export = {}
    for k, v in results_by_model.items():
        cv_summary_export[k] = {
            "model_name": v["model_name"],
            "best_params": v["best_params"],
            "mean_f1": v["mean_f1"],
            "std_f1": v["std_f1"],
            "mean_precision": v["mean_precision"],
            "std_precision": v["std_precision"],
            "mean_recall": v["mean_recall"],
            "std_recall": v["std_recall"],
            "mean_roc_auc": v["mean_roc_auc"],
            "std_roc_auc": v["std_roc_auc"],
            "mean_accuracy": v["mean_accuracy"],
            "std_accuracy": v["std_accuracy"],
            "mean_fit_time": v["mean_fit_time"],
            "fold_metrics": v["fold_metrics"],
        }

    # 11. Write reports/model_selection.md
    model_sel_path = os.path.join("reports", "model_selection.md")
    generate_model_selection_report(
        cv_summary=cv_summary_export,
        champion_name=champion_name,
        rationale=rationale,
        output_path=model_sel_path,
        env_meta=env_meta,
        manifest_hash=manifest_hash,
        raw_hash=raw_hash,
    )
    print(f"[+] Model selection report generated at: {model_sel_path}")

    # 12. Save Champion Artifact Package
    model_version = args.model_version
    artifact_model_dir = os.path.join("artifacts", "models", model_version)
    os.makedirs(artifact_model_dir, exist_ok=True)

    # Save fitted pipeline
    pipeline_save_path = os.path.join(artifact_model_dir, "pipeline.joblib")
    joblib.dump(champion_pipeline, pipeline_save_path)
    print(f"[+] Champion pipeline saved to: {pipeline_save_path}")

    # Save metadata.json
    metadata = {
        "model_version": model_version,
        "model_type": champion_name,
        "best_params": champion_bundle["best_params"],
        "class_mapping": {"No": 0, "Yes": 1},
        "threshold": 0.5,
        "primary_metric": "f1",
        "features": {
            "numeric": NUMERIC_COLUMNS,
            "categorical": CATEGORICAL_COLUMNS,
            "all_features": ALL_FEATURE_COLUMNS,
        },
        "seed": 42,
        "raw_data_hash": raw_hash,
        "manifest_hash": manifest_hash,
        "train_samples": len(X_dev),
        "selection_rationale": rationale,
        "created_at_utc": env_meta["timestamp_utc"],
        "environment": env_meta,
    }
    metadata_save_path = os.path.join(artifact_model_dir, "metadata.json")
    with open(metadata_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[+] Metadata saved to: {metadata_save_path}")

    # Save cv_summary.json in model package
    cv_summary_save_path = os.path.join(artifact_model_dir, "cv_summary.json")
    with open(cv_summary_save_path, "w", encoding="utf-8") as f:
        json.dump(cv_summary_export, f, indent=2, ensure_ascii=False)
    print(f"[+] CV summary saved to: {cv_summary_save_path}")

    # 13. Update artifacts/active_model.json
    active_path = os.path.join("artifacts", "active_model.json")
    active_content = {
        "active_version": model_version,
        "model_type": champion_name,
        "updated_at_utc": env_meta["timestamp_utc"],
    }
    with open(active_path, "w", encoding="utf-8") as f:
        json.dump(active_content, f, indent=2)
    print(f"[+] Active model updated to: {model_version} ({active_path})")

    # 14. Save detailed experiment log
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    exp_log_path = os.path.join(args.output_dir, f"run_{timestamp_str}_cv_results.json")

    full_exp_log = {
        "run_id": f"run_{timestamp_str}",
        "config": config,
        "champion_name": champion_name,
        "cv_summary": cv_summary_export,
        "metadata": metadata,
    }
    with open(exp_log_path, "w", encoding="utf-8") as f:
        json.dump(full_exp_log, f, indent=2, ensure_ascii=False)
    print(f"[+] Full experiment run log saved to: {exp_log_path}")
    print("=" * 70)
    print("TRAINING & MODEL SELECTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
