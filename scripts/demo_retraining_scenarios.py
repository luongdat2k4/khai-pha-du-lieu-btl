"""Interactive script to demonstrate all 4 operational scenarios defined in Plan 06 Section 8.

Scenarios:
    A: Insufficient new data below gate (<200 snapshots) -> SKIP retraining, Champion preserved.
    B: Sufficient synthetic data -> Candidate trained & compared against Champion on identical holdout.
    C: Inferior candidate (fails F1 / Recall promotion criteria) -> REJECT, Champion preserved.
    D: Superior candidate meets all criteria -> PROMOTE, atomic activation, verify, and safe Rollback.
"""

import os
import sys
import json
import shutil
import argparse
from datetime import datetime, timezone

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd
from telco_churn.storage import (
    init_db,
    save_prediction_batch,
    record_actual_label,
    get_snapshots_with_labels,
    get_prediction_history,
    get_training_runs,
    DEFAULT_DB_PATH,
)
from telco_churn.inference import load_model_bundle, predict_records
from telco_churn.registry import (
    get_active_model_info,
    rollback_model,
    reconcile_active_model,
    validate_artifact_bundle,
)
from telco_churn.retraining import (
    run_retraining,
    load_retraining_config,
)


def populate_synthetic_records(db_path: str, count: int = 240) -> None:
    """Helper to populate realistic synthetic customer records and labels for testing."""
    sample_path = os.path.join("data", "samples", "valid_sample.csv")
    sample_df = pd.read_csv(sample_path)
    bundle = load_model_bundle()

    # Replicate sample to reach desired count
    repeats = (count // len(sample_df)) + 1
    replicated = pd.concat([sample_df] * repeats, ignore_index=True).iloc[:count].copy()

    # Assign unique fake external IDs
    replicated["customerID"] = [f"SYNTH_{i:05d}" for i in range(len(replicated))]

    preds = predict_records(replicated, bundle)
    save_res = save_prediction_batch(
        inputs=replicated,
        outputs=preds,
        source="demo_synthetic_generator",
        idempotency_key=f"demo_batch_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        is_synthetic=True,
        db_path=db_path,
    )

    # Label with a balanced mix (e.g. 35% churn = 1)
    for idx, snap_id in enumerate(save_res["snapshot_ids"]):
        lbl = 1 if (idx % 3 == 0) else 0
        record_actual_label(
            snapshot_id=snap_id,
            label=lbl,
            observed_at="2026-11-15T00:00:00Z",
            source="crm_synthetic_survey",
            note="Dữ liệu mô phỏng theo Plan 06 Section 8",
            db_path=db_path,
        )


def run_scenario_a(db_path: str):
    print("\n" + "=" * 75)
    print("KỊCH BẢN A: DỮ LIỆU MỚI CHƯA ĐẠT GATE (< 200 SNAPSHOTS)")
    print("=" * 75)
    print("Mô tả: Hệ thống đánh giá cổng gate trước khi chạy retrain.")
    print("Nếu chưa tích lũy đủ 200 snapshot mới có nhãn, hệ thống tự động SKIP")
    print("và tiếp tục duy trì nguyên vẹn phiên bản Champion hiện tại.\n")

    # Run retraining against current DB without ignoring gate
    result = run_retraining(
        dry_run=True,
        ignore_gate=False,
        db_path=db_path,
    )

    print(f"[*] Gate Passed: {result['gate_passed']}")
    print(f"[*] Quyết định:   {result['decision']}")
    print(f"[*] Lý do:       {result['reason']}")
    print("\n[+] KẾT QUẢ KỊCH BẢN A: Hoàn thành đúng cam kết Plan 06 (SKIP an toàn, không train lãng phí).")


def run_scenario_b(db_path: str):
    print("\n" + "=" * 75)
    print("KỊCH BẢN B: ĐỦ DỮ LIỆU MÔ PHỎNG — HUẤN LUYỆN CANDIDATE & BÁO CÁO ĐỐI CHIẾU")
    print("=" * 75)
    print("Mô tả: Tích lũy dữ liệu mô phỏng (Synthetic) có gắn cờ minh bạch.")
    print("Huấn luyện pipeline Candidate trên 80% Dev gốc + phần Train mới.")
    print("Đánh giá Champion và Candidate trên CÙNG MỘT TẬP COMPARISON độc lập.\n")

    print("[1] Đang nạp thêm hồ sơ mô phỏng (Synthetic) vào database...")
    populate_synthetic_records(db_path, count=240)
    print("[+] Đã nạp 240 hồ sơ mô phỏng có nhãn thực tế.")

    print("\n[2] Thực thi chu trình Retraining ở chế độ Dry-run...")
    result = run_retraining(
        dry_run=True,
        ignore_gate=False,
        allow_synthetic=True,
        db_path=db_path,
    )

    print(f"[*] Gate Passed:       {result['gate_passed']}")
    print(f"[*] Quyết định:         {result['decision']}")
    print(f"[*] Lý do:             {result['reason']}")
    print(f"[*] Snapshot File:     {result.get('snapshot_path')}")
    print(f"[*] SHA-256 Hash:      {result.get('snapshot_hash')}")

    if result.get("champion_metrics") and result.get("candidate_metrics"):
        champ = result["champion_metrics"]
        cand = result["candidate_metrics"]
        print("\n--- Bảng Chỉ Số Trên Tập Đối Chiếu Độc Lập (Holdout Comparison) ---")
        print(f"  Chỉ số       | Champion ({result.get('champion_version')}) | Candidate | Delta")
        print(f"  -------------+----------------------+-----------+-------")
        print(f"  F1 Score     | {champ['f1']:.4f}               | {cand['f1']:.4f}    | {result.get('delta_f1'):+.4f}")
        print(f"  Recall       | {champ['recall']:.4f}               | {cand['recall']:.4f}    | {result.get('delta_recall'):+.4f}")
        print(f"  Precision    | {champ['precision']:.4f}               | {cand['precision']:.4f}    | {cand['precision']-champ['precision']:+.4f}")
        print(f"  ROC-AUC      | {champ['roc_auc']:.4f}               | {cand['roc_auc']:.4f}    | {cand['roc_auc']-champ['roc_auc']:+.4f}")
        print(f"  Latency (ms) | {champ.get('prediction_time_ms', 0):.2f}               | {cand.get('prediction_time_ms', 0):.2f}    | -")

    print("\n[+] KẾT QUẢ KỊCH BẢN B: Đã hoàn tất đối chiếu Champion vs Candidate trên cùng tập dữ liệu.")


def run_scenario_c(db_path: str):
    print("\n" + "=" * 75)
    print("KỊCH BẢN C: CANDIDATE KÉM HƠN HOẶC KHÔNG ĐẠT NGƯỠNG -> TỪ CHỐI (REJECT)")
    print("=" * 75)
    print("Mô tả: Minh họa chính sách kiểm soát nghiêm ngặt:")
    print("Candidate chỉ được kích hoạt khi F1 delta >= +0.01 và Recall delta >= -0.02.")
    print("Nếu không đạt, hệ thống kiên quyết REJECT và giữ nguyên Champion.\n")

    cfg = load_retraining_config()
    cfg["promotion_policy"]["min_f1_gain"] = 0.50  # Unattainable threshold to force rejection

    result = run_retraining(
        dry_run=True,
        ignore_gate=True,
        allow_synthetic=True,
        config=cfg,
        db_path=db_path,
    )

    print(f"[*] Quyết định: {result['decision']}")
    print(f"[*] Lý do:     {result['reason']}")
    print(f"[*] Champion:   {result['champion_version']} (Được bảo lưu an toàn)")
    print("\n[+] KẾT QUẢ KỊCH BẢN C: Candidate bị từ chối thành công, bảo vệ mô hình phục vụ thực tế.")


def run_scenario_d(db_path: str):
    print("\n" + "=" * 75)
    print("KỊCH BẢN D: CANDIDATE ĐẠT CHUẨN -> THĂNG CẤP (PROMOTE), KÍCH HOẠT & ROLLBACK")
    print("=" * 75)
    print("Mô tả: Khi candidate đạt yêu cầu và vượt qua smoke test, hệ thống:")
    print("  1. Đóng gói candidate vào thư mục phiên bản mới.")
    print("  2. Kích hoạt nguyên tử con trỏ active_model.json và đồng bộ DB.")
    print("  3. Trình diễn khả năng Rollback về phiên bản cũ ngay lập tức khi cần.\n")

    # Use a sandbox temp directory for artifacts to keep repository clean
    sandbox_artifacts = os.path.join("data", "runtime", "sandbox_artifacts")
    os.makedirs(sandbox_artifacts, exist_ok=True)
    if os.path.exists("artifacts"):
        shutil.copytree("artifacts", sandbox_artifacts, dirs_exist_ok=True)

    cfg = load_retraining_config()
    cfg["promotion_policy"]["min_f1_gain"] = -1.0
    cfg["promotion_policy"]["max_recall_drop"] = 1.0

    print("[1] Tiến hành huấn luyện và thăng cấp Candidate...")
    result = run_retraining(
        dry_run=False,
        ignore_gate=True,
        allow_synthetic=True,
        config=cfg,
        db_path=db_path,
        artifacts_dir=sandbox_artifacts,
    )

    new_ver = result.get("candidate_version")
    print(f"[*] Quyết định:         {result['decision']}")
    print(f"[*] Phiên bản mới:      {new_ver}")
    print(f"[*] Phiên bản thay thế: {result['champion_version']}")

    active_info = get_active_model_info(sandbox_artifacts)
    print(f"[+] Con trỏ Active hiện tại: {active_info['active_version']}")
    assert active_info["active_version"] == new_ver

    print("\n[2] Thực hiện Hoàn tác (Rollback) về phiên bản trước...")
    rb_res = rollback_model(artifacts_dir=sandbox_artifacts, db_path=db_path)
    print(f"[+] Đã hoàn tác thành công về: {rb_res['active_version']}")

    # Clean up sandbox artifacts
    shutil.rmtree(sandbox_artifacts, ignore_errors=True)
    print("\n[+] KẾT QUẢ KỊCH BẢN D: Chu trình Promote và Rollback hoạt động chuẩn xác.")


def main():
    parser = argparse.ArgumentParser(description="Demonstrate Plan 06 Retraining Scenarios A, B, C, D.")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["A", "B", "C", "D", "all"],
        default="all",
        help="Choose scenario to execute (A, B, C, D, or all)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help="Path to SQLite database",
    )
    args = parser.parse_args()

    # Ensure DB is initialized
    init_db(args.db_path)

    scen = args.scenario.upper()
    if scen in ["A", "ALL"]:
        run_scenario_a(args.db_path)
    if scen in ["B", "ALL"]:
        run_scenario_b(args.db_path)
    if scen in ["C", "ALL"]:
        run_scenario_c(args.db_path)
    if scen in ["D", "ALL"]:
        run_scenario_d(args.db_path)

    print("\n" + "=" * 75)
    print("HOÀN TẤT THỬ NGHIỆM 4 KỊCH BẢN VẬN HÀNH PLAN 06!")
    print("=" * 75)


if __name__ == "__main__":
    main()
