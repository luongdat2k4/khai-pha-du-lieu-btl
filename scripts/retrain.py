"""Script to execute monthly or manual retraining lifecycle conforming to Plan 04 and Plan 06."""

import os
import sys
import json
import argparse
from datetime import datetime, timezone

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from telco_churn.retraining import run_retraining, DEFAULT_CONFIG_PATH
from telco_churn.storage import DEFAULT_DB_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Run Telco Customer Churn model retraining lifecycle."
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Cutoff timestamp in ISO format (defaults to current UTC time)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform gate checking, candidate training, and evaluation without promoting or modifying active model",
    )
    parser.add_argument(
        "--ignore-gate",
        action="store_true",
        help="Force candidate training and comparison even if minimum gate count is not reached (demo/test only)",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Include synthetic demo records in retraining pool",
    )
    parser.add_argument(
        "--check-schedule",
        action="store_true",
        help="Enforce monthly execution schedule (Day 01 at 02:00 AM VN time) conforming to Plan 06 Section 4",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to retraining configuration file (default: configs/retraining.json)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Path to artifacts root directory",
    )
    args = parser.parse_args()

    as_of = args.as_of or datetime.now(timezone.utc).isoformat()

    print("=" * 70)
    print("TELCO CHURN RETRAINING LIFECYCLE")
    print(f"Timestamp as-of: {as_of}")
    print(f"Dry-run mode:    {args.dry_run}")
    print(f"Ignore gate:     {args.ignore_gate}")
    print(f"Check schedule:  {args.check_schedule}")
    print(f"Config path:     {args.config}")
    print("=" * 70)

    result = run_retraining(
        as_of=as_of,
        dry_run=args.dry_run,
        config=args.config,
        ignore_gate=args.ignore_gate,
        ignore_schedule=not args.check_schedule,
        allow_synthetic=args.allow_synthetic,
        db_path=args.db_path,
        artifacts_dir=args.artifacts_dir,
    )

    print("\n--- Gate Check Result ---")
    gate_details = result.get("gate_details", {})
    print(f"Gate Passed: {result.get('gate_passed')}")
    print(f"Total Eligible Snapshots: {gate_details.get('total_eligible_snapshots')}")
    print(f"Class 1 (Yes) Count:     {gate_details.get('class_1_count')}")
    print(f"Class 0 (No) Count:      {gate_details.get('class_0_count')}")
    if gate_details.get("failures"):
        print("Gate Failures:")
        for fail in gate_details["failures"]:
            print(f"  - {fail}")

    if result.get("snapshot_path"):
        print("\n--- Frozen Snapshot Dataset ---")
        print(f"Snapshot File: {result.get('snapshot_path')}")
        print(f"SHA-256 Hash:  {result.get('snapshot_hash')}")

    print("\n--- Candidate Evaluation & Decision ---")
    print(f"Decision: {result.get('decision')}")
    print(f"Reason:   {result.get('reason')}")

    if result.get("candidate_metrics") and result.get("champion_metrics"):
        champ = result["champion_metrics"]
        cand = result["candidate_metrics"]
        print("\nComparison Set Metrics:")
        print(f"  Metric       | Champion ({result.get('champion_version')}) | Candidate | Delta")
        print(f"  -------------+----------------------+-----------+-------")
        print(f"  F1 Score     | {champ['f1']:.4f}               | {cand['f1']:.4f}    | {result.get('delta_f1'):+.4f}")
        print(f"  Recall       | {champ['recall']:.4f}               | {cand['recall']:.4f}    | {result.get('delta_recall'):+.4f}")
        print(f"  Precision    | {champ['precision']:.4f}               | {cand['precision']:.4f}    | {cand['precision']-champ['precision']:+.4f}")
        print(f"  ROC-AUC      | {champ['roc_auc']:.4f}               | {cand['roc_auc']:.4f}    | {cand['roc_auc']-champ['roc_auc']:+.4f}")
        print(f"  Latency (ms) | {champ.get('prediction_time_ms', 0.0):.2f}               | {cand.get('prediction_time_ms', 0.0):.2f}    | -")

    if result.get("decision") == "PROMOTE":
        if args.dry_run:
            print("\n[!] Dry-run: Candidate met promotion criteria, but was NOT activated.")
        else:
            print(f"\n[+] Candidate promoted and activated as version: {result.get('candidate_version')}")

    print("=" * 70)


if __name__ == "__main__":
    main()
