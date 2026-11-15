"""Script to audit Telco Customer Churn dataset and produce reports/data_audit.md."""

import argparse
import json
import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from telco_churn.data import load_raw_data, audit_dataset, generate_audit_markdown


def main():
    parser = argparse.ArgumentParser(description="Audit Telco Customer Churn dataset.")
    parser.add_argument(
        "--config",
        default="configs/experiment.json",
        help="Path to experiment configuration JSON file",
    )
    parser.add_argument(
        "--output",
        default="reports/data_audit.md",
        help="Output path for data audit report",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    data_path = config.get("data_raw_path", "telco-customer-churn-on-icp4d/data/Telco-Customer-Churn.csv")
    print(f"[*] Auditing dataset: {data_path}")

    df_raw = load_raw_data(data_path)
    audit_results = audit_dataset(df_raw, source_file_path=data_path)

    output_path = args.output
    generate_audit_markdown(audit_results, output_path)
    print(f"[+] Audit report successfully generated at: {output_path}")
    print(f"    - Total records: {audit_results['total_rows']:,}")
    print(f"    - Columns: {audit_results['total_cols']}")
    print(f"    - File SHA-256: {audit_results['file_hash']}")
    print(f"    - Target distribution: {audit_results['target_dist']}")
    print(f"    - TotalCharges blank count: {audit_results['tc_blank_count']}")
    print(f"    - Duplicate rows: {audit_results['total_duplicate_rows']}")


if __name__ == "__main__":
    main()
