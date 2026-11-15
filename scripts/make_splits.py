"""Script to create stratified splits (80/20) and 5-fold CV manifest."""

import argparse
import json
import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from telco_churn.data import load_raw_data, create_splits
from telco_churn.schema import validate_records


def main():
    parser = argparse.ArgumentParser(description="Create stratified train/test split and CV manifest.")
    parser.add_argument(
        "--config",
        default="configs/experiment.json",
        help="Path to experiment configuration JSON file",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    data_path = config.get("data_raw_path", "telco-customer-churn-on-icp4d/data/Telco-Customer-Churn.csv")
    split_cfg = config.get("split", {})
    manifest_path = split_cfg.get("manifest_path", "data/splits/split_manifest.csv")
    test_size = split_cfg.get("test_size", 0.2)
    random_state = split_cfg.get("random_state", 42)
    n_splits = split_cfg.get("n_splits", 5)
    cleaning_version = config.get("cleaning_rule_version", "v1")

    print(f"[*] Loading raw dataset from: {data_path}")
    df_raw = load_raw_data(data_path)

    print("[*] Validating and standardizing records...")
    valid_df, errors = validate_records(df_raw, mode="training")
    if errors:
        print(f"[!] Found {len(errors)} validation notes/errors:")
        for err in errors[:5]:
            print(f"    - {err}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more.")

    print(f"[*] Valid records for splitting: {len(valid_df):,} / {len(df_raw):,}")

    dev_df, test_df, manifest_df = create_splits(
        df=valid_df,
        test_size=test_size,
        random_state=random_state,
        n_splits=n_splits,
        output_manifest_path=manifest_path,
        source_file_path=data_path,
        cleaning_version=cleaning_version,
    )

    print(f"[+] Split manifest created at: {manifest_path}")
    print(f"    - Development set (80%): {len(dev_df):,} records")
    print(f"      Churn=1: {(dev_df['Churn'] == 1).sum():,} ({(dev_df['Churn'] == 1).mean():.2%})")
    print(f"      Churn=0: {(dev_df['Churn'] == 0).sum():,} ({(dev_df['Churn'] == 0).mean():.2%})")
    print(f"    - Test set (20%): {len(test_df):,} records")
    print(f"      Churn=1: {(test_df['Churn'] == 1).sum():,} ({(test_df['Churn'] == 1).mean():.2%})")
    print(f"      Churn=0: {(test_df['Churn'] == 0).sum():,} ({(test_df['Churn'] == 0).mean():.2%})")

    # Verify folds in dev set
    dev_manifest = manifest_df[manifest_df["split"] == "dev"]
    fold_dist = dev_manifest.groupby(["fold", "Churn"]).size().unstack()
    print("[*] 5-Fold Stratified Distribution (Dev Set):")
    print(fold_dist)


if __name__ == "__main__":
    main()
