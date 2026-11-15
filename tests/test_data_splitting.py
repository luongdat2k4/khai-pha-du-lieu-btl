"""Unit tests for dataset splitting, stratification, and manifest integrity."""

import os
import sys
import pandas as pd
import pytest

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from telco_churn.data import create_splits
from telco_churn.schema import ID_COLUMN, TARGET_COLUMN


def test_split_manifest_integrity():
    manifest_path = os.path.join("data", "splits", "split_manifest.csv")
    assert os.path.exists(manifest_path), f"Manifest file missing: {manifest_path}"

    df_manifest = pd.read_csv(manifest_path)
    expected_cols = ["customerID", "split", "fold", "Churn", "seed", "data_hash", "cleaning_version"]
    for col in expected_cols:
        assert col in df_manifest.columns, f"Column '{col}' missing from manifest"

    # Total row count
    assert len(df_manifest) == 7043, f"Expected 7043 rows in manifest, got {len(df_manifest)}"

    # Check partition sizes
    dev_rows = df_manifest[df_manifest["split"] == "dev"]
    test_rows = df_manifest[df_manifest["split"] == "test"]
    assert len(dev_rows) == 5634, f"Expected 5,634 dev rows, got {len(dev_rows)}"
    assert len(test_rows) == 1409, f"Expected 1,409 test rows, got {len(test_rows)}"

    # Check strict disjointness
    dev_ids = set(dev_rows["customerID"])
    test_ids = set(test_rows["customerID"])
    overlap = dev_ids.intersection(test_ids)
    assert len(overlap) == 0, f"Data leakage detected! {len(overlap)} IDs overlap"

    # Check stratification
    dev_churn_rate = (dev_rows["Churn"] == 1).mean()
    test_churn_rate = (test_rows["Churn"] == 1).mean()
    assert abs(dev_churn_rate - test_churn_rate) < 0.005, (
        f"Stratification mismatch: dev={dev_churn_rate:.4f}, test={test_churn_rate:.4f}"
    )

    # Check folds in dev set
    dev_folds = dev_rows["fold"].unique()
    assert set(dev_folds) == {0, 1, 2, 3, 4}, f"Expected 5 folds (0-4), got {dev_folds}"
    for fold in range(5):
        fold_count = (dev_rows["fold"] == fold).sum()
        assert 1120 <= fold_count <= 1130, f"Fold {fold} size {fold_count} is out of expected range"


def test_create_splits_synthetic():
    # Test create_splits logic with small synthetic DataFrame
    records = []
    for i in range(100):
        records.append({
            ID_COLUMN: f"SYNTH-{i:03d}",
            TARGET_COLUMN: 1 if i < 30 else 0,
            "feature1": i * 1.5,
        })
    df_synth = pd.DataFrame(records)

    dev_df, test_df, manifest_df = create_splits(
        df_synth,
        test_size=0.2,
        random_state=42,
        n_splits=5,
    )

    assert len(dev_df) == 80
    assert len(test_df) == 20
    assert set(dev_df[ID_COLUMN]).isdisjoint(set(test_df[ID_COLUMN]))
    assert (test_df[TARGET_COLUMN] == 1).sum() == 6  # 30% of 20 = 6
    assert (dev_df[TARGET_COLUMN] == 1).sum() == 24  # 30% of 80 = 24
