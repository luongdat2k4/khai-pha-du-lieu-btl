"""Preprocessing pipelines for Telco Customer Churn models."""

from typing import List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from .schema import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS


def build_preprocessor(
    model_kind: str = "logistic",
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None
) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer tailored to the specified model type.

    Args:
        model_kind: One of 'logistic', 'tree', 'forest'
                    (or 'logistic_regression', 'decision_tree', 'random_forest').
        numeric_cols: List of numeric column names (defaults to schema.NUMERIC_COLUMNS).
        categorical_cols: List of categorical column names (defaults to schema.CATEGORICAL_COLUMNS).

    Returns:
        ColumnTransformer instance (unfitted).
    """
    if numeric_cols is None:
        numeric_cols = list(NUMERIC_COLUMNS)
    if categorical_cols is None:
        categorical_cols = list(CATEGORICAL_COLUMNS)

    norm_kind = model_kind.lower()

    # Numeric pipeline
    if norm_kind in ["logistic", "logistic_regression", "lr"]:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    elif norm_kind in ["tree", "decision_tree", "dt", "forest", "random_forest", "rf"]:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
    else:
        raise ValueError(
            f"Unknown model_kind: '{model_kind}'. Expected one of "
            "['logistic', 'tree', 'forest']."
        )

    # Categorical pipeline
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor
