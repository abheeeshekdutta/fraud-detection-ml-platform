from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fraud_platform.artifacts import ModelBundle, ModelMetadata, save_model_bundle


def _synthetic_training_frame() -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame(
        {
            "TransactionAmt": [20.0, 200.0, 35.0, 500.0, 75.0, 900.0, 15.0, 650.0],
            "ProductCD": ["W", "C", "W", "R", "H", "C", "W", "C"],
            "card1": [1001, 1002, 1001, 1003, 1004, 1002, 1001, 1003],
            "addr1": [100.0, 200.0, 100.0, 300.0, 300.0, 200.0, 100.0, 300.0],
            "P_emaildomain": [
                "a.test",
                "b.test",
                "a.test",
                "c.test",
                "a.test",
                "b.test",
                "a.test",
                "c.test",
            ],
            "DeviceType": [
                "desktop",
                "mobile",
                "desktop",
                "mobile",
                "desktop",
                "mobile",
                "desktop",
                "mobile",
            ],
            "id_31": [
                "chrome",
                "safari",
                "chrome",
                "firefox",
                "chrome",
                "safari",
                "chrome",
                "firefox",
            ],
        }
    )
    labels = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    return features, labels


def train_synthetic_model(output_dir: str | Path) -> ModelMetadata:
    features, labels = _synthetic_training_frame()
    categorical = ["ProductCD", "P_emaildomain", "DeviceType", "id_31"]
    numeric = ["TransactionAmt", "card1", "addr1"]
    model = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
                        ("numeric", "passthrough", numeric),
                    ]
                ),
            ),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )
    model.fit(features, labels)
    metadata = ModelMetadata(
        model_version="synthetic-fraud-model:1",
        feature_schema_version="v1",
        decision_policy_version="v1",
        model_type="logistic_regression_smoke",
    )
    save_model_bundle(ModelBundle(model=model, metadata=metadata), output_dir)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/model/latest")
    args = parser.parse_args()
    if not args.synthetic:
        raise SystemExit("Only --synthetic is implemented in the first training slice")
    train_synthetic_model(args.output_dir)


if __name__ == "__main__":
    main()
