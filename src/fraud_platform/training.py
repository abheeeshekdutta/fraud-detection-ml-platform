from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_platform.artifacts import ModelBundle, ModelMetadata, save_model_bundle
from fraud_platform.datasets import prepare_ieee_cis_splits

CATEGORICAL_FEATURES = ["ProductCD", "P_emaildomain", "DeviceType", "id_31"]
NUMERIC_FEATURES = ["TransactionAmt", "card1", "addr1"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


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
    model = _logistic_regression_pipeline()
    model.fit(features, labels)
    metadata = ModelMetadata(
        model_version="synthetic-fraud-model:1",
        feature_schema_version="v1",
        decision_policy_version="v1",
        model_type="logistic_regression_smoke",
    )
    save_model_bundle(ModelBundle(model=model, metadata=metadata), output_dir)
    return metadata


def train_ieee_baseline_model(
    processed_dir: str | Path = "data/processed",
    output_dir: str | Path = "artifacts/model/latest",
    max_train_rows: int | None = None,
) -> ModelMetadata:
    processed_path = Path(processed_dir)
    train = pd.read_parquet(processed_path / "train.parquet")
    validation = pd.read_parquet(processed_path / "validation.parquet")
    if max_train_rows is not None and len(train) > max_train_rows:
        train = train.tail(max_train_rows).copy()

    model = _logistic_regression_pipeline()
    model.fit(_features(train), train["isFraud"].astype(int))
    probabilities = model.predict_proba(_features(validation))[:, 1]
    labels = validation["isFraud"].astype(int)
    metadata = ModelMetadata(
        model_version="ieee-logistic-baseline:1",
        feature_schema_version="v1",
        decision_policy_version="v1",
        model_type="logistic_regression_ieee_baseline",
    )
    save_model_bundle(ModelBundle(model=model, metadata=metadata), output_dir)
    summary = {
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_fraud_rate": float(train["isFraud"].mean()),
        "validation_fraud_rate": float(labels.mean()),
        "validation_roc_auc": float(roc_auc_score(labels, probabilities)),
        "validation_pr_auc": float(average_precision_score(labels, probabilities)),
        "validation_brier_score": float(brier_score_loss(labels, probabilities)),
    }
    Path(output_dir, "training_summary.json").write_text(json.dumps(summary, indent=2))
    return metadata


def _logistic_regression_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "categorical",
                            Pipeline(
                                steps=[
                                    (
                                        "impute",
                                        SimpleImputer(strategy="constant", fill_value="missing"),
                                    ),
                                    ("encode", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            CATEGORICAL_FEATURES,
                        ),
                        (
                            "numeric",
                            Pipeline(
                                steps=[
                                    ("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            NUMERIC_FEATURES,
                        ),
                    ]
                ),
            ),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.reindex(columns=MODEL_FEATURES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--ieee-baseline", action="store_true")
    parser.add_argument("--prepare-ieee", action="store_true")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="artifacts/model/latest")
    parser.add_argument("--max-train-rows", type=int)
    args = parser.parse_args()
    if args.prepare_ieee:
        prepare_ieee_cis_splits(raw_dir=args.raw_dir, output_dir=args.processed_dir)
        return
    if args.synthetic:
        train_synthetic_model(args.output_dir)
        return
    if args.ieee_baseline:
        train_ieee_baseline_model(
            processed_dir=args.processed_dir,
            output_dir=args.output_dir,
            max_train_rows=args.max_train_rows,
        )
        return
    raise SystemExit("Choose --synthetic, --prepare-ieee, or --ieee-baseline")


if __name__ == "__main__":
    main()
