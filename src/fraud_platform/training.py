from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_platform.artifacts import ModelBundle, ModelMetadata, save_model_bundle
from fraud_platform.datasets import prepare_ieee_cis_splits

CATEGORICAL_FEATURES = ["ProductCD", "P_emaildomain", "DeviceType", "id_31"]
NUMERIC_FEATURES = ["TransactionAmt", "card1", "addr1"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MODEL_CANDIDATES = ("logistic_regression", "catboost", "lightgbm")


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
    model_candidate: str = "logistic_regression",
    tune_hyperparameters: bool = False,
    tuning_splits: int = 3,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "fraud-detection-ieee",
) -> ModelMetadata:
    processed_path = Path(processed_dir)
    train = pd.read_parquet(processed_path / "train.parquet")
    validation = pd.read_parquet(processed_path / "validation.parquet")
    if max_train_rows is not None and len(train) > max_train_rows:
        train = train.tail(max_train_rows).copy()

    model_params = _default_model_params(model_candidate)
    tuning_summary = None
    if tune_hyperparameters:
        tuning_summary = _tune_hyperparameters(
            train=train,
            model_candidate=model_candidate,
            n_splits=tuning_splits,
        )
        model_params = tuning_summary["best_params"]

    model = _model_pipeline(model_candidate, model_params)
    model.fit(_features(train), train["isFraud"].astype(int))
    probabilities = model.predict_proba(_features(validation))[:, 1]
    labels = validation["isFraud"].astype(int)
    metadata = ModelMetadata(
        model_version=_model_version(model_candidate),
        feature_schema_version="v1",
        decision_policy_version="v1",
        model_type=_model_type(model_candidate),
    )
    save_model_bundle(ModelBundle(model=model, metadata=metadata), output_dir)
    summary = {
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "model_candidate": model_candidate,
        "model_params": model_params,
        "train_fraud_rate": float(train["isFraud"].mean()),
        "validation_fraud_rate": float(labels.mean()),
        "validation_roc_auc": float(roc_auc_score(labels, probabilities)),
        "validation_pr_auc": float(average_precision_score(labels, probabilities)),
        "validation_brier_score": float(brier_score_loss(labels, probabilities)),
    }
    if tuning_summary is not None:
        summary["tuning"] = tuning_summary
    Path(output_dir, "training_summary.json").write_text(json.dumps(summary, indent=2))
    if mlflow_tracking_uri:
        _log_mlflow_run(
            model=model,
            metadata=metadata,
            summary=summary,
            tracking_uri=mlflow_tracking_uri,
            experiment_name=mlflow_experiment_name,
            max_train_rows=max_train_rows,
        )
    return metadata


def _log_mlflow_run(
    model: Pipeline,
    metadata: ModelMetadata,
    summary: dict[str, object],
    tracking_uri: str,
    experiment_name: str,
    max_train_rows: int | None,
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=metadata.model_version):
        mlflow.log_params(
            {
                "model_version": metadata.model_version,
                "model_type": metadata.model_type,
                "model_candidate": summary["model_candidate"],
                "feature_schema_version": metadata.feature_schema_version,
                "decision_policy_version": metadata.decision_policy_version,
                "max_train_rows": max_train_rows or summary["train_rows"],
            }
        )
        mlflow.log_params(
            {f"model__{name}": value for name, value in dict(summary["model_params"]).items()}
        )
        if "tuning" in summary:
            tuning = dict(summary["tuning"])
            mlflow.log_params(
                {
                    "tuning_strategy": tuning["strategy"],
                    "tuning_splits": tuning["n_splits"],
                    "tuning_trials": len(tuning["trials"]),
                }
            )
        mlflow.log_metrics(
            {
                "train_fraud_rate": float(summary["train_fraud_rate"]),
                "validation_fraud_rate": float(summary["validation_fraud_rate"]),
                "validation_roc_auc": float(summary["validation_roc_auc"]),
                "validation_pr_auc": float(summary["validation_pr_auc"]),
                "validation_brier_score": float(summary["validation_brier_score"]),
            }
        )
        mlflow.sklearn.log_model(model, name="model")


def _logistic_regression_pipeline(model_params: dict[str, object] | None = None) -> Pipeline:
    params = model_params or _default_model_params("logistic_regression")
    return Pipeline(
        steps=[
            ("preprocess", _preprocessor(sparse_output=True)),
            ("model", LogisticRegression(**params)),
        ]
    )


def _model_pipeline(
    model_candidate: str,
    model_params: dict[str, object] | None = None,
) -> Pipeline:
    params = model_params or _default_model_params(model_candidate)
    if model_candidate == "logistic_regression":
        return _logistic_regression_pipeline(params)
    if model_candidate == "catboost":
        return Pipeline(
            steps=[
                ("preprocess", _preprocessor(sparse_output=False)),
                (
                    "model",
                    CatBoostClassifier(
                        loss_function="Logloss",
                        random_seed=42,
                        verbose=False,
                        allow_writing_files=False,
                        **params,
                    ),
                ),
            ]
        )
    if model_candidate == "lightgbm":
        return Pipeline(
            steps=[
                ("preprocess", _preprocessor(sparse_output=False)),
                (
                    "model",
                    LGBMClassifier(
                        random_state=42,
                        verbose=-1,
                        **params,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model candidate: {model_candidate}")


def _model_version(model_candidate: str) -> str:
    versions = {
        "logistic_regression": "ieee-logistic-baseline:1",
        "catboost": "ieee-catboost:1",
        "lightgbm": "ieee-lightgbm:1",
    }
    return versions[model_candidate]


def _model_type(model_candidate: str) -> str:
    model_types = {
        "logistic_regression": "logistic_regression_ieee_baseline",
        "catboost": "catboost_ieee_candidate",
        "lightgbm": "lightgbm_ieee_candidate",
    }
    return model_types[model_candidate]


def _default_model_params(model_candidate: str) -> dict[str, object]:
    defaults: dict[str, dict[str, object]] = {
        "logistic_regression": {"max_iter": 1000},
        "catboost": {"iterations": 50, "depth": 4, "learning_rate": 0.1},
        "lightgbm": {"n_estimators": 50, "learning_rate": 0.1, "num_leaves": 7},
    }
    return defaults[model_candidate].copy()


def _hyperparameter_grid(model_candidate: str) -> list[dict[str, object]]:
    if model_candidate == "catboost":
        return [
            {"iterations": 50, "depth": 4, "learning_rate": 0.1},
            {"iterations": 100, "depth": 4, "learning_rate": 0.05},
            {"iterations": 100, "depth": 6, "learning_rate": 0.05},
        ]
    if model_candidate == "lightgbm":
        return [
            {"n_estimators": 50, "learning_rate": 0.1, "num_leaves": 7},
            {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 15},
            {"n_estimators": 150, "learning_rate": 0.03, "num_leaves": 31},
        ]
    return [_default_model_params(model_candidate)]


def _tune_hyperparameters(
    train: pd.DataFrame,
    model_candidate: str,
    n_splits: int,
) -> dict[str, object]:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    trials = []
    best_trial = None
    for params in _hyperparameter_grid(model_candidate):
        fold_metrics = []
        for train_index, validation_index in splitter.split(train):
            fold_train = train.iloc[train_index]
            fold_validation = train.iloc[validation_index]
            model = _model_pipeline(model_candidate, params)
            model.fit(_features(fold_train), fold_train["isFraud"].astype(int))
            probabilities = model.predict_proba(_features(fold_validation))[:, 1]
            labels = fold_validation["isFraud"].astype(int)
            fold_metrics.append(
                {
                    "validation_pr_auc": float(average_precision_score(labels, probabilities)),
                    "validation_roc_auc": float(roc_auc_score(labels, probabilities)),
                    "validation_brier_score": float(brier_score_loss(labels, probabilities)),
                }
            )
        trial = {
            "params": params,
            "mean_validation_pr_auc": float(
                sum(metric["validation_pr_auc"] for metric in fold_metrics) / len(fold_metrics)
            ),
            "mean_validation_roc_auc": float(
                sum(metric["validation_roc_auc"] for metric in fold_metrics) / len(fold_metrics)
            ),
            "mean_validation_brier_score": float(
                sum(metric["validation_brier_score"] for metric in fold_metrics) / len(fold_metrics)
            ),
            "folds": fold_metrics,
        }
        trials.append(trial)
        if best_trial is None or (
            trial["mean_validation_pr_auc"],
            -trial["mean_validation_brier_score"],
        ) > (
            best_trial["mean_validation_pr_auc"],
            -best_trial["mean_validation_brier_score"],
        ):
            best_trial = trial

    assert best_trial is not None
    return {
        "strategy": "time_series_split_grid_search",
        "n_splits": n_splits,
        "best_params": best_trial["params"],
        "best_mean_validation_pr_auc": best_trial["mean_validation_pr_auc"],
        "trials": trials,
    }


def _preprocessor(sparse_output: bool) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "impute",
                            SimpleImputer(strategy="constant", fill_value="missing"),
                        ),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output),
                        ),
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
    parser.add_argument(
        "--model-candidate",
        choices=MODEL_CANDIDATES,
        default="logistic_regression",
    )
    parser.add_argument("--tune-hyperparameters", action="store_true")
    parser.add_argument("--tuning-splits", type=int, default=3)
    parser.add_argument("--mlflow-tracking-uri")
    parser.add_argument("--mlflow-experiment-name", default="fraud-detection-ieee")
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
            model_candidate=args.model_candidate,
            tune_hyperparameters=args.tune_hyperparameters,
            tuning_splits=args.tuning_splits,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment_name=args.mlflow_experiment_name,
        )
        return
    raise SystemExit("Choose --synthetic, --prepare-ieee, or --ieee-baseline")


if __name__ == "__main__":
    main()
