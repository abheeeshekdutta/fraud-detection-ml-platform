from __future__ import annotations

import json

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from fraud_platform.artifacts import load_model_bundle
from fraud_platform.datasets import prepare_ieee_cis_splits
from fraud_platform.training import main, train_ieee_baseline_model


def test_prepare_ieee_cis_splits_writes_time_ordered_parquet_outputs(
    tmp_path,
    synthetic_transactions: pd.DataFrame,
    synthetic_identity: pd.DataFrame,
) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    synthetic_transactions.to_csv(raw_dir / "train_transaction.csv", index=False)
    synthetic_identity.to_csv(raw_dir / "train_identity.csv", index=False)

    summary = prepare_ieee_cis_splits(
        raw_dir=raw_dir,
        output_dir=processed_dir,
        train_fraction=0.50,
        calibration_fraction=0.17,
        validation_fraction=0.17,
    )

    assert summary["row_count"] == 6
    assert summary["identity_join_coverage"] == 0.5
    assert (processed_dir / "train.parquet").exists()
    assert (processed_dir / "calibration.parquet").exists()
    assert (processed_dir / "validation.parquet").exists()
    assert (processed_dir / "replay.parquet").exists()
    assert json.loads((processed_dir / "split_summary.json").read_text()) == summary

    train = pd.read_parquet(processed_dir / "train.parquet")
    replay = pd.read_parquet(processed_dir / "replay.parquet")

    assert train["TransactionDT"].max() < replay["TransactionDT"].min()
    assert "DeviceType" in train.columns


def test_train_ieee_baseline_model_writes_loadable_bundle(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    model_dir = tmp_path / "model"
    processed_dir.mkdir()
    train = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [10, 20, 30, 40],
            "TransactionAmt": [20.0, 200.0, 35.0, 500.0],
            "ProductCD": ["W", "C", "W", "R"],
            "card1": [1001, 1002, 1001, 1003],
            "addr1": [100.0, 200.0, None, 300.0],
            "P_emaildomain": ["a.test", "b.test", None, "c.test"],
            "DeviceType": ["desktop", "mobile", None, "mobile"],
            "id_31": ["chrome", "safari", None, "firefox"],
            "isFraud": [0, 1, 0, 1],
        }
    )
    validation = pd.DataFrame(
        {
            "TransactionID": [5, 6],
            "TransactionDT": [50, 60],
            "TransactionAmt": [75.0, 900.0],
            "ProductCD": ["H", "C"],
            "card1": [1004, 1002],
            "addr1": [300.0, 200.0],
            "P_emaildomain": ["a.test", "b.test"],
            "DeviceType": ["desktop", "mobile"],
            "id_31": ["chrome", "safari"],
            "isFraud": [0, 1],
        }
    )
    train.to_parquet(processed_dir / "train.parquet", index=False)
    validation.to_parquet(processed_dir / "validation.parquet", index=False)

    metadata = train_ieee_baseline_model(processed_dir=processed_dir, output_dir=model_dir)
    bundle = load_model_bundle(model_dir)
    training_summary = json.loads((model_dir / "training_summary.json").read_text())

    assert metadata.model_version == "ieee-logistic-baseline:1"
    assert bundle.metadata.model_type == "logistic_regression_ieee_baseline"
    assert 0.0 <= training_summary["validation_roc_auc"] <= 1.0
    assert 0.0 <= training_summary["validation_pr_auc"] <= 1.0


def test_train_ieee_baseline_model_logs_mlflow_run(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    model_dir = tmp_path / "model"
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    processed_dir.mkdir()
    frame = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "TransactionDT": [10, 20, 30, 40, 50, 60],
            "TransactionAmt": [20.0, 200.0, 35.0, 500.0, 75.0, 900.0],
            "ProductCD": ["W", "C", "W", "R", "H", "C"],
            "card1": [1001, 1002, 1001, 1003, 1004, 1002],
            "addr1": [100.0, 200.0, None, 300.0, 300.0, 200.0],
            "P_emaildomain": ["a.test", "b.test", None, "c.test", "a.test", "b.test"],
            "DeviceType": ["desktop", "mobile", None, "mobile", "desktop", "mobile"],
            "id_31": ["chrome", "safari", None, "firefox", "chrome", "safari"],
            "isFraud": [0, 1, 0, 1, 0, 1],
        }
    )
    frame.iloc[:4].to_parquet(processed_dir / "train.parquet", index=False)
    frame.iloc[4:].to_parquet(processed_dir / "validation.parquet", index=False)

    train_ieee_baseline_model(
        processed_dir=processed_dir,
        output_dir=model_dir,
        mlflow_tracking_uri=tracking_uri,
        mlflow_experiment_name="fraud-test",
    )

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("fraud-test")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].data.params["model_version"] == "ieee-logistic-baseline:1"
    assert "validation_pr_auc" in runs[0].data.metrics


def test_prepare_ieee_cli_mode_exits_after_writing_splits(
    tmp_path,
    synthetic_transactions: pd.DataFrame,
    synthetic_identity: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    synthetic_transactions.to_csv(raw_dir / "train_transaction.csv", index=False)
    synthetic_identity.to_csv(raw_dir / "train_identity.csv", index=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "fraud-train",
            "--prepare-ieee",
            "--raw-dir",
            str(raw_dir),
            "--processed-dir",
            str(processed_dir),
        ],
    )

    main()

    assert (processed_dir / "split_summary.json").exists()
