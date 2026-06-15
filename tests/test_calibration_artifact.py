from __future__ import annotations

import json

import pandas as pd

from fraud_platform.calibration import fit_calibrator_artifact, load_calibrator
from fraud_platform.conformal import fit_conformal_artifact, load_conformal
from fraud_platform.training import train_ieee_baseline_model


def test_fit_calibrator_artifact_writes_calibrator_and_summary(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "calibration"
    processed_dir.mkdir()
    frame = pd.DataFrame(
        {
            "TransactionID": range(1, 17),
            "TransactionDT": range(10, 170, 10),
            "TransactionAmt": [
                20.0,
                200.0,
                35.0,
                500.0,
                75.0,
                900.0,
                15.0,
                650.0,
                40.0,
                700.0,
                55.0,
                850.0,
                45.0,
                300.0,
                65.0,
                950.0,
            ],
            "ProductCD": ["W", "C", "W", "R"] * 4,
            "card1": [1001, 1002, 1001, 1003] * 4,
            "addr1": [100.0, 200.0, None, 300.0] * 4,
            "P_emaildomain": ["a.test", "b.test", None, "c.test"] * 4,
            "DeviceType": ["desktop", "mobile", None, "mobile"] * 4,
            "id_31": ["chrome", "safari", None, "firefox"] * 4,
            "isFraud": [0, 1, 0, 1] * 4,
        }
    )
    frame.iloc[:8].to_parquet(processed_dir / "train.parquet", index=False)
    frame.iloc[8:12].to_parquet(processed_dir / "calibration.parquet", index=False)
    frame.iloc[12:].to_parquet(processed_dir / "validation.parquet", index=False)
    train_ieee_baseline_model(processed_dir=processed_dir, output_dir=model_dir)

    summary = fit_calibrator_artifact(
        processed_dir=processed_dir,
        model_dir=model_dir,
        output_dir=output_dir,
        method="isotonic",
    )

    saved_summary = json.loads((output_dir / "calibration_summary.json").read_text())
    calibrator = load_calibrator(output_dir / "calibrator.pkl")
    assert saved_summary == summary
    assert calibrator.method == "isotonic"
    assert summary["model_version"] == "ieee-logistic-baseline:1"
    assert summary["calibration_rows"] == 4
    assert summary["validation_rows"] == 4
    assert "validation_raw_brier_score" in summary
    assert "validation_calibrated_brier_score" in summary
    assert "validation_raw_calibration_error" in summary
    assert "validation_calibrated_calibration_error" in summary


def test_fit_conformal_artifact_writes_conformal_and_summary(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "conformal"
    _write_tiny_processed_splits(processed_dir)
    train_ieee_baseline_model(processed_dir=processed_dir, output_dir=model_dir)

    summary = fit_conformal_artifact(
        processed_dir=processed_dir,
        model_dir=model_dir,
        output_dir=output_dir,
        alpha=0.25,
    )

    saved_summary = json.loads((output_dir / "conformal_summary.json").read_text())
    conformal = load_conformal(output_dir / "conformal.pkl")
    assert saved_summary == summary
    assert conformal.alpha == 0.25
    assert summary["model_version"] == "ieee-logistic-baseline:1"
    assert summary["calibration_rows"] == 4
    assert summary["validation_rows"] == 4
    assert "validation_conformal_coverage" in summary
    assert "conformal_path" in summary


def _write_tiny_processed_splits(processed_dir) -> None:
    processed_dir.mkdir()
    frame = pd.DataFrame(
        {
            "TransactionID": range(1, 17),
            "TransactionDT": range(10, 170, 10),
            "TransactionAmt": [
                20.0,
                200.0,
                35.0,
                500.0,
                75.0,
                900.0,
                15.0,
                650.0,
                40.0,
                700.0,
                55.0,
                850.0,
                45.0,
                300.0,
                65.0,
                950.0,
            ],
            "ProductCD": ["W", "C", "W", "R"] * 4,
            "card1": [1001, 1002, 1001, 1003] * 4,
            "addr1": [100.0, 200.0, None, 300.0] * 4,
            "P_emaildomain": ["a.test", "b.test", None, "c.test"] * 4,
            "DeviceType": ["desktop", "mobile", None, "mobile"] * 4,
            "id_31": ["chrome", "safari", None, "firefox"] * 4,
            "isFraud": [0, 1, 0, 1] * 4,
        }
    )
    frame.iloc[:8].to_parquet(processed_dir / "train.parquet", index=False)
    frame.iloc[8:12].to_parquet(processed_dir / "calibration.parquet", index=False)
    frame.iloc[12:].to_parquet(processed_dir / "validation.parquet", index=False)
