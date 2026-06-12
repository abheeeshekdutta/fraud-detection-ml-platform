from __future__ import annotations

import json

import pandas as pd

from fraud_platform.calibration import ProbabilityCalibrator, save_calibrator
from fraud_platform.threshold_analysis import run_threshold_analysis
from fraud_platform.training import train_ieee_baseline_model


def test_run_threshold_analysis_writes_sorted_report(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    model_dir = tmp_path / "model"
    report_path = tmp_path / "thresholds.json"
    processed_dir.mkdir()
    frame = pd.DataFrame(
        {
            "TransactionID": range(1, 13),
            "TransactionDT": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
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
            ],
            "ProductCD": ["W", "C", "W", "R", "H", "C", "W", "C", "W", "C", "H", "R"],
            "card1": [1001, 1002, 1001, 1003, 1004, 1002, 1001, 1003, 1004, 1002, 1001, 1003],
            "addr1": [
                100.0,
                200.0,
                None,
                300.0,
                300.0,
                200.0,
                100.0,
                300.0,
                100.0,
                200.0,
                None,
                300.0,
            ],
            "P_emaildomain": [
                "a.test",
                "b.test",
                None,
                "c.test",
                "a.test",
                "b.test",
                "a.test",
                "b.test",
                None,
                "c.test",
                "a.test",
                "b.test",
            ],
            "DeviceType": [
                "desktop",
                "mobile",
                None,
                "mobile",
                "desktop",
                "mobile",
                "desktop",
                "mobile",
                None,
                "mobile",
                "desktop",
                "mobile",
            ],
            "id_31": [
                "chrome",
                "safari",
                None,
                "firefox",
                "chrome",
                "safari",
                "chrome",
                "safari",
                None,
                "firefox",
                "chrome",
                "safari",
            ],
            "isFraud": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    frame.iloc[:8].to_parquet(processed_dir / "train.parquet", index=False)
    frame.iloc[8:].to_parquet(processed_dir / "validation.parquet", index=False)
    train_ieee_baseline_model(processed_dir=processed_dir, output_dir=model_dir)
    calibrator_path = tmp_path / "calibrator.pkl"
    calibrator = ProbabilityCalibrator(method="isotonic").fit(
        pd.Series([0.1, 0.2, 0.8, 0.9]).to_numpy(),
        pd.Series([0, 0, 1, 1]).to_numpy(),
    )
    save_calibrator(calibrator, calibrator_path)

    report = run_threshold_analysis(
        processed_dir=processed_dir,
        model_dir=model_dir,
        output_path=report_path,
        calibrator_path=calibrator_path,
        approve_thresholds=[0.1, 0.3],
        block_thresholds=[0.6, 0.8],
        fraud_loss=100.0,
        review_cost=5.0,
        false_block_cost=25.0,
        top_k=3,
        max_false_block_rate=0.50,
        max_review_rate=0.75,
        min_block_precision=0.0,
    )

    saved = json.loads(report_path.read_text())
    assert saved == report
    assert saved["model_version"] == "ieee-logistic-baseline:1"
    assert saved["score_type"] == "calibrated"
    assert saved["calibrator_path"] == str(calibrator_path)
    assert saved["constraints"] == {
        "max_false_block_rate": 0.5,
        "max_review_rate": 0.75,
        "min_block_precision": 0.0,
    }
    assert saved["validation_rows"] == 4
    assert saved["best_thresholds"] == saved["threshold_reports"][0]
    assert saved["best_constrained_thresholds"] is not None
    assert saved["best_constrained_thresholds"]["false_block_rate"] <= 0.5
    assert saved["best_constrained_thresholds"]["review_rate"] <= 0.75
    assert len(saved["threshold_reports"]) == 3
    assert (
        saved["threshold_reports"][0]["expected_utility"]
        >= saved["threshold_reports"][1]["expected_utility"]
    )
