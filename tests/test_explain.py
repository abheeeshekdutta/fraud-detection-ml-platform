from __future__ import annotations

import json

import pandas as pd

from fraud_platform.explain import fallback_reason_codes, fit_explanation_artifact
from fraud_platform.training import train_ieee_baseline_model


def test_fallback_reason_codes_are_stable_and_analyst_readable() -> None:
    rows = pd.DataFrame(
        {
            "TransactionAmt": [900.0],
            "ProductCD": ["C"],
            "card1": [1002],
        }
    )

    reason_codes = fallback_reason_codes(rows, max_reasons=2)

    assert reason_codes[0].feature == "TransactionAmt"
    assert reason_codes[0].direction == "increases_risk"


def test_fit_explanation_artifact_writes_global_shap_summary(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "explain"
    _write_tiny_processed_splits(processed_dir)
    train_ieee_baseline_model(processed_dir=processed_dir, output_dir=model_dir)

    summary = fit_explanation_artifact(
        processed_dir=processed_dir,
        model_dir=model_dir,
        output_dir=output_dir,
        max_background_rows=4,
        max_explain_rows=4,
    )

    saved_summary = json.loads((output_dir / "global_shap_summary.json").read_text())
    assert saved_summary == summary
    assert summary["model_version"] == "ieee-logistic-baseline:1"
    assert summary["method"] == "shap_kernel"
    assert summary["explained_rows"] == 4
    assert summary["top_features"]
    assert {"feature", "mean_abs_shap"}.issubset(summary["top_features"][0])


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
