from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from fraud_platform.data_profile import (
    RawDataPaths,
    build_profile,
    check_raw_data_paths,
    write_profile_outputs,
)

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "profile_ieee_cis.py"


def _load_profile_script():
    spec = importlib.util.spec_from_file_location("profile_ieee_cis", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_profile_summarizes_fraud_and_amount_patterns() -> None:
    transactions = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [10, 20, 30, 40],
            "TransactionAmt": [10.0, 20.0, 100.0, 200.0],
            "ProductCD": ["W", "W", "C", "C"],
            "isFraud": [0, 1, 0, 1],
            "card1": [1001, 1002, None, 1004],
            "P_emaildomain": ["a.test", "b.test", "a.test", None],
        }
    )
    identity = pd.DataFrame(
        {
            "TransactionID": [2, 4],
            "DeviceType": ["mobile", None],
            "id_31": ["safari", "chrome"],
        }
    )

    profile = build_profile(transactions, identity, drift_windows=2)

    assert profile.row_count == 4
    assert profile.fraud_rate == 0.5
    assert profile.amount_summary["p50"] == 60.0
    assert profile.product_fraud_rates.loc["C", "fraud_rate"] == 0.5
    assert profile.identity_join_coverage == 0.5
    assert profile.missingness.loc["card1", "missing_rate"] == 0.25
    assert profile.categorical_cardinality.loc["ProductCD", "unique_values"] == 2
    assert profile.time_window_fraud_rates["fraud_rate"].tolist() == [0.5, 0.5]


def test_build_profile_flags_leakage_and_serving_risks() -> None:
    transactions = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "TransactionDT": [10, 20],
            "TransactionAmt": [10.0, 20.0],
            "ProductCD": ["W", "C"],
            "isFraud": [0, 1],
            "fraud_label_snapshot": [0, 1],
            "TransactionAmt_mean_target": [0.2, 0.8],
        }
    )

    profile = build_profile(transactions, identity=None)

    assert "fraud_label_snapshot" in profile.target_leakage_columns
    assert "TransactionAmt_mean_target" in profile.target_leakage_columns
    assert "isFraud" in profile.serving_incompatible_columns


def test_check_raw_data_paths_reports_missing_ieee_files(tmp_path) -> None:
    paths = RawDataPaths(raw_dir=tmp_path)

    with pytest.raises(FileNotFoundError, match="train_transaction.csv"):
        check_raw_data_paths(paths)


def test_profile_script_exits_cleanly_when_raw_files_are_missing(tmp_path) -> None:
    profile_ieee_cis = _load_profile_script()

    with pytest.raises(SystemExit, match="Missing IEEE-CIS raw data files"):
        profile_ieee_cis.main(["--raw-dir", str(tmp_path)])


def test_write_profile_outputs_creates_tables_charts_and_modeling_guidance(tmp_path) -> None:
    transactions = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "TransactionDT": [10, 20],
            "TransactionAmt": [10.0, 20.0],
            "ProductCD": ["W", "C"],
            "isFraud": [0, 1],
        }
    )
    profile = build_profile(transactions, identity=None)
    reports_dir = tmp_path / "reports"
    docs_path = tmp_path / "docs" / "data-profile.md"

    write_profile_outputs(profile, reports_dir, docs_path)

    assert (reports_dir / "product_fraud_rates.csv").exists()
    assert (reports_dir / "product_fraud_rates.svg").exists()
    report_text = docs_path.read_text(encoding="utf-8")
    assert "CatBoost" in report_text
    assert "LightGBM" in report_text
    assert "Calibration" in report_text
