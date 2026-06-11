from __future__ import annotations

import pandas as pd

from fraud_platform.features.ieee import (
    build_transaction_event,
    join_transaction_identity,
    time_ordered_split,
    validate_training_frame,
)
from fraud_platform.features.transformers import FraudFeatureTransformer


def test_join_preserves_transactions_with_missing_identity(
    synthetic_transactions: pd.DataFrame,
    synthetic_identity: pd.DataFrame,
) -> None:
    joined = join_transaction_identity(synthetic_transactions, synthetic_identity)

    assert len(joined) == len(synthetic_transactions)
    assert joined.loc[joined["TransactionID"] == 3, "DeviceType"].isna().all()


def test_time_ordered_split_uses_transaction_dt(synthetic_transactions: pd.DataFrame) -> None:
    splits = time_ordered_split(
        synthetic_transactions,
        train_fraction=0.50,
        calibration_fraction=0.17,
        validation_fraction=0.17,
    )

    assert splits.train["TransactionDT"].max() < splits.calibration["TransactionDT"].min()
    assert splits.calibration["TransactionDT"].max() < splits.validation["TransactionDT"].min()
    assert splits.validation["TransactionDT"].max() < splits.replay["TransactionDT"].min()


def test_validate_training_frame_requires_expected_columns(
    synthetic_transactions: pd.DataFrame,
) -> None:
    result = validate_training_frame(synthetic_transactions)

    assert result.valid is True
    assert result.errors == []


def test_transformer_produces_stable_feature_columns(
    synthetic_transactions: pd.DataFrame,
    synthetic_identity: pd.DataFrame,
) -> None:
    joined = join_transaction_identity(synthetic_transactions, synthetic_identity)
    transformer = FraudFeatureTransformer()

    features = transformer.fit_transform(joined)

    assert list(features.columns) == [
        "TransactionAmt",
        "ProductCD",
        "card1",
        "addr1",
        "P_emaildomain",
        "DeviceType",
        "id_31",
    ]
    assert features["ProductCD"].dtype.name == "category"


def test_build_transaction_event_maps_serving_safe_groups(
    synthetic_transactions: pd.DataFrame,
    synthetic_identity: pd.DataFrame,
) -> None:
    joined = join_transaction_identity(synthetic_transactions, synthetic_identity)
    event = build_transaction_event(joined.iloc[0], event_time_base="2026-06-10T12:00:00Z")

    assert event.amount == 20.0
    assert event.card_features == {"card1": 1001}
    assert event.identity_features == {"DeviceType": "desktop", "id_31": "chrome"}
    assert event.schema_version == "v1"
