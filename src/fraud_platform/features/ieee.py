from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import pandas as pd

from fraud_platform.contracts import TransactionEvent


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str]


@dataclass(frozen=True)
class TimeSplits:
    train: pd.DataFrame
    calibration: pd.DataFrame
    validation: pd.DataFrame
    replay: pd.DataFrame


REQUIRED_COLUMNS = {"TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "isFraud"}


def join_transaction_identity(
    transactions: pd.DataFrame,
    identity: pd.DataFrame | None,
) -> pd.DataFrame:
    if identity is None or identity.empty:
        return transactions.copy()
    if identity["TransactionID"].duplicated().any():
        raise ValueError("identity TransactionID values must be unique")
    return transactions.merge(identity, on="TransactionID", how="left", validate="one_to_one")


def validate_training_frame(frame: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        errors.append(f"missing required columns: {missing}")
    if "isFraud" in frame and not set(frame["isFraud"].dropna().unique()).issubset({0, 1}):
        errors.append("isFraud must contain only 0 and 1")
    if "TransactionDT" in frame and not frame["TransactionDT"].is_monotonic_increasing:
        ordered = frame.sort_values("TransactionDT")["TransactionDT"].tolist()
        if ordered != frame["TransactionDT"].tolist():
            errors.append("TransactionDT must be sorted for final split inputs")
    return ValidationResult(valid=not errors, errors=errors)


def time_ordered_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.15,
    validation_fraction: float = 0.15,
) -> TimeSplits:
    if train_fraction + calibration_fraction + validation_fraction >= 1:
        raise ValueError("fractions must leave at least one replay segment")
    ordered = frame.sort_values("TransactionDT").reset_index(drop=True)
    n_rows = len(ordered)
    train_end = max(1, int(n_rows * train_fraction))
    calibration_end = max(train_end + 1, int(n_rows * (train_fraction + calibration_fraction)))
    validation_end = max(
        calibration_end + 1,
        int(n_rows * (train_fraction + calibration_fraction + validation_fraction)),
    )
    validation_end = min(validation_end, n_rows - 1)
    return TimeSplits(
        train=ordered.iloc[:train_end].copy(),
        calibration=ordered.iloc[train_end:calibration_end].copy(),
        validation=ordered.iloc[calibration_end:validation_end].copy(),
        replay=ordered.iloc[validation_end:].copy(),
    )


def _clean_mapping(values: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in values.items():
        if pd.isna(value):
            continue
        cleaned[key] = value.item() if hasattr(value, "item") else value
    return cleaned


def build_transaction_event(row: pd.Series, event_time_base: str) -> TransactionEvent:
    base_time = datetime.fromisoformat(event_time_base.replace("Z", "+00:00"))
    event_time = base_time + timedelta(seconds=int(row["TransactionDT"]))
    return TransactionEvent(
        event_id=str(uuid4()),
        transaction_id=int(row["TransactionID"]),
        event_time=event_time,
        amount=float(row["TransactionAmt"]),
        product_cd=str(row["ProductCD"]),
        card_features=_clean_mapping({"card1": row.get("card1")}),
        address_features=_clean_mapping({"addr1": row.get("addr1")}),
        email_domain_features=_clean_mapping({"P_emaildomain": row.get("P_emaildomain")}),
        identity_features=_clean_mapping(
            {"DeviceType": row.get("DeviceType"), "id_31": row.get("id_31")}
        ),
        schema_version="v1",
    )
