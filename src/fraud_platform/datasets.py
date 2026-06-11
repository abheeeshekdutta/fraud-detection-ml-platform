from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_platform.features.ieee import (
    join_transaction_identity,
    time_ordered_split,
    validate_training_frame,
)


def prepare_ieee_cis_splits(
    raw_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/processed",
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.15,
    validation_fraction: float = 0.15,
) -> dict[str, Any]:
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    transactions_path = raw_path / "train_transaction.csv"
    identity_path = raw_path / "train_identity.csv"
    if not transactions_path.exists() or not identity_path.exists():
        raise FileNotFoundError(
            "Expected data/raw/train_transaction.csv and data/raw/train_identity.csv. "
            "Download the IEEE-CIS files from Kaggle before preparing splits."
        )

    transactions = pd.read_csv(transactions_path)
    identity = pd.read_csv(identity_path)
    joined = join_transaction_identity(transactions, identity)
    ordered = joined.sort_values("TransactionDT").reset_index(drop=True)
    validation = validate_training_frame(ordered)
    if not validation.valid:
        raise ValueError("; ".join(validation.errors))

    splits = time_ordered_split(
        ordered,
        train_fraction=train_fraction,
        calibration_fraction=calibration_fraction,
        validation_fraction=validation_fraction,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    split_frames = {
        "train": splits.train,
        "calibration": splits.calibration,
        "validation": splits.validation,
        "replay": splits.replay,
    }
    for split_name, frame in split_frames.items():
        frame.to_parquet(output_path / f"{split_name}.parquet", index=False)

    summary = {
        "row_count": int(len(ordered)),
        "fraud_count": int(ordered["isFraud"].sum()),
        "fraud_rate": float(ordered["isFraud"].mean()),
        "identity_join_coverage": float(
            ordered["TransactionID"].isin(identity["TransactionID"]).mean()
        ),
        "splits": {
            split_name: _split_summary(frame) for split_name, frame in split_frames.items()
        },
    }
    (output_path / "split_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "row_count": int(len(frame)),
        "fraud_count": int(frame["isFraud"].sum()),
        "fraud_rate": float(frame["isFraud"].mean()) if len(frame) else 0.0,
        "transaction_dt_min": int(frame["TransactionDT"].min()) if len(frame) else None,
        "transaction_dt_max": int(frame["TransactionDT"].max()) if len(frame) else None,
    }
