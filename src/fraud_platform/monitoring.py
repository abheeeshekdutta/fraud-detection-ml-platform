from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd

from fraud_platform.contracts import AlertEvent


def missingness_rate(frame: pd.DataFrame) -> dict[str, float]:
    return {column: float(frame[column].isna().mean()) for column in frame.columns}


def conformal_coverage(frame: pd.DataFrame) -> float:
    covered: list[bool] = []
    for _, row in frame.iterrows():
        label = "fraud" if int(row["is_fraud"]) == 1 else "legit"
        covered.append(label in row["conformal_prediction_set"])
    return float(sum(covered) / len(covered)) if covered else 0.0


def decision_rate_shift_alert(
    reference_rates: dict[str, float],
    current_rates: dict[str, float],
    threshold_multiplier: float,
) -> AlertEvent | None:
    reference_review = reference_rates.get("review", 0.0)
    current_review = current_rates.get("review", 0.0)
    if reference_review == 0:
        return None
    if current_review >= reference_review * threshold_multiplier:
        return AlertEvent(
            alert_id=str(uuid4()),
            created_at=datetime.now(UTC),
            severity="warning",
            alert_type="decision_rate_shift",
            message=f"Review rate increased from {reference_review:.3f} to {current_review:.3f}",
            metadata={
                "reference_review_rate": reference_review,
                "current_review_rate": current_review,
            },
        )
    return None


def main() -> None:
    print("Monitoring worker entrypoint will run scheduled checks in the compose stack.")
