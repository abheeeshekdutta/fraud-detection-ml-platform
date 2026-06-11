from __future__ import annotations

import pandas as pd

from fraud_platform.monitoring import (
    conformal_coverage,
    decision_rate_shift_alert,
    missingness_rate,
)


def test_missingness_rate_by_column() -> None:
    frame = pd.DataFrame({"identity": [None, "mobile", None], "amount": [1.0, 2.0, 3.0]})

    rates = missingness_rate(frame)

    assert rates["identity"] == 2 / 3
    assert rates["amount"] == 0.0


def test_conformal_coverage_counts_true_label_inside_set() -> None:
    frame = pd.DataFrame(
        {
            "is_fraud": [0, 1, 1],
            "conformal_prediction_set": [["legit"], ["fraud"], ["legit", "fraud"]],
        }
    )

    coverage = conformal_coverage(frame)

    assert coverage == 1.0


def test_decision_rate_shift_alert_when_review_rate_doubles() -> None:
    alert = decision_rate_shift_alert(
        reference_rates={"review": 0.10},
        current_rates={"review": 0.25},
        threshold_multiplier=2.0,
    )

    assert alert is not None
    assert alert.alert_type == "decision_rate_shift"
    assert alert.severity == "warning"
    assert alert.metadata == {
        "reference_review_rate": 0.10,
        "current_review_rate": 0.25,
    }


def test_decision_rate_shift_alert_stays_quiet_below_threshold() -> None:
    alert = decision_rate_shift_alert(
        reference_rates={"review": 0.10},
        current_rates={"review": 0.15},
        threshold_multiplier=2.0,
    )

    assert alert is None
