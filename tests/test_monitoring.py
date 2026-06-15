from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from fraud_platform.contracts import DecisionEvent
from fraud_platform.monitoring import (
    conformal_coverage,
    decision_rate_shift_alert,
    missingness_rate,
    run_monitoring_check,
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


class FakePredictionRepository:
    def __init__(self, decisions: list[DecisionEvent]) -> None:
        self.decisions = decisions

    def latest(self, limit: int = 100) -> list[DecisionEvent]:
        return self.decisions[:limit]


class FakeAlertRepository:
    def __init__(self) -> None:
        self.saved = []

    def save(self, alert) -> None:
        self.saved.append(alert)


def test_run_monitoring_check_saves_decision_rate_shift_alert() -> None:
    decisions = [
        _decision("evt-1", "review"),
        _decision("evt-2", "review"),
        _decision("evt-3", "approve"),
        _decision("evt-4", "block"),
    ]
    alert_repository = FakeAlertRepository()

    alert = run_monitoring_check(
        prediction_repository=FakePredictionRepository(decisions),
        alert_repository=alert_repository,
        reference_review_rate=0.10,
        threshold_multiplier=2.0,
        limit=100,
    )

    assert alert is not None
    assert alert.alert_type == "decision_rate_shift"
    assert alert.metadata["current_review_rate"] == 0.5
    assert alert_repository.saved == [alert]


def test_run_monitoring_check_stays_quiet_without_recent_predictions() -> None:
    alert_repository = FakeAlertRepository()

    alert = run_monitoring_check(
        prediction_repository=FakePredictionRepository([]),
        alert_repository=alert_repository,
        reference_review_rate=0.10,
        threshold_multiplier=2.0,
        limit=100,
    )

    assert alert is None
    assert alert_repository.saved == []


def _decision(event_id: str, decision: str) -> DecisionEvent:
    return DecisionEvent(
        event_id=event_id,
        transaction_id=int(event_id.rsplit("-", maxsplit=1)[1]),
        scored_at=datetime(2026, 6, 10, 12, tzinfo=UTC),
        model_version="fraud-model:1",
        feature_schema_version="v1",
        decision_policy_version="v1",
        fraud_probability=0.5,
        calibrated_probability=0.5,
        conformal_prediction_set=["legit", "fraud"],
        uncertainty="medium",
        decision=decision,
        latency_ms=10.0,
    )
