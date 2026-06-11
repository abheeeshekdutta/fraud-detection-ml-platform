from __future__ import annotations

from datetime import UTC, datetime

from fraud_platform.contracts import AlertEvent, DecisionEvent
from fraud_platform.repositories import AlertRepository, PredictionRepository
from fraud_platform.storage import create_session_factory, create_tables


def test_prediction_repository_round_trips_decision() -> None:
    session_factory = create_session_factory("sqlite+pysqlite:///:memory:")
    create_tables(session_factory)
    decision = DecisionEvent(
        event_id="evt-1",
        transaction_id=1,
        scored_at=datetime(2026, 6, 10, 12, tzinfo=UTC),
        model_version="fraud-model:1",
        feature_schema_version="v1",
        decision_policy_version="v1",
        fraud_probability=0.9,
        calibrated_probability=0.85,
        conformal_prediction_set=["fraud"],
        uncertainty="low",
        decision="block",
        latency_ms=10.0,
    )

    PredictionRepository(session_factory).save(decision)
    rows = PredictionRepository(session_factory).latest(limit=10)

    assert rows[0].event_id == "evt-1"
    assert rows[0].decision == "block"


def test_alert_repository_round_trips_alert() -> None:
    session_factory = create_session_factory("sqlite+pysqlite:///:memory:")
    create_tables(session_factory)
    alert = AlertEvent(
        alert_id="alert-1",
        created_at=datetime(2026, 6, 10, 12, tzinfo=UTC),
        severity="warning",
        alert_type="dead_letter_rate",
        message="Dead letter rate exceeded threshold",
    )

    AlertRepository(session_factory).save(alert)
    rows = AlertRepository(session_factory).latest(limit=10)

    assert rows[0].alert_type == "dead_letter_rate"
