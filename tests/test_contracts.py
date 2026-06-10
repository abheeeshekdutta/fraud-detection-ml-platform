from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from fraud_platform.contracts import (
    AlertEvent,
    DecisionEvent,
    ReasonCode,
    TransactionEvent,
)


def test_transaction_event_accepts_production_safe_payload() -> None:
    event = TransactionEvent(
        event_id="evt-1",
        transaction_id=2987000,
        event_time=datetime(2026, 6, 10, 12, tzinfo=UTC),
        amount=68.5,
        product_cd="W",
        card_features={"card1": 1001},
        address_features={"addr1": 100.0},
        email_domain_features={"P_emaildomain": "example.test"},
        identity_features={},
        schema_version="v1",
    )

    assert event.transaction_id == 2987000
    assert event.schema_version == "v1"


def test_transaction_event_rejects_negative_amount() -> None:
    with pytest.raises(ValidationError):
        TransactionEvent(
            event_id="evt-1",
            transaction_id=1,
            event_time=datetime(2026, 6, 10, 12, tzinfo=UTC),
            amount=-1.0,
            product_cd="W",
            schema_version="v1",
        )


def test_transaction_event_rejects_naive_event_time() -> None:
    with pytest.raises(ValidationError):
        TransactionEvent(
            event_id="evt-1",
            transaction_id=1,
            event_time=datetime(2026, 6, 10, 12),
            amount=68.5,
            product_cd="W",
            schema_version="v1",
        )


def test_transaction_event_rejects_string_numeric_amount() -> None:
    with pytest.raises(ValidationError):
        TransactionEvent(
            event_id="evt-1",
            transaction_id=1,
            event_time=datetime(2026, 6, 10, 12, tzinfo=UTC),
            amount="68.5",
            product_cd="W",
            schema_version="v1",
        )


def test_transaction_event_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TransactionEvent(
            event_id="evt-1",
            transaction_id=1,
            event_time=datetime(2026, 6, 10, 12, tzinfo=UTC),
            amount=68.5,
            product_cd="W",
            schema_version="v1",
            unexpected=True,
        )


def test_decision_event_contains_governance_metadata() -> None:
    decision = DecisionEvent(
        event_id="evt-1",
        transaction_id=2987000,
        scored_at=datetime(2026, 6, 10, 12, 0, 1, tzinfo=UTC),
        model_version="fraud-model:1",
        feature_schema_version="v1",
        decision_policy_version="v1",
        fraud_probability=0.82,
        calibrated_probability=0.76,
        conformal_prediction_set=["fraud"],
        uncertainty="low",
        decision="block",
        reason_codes=[ReasonCode(feature="TransactionAmt", direction="increases_risk")],
        latency_ms=42.0,
    )

    assert decision.decision == "block"
    assert decision.reason_codes[0].direction == "increases_risk"


def test_decision_event_rejects_invalid_decision() -> None:
    with pytest.raises(ValidationError):
        DecisionEvent(
            event_id="evt-1",
            transaction_id=2987000,
            scored_at=datetime(2026, 6, 10, 12, 0, 1, tzinfo=UTC),
            model_version="fraud-model:1",
            feature_schema_version="v1",
            decision_policy_version="v1",
            fraud_probability=0.5,
            calibrated_probability=0.5,
            conformal_prediction_set=["legit", "fraud"],
            uncertainty="high",
            decision="escalate",
            latency_ms=42.0,
        )


def test_decision_event_rejects_naive_scored_at() -> None:
    with pytest.raises(ValidationError):
        DecisionEvent(
            event_id="evt-1",
            transaction_id=2987000,
            scored_at=datetime(2026, 6, 10, 12, 0, 1),
            model_version="fraud-model:1",
            feature_schema_version="v1",
            decision_policy_version="v1",
            fraud_probability=0.5,
            calibrated_probability=0.5,
            conformal_prediction_set=["legit", "fraud"],
            uncertainty="high",
            decision="review",
            latency_ms=42.0,
        )


def test_decision_event_rejects_string_numeric_probability() -> None:
    with pytest.raises(ValidationError):
        DecisionEvent(
            event_id="evt-1",
            transaction_id=2987000,
            scored_at=datetime(2026, 6, 10, 12, 0, 1, tzinfo=UTC),
            model_version="fraud-model:1",
            feature_schema_version="v1",
            decision_policy_version="v1",
            fraud_probability="0.5",
            calibrated_probability=0.5,
            conformal_prediction_set=["legit", "fraud"],
            uncertainty="high",
            decision="review",
            latency_ms=42.0,
        )


def test_decision_event_rejects_invalid_conformal_label() -> None:
    with pytest.raises(ValidationError):
        DecisionEvent(
            event_id="evt-1",
            transaction_id=2987000,
            scored_at=datetime(2026, 6, 10, 12, 0, 1, tzinfo=UTC),
            model_version="fraud-model:1",
            feature_schema_version="v1",
            decision_policy_version="v1",
            fraud_probability=0.5,
            calibrated_probability=0.5,
            conformal_prediction_set=["unknown"],
            uncertainty="high",
            decision="review",
            latency_ms=42.0,
        )


def test_decision_event_rejects_duplicate_conformal_labels() -> None:
    with pytest.raises(ValidationError):
        DecisionEvent(
            event_id="evt-1",
            transaction_id=2987000,
            scored_at=datetime(2026, 6, 10, 12, 0, 1, tzinfo=UTC),
            model_version="fraud-model:1",
            feature_schema_version="v1",
            decision_policy_version="v1",
            fraud_probability=0.5,
            calibrated_probability=0.5,
            conformal_prediction_set=["fraud", "fraud"],
            uncertainty="high",
            decision="review",
            latency_ms=42.0,
        )


def test_reason_code_rejects_coerced_feature() -> None:
    with pytest.raises(ValidationError):
        ReasonCode(feature=123, direction="increases_risk")


def test_alert_event_schema() -> None:
    alert = AlertEvent(
        alert_id="alert-1",
        created_at=datetime(2026, 6, 10, 12, tzinfo=UTC),
        severity="warning",
        alert_type="review_rate_shift",
        message="Review rate increased from 0.10 to 0.25",
        metadata={"current_rate": 0.25},
    )

    assert alert.severity == "warning"


def test_alert_event_rejects_naive_created_at() -> None:
    with pytest.raises(ValidationError):
        AlertEvent(
            alert_id="alert-1",
            created_at=datetime(2026, 6, 10, 12),
            severity="warning",
            alert_type="review_rate_shift",
            message="Review rate increased from 0.10 to 0.25",
        )
