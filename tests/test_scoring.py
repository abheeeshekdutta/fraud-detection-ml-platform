from __future__ import annotations

from datetime import UTC, datetime

from fraud_platform.contracts import TransactionEvent
from fraud_platform.policy import DecisionPolicy, PolicyConfig
from fraud_platform.scoring import ScoringEngine
from fraud_platform.training import train_synthetic_model


def test_scoring_engine_returns_decision_event(tmp_path) -> None:
    model_dir = tmp_path / "model"
    train_synthetic_model(model_dir)
    engine = ScoringEngine.from_paths(
        model_path=model_dir,
        policy=DecisionPolicy(
            PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8)
        ),
    )
    event = TransactionEvent(
        event_id="evt-1",
        transaction_id=1,
        event_time=datetime(2026, 6, 10, 12, tzinfo=UTC),
        amount=900.0,
        product_cd="C",
        card_features={"card1": 1002},
        address_features={"addr1": 200.0},
        email_domain_features={"P_emaildomain": "b.test"},
        identity_features={"DeviceType": "mobile", "id_31": "safari"},
        schema_version="v1",
    )

    decision = engine.score(event)

    assert decision.event_id == "evt-1"
    assert decision.model_version == "synthetic-fraud-model:1"
    assert decision.feature_schema_version == "v1"
    assert decision.decision in {"approve", "review", "block"}
    assert decision.latency_ms >= 0
