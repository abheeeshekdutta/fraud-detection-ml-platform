from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from fraud_platform.calibration import ProbabilityCalibrator, save_calibrator
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


def test_scoring_engine_uses_loaded_calibrator(tmp_path) -> None:
    model_dir = tmp_path / "model"
    calibrator_path = tmp_path / "calibrator.pkl"
    train_synthetic_model(model_dir)
    calibrator = ProbabilityCalibrator(method="isotonic").fit(
        np.array([0.0, 1.0]),
        np.array([0, 1]),
    )
    save_calibrator(calibrator, calibrator_path)
    engine = ScoringEngine.from_paths(
        model_path=model_dir,
        policy=DecisionPolicy(
            PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8)
        ),
        calibrator_path=calibrator_path,
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

    assert decision.calibrated_probability == calibrator.predict(
        np.array([decision.fraud_probability])
    )[0]
