from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from fraud_platform.calibration import ProbabilityCalibrator, save_calibrator
from fraud_platform.conformal import SplitConformalClassifier, save_conformal
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

    assert (
        decision.calibrated_probability
        == calibrator.predict(np.array([decision.fraud_probability]))[0]
    )


def test_scoring_engine_uses_loaded_conformal_artifact(tmp_path) -> None:
    model_dir = tmp_path / "model"
    conformal_path = tmp_path / "conformal.pkl"
    train_synthetic_model(model_dir)
    conformal = SplitConformalClassifier(alpha=0.25).fit(
        np.array([0.05, 0.95, 0.50, 0.60]),
        np.array([0, 1, 0, 1]),
    )
    save_conformal(conformal, conformal_path)
    engine = ScoringEngine.from_paths(
        model_path=model_dir,
        policy=DecisionPolicy(
            PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8)
        ),
        conformal_path=conformal_path,
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

    assert (
        decision.conformal_prediction_set
        == conformal.predict_sets(np.array([decision.calibrated_probability]))[0]
    )


def test_replayed_event_preserves_offline_model_probability(tmp_path) -> None:
    import pandas as pd

    from fraud_platform.features.ieee import build_transaction_event

    model_dir = tmp_path / "model"
    train_synthetic_model(model_dir)
    engine = ScoringEngine.from_paths(
        model_dir,
        DecisionPolicy(PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8)),
    )
    frame = pd.DataFrame(
        [
            {
                "TransactionID": 42,
                "TransactionDT": 123456.0,
                "TransactionAmt": 75.0,
                "ProductCD": "C",
                "card1": 1002,
                "addr1": 200.0,
                "P_emaildomain": "b.test",
                "DeviceType": "mobile",
                "id_31": "safari",
            }
        ]
    )
    event = build_transaction_event(frame.iloc[0], "2026-06-10T12:00:00Z")
    assert event.transaction_dt == 123456.0
    assert np.isclose(
        engine.score(event).fraud_probability, engine.bundle.predict_raw_probability(frame)[0]
    )

    # Arbitrary enrichment keys cannot replace the contract's canonical fields.
    event.card_features["TransactionAmt"] = 999999.0
    event.identity_features["TransactionDT"] = 0.0
    assert np.isclose(
        engine.score(event).fraud_probability, engine.bundle.predict_raw_probability(frame)[0]
    )


def test_conformal_keeps_raw_score_scale_with_probability_calibration(tmp_path) -> None:
    class Calibrator:
        def predict(self, values):
            return np.array([0.5])

    class Conformal:
        def predict_sets(self, values):
            self.received = values[0]
            return [["legit", "fraud"]]

    model_dir = tmp_path / "model"
    train_synthetic_model(model_dir)
    engine = ScoringEngine.from_paths(
        model_dir,
        DecisionPolicy(PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8)),
    )
    engine.calibrator = Calibrator()
    engine.conformal = Conformal()
    event = TransactionEvent(
        event_id="evt",
        transaction_id=1,
        event_time=datetime.now(UTC),
        amount=20.0,
        product_cd="W",
        schema_version="v1",
    )
    decision = engine.score(event)
    assert decision.calibrated_probability == 0.5
    assert engine.conformal.received == decision.fraud_probability
    assert decision.decision == "review"
