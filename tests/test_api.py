from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from fraud_platform.api import create_app
from fraud_platform.policy import DecisionPolicy, PolicyConfig
from fraud_platform.scoring import ScoringEngine
from fraud_platform.training import train_synthetic_model


def test_health_endpoint(tmp_path) -> None:
    model_dir = tmp_path / "model"
    train_synthetic_model(model_dir)
    engine = ScoringEngine.from_paths(
        model_dir,
        DecisionPolicy(
            PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8)
        ),
    )
    client = TestClient(create_app(scoring_engine=engine))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_score_endpoint_returns_decision(tmp_path) -> None:
    model_dir = tmp_path / "model"
    train_synthetic_model(model_dir)
    engine = ScoringEngine.from_paths(
        model_dir,
        DecisionPolicy(
            PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8)
        ),
    )
    client = TestClient(create_app(scoring_engine=engine))
    payload = {
        "event_id": "evt-1",
        "transaction_id": 1,
        "event_time": datetime(2026, 6, 10, 12, tzinfo=UTC).isoformat(),
        "amount": 900.0,
        "product_cd": "C",
        "card_features": {"card1": 1002},
        "address_features": {"addr1": 200.0},
        "email_domain_features": {"P_emaildomain": "b.test"},
        "identity_features": {"DeviceType": "mobile", "id_31": "safari"},
        "schema_version": "v1",
    }

    response = client.post("/score", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["event_id"] == "evt-1"
    assert body["decision"] in {"approve", "review", "block"}


def test_model_info_endpoint(tmp_path) -> None:
    model_dir = tmp_path / "model"
    train_synthetic_model(model_dir)
    engine = ScoringEngine.from_paths(
        model_dir,
        DecisionPolicy(
            PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8)
        ),
    )
    client = TestClient(create_app(scoring_engine=engine))

    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model_version"] == "synthetic-fraud-model:1"
