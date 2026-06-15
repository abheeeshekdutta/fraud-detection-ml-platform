from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from fastapi.testclient import TestClient

from fraud_platform.api import create_app
from fraud_platform.calibration import ProbabilityCalibrator, save_calibrator
from fraud_platform.contracts import AlertEvent, DecisionEvent
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


def test_create_app_loads_configured_calibrator(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "model"
    policy_path = tmp_path / "decision_policy.yaml"
    calibrator_path = tmp_path / "calibrator.pkl"
    train_synthetic_model(model_dir)
    policy_path.write_text(
        "\n".join(
            [
                "version: v1",
                "approve_threshold: 0.2",
                "block_threshold: 0.8",
            ]
        )
    )
    calibrator = ProbabilityCalibrator(method="isotonic").fit(
        np.array([0.0, 1.0]),
        np.array([0, 1]),
    )
    save_calibrator(calibrator, calibrator_path)
    monkeypatch.setenv("MODEL_BUNDLE_PATH", str(model_dir))
    monkeypatch.setenv("DECISION_POLICY_PATH", str(policy_path))
    monkeypatch.setenv("CALIBRATOR_PATH", str(calibrator_path))

    client = TestClient(create_app())

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
    assert body["calibrated_probability"] == calibrator.predict(
        np.array([body["fraud_probability"]])
    )[0]


def test_dashboard_endpoints_return_stored_predictions_and_alerts(tmp_path) -> None:
    model_dir = tmp_path / "model"
    train_synthetic_model(model_dir)
    engine = ScoringEngine.from_paths(
        model_dir,
        DecisionPolicy(
            PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8)
        ),
    )
    stored_decision = DecisionEvent(
        event_id="evt-stored",
        transaction_id=42,
        scored_at=datetime(2026, 6, 10, 12, tzinfo=UTC),
        model_version="fraud-model:1",
        feature_schema_version="v1",
        decision_policy_version="v1",
        fraud_probability=0.9,
        calibrated_probability=0.85,
        conformal_prediction_set=["fraud"],
        uncertainty="low",
        decision="block",
        reason_codes=[{"feature": "TransactionAmt", "direction": "increases_risk"}],
        latency_ms=12.0,
    )
    stored_alert = AlertEvent(
        alert_id="alert-stored",
        created_at=datetime(2026, 6, 10, 12, 5, tzinfo=UTC),
        severity="warning",
        alert_type="decision_rate_shift",
        message="Block rate shifted above threshold",
        metadata={"window": "15m"},
    )
    client = TestClient(
        create_app(
            scoring_engine=engine,
            predictions=[stored_decision],
            alerts=[stored_alert],
        )
    )

    predictions_response = client.get("/predictions")
    alerts_response = client.get("/alerts")

    assert predictions_response.status_code == 200
    assert predictions_response.json()[0]["event_id"] == "evt-stored"
    assert predictions_response.json()[0]["reason_codes"][0]["feature"] == "TransactionAmt"
    assert alerts_response.status_code == 200
    assert alerts_response.json()[0]["alert_id"] == "alert-stored"
    assert alerts_response.json()[0]["metadata"] == {"window": "15m"}


def test_api_allows_dashboard_browser_origin(tmp_path) -> None:
    model_dir = tmp_path / "model"
    train_synthetic_model(model_dir)
    engine = ScoringEngine.from_paths(
        model_dir,
        DecisionPolicy(
            PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8)
        ),
    )
    client = TestClient(create_app(scoring_engine=engine))

    response = client.options(
        "/predictions",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
