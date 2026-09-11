from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from fraud_platform.api import create_app
from fraud_platform.demo import prepare_demo
from fraud_platform.features.ieee import build_transaction_event
from fraud_platform.policy import load_policy
from fraud_platform.repositories import AlertRepository, PredictionRepository
from fraud_platform.scoring import ScoringEngine
from fraud_platform.storage import create_session_factory, create_tables


def test_demo_model_to_api_to_persisted_dashboard_feed(tmp_path) -> None:
    target = prepare_demo(tmp_path / "demo", rows=5)
    frame = pd.read_parquet(target / "replay.parquet")
    factory = create_session_factory(f"sqlite:///{tmp_path / 'test.db'}")
    create_tables(factory)
    engine = ScoringEngine.from_paths(target / "model", load_policy("configs/decision_policy.yaml"))
    app = create_app(
        scoring_engine=engine,
        prediction_repository=PredictionRepository(factory),
        alert_repository=AlertRepository(factory),
    )
    with TestClient(app) as client:
        assert client.get("/predictions").json() == []
        for _, row in frame.iterrows():
            event = build_transaction_event(row, "2026-06-10T12:00:00Z")
            response = client.post("/score", json=event.model_dump(mode="json"))
            assert response.status_code == 200
            assert response.json()["model_version"] == "synthetic-fraud-model:1"
        # An event retry upserts the prediction rather than creating a duplicate.
        assert client.post("/score", json=event.model_dump(mode="json")).status_code == 200
        feed = client.get("/predictions")
        assert feed.status_code == 200
        assert len(feed.json()) == 5
        assert client.get("/alerts").json() == []
        assert "fraud_api_scoring_latency_ms_count" in client.get("/metrics").text
