from __future__ import annotations

import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from fraud_platform.config import Settings
from fraud_platform.contracts import AlertEvent, DecisionEvent, TransactionEvent
from fraud_platform.policy import load_policy
from fraud_platform.repositories import AlertRepository, PredictionRepository
from fraud_platform.scoring import ScoringEngine
from fraud_platform.storage import AlertRecord, PredictionRecord, create_session_factory

REQUEST_COUNT = Counter("fraud_api_requests_total", "Fraud API request count", ["endpoint"])
SCORING_LATENCY = Histogram("fraud_api_scoring_latency_ms", "Scoring latency in milliseconds")


class _InMemoryPredictionRepository:
    def __init__(self, predictions: list[DecisionEvent]) -> None:
        self.predictions = predictions

    def latest(self, limit: int = 100) -> list[DecisionEvent]:
        return self.predictions[:limit]


class _InMemoryAlertRepository:
    def __init__(self, alerts: list[AlertEvent]) -> None:
        self.alerts = alerts

    def latest(self, limit: int = 100) -> list[AlertEvent]:
        return self.alerts[:limit]


def create_app(
    scoring_engine: ScoringEngine | None = None,
    prediction_repository: PredictionRepository | None = None,
    alert_repository: AlertRepository | None = None,
    predictions: list[DecisionEvent] | None = None,
    alerts: list[AlertEvent] | None = None,
) -> FastAPI:
    settings = Settings()
    if scoring_engine is None:
        scoring_engine = ScoringEngine.from_paths(
            model_path=settings.model_bundle_path,
            policy=load_policy(settings.decision_policy_path),
            calibrator_path=settings.calibrator_path,
        )
    if prediction_repository is None:
        prediction_repository = (
            _InMemoryPredictionRepository(predictions)
            if predictions is not None
            else PredictionRepository(create_session_factory(settings.database_url))
        )
    if alert_repository is None:
        alert_repository = (
            _InMemoryAlertRepository(alerts)
            if alerts is not None
            else AlertRepository(create_session_factory(settings.database_url))
        )
    app = FastAPI(title="Fraud Detection API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.state.scoring_engine = scoring_engine
    app.state.prediction_repository = prediction_repository
    app.state.alert_repository = alert_repository

    @app.get("/health")
    def health() -> dict[str, str]:
        REQUEST_COUNT.labels(endpoint="/health").inc()
        return {"status": "ok"}

    @app.get("/model-info")
    def model_info() -> dict[str, str]:
        REQUEST_COUNT.labels(endpoint="/model-info").inc()
        bundle = app.state.scoring_engine.bundle
        return bundle.metadata.model_dump()

    @app.post("/score", response_model=DecisionEvent)
    def score(event: TransactionEvent) -> DecisionEvent:
        REQUEST_COUNT.labels(endpoint="/score").inc()
        decision = app.state.scoring_engine.score(event)
        SCORING_LATENCY.observe(decision.latency_ms)
        return decision

    @app.get("/predictions", response_model=list[DecisionEvent])
    def predictions(limit: int = 100) -> list[DecisionEvent]:
        REQUEST_COUNT.labels(endpoint="/predictions").inc()
        rows = app.state.prediction_repository.latest(limit=max(1, min(limit, 500)))
        return [_decision_from_record(row) for row in rows]

    @app.get("/alerts")
    def alerts(limit: int = 100) -> list[AlertEvent | dict]:
        REQUEST_COUNT.labels(endpoint="/alerts").inc()
        rows = app.state.alert_repository.latest(limit=max(1, min(limit, 500)))
        return [_alert_from_record(row) for row in rows]

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def _decision_from_record(record) -> DecisionEvent:
    if isinstance(record, DecisionEvent):
        return record
    if not isinstance(record, PredictionRecord):
        return DecisionEvent.model_validate(record)
    return DecisionEvent(
        event_id=record.event_id,
        transaction_id=record.transaction_id,
        scored_at=record.scored_at,
        model_version=record.model_version,
        feature_schema_version=record.feature_schema_version,
        decision_policy_version=record.decision_policy_version,
        fraud_probability=record.fraud_probability,
        calibrated_probability=record.calibrated_probability,
        conformal_prediction_set=record.conformal_prediction_set,
        uncertainty=record.uncertainty,
        decision=record.decision,
        reason_codes=record.reason_codes,
        latency_ms=record.latency_ms,
    )


def _alert_from_record(record) -> AlertEvent | dict:
    if not isinstance(record, AlertRecord):
        return record
    return {
        "alert_id": record.alert_id,
        "created_at": record.created_at,
        "severity": record.severity,
        "alert_type": record.alert_type,
        "message": record.message,
        "metadata": record.metadata_json,
    }


def main() -> None:
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
