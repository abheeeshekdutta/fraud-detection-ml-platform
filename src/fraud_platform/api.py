from __future__ import annotations

import uvicorn
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from fraud_platform.config import Settings
from fraud_platform.contracts import DecisionEvent, TransactionEvent
from fraud_platform.policy import load_policy
from fraud_platform.scoring import ScoringEngine

REQUEST_COUNT = Counter("fraud_api_requests_total", "Fraud API request count", ["endpoint"])
SCORING_LATENCY = Histogram("fraud_api_scoring_latency_ms", "Scoring latency in milliseconds")


def create_app(scoring_engine: ScoringEngine | None = None) -> FastAPI:
    settings = Settings()
    if scoring_engine is None:
        scoring_engine = ScoringEngine.from_paths(
            model_path=settings.model_bundle_path,
            policy=load_policy(settings.decision_policy_path),
        )
    app = FastAPI(title="Fraud Detection API", version="0.1.0")
    app.state.scoring_engine = scoring_engine

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

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def main() -> None:
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
