from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd
from confluent_kafka import Producer

from fraud_platform.config import Settings
from fraud_platform.contracts import AlertEvent
from fraud_platform.repositories import AlertRepository, PredictionRepository
from fraud_platform.storage import create_session_factory
from fraud_platform.streaming import serialize_event


def missingness_rate(frame: pd.DataFrame) -> dict[str, float]:
    return {column: float(frame[column].isna().mean()) for column in frame.columns}


def conformal_coverage(frame: pd.DataFrame) -> float:
    covered: list[bool] = []
    for _, row in frame.iterrows():
        label = "fraud" if int(row["is_fraud"]) == 1 else "legit"
        covered.append(label in row["conformal_prediction_set"])
    return float(sum(covered) / len(covered)) if covered else 0.0


def decision_rate_shift_alert(
    reference_rates: dict[str, float],
    current_rates: dict[str, float],
    threshold_multiplier: float,
) -> AlertEvent | None:
    reference_review = reference_rates.get("review", 0.0)
    current_review = current_rates.get("review", 0.0)
    if reference_review == 0:
        return None
    if current_review >= reference_review * threshold_multiplier:
        return AlertEvent(
            alert_id=str(uuid4()),
            created_at=datetime.now(UTC),
            severity="warning",
            alert_type="decision_rate_shift",
            message=f"Review rate increased from {reference_review:.3f} to {current_review:.3f}",
            metadata={
                "reference_review_rate": reference_review,
                "current_review_rate": current_review,
            },
        )
    return None


def run_monitoring_check(
    prediction_repository,
    alert_repository,
    reference_review_rate: float,
    threshold_multiplier: float,
    limit: int,
    producer=None,
    alert_topic: str | None = None,
) -> AlertEvent | None:
    predictions = prediction_repository.latest(limit=limit)
    if not predictions:
        return None

    decision_counts: dict[str, int] = {}
    for prediction in predictions:
        decision_counts[prediction.decision] = decision_counts.get(prediction.decision, 0) + 1
    total = len(predictions)
    current_rates = {
        decision: count / total for decision, count in decision_counts.items()
    }
    alert = decision_rate_shift_alert(
        reference_rates={"review": reference_review_rate},
        current_rates=current_rates,
        threshold_multiplier=threshold_multiplier,
    )
    if alert is not None:
        alert_repository.save(alert)
        if producer is not None and alert_topic is not None:
            producer.produce(alert_topic, key=alert.alert_id, value=serialize_event(alert))
            producer.poll(0)
            producer.flush()
    return alert


def run_monitoring_loop(
    database_url: str,
    reference_review_rate: float,
    threshold_multiplier: float,
    limit: int,
    interval_seconds: float,
    bootstrap_servers: str | None = None,
    alert_topic: str | None = None,
    once: bool = False,
    sleep=time.sleep,
) -> None:
    session_factory = create_session_factory(database_url)
    prediction_repository = PredictionRepository(session_factory)
    alert_repository = AlertRepository(session_factory)
    producer = (
        Producer({"bootstrap.servers": bootstrap_servers})
        if bootstrap_servers is not None and alert_topic is not None
        else None
    )

    while True:
        run_monitoring_check(
            prediction_repository=prediction_repository,
            alert_repository=alert_repository,
            reference_review_rate=reference_review_rate,
            threshold_multiplier=threshold_multiplier,
            limit=limit,
            producer=producer,
            alert_topic=alert_topic,
        )
        if once:
            return
        sleep(interval_seconds)


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Run scheduled fraud platform monitoring checks.")
    parser.add_argument("--database-url", default=settings.database_url)
    parser.add_argument(
        "--reference-review-rate",
        type=float,
        default=settings.monitoring_reference_review_rate,
    )
    parser.add_argument(
        "--threshold-multiplier",
        type=float,
        default=settings.monitoring_review_rate_multiplier,
    )
    parser.add_argument("--limit", type=int, default=settings.monitoring_prediction_limit)
    parser.add_argument("--bootstrap-servers", default=settings.kafka_bootstrap_servers)
    parser.add_argument("--alert-topic", default=settings.model_alerts_topic)
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=settings.monitoring_interval_seconds,
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run_monitoring_loop(
        database_url=args.database_url,
        reference_review_rate=args.reference_review_rate,
        threshold_multiplier=args.threshold_multiplier,
        limit=args.limit,
        interval_seconds=args.interval_seconds,
        bootstrap_servers=args.bootstrap_servers,
        alert_topic=args.alert_topic,
        once=args.once,
    )
