from __future__ import annotations

import argparse
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from confluent_kafka import Consumer, Producer

from fraud_platform.config import Settings
from fraud_platform.contracts import DeadLetterEvent, TransactionEvent
from fraud_platform.policy import load_policy
from fraud_platform.repositories import PredictionRepository
from fraud_platform.scoring import ScoringEngine
from fraud_platform.storage import create_session_factory
from fraud_platform.streaming import deserialize_event, serialize_event


def consume_available_messages(
    consumer,
    producer,
    engine,
    input_topic: str,
    output_topic: str,
    prediction_repository: PredictionRepository | None = None,
    dead_letter_topic: str | None = None,
    max_messages: int | None = None,
) -> int:
    consumer.subscribe([input_topic])
    processed = 0
    while max_messages is None or processed < max_messages:
        message = consumer.poll(1.0)
        if message is None:
            if max_messages is not None:
                break
            continue
        if message.error():
            continue
        try:
            event = deserialize_event(message.value(), TransactionEvent)
            decision = engine.score(event)
            if prediction_repository is not None:
                prediction_repository.save(decision)
            producer.produce(
                output_topic,
                key=str(_transaction_id(decision)),
                value=serialize_event(decision),
            )
            producer.poll(0)
            processed += 1
        except Exception as exc:
            if dead_letter_topic is not None:
                dead_letter = _dead_letter_event(
                    payload=message.value(),
                    source_topic=input_topic,
                    error=exc,
                )
                producer.produce(
                    dead_letter_topic,
                    key=dead_letter.event_id,
                    value=serialize_event(dead_letter),
                )
                producer.poll(0)
        finally:
            consumer.commit(message)
    producer.flush()
    return processed


def run_consumer(
    bootstrap_servers: str,
    input_topic: str,
    output_topic: str,
    group_id: str,
    model_path: str,
    policy_path: str,
    calibrator_path: str | None = None,
    database_url: str | None = None,
    dead_letter_topic: str | None = None,
) -> None:
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
        }
    )
    producer = Producer({"bootstrap.servers": bootstrap_servers})
    engine = ScoringEngine.from_paths(
        model_path,
        load_policy(policy_path),
        calibrator_path=calibrator_path,
    )
    prediction_repository = (
        PredictionRepository(create_session_factory(database_url)) if database_url else None
    )
    try:
        consume_available_messages(
            consumer=consumer,
            producer=producer,
            engine=engine,
            input_topic=input_topic,
            output_topic=output_topic,
            prediction_repository=prediction_repository,
            dead_letter_topic=dead_letter_topic,
        )
    finally:
        consumer.close()


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Score transaction events from Kafka.")
    parser.add_argument("--bootstrap-servers", default=settings.kafka_bootstrap_servers)
    parser.add_argument("--input-topic", default="transaction-events")
    parser.add_argument("--output-topic", default="fraud-decisions")
    parser.add_argument("--group-id", default="fraud-consumer")
    parser.add_argument("--model-path", default=settings.model_bundle_path)
    parser.add_argument("--policy-path", default=settings.decision_policy_path)
    parser.add_argument("--calibrator-path", default=settings.calibrator_path)
    parser.add_argument("--database-url", default=settings.database_url)
    parser.add_argument("--dead-letter-topic", default=settings.dead_letter_events_topic)
    args = parser.parse_args()
    run_consumer(
        args.bootstrap_servers,
        args.input_topic,
        args.output_topic,
        args.group_id,
        args.model_path,
        args.policy_path,
        args.calibrator_path,
        args.database_url,
        args.dead_letter_topic,
    )


def _transaction_id(decision: Any) -> int:
    if isinstance(decision, dict):
        return int(decision["transaction_id"])
    return int(decision.transaction_id)


def _dead_letter_event(payload: bytes, source_topic: str, error: Exception) -> DeadLetterEvent:
    return DeadLetterEvent(
        event_id=str(uuid4()),
        failed_at=datetime.now(UTC),
        source_topic=source_topic,
        error_type=type(error).__name__,
        error_message=str(error)[:500] or type(error).__name__,
        payload=payload.decode("utf-8", errors="replace"),
        schema_version="v1",
    )


if __name__ == "__main__":
    main()
