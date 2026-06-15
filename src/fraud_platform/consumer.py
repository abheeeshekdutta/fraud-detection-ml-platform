from __future__ import annotations

import argparse
from typing import Any

from confluent_kafka import Consumer, Producer

from fraud_platform.config import Settings
from fraud_platform.contracts import TransactionEvent
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
        consumer.commit(message)
        processed += 1
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
    )


def _transaction_id(decision: Any) -> int:
    if isinstance(decision, dict):
        return int(decision["transaction_id"])
    return int(decision.transaction_id)


if __name__ == "__main__":
    main()
