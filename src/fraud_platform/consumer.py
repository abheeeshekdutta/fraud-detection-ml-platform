from __future__ import annotations

import argparse
from typing import Any

from confluent_kafka import Consumer, Producer

from fraud_platform.contracts import TransactionEvent
from fraud_platform.policy import load_policy
from fraud_platform.scoring import ScoringEngine
from fraud_platform.streaming import deserialize_event, serialize_event


def consume_available_messages(
    consumer,
    producer,
    engine,
    input_topic: str,
    output_topic: str,
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
    try:
        consume_available_messages(
            consumer=consumer,
            producer=producer,
            engine=engine,
            input_topic=input_topic,
            output_topic=output_topic,
        )
    finally:
        consumer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score transaction events from Kafka.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--input-topic", default="transaction-events")
    parser.add_argument("--output-topic", default="fraud-decisions")
    parser.add_argument("--group-id", default="fraud-consumer")
    parser.add_argument("--model-path", default="artifacts/model/latest")
    parser.add_argument("--policy-path", default="configs/decision_policy.yaml")
    parser.add_argument("--calibrator-path")
    args = parser.parse_args()
    run_consumer(
        args.bootstrap_servers,
        args.input_topic,
        args.output_topic,
        args.group_id,
        args.model_path,
        args.policy_path,
        args.calibrator_path,
    )


def _transaction_id(decision: Any) -> int:
    if isinstance(decision, dict):
        return int(decision["transaction_id"])
    return int(decision.transaction_id)


if __name__ == "__main__":
    main()
