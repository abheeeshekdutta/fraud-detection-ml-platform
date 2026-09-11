"""Live Kafka contract test; enable with RUN_KAFKA_INTEGRATION=1."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from time import monotonic
from uuid import uuid4

import pytest
from confluent_kafka import Consumer, Producer
from confluent_kafka.admin import AdminClient, NewTopic

from fraud_platform.consumer import consume_available_messages
from fraud_platform.contracts import DeadLetterEvent, DecisionEvent, TransactionEvent
from fraud_platform.policy import load_policy
from fraud_platform.scoring import ScoringEngine
from fraud_platform.streaming import deserialize_event, publish_confirmed
from fraud_platform.training import train_synthetic_model

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_KAFKA_INTEGRATION") != "1",
        reason="Set RUN_KAFKA_INTEGRATION=1 with a running Kafka broker",
    ),
]


def test_kafka_scoring_delivery_and_dead_letter(tmp_path) -> None:
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    namespace = f"fraud-test-{uuid4().hex}"
    topics = [f"{namespace}-{suffix}" for suffix in ("in", "out", "dead")]
    admin = AdminClient({"bootstrap.servers": bootstrap})
    for future in admin.create_topics([NewTopic(t, 1, 1) for t in topics]).values():
        future.result(timeout=30)
    config = {
        "bootstrap.servers": bootstrap,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    }
    consumer = Consumer({**config, "group.id": namespace})
    reader = Consumer({**config, "group.id": f"{namespace}-reader"})
    producer = Producer({"bootstrap.servers": bootstrap})
    try:
        train_synthetic_model(tmp_path / "model")
        engine = ScoringEngine.from_paths(
            tmp_path / "model", load_policy("configs/decision_policy.yaml")
        )
        event = TransactionEvent(
            event_id=namespace,
            transaction_id=42,
            event_time=datetime.now(UTC),
            amount=20.0,
            transaction_dt=60.0,
            product_cd="W",
            schema_version="v1",
        )
        producer.produce(topics[0], key="invalid", value=b"not-json")
        publish_confirmed(producer, topics[0], "42", event)
        deadline = monotonic() + 45
        processed = 0
        while processed == 0 and monotonic() < deadline:
            processed += consume_available_messages(
                consumer,
                producer,
                engine,
                topics[0],
                topics[1],
                dead_letter_topic=topics[2],
                max_messages=1,
            )
        assert processed == 1
        reader.subscribe(topics[1:])
        received = {}
        deadline = monotonic() + 30
        while len(received) < 2 and monotonic() < deadline:
            message = reader.poll(1)
            if message is not None:
                assert not message.error()
                received[message.topic()] = message.value()
        decision = deserialize_event(received[topics[1]], DecisionEvent)
        dead_letter = deserialize_event(received[topics[2]], DeadLetterEvent)
        assert decision.event_id == namespace
        assert dead_letter.payload == "not-json"
        assert consumer.committed(consumer.assignment(), timeout=10)[0].offset == 2
    finally:
        consumer.close()
        reader.close()
        for future in admin.delete_topics(topics).values():
            future.result(timeout=30)
