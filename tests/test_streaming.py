from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from fraud_platform.consumer import consume_available_messages, run_consumer
from fraud_platform.contracts import DecisionEvent, FraudLabelEvent, TransactionEvent
from fraud_platform.replay import replay_frame
from fraud_platform.streaming import deserialize_event, serialize_event


def test_serialize_deserialize_transaction_event() -> None:
    event = TransactionEvent(
        event_id="evt-1",
        transaction_id=1,
        event_time=datetime(2026, 6, 10, 12, tzinfo=UTC),
        amount=20.0,
        product_cd="W",
        schema_version="v1",
    )

    payload = serialize_event(event)
    restored = deserialize_event(payload, TransactionEvent)

    assert restored == event


def test_serialize_deserialize_fraud_label_event() -> None:
    event = FraudLabelEvent(
        event_id="label-1",
        transaction_id=1,
        labeled_at=datetime(2026, 6, 10, 12, 30, tzinfo=UTC),
        is_fraud=True,
        label_source="ieee_replay",
        schema_version="v1",
    )

    payload = serialize_event(event)
    restored = deserialize_event(payload, FraudLabelEvent)

    assert restored == event


class FakeProducer:
    def __init__(self) -> None:
        self.produced: list[tuple[str, str | None, bytes]] = []
        self.flush_called = False

    def produce(self, topic: str, key: str | None = None, value: bytes | None = None) -> None:
        assert value is not None
        self.produced.append((topic, key, value))

    def poll(self, timeout: float) -> None:
        assert timeout == 0

    def flush(self) -> None:
        self.flush_called = True


def test_replay_frame_publishes_transactions_in_time_order() -> None:
    frame = pd.DataFrame(
        {
            "TransactionID": [2, 1],
            "TransactionDT": [20, 10],
            "TransactionAmt": [200.0, 20.0],
            "ProductCD": ["C", "W"],
        }
    )
    producer = FakeProducer()

    count = replay_frame(
        frame,
        producer=producer,
        topic="transaction-events",
        speed_multiplier=1000.0,
        sleep=lambda _: None,
    )

    assert count == 2
    assert producer.flush_called is True
    assert [key for _, key, _ in producer.produced] == ["1", "2"]
    first_event = deserialize_event(producer.produced[0][2], TransactionEvent)
    assert first_event.transaction_id == 1


def test_replay_frame_publishes_delayed_labels_when_topic_is_configured() -> None:
    frame = pd.DataFrame(
        {
            "TransactionID": [1],
            "TransactionDT": [10],
            "TransactionAmt": [20.0],
            "ProductCD": ["W"],
            "isFraud": [1],
        }
    )
    producer = FakeProducer()

    count = replay_frame(
        frame,
        producer=producer,
        topic="transaction-events",
        label_topic="fraud-labels",
        label_delay_seconds=30,
        speed_multiplier=1000.0,
        sleep=lambda _: None,
    )

    assert count == 1
    assert [topic for topic, _, _ in producer.produced] == [
        "transaction-events",
        "fraud-labels",
    ]
    label = deserialize_event(producer.produced[1][2], FraudLabelEvent)
    assert label.transaction_id == 1
    assert label.is_fraud is True
    assert label.label_source == "ieee_replay"


class FakeMessage:
    def __init__(self, value: bytes, error: object | None = None) -> None:
        self._value = value
        self._error = error

    def value(self) -> bytes:
        return self._value

    def error(self) -> object | None:
        return self._error


class FakeConsumer:
    def __init__(self, messages: list[FakeMessage]) -> None:
        self.messages = messages
        self.subscribed_topics: list[str] = []
        self.committed = 0

    def subscribe(self, topics: list[str]) -> None:
        self.subscribed_topics = topics

    def poll(self, timeout: float) -> FakeMessage | None:
        assert timeout == 1.0
        if self.messages:
            return self.messages.pop(0)
        return None

    def commit(self, message: FakeMessage) -> None:
        self.committed += 1


class FakeEngine:
    def score(self, event: TransactionEvent) -> DecisionEvent:
        return DecisionEvent(
            event_id=event.event_id,
            transaction_id=event.transaction_id,
            scored_at=datetime(2026, 6, 10, 12, tzinfo=UTC),
            model_version="test-model:1",
            feature_schema_version="v1",
            decision_policy_version="v1",
            fraud_probability=0.1,
            calibrated_probability=0.1,
            conformal_prediction_set=["legit"],
            uncertainty="low",
            decision="approve",
            latency_ms=1.0,
        )


class FakePredictionRepository:
    def __init__(self) -> None:
        self.saved: list[DecisionEvent] = []

    def save(self, decision: DecisionEvent) -> None:
        self.saved.append(decision)


def test_consume_available_messages_scores_and_publishes_decisions() -> None:
    event = TransactionEvent(
        event_id="evt-1",
        transaction_id=1,
        event_time=datetime(2026, 6, 10, 12, tzinfo=UTC),
        amount=20.0,
        product_cd="W",
        schema_version="v1",
    )
    consumer = FakeConsumer([FakeMessage(serialize_event(event))])
    producer = FakeProducer()

    processed = consume_available_messages(
        consumer=consumer,
        producer=producer,
        engine=FakeEngine(),
        input_topic="transaction-events",
        output_topic="fraud-decisions",
        max_messages=1,
    )

    assert processed == 1
    assert consumer.subscribed_topics == ["transaction-events"]
    assert consumer.committed == 1
    assert producer.produced[0][0] == "fraud-decisions"
    assert producer.produced[0][1] == "1"


def test_consume_available_messages_persists_decisions_when_repository_is_configured() -> None:
    event = TransactionEvent(
        event_id="evt-1",
        transaction_id=1,
        event_time=datetime(2026, 6, 10, 12, tzinfo=UTC),
        amount=20.0,
        product_cd="W",
        schema_version="v1",
    )
    consumer = FakeConsumer([FakeMessage(serialize_event(event))])
    producer = FakeProducer()
    repository = FakePredictionRepository()

    processed = consume_available_messages(
        consumer=consumer,
        producer=producer,
        engine=FakeEngine(),
        input_topic="transaction-events",
        output_topic="fraud-decisions",
        prediction_repository=repository,
        max_messages=1,
    )

    assert processed == 1
    assert repository.saved[0].event_id == "evt-1"
    assert repository.saved[0].decision == "approve"


def test_run_consumer_passes_calibrator_path(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class EmptyConsumer(FakeConsumer):
        def close(self) -> None:
            captured["closed"] = True

    def fake_consumer(config: dict[str, str]) -> EmptyConsumer:
        captured["consumer_config"] = config
        return EmptyConsumer([])

    def fake_producer(config: dict[str, str]) -> FakeProducer:
        captured["producer_config"] = config
        return FakeProducer()

    def fake_load_policy(path: str) -> object:
        captured["policy_path"] = path
        return object()

    def fake_from_paths(
        model_path: str,
        policy: object,
        calibrator_path: str | None = None,
    ) -> FakeEngine:
        captured["model_path"] = model_path
        captured["policy"] = policy
        captured["calibrator_path"] = calibrator_path
        return FakeEngine()

    def fake_consume_available_messages(**kwargs: object) -> int:
        captured["engine"] = kwargs["engine"]
        captured["input_topic"] = kwargs["input_topic"]
        captured["output_topic"] = kwargs["output_topic"]
        return 0

    monkeypatch.setattr("fraud_platform.consumer.Consumer", fake_consumer)
    monkeypatch.setattr("fraud_platform.consumer.Producer", fake_producer)
    monkeypatch.setattr("fraud_platform.consumer.load_policy", fake_load_policy)
    monkeypatch.setattr("fraud_platform.consumer.ScoringEngine.from_paths", fake_from_paths)
    monkeypatch.setattr(
        "fraud_platform.consumer.consume_available_messages",
        fake_consume_available_messages,
    )

    run_consumer(
        bootstrap_servers="localhost:9092",
        input_topic="transaction-events",
        output_topic="fraud-decisions",
        group_id="fraud-consumer",
        model_path="artifacts/model/latest",
        policy_path="configs/decision_policy.yaml",
        calibrator_path="artifacts/calibration/latest/calibrator.pkl",
    )

    assert captured["calibrator_path"] == "artifacts/calibration/latest/calibrator.pkl"
    assert captured["input_topic"] == "transaction-events"
    assert captured["output_topic"] == "fraud-decisions"
    assert captured["closed"] is True
