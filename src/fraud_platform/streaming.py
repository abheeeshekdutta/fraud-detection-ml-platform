from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def serialize_event(event: BaseModel) -> bytes:
    return event.model_dump_json().encode("utf-8")


def deserialize_event(payload: bytes, model: type[T]) -> T:
    return model.model_validate_json(payload.decode("utf-8"))


def publish_confirmed(producer, topic: str, key: str, event: BaseModel) -> None:
    """Wait for broker acknowledgement before allowing a source offset to advance."""
    delivered = []

    def on_delivery(error, message) -> None:
        delivered.append(error)

    producer.produce(topic, key=key, value=serialize_event(event), on_delivery=on_delivery)
    remaining = producer.flush(10)
    if remaining or not delivered:
        raise TimeoutError(f"Kafka delivery timed out for {topic}")
    if delivered[0] is not None:
        raise RuntimeError(f"Kafka delivery failed for {topic}: {delivered[0]}")
