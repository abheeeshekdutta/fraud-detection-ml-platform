from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from confluent_kafka import Producer

from fraud_platform.config import Settings
from fraud_platform.contracts import FraudLabelEvent
from fraud_platform.features.ieee import build_transaction_event
from fraud_platform.streaming import serialize_event


def replay_frame(
    frame: pd.DataFrame,
    producer,
    topic: str,
    label_topic: str | None = None,
    label_delay_seconds: float = 30.0,
    speed_multiplier: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    event_time_base: str = "2026-06-10T12:00:00Z",
) -> int:
    ordered = frame.sort_values("TransactionDT")
    previous_dt: float | None = None
    count = 0
    for _, row in ordered.iterrows():
        current_dt = float(row["TransactionDT"])
        if previous_dt is not None:
            delay = max(0.0, (current_dt - previous_dt) / speed_multiplier)
            sleep(min(delay, 1.0))
        event = build_transaction_event(row, event_time_base=event_time_base)
        producer.produce(topic, key=str(event.transaction_id), value=serialize_event(event))
        producer.poll(0)
        if label_topic is not None and "isFraud" in row:
            if label_delay_seconds > 0:
                sleep(min(label_delay_seconds / speed_multiplier, 1.0))
            label = FraudLabelEvent(
                event_id=f"label-{event.transaction_id}",
                transaction_id=event.transaction_id,
                labeled_at=datetime.now(UTC) + timedelta(seconds=label_delay_seconds),
                is_fraud=bool(int(row["isFraud"])),
                label_source="ieee_replay",
                schema_version=event.schema_version,
            )
            producer.produce(
                label_topic,
                key=str(event.transaction_id),
                value=serialize_event(label),
            )
            producer.poll(0)
        previous_dt = current_dt
        count += 1
    producer.flush()
    return count


def replay_transactions(
    replay_path: str | Path,
    bootstrap_servers: str,
    topic: str,
    label_topic: str | None = None,
    label_delay_seconds: float = 30.0,
    speed_multiplier: float = 60.0,
) -> int:
    producer = Producer({"bootstrap.servers": bootstrap_servers})
    frame = pd.read_parquet(replay_path)
    return replay_frame(
        frame,
        producer=producer,
        topic=topic,
        label_topic=label_topic,
        label_delay_seconds=label_delay_seconds,
        speed_multiplier=speed_multiplier,
    )


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Replay processed transactions into Kafka.")
    parser.add_argument("--replay-path", default=settings.replay_data_path)
    parser.add_argument("--bootstrap-servers", default=settings.kafka_bootstrap_servers)
    parser.add_argument("--topic", default="transaction-events")
    parser.add_argument("--label-topic", default=None)
    parser.add_argument("--label-delay-seconds", type=float, default=settings.label_delay_seconds)
    parser.add_argument("--speed-multiplier", type=float, default=settings.replay_speed_multiplier)
    args = parser.parse_args()
    replay_transactions(
        args.replay_path,
        args.bootstrap_servers,
        args.topic,
        args.label_topic,
        args.label_delay_seconds,
        args.speed_multiplier,
    )


if __name__ == "__main__":
    main()
