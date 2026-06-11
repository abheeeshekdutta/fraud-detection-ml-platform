from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from confluent_kafka import Producer

from fraud_platform.features.ieee import build_transaction_event
from fraud_platform.streaming import serialize_event


def replay_frame(
    frame: pd.DataFrame,
    producer,
    topic: str,
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
        previous_dt = current_dt
        count += 1
    producer.flush()
    return count


def replay_transactions(
    replay_path: str | Path,
    bootstrap_servers: str,
    topic: str,
    speed_multiplier: float = 60.0,
) -> int:
    producer = Producer({"bootstrap.servers": bootstrap_servers})
    frame = pd.read_parquet(replay_path)
    return replay_frame(
        frame,
        producer=producer,
        topic=topic,
        speed_multiplier=speed_multiplier,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay processed transactions into Kafka.")
    parser.add_argument("--replay-path", default="data/processed/replay.parquet")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default="transaction-events")
    parser.add_argument("--speed-multiplier", type=float, default=60.0)
    args = parser.parse_args()
    replay_transactions(
        args.replay_path,
        args.bootstrap_servers,
        args.topic,
        args.speed_multiplier,
    )


if __name__ == "__main__":
    main()
