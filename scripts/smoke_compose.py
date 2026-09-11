"""Verify the running synthetic Compose deployment using its public HTTP contracts."""

from __future__ import annotations

import json
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def request(path: str, payload: dict | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(
        f"http://localhost:8000{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urlopen(req, timeout=10) as response:
        return json.load(response)


def main() -> None:
    deadline = time.monotonic() + 180
    last_error = "No predictions returned"
    while time.monotonic() < deadline:
        try:
            decisions = request("/predictions")
            if len(decisions) >= 10:
                break
        except (URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(2)
    else:
        raise RuntimeError(f"Replay did not reach the persisted API feed: {last_error}")
    assert all(row["model_version"] == "synthetic-fraud-model:1" for row in decisions)
    decision = request(
        "/score",
        {
            "event_id": "compose-smoke",
            "transaction_id": 10001,
            "event_time": "2026-09-11T12:00:00Z",
            "transaction_dt": 3600.0,
            "amount": 75.0,
            "product_cd": "W",
            "schema_version": "v1",
        },
    )
    assert decision["event_id"] == "compose-smoke"
    assert any(row["event_id"] == "compose-smoke" for row in request("/predictions"))
    assert isinstance(request("/alerts"), list)
    with urlopen("http://localhost:5173", timeout=10) as response:
        assert "Fraud Operations" in response.read().decode()
    print("Compose smoke passed: Kafka replay, persisted scoring, HTTP API, and dashboard")


if __name__ == "__main__":
    main()
