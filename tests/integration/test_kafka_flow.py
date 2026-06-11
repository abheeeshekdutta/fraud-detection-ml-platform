from __future__ import annotations

import pytest


@pytest.mark.integration
def test_kafka_flow_documented_for_compose_stack() -> None:
    pytest.skip(
        "Run after the Docker Compose stack exists: replay publishes transaction-events "
        "and consumer emits fraud-decisions."
    )
