# Local walkthrough

## Start

```bash
uv sync --locked --extra dev
make demo-up
```

Open the console at `http://localhost:5173`. The replay producer sends 200 synthetic transactions;
the consumer scores and persists them. Allow time for container builds and initial Kafka startup.
The console shows connection state, a decision feed, windowed rates, and p95 model-scoring latency.
Select a transaction to inspect its decision, probability, prediction set, policy, and reason codes.

## Submit an API request

```bash
curl --fail-with-body http://localhost:8000/score \
  -H 'Content-Type: application/json' \
  -d '{"event_id":"walkthrough-001","transaction_id":10001,"event_time":"2026-09-11T12:00:00Z","transaction_dt":3600.0,"amount":75.0,"product_cd":"W","schema_version":"v1"}'
curl --fail http://localhost:8000/predictions
```

The HTTP response includes model and policy versions and is persisted before success is returned.
The dashboard refreshes automatically. Repeating this event ID updates the same stored row.

## Inspect operations

- API schema and request validation: `http://localhost:8000/docs`
- API metrics: `http://localhost:8000/metrics`
- Grafana dashboards: `http://localhost:3000`
- Prometheus scrape target: `http://localhost:9090`
- MLflow experiment UI: `http://localhost:5001` (empty until a training run enables logging)

The synthetic model has no calibrated or conformal artifacts configured. Its uncertainty sets are
threshold-derived. Actual calibration and uncertainty evaluation use the IEEE-CIS workflows in
the [runbook](runbook.md). Neither synthetic scores nor synthetic latency demonstrate production
fraud-detection performance.

## Replay and stop

```bash
docker compose -f docker-compose.yml -f docker-compose.demo.yml run --rm transaction-producer
docker compose -f docker-compose.yml -f docker-compose.demo.yml down
```

PostgreSQL retains history across restarts. Each replay creates new event IDs. The console shows
up to 100 latest records; API callers can request up to 500 using `?limit=500`.
