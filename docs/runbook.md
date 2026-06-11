# Fraud Platform Runbook

## Local Startup

Run the smoke training command before starting the API if no model artifact exists:

```bash
uv run fraud-train --synthetic --output-dir artifacts/model/latest
```

After downloading IEEE-CIS data, train the current real-data baseline with:

```bash
uv run fraud-train --prepare-ieee --raw-dir data/raw --processed-dir data/processed
uv run fraud-train --ieee-baseline --processed-dir data/processed --output-dir artifacts/model/latest --max-train-rows 100000
```

Start the local stack:

```bash
docker compose up --build
```

Docker's Compose reference describes `docker compose up` as the command that builds, creates,
starts, and attaches to services defined in the Compose file.

## Health Checks

- API health: `curl http://localhost:8000/health`
- API docs: `http://localhost:8000/docs`
- API metrics: `curl http://localhost:8000/metrics`
- Dashboard: `http://localhost:5173`
- MLflow: `http://localhost:5001`
- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`

FastAPI exposes Swagger UI docs at `/docs` with its default configuration.

## Common Issues

### API Cannot Load A Model

Rerun:

```bash
uv run fraud-train --synthetic --output-dir artifacts/model/latest
```

Then restart the API container.

### Kafka Is Unavailable

Restart Kafka and dependent services:

```bash
docker compose up --build kafka fraud-consumer transaction-producer
```

### Postgres Tables Are Missing

Reset local volumes and start again:

```bash
docker compose down -v
docker compose up --build postgres fraud-api
```

### Dashboard Shows Fallback Data

Fallback data means the dashboard is reachable but the API prediction and alert endpoints are not
available yet. This is expected until the dashboard-specific API endpoints are wired to Postgres.

### Prometheus Has No Fraud API Data

Confirm the API is healthy, then open:

```bash
curl http://localhost:8000/metrics
```

Prometheus scrapes `fraud-api:8000/metrics` inside the Compose network.
