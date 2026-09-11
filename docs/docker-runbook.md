# Docker operations

## Prepare and start

Check that `docker info` succeeds. Install Python 3.11 dependencies and start the complete synthetic
path with:

```bash
uv sync --locked --extra dev
make demo-up
```

For IEEE-CIS, prepare the data and model using the [runbook](runbook.md), then use the base
`docker compose up --build -d` command. Synthetic demo commands use both Compose files; keep using
that same pair for subsequent operations so artifact selection stays consistent.

## Inspect

```bash
docker compose -f docker-compose.yml -f docker-compose.demo.yml ps -a
docker compose -f docker-compose.yml -f docker-compose.demo.yml logs --tail=100 fraud-consumer transaction-producer
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/model-info
curl --fail http://localhost:8000/predictions
curl --fail http://localhost:8000/alerts
curl --fail http://localhost:9090/-/ready
```

`/health` returns `{"status":"ok"}` after model loading. It does not check database connectivity.
The console runs at `http://localhost:5173`; interactive API docs run at `http://localhost:8000/docs`.
MLflow, Grafana, and Prometheus ports are listed in [deployment](deployment.md).

The replay producer is a finite job. An exit code of zero after processing its input is expected.
The consumer and monitoring worker remain running. To publish another synthetic batch:

```bash
docker compose -f docker-compose.yml -f docker-compose.demo.yml run --rm transaction-producer
```

## Recover

### Model or replay input missing

For the demo, rerun `make demo` and restart the affected service with the demo overlay. For real
data, regenerate the model and replay partition with `fraud-train`. Synthetic files are isolated
under `artifacts/demo/`; the base stack does not use them.

### API feeds fail or remain empty

A visible connection error means a feed request failed. An empty feed means no predictions have
been stored. Inspect API, PostgreSQL, consumer, and producer logs. API scores also populate the
feed; use the sample request in the [walkthrough](demo-script.md).

### Database schema missing

Apply the idempotent schema script without removing stored data:

```bash
docker compose exec -T postgres psql -U fraud -d fraud < docker/postgres/init.sql
```

### Consumer stops on a processing failure

Scoring, database, and broker failures intentionally leave the source offset uncommitted. Fix the
underlying dependency or artifact, then restart:

```bash
docker compose -f docker-compose.yml -f docker-compose.demo.yml restart fraud-consumer
```

Malformed event contracts are sent to the dead-letter topic, with the offset committed after
confirmed delivery. Infrastructure failures are not classified as bad input.

### Optional artifact path is wrong

Real-data services fail at startup if `CALIBRATOR_PATH` or `CONFORMAL_PATH` is set to a missing file.
Clear the path or regenerate the matching artifact. Refit conformal artifacts after the quantile
correction documented in the [release notes](../CHANGELOG.md).

## Stop

```bash
docker compose -f docker-compose.yml -f docker-compose.demo.yml down
```

This preserves PostgreSQL's named volume. Adding `-v` deletes stored predictions and alerts.
Kafka and MLflow do not have durable volumes in the current local configuration.
