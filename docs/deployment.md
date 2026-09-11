# Deployment

The default deployment is a local Docker Compose stack. It needs no managed cloud services or
external API accounts. Use a Docker-compatible runtime with Compose and a running daemon.

## Synthetic deployment

```bash
uv sync --locked --extra dev
make demo-up
```

`docker-compose.demo.yml` overlays the base stack to use `artifacts/demo/model` and
`artifacts/demo/replay.parquet`. It disables optional calibration/conformal paths so real-data
artifacts cannot accidentally be applied to a demonstration model.

## IEEE-CIS deployment

Prepare chronological splits and a model with the [operator runbook](runbook.md), then run:

```bash
docker compose up --build -d
```

The base stack expects `artifacts/model/latest` and `data/processed/replay.parquet`. Optionally
set `CALIBRATOR_PATH` and `CONFORMAL_PATH` in `.env` or the shell. Paths are relative to `/app`
in the container and should point into the mounted `artifacts/` directory.

## Services and ports

| Service | Host port | Role |
| --- | ---: | --- |
| dashboard | 5173 | React console served by nginx |
| fraud-api | 8000 | Scoring, feeds, OpenAPI, metrics |
| kafka | 9092 | Host Kafka listener; containers use `kafka:29092` |
| postgres | 5432 | Predictions and alerts |
| mlflow | 5001 | Experiment tracking |
| prometheus | 9090 | Metric collection |
| grafana | 3000 | Operational dashboards |

The consumer, producer, and monitoring worker have no exposed HTTP ports. Kafka runs in KRaft
mode using the checked-in Confluent image. Backend images install from `uv.lock`; the frontend
uses `npm ci`. Kafka and PostgreSQL health checks gate dependent service startup.

## Configuration and persistence

Compose loads baseline values from `.env.example`. Only fields explicitly interpolated in
`docker-compose.yml` can be overridden by `.env`; changing an arbitrary variable there does not
replace a service's explicit `environment` entry. Direct Python CLI commands read `.env` normally.

PostgreSQL records persist in the `postgres-data` named volume. Kafka and MLflow use container
storage and are not durable across replacement. Local model files remain in bind mounts. Do not
use `docker compose down -v` unless deleting the database volume is intentional.

This deployment has development credentials, no API authentication, and no TLS termination. It
is designed for a trusted machine rather than public ingress. See [architecture](architecture.md)
for delivery guarantees and [Docker operations](docker-runbook.md) for recovery steps.
