# Deployment Plan

## Requirement

The project must run locally with no required paid services.

Default deployment target:

- Docker Compose

Recommended macOS container runtime:

- Colima + Docker CLI

Docker Desktop is not required.

## Services

Docker Compose services:

- `kafka`
- `postgres`
- `mlflow`
- `fraud-api`
- `fraud-consumer`
- `transaction-producer`
- `monitoring-worker`
- `prometheus`
- `grafana`
- `dashboard`

## Kafka

Use Apache Kafka in KRaft mode.

Do not use Redpanda in the default stack. Redpanda is convenient and Kafka-compatible, but the project should avoid source-available licensing ambiguity and use Apache Kafka directly.

The local Compose stack uses Confluent's `cp-kafka` image because it is an official Apache Kafka image for Confluent Platform and supports local KRaft mode. Bitnami's public Kafka image availability changed, so it is not used as the default.

## Local Ports

Defaults:

- dashboard: `localhost:5173`
- fraud API: `localhost:8000`
- MLflow: `localhost:5000`
- Grafana: `localhost:3000`
- Prometheus: `localhost:9090`
- Kafka broker: `localhost:9092`
- Postgres: `localhost:5432`

## Configuration

Use environment variables and checked-in example files:

- `.env.example`
- service-specific config files
- model decision policy YAML
- topic configuration YAML

Secrets are not expected for the local project.

Before starting the full stack, create the current smoke model artifact:

```bash
uv run fraud-train --synthetic --output-dir artifacts/model/latest
```

Then start the local services:

```bash
docker compose up --build
```

## Health Checks

After the stack starts, check:

- API: `curl http://localhost:8000/health`
- API docs: `http://localhost:8000/docs`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Dashboard: `http://localhost:5173`

If Docker Compose is unavailable on the local machine, the checked-in deployment tests still parse
the Compose, Prometheus, Grafana, and Postgres configuration files.

## Cost

Required cost: `$0`.

The project should not require:

- cloud compute
- managed Kafka
- managed database
- paid observability tooling
- paid APIs
- paid Docker Desktop subscription

The user may optionally deploy to cloud later, but that is out of scope for the default flagship project.
