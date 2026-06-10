# Deployment Plan

## Requirement

The project must run locally with no required paid services.

Default deployment target:

- Docker Compose

Recommended macOS container runtime:

- Colima + Docker CLI

Docker Desktop is not required.

## Services

Planned Docker Compose services:

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

## Local Ports

Planned defaults:

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
