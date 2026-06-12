# Architecture

## Goal

This document describes a local fraud detection platform that scores transactions in near real time, explains risk decisions, and monitors model health over time.

The architecture separates offline model development, online scoring, monitoring, and analyst review into distinct components.

## Event Flow

```mermaid
flowchart TB
    raw["IEEE-CIS transaction<br/>and identity files"]
    split["Validation +<br/>time-aware split"]
    replay["Transaction producer<br/>holdout replay"]
    transaction_events[["Kafka<br/>transaction-events"]]
    consumer["Fraud consumer<br/>stream scoring"]
    policy{"Decision policy<br/>approve / review / block"}
    fraud_decisions[["Kafka<br/>fraud-decisions"]]
    postgres[("Postgres<br/>predictions + alerts")]
    dashboard["React operations console<br/>analyst review"]

    raw --> split --> replay --> transaction_events --> consumer --> policy --> fraud_decisions
    fraud_decisions --> postgres --> dashboard
    fraud_decisions --> dashboard

    mlflow["MLflow<br/>experiments + registry"]
    artifacts["Local artifact bundle<br/>artifacts/model/latest"]
    model["Active model bundle<br/>features + calibration + SHAP"]
    api["Fraud API<br/>synchronous scoring"]
    api_policy{"API decision policy<br/>same approve / review / block rules"}
    checkout["Checkout or analyst workflow"]

    mlflow --> artifacts --> model
    model --> consumer
    model --> api
    checkout --> api --> api_policy
    api_policy --> postgres
    policy --> postgres

    fraud_labels[["Kafka<br/>fraud-labels"]]
    monitor["Monitoring worker<br/>drift + quality checks"]
    model_alerts[["Kafka<br/>model-alerts"]]
    dead_letters[["Kafka<br/>dead-letter-events"]]
    prometheus["Prometheus<br/>API metrics"]
    grafana["Grafana<br/>observability dashboards"]

    transaction_events -. invalid payload .-> dead_letters
    fraud_decisions --> monitor
    fraud_labels --> monitor
    monitor --> model_alerts
    monitor --> postgres
    monitor -. health feedback .-> mlflow
    model_alerts --> dashboard
    api --> prometheus --> grafana

    classDef dataNode fill:#e0f2fe,stroke:#0284c7,color:#0f172a,stroke-width:1.5px
    classDef kafkaNode fill:#ffedd5,stroke:#ea580c,color:#0f172a,stroke-width:1.5px
    classDef serviceNode fill:#f5f3ff,stroke:#7c3aed,color:#0f172a,stroke-width:1.5px
    classDef decisionNode fill:#fee2e2,stroke:#dc2626,color:#0f172a,stroke-width:1.5px
    classDef storeNode fill:#fef9c3,stroke:#ca8a04,color:#0f172a,stroke-width:1.5px
    classDef opsNode fill:#f1f5f9,stroke:#475569,color:#0f172a,stroke-width:1.5px

    class raw,split,replay dataNode
    class transaction_events,fraud_decisions,fraud_labels,model_alerts,dead_letters kafkaNode
    class consumer,api,checkout,model serviceNode
    class policy,api_policy decisionNode
    class mlflow,artifacts,postgres storeNode
    class monitor,prometheus,grafana,dashboard opsNode
```

## Major Components

### Transaction Producer

Reads held-out IEEE-CIS transactions in timestamp order and publishes them to Kafka as simulated live payment events.

Responsibilities:

- load production replay split
- preserve event-time ordering
- optionally control replay speed
- optionally inject drift scenarios
- publish valid transaction events to `transaction-events`

### Kafka

Apache Kafka provides the event backbone.

Initial topics:

- `transaction-events`: incoming transaction scoring requests
- `fraud-decisions`: model scores, decisions, reason codes, and uncertainty
- `fraud-labels`: delayed labels for monitoring and retraining simulation
- `model-alerts`: drift and operational alerts
- `dead-letter-events`: invalid or unprocessable messages

Kafka should run in KRaft mode through Docker Compose.

### Fraud Consumer

Consumes `transaction-events`, scores each transaction, writes the result to `fraud-decisions`, and persists prediction records.

Responsibilities:

- deserialize and validate transaction payloads
- apply production feature pipeline
- load active model artifact
- calculate calibrated fraud probability
- calculate conformal prediction set or uncertainty flag
- generate SHAP-based reason codes
- apply decision policy
- persist prediction metadata and emit decision events

### Fraud API

FastAPI service for synchronous scoring and operational endpoints.

Endpoints:

- `POST /score`: score a transaction synchronously
- `GET /health`: service health
- `GET /model-info`: active model metadata
- `GET /metrics`: Prometheus-compatible metrics

This mirrors the checkout service path while Kafka powers replay, streaming, and downstream monitoring.

### Training Pipeline

Builds reproducible offline models.

Responsibilities:

- data loading and joining
- time-aware split
- data validation
- feature engineering
- baseline model
- CatBoost and LightGBM candidates
- calibration
- conformal calibration
- threshold and decision policy selection
- MLflow experiment tracking
- model artifact packaging

### Monitoring Worker

Evaluates production-like traffic and emits alerts.

Responsibilities:

- data drift
- missingness drift
- score distribution drift
- latency and throughput checks
- delayed-label performance checks
- conformal coverage checks
- alert writing to Postgres and Kafka

### Dashboard

React + Vite fraud operations console.

Views:

- live transaction decision feed
- fraud score distribution
- approve/review/block rates
- model version and serving health
- latency and throughput
- drift and alert panel
- transaction detail drawer with reason codes

## Decision Flow

```text
incoming event
   |
schema validation
   |
feature pipeline
   |
model probability
   |
probability calibration
   |
conformal uncertainty
   |
decision policy
   |
approve / review / block
```

## Production Principles

- Offline and online feature code should share the same transformations where practical.
- Time-based validation is preferred over random splitting.
- Every prediction should carry model version, feature schema version, and decision policy version.
- Uncertainty should be routed to review rather than hidden.
- Monitoring should evaluate both model quality and system behavior.
