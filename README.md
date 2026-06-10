# Production-Ready Fraud Detection ML Platform

Flagship data science project for real-time e-commerce payment fraud detection.

This project is designed to showcase mid-level data science and ML engineering skills:

- time-aware fraud modeling on realistic tabular data
- Kafka-based real-time transaction scoring
- calibrated probabilities and conformal uncertainty
- explainable fraud decisions for analyst review
- model monitoring, drift checks, and operational observability
- reproducible local deployment with free/open-source tooling

## Use Case

The system scores incoming e-commerce payment transactions and returns one of three decisions:

- `approve`: low fraud risk and low uncertainty
- `review`: uncertain, borderline, or operationally suspicious
- `block`: high fraud risk and low uncertainty

The project uses the IEEE-CIS Fraud Detection dataset as the historical data source. The dataset contains transaction and identity files joined by `TransactionID`; not every transaction has identity data, which creates realistic missingness and enrichment behavior.

## Runtime Philosophy

Required spend: `$0`.

The default stack is self-hosted locally with Docker Compose and open/free tooling. No managed cloud services, paid APIs, paid monitoring products, or proprietary databases are required.

Recommended local container runtime on macOS:

- Colima + Docker CLI, to avoid Docker Desktop licensing concerns

## Core Stack

- Python 3.11
- uv for Python dependency and environment management
- Apache Kafka in KRaft mode
- FastAPI
- PostgreSQL
- MLflow OSS
- CatBoost, LightGBM, and scikit-learn
- MAPIE for conformal prediction
- SHAP for explainability
- Evidently OSS for model/data monitoring
- Prometheus and Grafana OSS
- React + Vite for the fraud operations console
- Docker Compose for local deployment

## System Shape

```text
historical IEEE-CIS data
        |
        v
data validation + time-aware split
        |
        +--> offline training --> MLflow model registry
        |
        +--> replay holdout transactions --> Kafka transaction-events
                                             |
                                             v
                                      fraud-consumer
                                             |
                                             v
                                  fraud-decisions topic
                                             |
                                             +--> Postgres prediction store
                                             +--> monitoring-worker
                                             +--> React operations dashboard
```

## Documentation

- [Architecture](docs/architecture.md)
- [Data Contracts](docs/data-contracts.md)
- [Modeling Plan](docs/modeling.md)
- [Monitoring Plan](docs/monitoring.md)
- [Deployment Plan](docs/deployment.md)
- [Model Card](docs/model-card.md)
- [Superpowers Design Spec](docs/superpowers/specs/2026-06-10-fraud-detection-platform-design.md)

## Status

Design and project scaffold are in progress. Implementation will follow the approved design.
