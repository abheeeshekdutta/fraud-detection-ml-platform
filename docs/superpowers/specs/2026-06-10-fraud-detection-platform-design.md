# Fraud Detection ML Platform Design

Date: 2026-06-10

## Summary

Build a local fraud detection platform for real-time transaction scoring, monitoring, and analyst review.

The project will use historical IEEE-CIS fraud data to train a tabular ML model, then replay held-out transactions through Apache Kafka to simulate real-time payment traffic. A scoring service will classify each transaction as `approve`, `review`, or `block`, with calibrated probabilities, conformal uncertainty, and SHAP-based reason codes.

The entire system must run locally with free/open-source tooling and no required paid services.

## Goals

- Show production ML judgment beyond notebook modeling.
- Support real-time event-driven scoring with Apache Kafka.
- Use strong classical tabular models, especially CatBoost and LightGBM.
- Include calibration, thresholding, and conformal uncertainty.
- Provide practical explainability for fraud analysts.
- Include monitoring for drift, performance, latency, and coverage.
- Deliver a clean React-based analyst dashboard.
- Keep deployment runnable with Docker Compose.

## Non-Goals

- No paid cloud deployment in the default project.
- No managed Kafka, managed database, or paid observability SaaS.
- No deep learning unless later evidence justifies it.
- No Streamlit dashboard.
- No Redpanda default deployment.

## Data

Use IEEE-CIS Fraud Detection.

The transaction and identity files join on `TransactionID`, and identity data is not available for every transaction. The system will treat this as a realistic enrichment pattern rather than a data defect.

Splits will be based on `TransactionDT`:

- train
- calibration
- validation
- production replay

The replay split will simulate live transaction traffic through Kafka.

## Architecture

Components:

- Apache Kafka in KRaft mode
- transaction replay producer
- FastAPI synchronous scoring API
- Kafka fraud scoring consumer
- offline training pipeline
- MLflow tracking and registry
- PostgreSQL prediction and alert store
- monitoring worker
- Prometheus and Grafana OSS
- React + Vite fraud operations dashboard

Kafka topics:

- `transaction-events`
- `fraud-decisions`
- `fraud-labels`
- `model-alerts`
- `dead-letter-events`

## Modeling

Baseline:

- logistic regression

Primary candidates:

- CatBoostClassifier
- LightGBMClassifier

CatBoost is a first-class model candidate because the dataset contains mixed tabular features, missingness, and categorical variables. LightGBM is included as a strong benchmark and challenger.

Python version:

- Python 3.11

Package manager:

- uv

Primary model selection criteria:

- PR-AUC
- recall at fixed precision
- cost-sensitive business utility
- calibration quality
- false positive rate
- inference latency
- conformal uncertainty behavior

## Calibration And Conformal Prediction

Probability calibration will use a dedicated calibration split.

Conformal prediction will be added as an uncertainty layer. For classification, the output is a prediction set:

- `{legit}`
- `{fraud}`
- `{legit, fraud}`

Decision policy:

- low-risk `{legit}` transactions are approved
- high-risk `{fraud}` transactions are blocked
- ambiguous `{legit, fraud}` transactions go to review

This makes uncertainty operationally meaningful and gives the system a realistic human-in-the-loop fraud workflow.

## Explainability

Use SHAP for:

- global feature importance
- local transaction explanations
- dashboard reason codes

Reason codes should be simplified for analyst readability and versioned with the decision policy.

## Monitoring

Model monitoring:

- feature drift
- missingness drift
- score drift
- delayed-label model performance
- calibration drift
- conformal coverage
- review/block/approve rate changes

Service monitoring:

- latency
- throughput
- Kafka consumer lag
- dead-letter events
- API error rates

Tools:

- Evidently OSS
- Prometheus
- Grafana OSS

## Dashboard

Build a React + Vite internal operations console.

Views:

- live scored transactions
- transaction detail drawer
- fraud score and decision
- reason codes
- model version
- approve/review/block rates
- drift and alert panel
- latency and throughput

The dashboard should look like a clean financial operations tool, not a notebook dashboard.

## Deployment

Use Docker Compose as the default deployment path.

No required paid services.

Recommended local container runtime:

- Colima + Docker CLI on macOS

Default services:

- Kafka
- Postgres
- MLflow
- fraud API
- fraud consumer
- transaction producer
- monitoring worker
- Prometheus
- Grafana
- dashboard

## Testing

Required test categories:

- unit tests for feature transformations
- data contract tests
- model pipeline smoke tests
- API contract tests
- Kafka integration tests
- decision policy tests
- monitoring calculation tests

## Success Criteria

The final project should let a user:

- run the stack locally
- see transactions flowing through Kafka
- inspect fraud decisions in a clean dashboard
- review model metrics and MLflow runs
- understand the architecture from documentation
- see evidence of monitoring, testing, and production thinking

## Open Decisions

- exact feature subset after profiling
- final model choice after benchmark
- exact threshold values after utility analysis
- conformal method choice after calibration experiments
