# Demo Script

## Setup

1. Run `uv sync --extra dev`.
2. Run `uv run fraud-train --synthetic --output-dir artifacts/model/latest`.
3. Run `docker compose up --build`.

## Walkthrough

1. Show the architecture diagram in `README.md`.
2. Open the dashboard at `http://localhost:5173`.
3. Point out the KPI strip, decision feed, alert panel, and transaction detail drawer.
4. Open `http://localhost:8000/docs` and show the FastAPI `/score`, `/health`, and `/metrics` endpoints.
5. Submit or describe a sample `/score` request using the strict `TransactionEvent` contract.
6. Open MLflow at `http://localhost:5000` and explain that the current artifact is a synthetic smoke model.
7. Open Grafana at `http://localhost:3000` and show the API request-rate and p95 latency panels.
8. Open Prometheus at `http://localhost:9090` and show that it scrapes the fraud API metrics endpoint.
9. Explain why ambiguous cases are routed to `review`: calibrated probability and conformal uncertainty are used to avoid silently approving or blocking uncertain transactions.

## Talking Points

- The project separates offline model development from online scoring.
- Every score carries model, feature schema, and decision policy versions for auditability.
- Kafka replay simulates real-time transaction flow without paid services.
- Postgres stores prediction and alert records for dashboard and monitoring use.
- The first model is intentionally a smoke artifact; later expansion replaces it with CatBoost and LightGBM benchmark runs over IEEE-CIS.
