# Execution Runbook

This project is built so each stage can be run locally and inspected through files.

## 1. Fetch Or Place Raw Data

The project does not automatically download the IEEE-CIS dataset yet. Download the dataset from
Kaggle and place these files locally:

- `data/raw/train_transaction.csv`
- `data/raw/train_identity.csv`

Expected output after this manual step:

- Raw CSV files under `data/raw/`.

## 2. Run EDA And Data Profiling

```bash
uv run python scripts/profile_ieee_cis.py
```

Expected outputs:

- `docs/data-profile.md`
- `reports/eda/product_fraud_rates.csv`
- `reports/eda/missingness.csv`
- `reports/eda/categorical_cardinality.csv`
- `reports/eda/time_window_fraud_rates.csv`
- `reports/eda/product_fraud_rates.svg`
- `reports/eda/time_window_fraud_rates.svg`

Purpose:

- Understand imbalance, missingness, categorical cardinality, identity coverage, and time drift.
- Use those findings to decide CatBoost and LightGBM preprocessing choices.
- Decide how to structure time-based train, calibration, validation, and replay splits.

## 3. Train The Current Smoke Model

```bash
uv run fraud-train --synthetic --output-dir artifacts/model/latest
```

Expected outputs:

- `artifacts/model/latest/model.joblib`
- `artifacts/model/latest/metadata.json`

Purpose:

- Create a small local model artifact so API and streaming slices can be tested before full IEEE-CIS
  training is implemented.

## 4. Run The Scoring API

```bash
uv run fraud-api
```

Expected runtime endpoints:

- `GET /health`
- `GET /model-info`
- `POST /score`
- `GET /metrics`

Purpose:

- Score a single transaction through the same contracts used by streaming.
- Expose Prometheus metrics for service behavior.

## 5. Start Kafka Services

This task adds topic definitions in:

- `configs/kafka_topics.yaml`

The Docker Compose stack runs Kafka in KRaft mode alongside Postgres, MLflow, Prometheus, Grafana,
the API, the consumer, the replay producer, the monitoring worker, and the dashboard.

Before starting the full stack, make sure the smoke model exists:

```bash
uv run fraud-train --synthetic --output-dir artifacts/model/latest
```

Then run:

```bash
docker compose up --build
```

Expected local services:

- Kafka topics named `transaction-events`, `fraud-decisions`, `fraud-labels`, `model-alerts`, and
  `dead-letter-events`.
- Postgres at `localhost:5432`.
- Fraud API at `http://localhost:8000`.
- MLflow at `http://localhost:5001`.
- Prometheus at `http://localhost:9090`.
- Grafana at `http://localhost:3000`.
- Dashboard at `http://localhost:5173`.

## 6. Replay Transactions Into Kafka

After a processed replay dataset exists:

```bash
uv run fraud-replay \
  --replay-path data/processed/replay.parquet \
  --bootstrap-servers localhost:9092 \
  --topic transaction-events
```

Expected output:

- Transaction messages published to the Kafka `transaction-events` topic.

Purpose:

- Simulate production-like event flow by replaying historical transactions in time order.

## 7. Run The Fraud Consumer

```bash
uv run fraud-consumer \
  --bootstrap-servers localhost:9092 \
  --input-topic transaction-events \
  --output-topic fraud-decisions \
  --model-path artifacts/model/latest \
  --policy-path configs/decision_policy.yaml
```

Expected output:

- Scored decision messages published to the Kafka `fraud-decisions` topic.

Purpose:

- Consume transaction events, score them with the model bundle, apply the decision policy, and emit
  governed fraud decisions.

## 8. Store Predictions And Alerts

The storage repositories are implemented, but wiring Kafka decisions into Postgres persistence is a
later slice.

Expected output after that wiring exists:

- Rows in the `predictions` table.
- Rows in the `alerts` table.

## 9. Run The Fraud Operations Dashboard

```bash
cd dashboard
npm install
npm run dev -- --host 127.0.0.1
```

Expected runtime URL:

- `http://127.0.0.1:5173/`

Expected UI:

- KPI strip for approve, review, block, and p95 latency.
- Decision feed table with transaction ID, decision, probability, uncertainty, latency, and model.
- Alerts panel.
- Transaction detail drawer with calibrated probability, conformal set, policy/schema versions, and
  reason codes.

Purpose:

- Give fraud analysts a local operations console for inspecting model decisions and alert status.
- Use fallback demo data when backend prediction and alert endpoints are not available yet.

## Current Sequence Summary

```text
manual Kaggle download
  -> data/raw/*.csv
  -> uv run python scripts/profile_ieee_cis.py
  -> reports/eda/* and docs/data-profile.md
  -> uv run fraud-train --synthetic
  -> artifacts/model/latest/*
  -> uv run fraud-api
  -> uv run fraud-replay
  -> Kafka transaction-events
  -> uv run fraud-consumer
  -> Kafka fraud-decisions
  -> cd dashboard && npm run dev -- --host 127.0.0.1
  -> browser dashboard at http://127.0.0.1:5173/
```
