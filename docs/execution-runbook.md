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

## 3. Train A Local Model Artifact

```bash
uv run fraud-train --synthetic --output-dir artifacts/model/latest
```

Expected outputs:

- `artifacts/model/latest/model.pkl`
- `artifacts/model/latest/metadata.json`

Purpose:

- Create a small local model artifact so API and streaming slices can be tested before full IEEE-CIS
  training is run.

For the real-data baseline after processed splits exist:

```bash
uv run fraud-train --ieee-baseline --processed-dir data/processed --output-dir artifacts/model/latest
```

Optional follow-on artifacts:

```bash
uv run fraud-calibrate --processed-dir data/processed --model-dir artifacts/model/latest --output-dir artifacts/calibration/latest
uv run fraud-conformal --processed-dir data/processed --model-dir artifacts/model/latest --output-dir artifacts/conformal/latest
uv run fraud-explain --processed-dir data/processed --model-dir artifacts/model/latest --output-dir artifacts/explain/latest
```

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
  --topic transaction-events \
  --label-topic fraud-labels \
  --label-delay-seconds 30
```

Expected output:

- Transaction messages published to the Kafka `transaction-events` topic.
- Delayed label messages published to the Kafka `fraud-labels` topic when `isFraud` is present.

Purpose:

- Simulate production-like event flow by replaying historical transactions in time order.

## 7. Run The Fraud Consumer

```bash
uv run fraud-consumer \
  --bootstrap-servers localhost:9092 \
  --input-topic transaction-events \
  --output-topic fraud-decisions \
  --model-path artifacts/model/latest \
  --policy-path configs/decision_policy.yaml \
  --database-url postgresql+psycopg://fraud:fraud@localhost:5432/fraud \
  --dead-letter-topic dead-letter-events
```

Expected output:

- Scored decision messages published to the Kafka `fraud-decisions` topic.
- Prediction rows persisted to Postgres when `--database-url` is set.
- Invalid transaction messages published to `dead-letter-events`.

Purpose:

- Consume transaction events, score them with the model bundle, apply the decision policy, and emit
  governed fraud decisions.

## 8. Monitor Decisions And Store Alerts

Run the monitoring worker once against persisted predictions:

```bash
uv run fraud-monitor --once
```

Generate an offline monitoring report:

```bash
uv run fraud-monitor-report \
  --reference-path data/processed/validation.parquet \
  --current-path data/processed/replay.parquet \
  --output-path reports/generated/monitoring_report.json
```

Expected outputs:

- Rows in the `predictions` table.
- Rows in the `alerts` table.
- `reports/generated/monitoring_report.json`

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
- Read live prediction and alert feeds from the API, with fallback demo data only when no feed rows
  are available.

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
  -> Kafka transaction-events and fraud-labels
  -> uv run fraud-consumer
  -> Kafka fraud-decisions, Postgres predictions, optional dead-letter-events
  -> uv run fraud-monitor
  -> Postgres alerts and Kafka model-alerts
  -> uv run fraud-monitor-report
  -> reports/generated/monitoring_report.json
  -> cd dashboard && npm run dev -- --host 127.0.0.1
  -> browser dashboard at http://127.0.0.1:5173/
```
