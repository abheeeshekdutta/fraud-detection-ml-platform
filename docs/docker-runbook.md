# Docker Runbook

This guide runs the local fraud detection platform with Docker Compose.

## Prerequisites

- Python 3.11
- `uv`
- Docker Compose through Docker CLI, Colima, or another local container runtime

The stack is designed to run locally with no paid services.

## 1. Create Local Environment File

```bash
cp .env.example .env
```

Most Docker services read `.env.example` directly, but keeping `.env` lets you override optional
runtime artifacts:

```bash
CALIBRATOR_PATH=artifacts/calibration/latest/calibrator.pkl
CONFORMAL_PATH=artifacts/conformal/latest/conformal.pkl
```

Leave these blank for the synthetic smoke path.

## 2. Install Dependencies

```bash
uv sync --extra dev
```

This creates the local Python environment used to prepare model artifacts before containers start.

## 3. Create A Model Artifact

For the fastest Docker smoke run:

```bash
uv run fraud-train --synthetic --output-dir artifacts/model/latest
```

This writes:

- `artifacts/model/latest/model.pkl`
- `artifacts/model/latest/metadata.json`

The API and consumer containers mount `./artifacts` and load `artifacts/model/latest`.

## 4. Optional Real-Data Artifacts

If IEEE-CIS files are available under `data/raw`, prepare splits and train a real-data candidate:

```bash
uv run fraud-train --prepare-ieee --raw-dir data/raw --processed-dir data/processed
uv run fraud-train --ieee-baseline --processed-dir data/processed --output-dir artifacts/model/latest --max-train-rows 100000
```

Optional follow-on artifacts:

```bash
uv run fraud-calibrate --processed-dir data/processed --model-dir artifacts/model/latest --output-dir artifacts/calibration/latest
uv run fraud-conformal --processed-dir data/processed --model-dir artifacts/model/latest --output-dir artifacts/conformal/latest
uv run fraud-explain --processed-dir data/processed --model-dir artifacts/model/latest --output-dir artifacts/explain/latest
```

To load calibration and conformal artifacts at runtime, set `CALIBRATOR_PATH` and `CONFORMAL_PATH`
before starting Compose.

## 5. Start Docker Compose

```bash
docker compose up --build
```

Run in detached mode if preferred:

```bash
docker compose up --build -d
```

## 6. Open Services

- Dashboard: `http://localhost:5173`
- Fraud API docs: `http://localhost:8000/docs`
- Fraud API health: `http://localhost:8000/health`
- Fraud API metrics: `http://localhost:8000/metrics`
- MLflow: `http://localhost:5001`
- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`
- Kafka external listener: `localhost:9092`
- Postgres: `localhost:5432`

## 7. Health Checks

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model-info
curl http://localhost:8000/predictions
curl http://localhost:8000/alerts
curl http://localhost:9090/-/ready
```

Expected API health response:

```json
{"status":"ok"}
```

## 8. What Compose Starts

| Service | Purpose |
| --- | --- |
| `kafka` | Local Apache Kafka broker in KRaft mode |
| `postgres` | Prediction and alert storage |
| `mlflow` | Local model experiment tracking UI |
| `fraud-api` | FastAPI synchronous scoring and dashboard feed API |
| `fraud-consumer` | Kafka transaction scoring worker |
| `transaction-producer` | Replays processed transactions into Kafka |
| `monitoring-worker` | Detects review-rate shifts and writes/publishes alerts |
| `prometheus` | Scrapes API metrics |
| `grafana` | Displays Prometheus dashboards |
| `dashboard` | Fraud operations console |

## 9. Application Flow

1. Offline training writes a model bundle under `artifacts/model/latest`.
2. Optional calibration and conformal workflows write runtime artifacts under `artifacts/`.
3. Docker Compose starts Kafka, Postgres, MLflow, API, workers, monitoring, and dashboard.
4. `transaction-producer` reads `data/processed/replay.parquet`.
5. It publishes transaction messages to Kafka topic `transaction-events`.
6. If replay rows include `isFraud`, it also publishes delayed labels to `fraud-labels`.
7. `fraud-consumer` reads `transaction-events`.
8. The consumer loads the model bundle, optional calibrator, optional conformal artifact, and decision policy.
9. It scores each transaction and applies approve/review/block policy.
10. It publishes decisions to `fraud-decisions`.
11. It persists decisions into Postgres `predictions`.
12. If a message is malformed or unprocessable, it publishes a `DeadLetterEvent` to `dead-letter-events`.
13. `fraud-api` exposes synchronous `POST /score`, model metadata, Prometheus metrics, and dashboard feed endpoints.
14. `monitoring-worker` reads recent Postgres predictions and detects review-rate shifts.
15. Monitoring alerts are saved to Postgres `alerts` and published to `model-alerts`.
16. Prometheus scrapes API metrics; Grafana visualizes request count and latency.
17. The React dashboard reads `/predictions` and `/alerts` from the API.

## 10. Stop Or Reset

Stop containers:

```bash
docker compose down
```

Stop containers and remove volumes:

```bash
docker compose down -v
```

Rebuild from scratch:

```bash
docker compose down -v
uv run fraud-train --synthetic --output-dir artifacts/model/latest
docker compose up --build
```

## Troubleshooting

### API Cannot Load A Model

Create the model artifact before starting Compose:

```bash
uv run fraud-train --synthetic --output-dir artifacts/model/latest
```

Then restart:

```bash
docker compose up --build fraud-api fraud-consumer
```

### Dashboard Shows Only Fallback Data

Check API feed endpoints:

```bash
curl http://localhost:8000/predictions
curl http://localhost:8000/alerts
```

If predictions are empty, confirm `transaction-producer`, `fraud-consumer`, and `postgres` are
running:

```bash
docker compose ps
```

### Replay Data Missing

The producer expects:

```text
data/processed/replay.parquet
```

Create real-data splits from IEEE-CIS:

```bash
uv run fraud-train --prepare-ieee --raw-dir data/raw --processed-dir data/processed
```

### Optional Artifact Path Is Wrong

If `CALIBRATOR_PATH` or `CONFORMAL_PATH` points to a missing file, the API or consumer will fail at
startup. Clear the variable or regenerate the artifact.

