# Fraud Platform Runbook

## Local Startup

Run the smoke training command before starting the API if no model artifact exists:

```bash
uv run fraud-train --synthetic --output-dir artifacts/model/latest
```

After downloading IEEE-CIS data, train the current real-data baseline with:

```bash
uv run fraud-train --prepare-ieee --raw-dir data/raw --processed-dir data/processed
uv run fraud-train --ieee-baseline --processed-dir data/processed --output-dir artifacts/model/latest --max-train-rows 100000
```

To benchmark a tree-based candidate, add `--model-candidate catboost` or
`--model-candidate lightgbm` to the training command.

To run a small time-aware hyperparameter search before fitting the final candidate, add
`--tune-hyperparameters`:

```bash
uv run fraud-train --ieee-baseline --processed-dir data/processed --output-dir artifacts/model/candidates/lightgbm-tuned --max-train-rows 100000 --model-candidate lightgbm --tune-hyperparameters
```

To record the baseline run in local MLflow, start MLflow first and rerun the training command with a
tracking URI:

```bash
docker compose up -d mlflow
uv run fraud-train --ieee-baseline --processed-dir data/processed --output-dir artifacts/model/latest --max-train-rows 100000 --model-candidate catboost --mlflow-tracking-uri http://localhost:5001
```

Without `--mlflow-tracking-uri`, training still writes the model bundle and `training_summary.json`,
but no MLflow run is created.

To compare approve/review/block operating points for a saved model bundle, run:

```bash
uv run fraud-thresholds \
  --processed-dir data/processed \
  --model-dir artifacts/model/candidates/lightgbm-features \
  --output-path reports/generated/lightgbm_threshold_analysis.json \
  --approve-thresholds 0.01,0.02,0.03,0.04 \
  --block-thresholds 0.05,0.08,0.10,0.15,0.20,0.30 \
  --fraud-loss 500 \
  --review-cost 5 \
  --false-block-cost 25 \
  --max-false-block-rate 0.02 \
  --max-review-rate 0.30 \
  --min-block-precision 0.20
```

The report is generated locally under `reports/generated/` and is not committed.

To fit a probability calibrator on the dedicated calibration split:

```bash
uv run fraud-calibrate \
  --processed-dir data/processed \
  --model-dir artifacts/model/candidates/lightgbm-features \
  --output-dir artifacts/calibration/lightgbm-isotonic \
  --method isotonic
```

Then rerun threshold analysis with the saved calibrator:

```bash
uv run fraud-thresholds \
  --processed-dir data/processed \
  --model-dir artifacts/model/candidates/lightgbm-features \
  --calibrator-path artifacts/calibration/lightgbm-isotonic/calibrator.pkl \
  --output-path reports/generated/lightgbm_threshold_analysis_isotonic.json \
  --approve-thresholds 0.01,0.02,0.03,0.04 \
  --block-thresholds 0.05,0.08,0.10,0.15,0.20,0.30 \
  --fraud-loss 500 \
  --review-cost 5 \
  --false-block-cost 25 \
  --max-false-block-rate 0.02 \
  --max-review-rate 0.30 \
  --min-block-precision 0.20
```

To use the same calibrator in runtime scoring, set:

```bash
CALIBRATOR_PATH=artifacts/calibration/lightgbm-isotonic/calibrator.pkl
```

`fraud-api` reads `CALIBRATOR_PATH` from settings. `fraud-consumer` receives the same value through
its `--calibrator-path` argument in Compose.

To fit a split-conformal uncertainty artifact on the dedicated calibration split:

```bash
uv run fraud-conformal \
  --processed-dir data/processed \
  --model-dir artifacts/model/candidates/lightgbm-features \
  --output-dir artifacts/conformal/lightgbm-alpha10 \
  --alpha 0.10
```

To use the same conformal artifact in runtime scoring, set:

```bash
CONFORMAL_PATH=artifacts/conformal/lightgbm-alpha10/conformal.pkl
```

`fraud-api` reads `CONFORMAL_PATH` from settings. `fraud-consumer` receives the same value through
its `--conformal-path` argument in Compose.

To generate a global SHAP explanation summary for a saved model bundle:

```bash
uv run fraud-explain \
  --processed-dir data/processed \
  --model-dir artifacts/model/candidates/lightgbm-features \
  --output-dir artifacts/explain/lightgbm-global \
  --max-background-rows 100 \
  --max-explain-rows 100
```

The workflow writes `global_shap_summary.json` with ranked mean absolute SHAP values.

The monitoring worker reads recent persisted decisions and writes review-rate shift alerts to
Postgres and `model-alerts`. Configure its first local guardrail with:

```bash
MONITORING_REFERENCE_REVIEW_RATE=0.10
MONITORING_REVIEW_RATE_MULTIPLIER=2.0
MONITORING_PREDICTION_LIMIT=500
MONITORING_INTERVAL_SECONDS=60
```

Start the local stack:

```bash
docker compose up --build
```

The transaction producer publishes replayed transactions to `transaction-events` and, when labels are
present in the replay data, delayed outcomes to `fraud-labels`. Tune the simulated label delay with:

```bash
LABEL_DELAY_SECONDS=30
```

The fraud consumer publishes malformed or unprocessable transaction messages to
`dead-letter-events` and commits them so one bad payload does not block the stream.

Docker's Compose reference describes `docker compose up` as the command that builds, creates,
starts, and attaches to services defined in the Compose file.

## Health Checks

- API health: `curl http://localhost:8000/health`
- API docs: `http://localhost:8000/docs`
- API metrics: `curl http://localhost:8000/metrics`
- Dashboard feed: `curl http://localhost:8000/predictions`
- Alert feed: `curl http://localhost:8000/alerts`
- Dashboard: `http://localhost:5173`
- MLflow: `http://localhost:5001`
- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`

FastAPI exposes Swagger UI docs at `/docs` with its default configuration.

## Common Issues

### API Cannot Load A Model

Rerun:

```bash
uv run fraud-train --synthetic --output-dir artifacts/model/latest
```

Then restart the API container.

### Kafka Is Unavailable

Restart Kafka and dependent services:

```bash
docker compose up --build kafka fraud-consumer transaction-producer
```

### Postgres Tables Are Missing

Reset local volumes and start again:

```bash
docker compose down -v
docker compose up --build postgres fraud-api
```

### Dashboard Shows Fallback Data

Fallback data means the dashboard is reachable but the API has no stored predictions to return, or
the browser cannot reach the API. Confirm the API is running and returning dashboard feed data:

```bash
curl http://localhost:8000/predictions
curl http://localhost:8000/alerts
```

### Prometheus Has No Fraud API Data

Confirm the API is healthy, then open:

```bash
curl http://localhost:8000/metrics
```

Prometheus scrapes `fraud-api:8000/metrics` inside the Compose network.
