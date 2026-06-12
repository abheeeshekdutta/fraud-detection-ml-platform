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

Start the local stack:

```bash
docker compose up --build
```

Docker's Compose reference describes `docker compose up` as the command that builds, creates,
starts, and attaches to services defined in the Compose file.

## Health Checks

- API health: `curl http://localhost:8000/health`
- API docs: `http://localhost:8000/docs`
- API metrics: `curl http://localhost:8000/metrics`
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

Fallback data means the dashboard is reachable but the API prediction and alert endpoints are not
available yet. This is expected until the dashboard-specific API endpoints are wired to Postgres.

### Prometheus Has No Fraud API Data

Confirm the API is healthy, then open:

```bash
curl http://localhost:8000/metrics
```

Prometheus scrapes `fraud-api:8000/metrics` inside the Compose network.
