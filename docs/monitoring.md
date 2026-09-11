# Monitoring

## Online review-rate alerts

`fraud-monitor` polls the latest persisted predictions. If the review rate reaches the configured
reference rate multiplied by the threshold multiplier, it persists a `decision_rate_shift` alert
and publishes it to `model-alerts`. The dashboard reads the persisted alert feed through FastAPI.

```bash
uv run fraud-monitor --once
```

| Setting | Default | Meaning |
| --- | ---: | --- |
| `MONITORING_INTERVAL_SECONDS` | 60 | Time between checks |
| `MONITORING_PREDICTION_LIMIT` | 500 | Latest predictions in the comparison window |
| `MONITORING_REFERENCE_REVIEW_RATE` | 0.10 | Fixed reference review rate |
| `MONITORING_REVIEW_RATE_MULTIPLIER` | 2.0 | Alert threshold multiplier |

The window is count-based, not time-based. Repeated checks can create repeated alerts for an
unchanged breach; alerts have no acknowledgement/resolution workflow. A zero reference rate
currently disables this comparison. These details matter when interpreting the console.

## Offline drift report

```bash
uv run fraud-monitor-report \
  --reference-path data/processed/validation.parquet \
  --current-path data/processed/replay.parquet \
  --output-path reports/generated/monitoring_report.json
```

Reports include row counts, per-column missingness, numeric mean differences, and categorical
total variation distance. They are descriptive comparisons, not significance tests or automatic
model promotion decisions.

## Service metrics

Prometheus scrapes `fraud-api:8000/metrics`. The checked-in Grafana dashboard displays request rate
and scoring latency. `fraud_api_scoring_latency_ms` measures the scoring engine, excluding database
persistence and total HTTP round-trip time. Dashboard p95 is calculated from its latest feed window.

Kafka lag, delayed-label performance, and online conformal coverage are not exported by the current
worker. The replay producer can publish `fraud-labels`, and a coverage helper exists for offline
analysis, but no background service joins those labels to decisions. Evidently is available as a
dependency; the committed report generator uses pandas-based summaries.
