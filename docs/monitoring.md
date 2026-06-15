# Monitoring Plan

## Goals

The monitoring layer should make model behavior visible after deployment.

It should answer:

- Are incoming transactions similar to training data?
- Are fraud scores shifting unexpectedly?
- Is the model becoming less calibrated?
- Are conformal prediction sets still achieving expected coverage?
- Is serving latency acceptable?
- Are invalid events or schema failures increasing?

## Monitoring Sources

- Kafka transaction and decision topics
- Postgres prediction records
- delayed labels
- Prometheus service metrics
- training reference data

## Model Monitoring

Track:

- feature drift
- missingness drift
- categorical cardinality drift
- prediction distribution drift
- approve/review/block rate changes
- delayed-label PR-AUC
- delayed-label recall at fixed precision
- calibration error
- conformal coverage

The current local report workflow writes JSON drift and missingness summaries with
`fraud-monitor-report`. Evidently OSS remains available in the environment for deeper report
templates once the monitored production schema stabilizes.

## Service Monitoring

Track:

- request count
- Kafka consumer lag
- scoring latency
- error rate
- invalid event count
- dead-letter count
- model load failures
- prediction throughput

Prometheus should scrape service metrics. Grafana should visualize operational health.

## Alert Examples

- missingness in identity features increases beyond threshold
- review rate doubles compared with the configured reference period
- conformal coverage falls below target
- p95 scoring latency exceeds target
- dead-letter event rate exceeds threshold
- fraud score distribution shifts materially

## Implemented Worker

The current `fraud-monitor` worker reads recent persisted prediction records from Postgres, computes
the current approve/review/block mix, and saves a `decision_rate_shift` alert when the review rate is
greater than or equal to `MONITORING_REFERENCE_REVIEW_RATE * MONITORING_REVIEW_RATE_MULTIPLIER`.
When Kafka settings are configured, the same alert is also published to `model-alerts`.

It runs continuously in Docker Compose and can be run once for local checks:

```bash
uv run fraud-monitor --once
```

## Implemented Report

Generate a local monitoring report comparing a reference parquet file with a current parquet file:

```bash
uv run fraud-monitor-report \
  --reference-path data/processed/validation.parquet \
  --current-path data/processed/replay.parquet \
  --output-path reports/generated/monitoring_report.json
```

The report records row counts, missingness by column, numeric mean shifts, and categorical total
variation distance.

## Alert Routing

Alerts should be written to:

- Postgres alert table
- dashboard alert panel
- `model-alerts` Kafka topic

No paid paging service is required.
