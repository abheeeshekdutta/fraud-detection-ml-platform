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

Evidently OSS should be used for offline and scheduled monitoring reports.

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

It runs continuously in Docker Compose and can be run once for local checks:

```bash
uv run fraud-monitor --once
```

## Alert Routing

Alerts should be written to:

- Postgres alert table
- dashboard alert panel
- `model-alerts` Kafka topic

No paid paging service is required.
