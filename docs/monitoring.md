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
- review rate doubles compared with reference period
- conformal coverage falls below target
- p95 scoring latency exceeds target
- dead-letter event rate exceeds threshold
- fraud score distribution shifts materially

## Alert Routing

Alerts should be written to:

- `model-alerts` Kafka topic
- Postgres alert table
- dashboard alert panel

No paid paging service is required.
