# Architecture

The platform separates model development, event processing, synchronous scoring, and operational
review. The diagram in the [README](../README.md#architecture) reflects the implemented runtime.

## Offline learning

IEEE-CIS transaction and identity data are left-joined by `TransactionID` and split chronologically
into train, calibration, validation, and replay partitions. Preprocessing and model are packaged
together in a local bundle. Logistic regression, CatBoost, and LightGBM share the feature schema.
MLflow logging is optional; serving reads the configured local artifact, not the MLflow registry.

The probability calibrator is fitted on held-out raw scores. The conformal artifact is also fitted
on raw scores and continues to use that scale at runtime even when a probability calibrator is
loaded. Decision thresholds use the calibrated probability. Without optional artifacts, raw scores
and threshold-derived sets provide the smoke path; those sets have no conformal coverage claim.

## Runtime paths

1. The replay producer converts rows into strict `TransactionEvent` messages. `transaction_dt`
   preserves the dataset-relative time feature independently of the wall-clock `event_time`.
2. The consumer validates each event and uses the shared scoring engine to generate a decision.
3. It saves the decision to PostgreSQL, publishes to `fraud-decisions`, waits for broker
   acknowledgement, then synchronously commits the input offset.
4. `POST /score` runs the same engine and persists the decision before returning. It does not
   publish a Kafka decision event.
5. The React console polls `/predictions` and `/alerts` approximately every five seconds after the
   previous refresh completes. An API error is visible; stale data retains its last-refresh timestamp.

## Delivery and failure semantics

Kafka automatic offset commits and automatic offset storage are disabled. Invalid JSON or event
contracts are acknowledged only after confirmed publication to `dead-letter-events`. Without a
configured dead-letter destination, validation errors stop the consumer with the offset uncommitted.
Scoring, database, broker, and acknowledgement errors also stop processing without advancing that
offset; restart the consumer after resolving the cause.

Delivery is **at least once**. PostgreSQL and Kafka do not share a transaction. A crash after a
successful write or publish but before the offset commit can replay that event. Prediction writes
upsert by `event_id`; downstream Kafka consumers must deduplicate by the same field. The same
transaction can have multiple event IDs. Concurrent requests with a new, identical event ID can
race at the database primary key; clients should retry failures with the same ID.

The single-message delivery wait favors a clear recovery boundary over peak throughput. There is
no measured throughput or exactly-once guarantee. Replay and monitoring producers retain their
simpler queued-publish behavior; consumer acknowledgement guarantees do not extend to them.

## Monitoring and observability

The monitoring worker polls persisted predictions and emits review-rate shift alerts to PostgreSQL
and `model-alerts`. Offline reports compare missingness, numeric means, and categorical distributions.
Prometheus scrapes API request counts and scoring latency; Grafana is provisioned from source.

`fraud-labels` can carry simulated delayed outcomes, but there is no label-consuming performance
worker, automatic retraining, or model promotion. Online reason codes are deterministic heuristics;
SHAP reports are offline artifacts. These boundaries avoid confusing available helper functions
with continuously operated services.

## Storage and deployment

Compose uses PostgreSQL, Kafka in KRaft mode, MLflow, FastAPI, the scoring consumer, replay producer,
monitoring worker, Prometheus, Grafana, and the React console. Kafka and PostgreSQL health checks
hold dependent services until those dependencies are ready. PostgreSQL uses a named volume.
Model bundles remain bind-mounted local files. The demo overlay selects independent synthetic
artifacts and disables real-data calibrators and conformal artifacts.

API `/health` is liveness after model initialization, not a database or broker readiness check.
A database failure can therefore leave `/health` green while scoring and feeds fail. Kafka and
MLflow storage are not configured for durable recovery across container replacement. The local
stack has no service authentication or TLS and is intended for a trusted development machine.
