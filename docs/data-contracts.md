# Data Contracts

## Dataset

The historical data source is the IEEE-CIS Fraud Detection dataset.

Relevant files:

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`

The transaction and identity files join on `TransactionID`. Identity rows are not guaranteed for every transaction, so the feature pipeline must handle missing identity enrichment.

## Split Strategy

Use `TransactionDT` as the event-time proxy.

Planned splits:

- training: earliest transactions
- calibration: later training-period transactions used for probability and conformal calibration
- validation: later labeled transactions for model/threshold selection
- replay: latest labeled transactions used to simulate production traffic

No random split should be used for final reported performance.

## Event Schema

`transaction-events` should contain a production-safe subset of transaction data plus metadata.

```json
{
  "event_id": "uuid",
  "transaction_id": 2987000,
  "event_time": "2026-06-10T12:00:00Z",
  "amount": 68.5,
  "product_cd": "W",
  "card_features": {},
  "address_features": {},
  "email_domain_features": {},
  "identity_features": {},
  "schema_version": "v1"
}
```

Exact feature names will be finalized during data profiling.

## Decision Schema

`fraud-decisions` should contain the model output and operational metadata.

```json
{
  "event_id": "uuid",
  "transaction_id": 2987000,
  "scored_at": "2026-06-10T12:00:00.120Z",
  "model_version": "fraud-model:1",
  "feature_schema_version": "v1",
  "decision_policy_version": "v1",
  "fraud_probability": 0.82,
  "calibrated_probability": 0.76,
  "conformal_prediction_set": ["fraud"],
  "uncertainty": "low",
  "decision": "block",
  "reason_codes": [
    {"feature": "TransactionAmt", "direction": "increases_risk"},
    {"feature": "card_velocity_24h", "direction": "increases_risk"}
  ],
  "latency_ms": 42
}
```

## Data Quality Checks

Minimum checks:

- required columns exist
- join key uniqueness where expected
- valid label values
- valid timestamp ordering
- missingness thresholds
- categorical cardinality bounds
- numeric range checks
- no target leakage fields in serving payload
- training/serving schema compatibility

## Label Delay Simulation

Production fraud labels often arrive after chargebacks, reviews, or investigations. This project should simulate delayed labels by publishing replay labels after a configurable delay to `fraud-labels`.

The replay label event includes:

- `event_id`
- `transaction_id`
- `labeled_at`
- `is_fraud`
- `label_source`
- `schema_version`
