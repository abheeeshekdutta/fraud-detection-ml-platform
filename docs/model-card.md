# Model Card

## Model Name

Fraud Detection Classifier

## Intended Use

Score e-commerce payment transactions for fraud risk and route each transaction to:

- approve
- manual review
- block

## Not Intended For

- final legal determination of fraud
- use without human review for uncertain cases
- deployment on materially different payment data without validation
- consumer-facing explanation without additional compliance review

## Training Data

IEEE-CIS Fraud Detection dataset.

The dataset includes transaction and identity files joined by `TransactionID`. Identity information is incomplete for some transactions.

## Model Candidates

- Logistic regression baseline
- CatBoostClassifier
- LightGBMClassifier
- optional XGBoostClassifier

## Evaluation

Primary metrics:

- PR-AUC
- recall at fixed precision
- expected fraud loss saved
- false positive rate
- review rate

Operational metrics:

- latency
- invalid event rate
- model load success
- Kafka consumer lag

Uncertainty metrics:

- conformal coverage
- prediction set size distribution
- review rate from uncertain predictions

## Explainability

Use SHAP to generate global explanations and per-transaction reason codes.

Reason codes should be reviewed for stability and clarity before being shown in the dashboard.

## Limitations

- Historical replay is not a true live payment feed.
- Dataset features are anonymized, limiting domain-specific interpretation.
- Labels may not reflect modern fraud patterns.
- Offline performance can overstate real-world performance if drift is not monitored.
- Conformal guarantees rely on calibration data being exchangeable with production traffic; monitoring is required when traffic drifts.

## Governance

Every production prediction should record:

- model version
- feature schema version
- decision policy version
- calibrated probability
- conformal uncertainty output
- reason codes
- latency

## Implemented Artifact Metadata

Every packaged model bundle includes:

- `model_version`
- `feature_schema_version`
- `decision_policy_version`
- `model_type`

The initial synthetic model is a smoke-test artifact, not the final IEEE-CIS production candidate.

## Current IEEE-CIS Baseline

The first real-data artifact is:

- `model_version`: `ieee-logistic-baseline:1`
- `model_type`: `logistic_regression_ieee_baseline`
- training sample: most recent 100,000 rows from the time-ordered training split
- validation split: 88,581 later transactions
- MLflow experiment: `fraud-detection-ieee`
- latest logged MLflow run: `537422fa43054c8b9c58c0c49ab867f6`

Validation metrics:

- ROC-AUC: 0.7023
- PR-AUC: 0.0977
- Brier score: 0.0307

This is a baseline for platform validation and comparison. It is not the final recommended fraud
model; CatBoost and LightGBM candidate training is available for follow-up benchmark runs.
