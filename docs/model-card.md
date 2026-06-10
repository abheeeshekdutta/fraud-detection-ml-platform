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
