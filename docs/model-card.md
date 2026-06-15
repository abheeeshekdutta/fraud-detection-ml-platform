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

## Current IEEE-CIS Baseline And Candidates

The first real-data artifact is:

- `model_version`: `ieee-logistic-baseline:1`
- `model_type`: `logistic_regression_ieee_baseline`
- training sample: most recent 100,000 rows from the time-ordered training split
- validation split: 88,581 later transactions
- MLflow experiment: `fraud-detection-ieee`
- latest logged MLflow run: `d27e4c3990234b6181d97ad606495352`

First-pass validation comparison:

| Candidate | Model version | ROC-AUC | PR-AUC | Brier score |
| --- | --- | ---: | ---: | ---: |
| Logistic regression | `ieee-logistic-baseline:1` | 0.7543 | 0.1111 | 0.0303 |
| CatBoost | `ieee-catboost:1` | 0.7526 | 0.1309 | 0.0300 |
| LightGBM | `ieee-lightgbm:1` | 0.7677 | 0.1503 | 0.0297 |

These runs use the first-pass leakage-safe transaction feature set. LightGBM is the strongest
candidate on the current validation split, but it is not the final recommended fraud model until
threshold, calibration, latency, and segment behavior are evaluated.

### Calibration

LightGBM calibration was fit on the calibration split and evaluated on the validation split:

| Method | Validation Brier | Calibration error |
| --- | ---: | ---: |
| Raw scores | 0.029665 | 0.005643 |
| Isotonic | 0.029763 | 0.003989 |
| Platt | 0.030220 | 0.007460 |

Isotonic is the better calibration-error candidate in this pass, though raw scores still have a
slightly better Brier score. Runtime scoring can load the selected calibrator through
`CALIBRATOR_PATH`.

### Threshold Analysis

An initial threshold-grid report was run against the LightGBM candidate on the validation split with
illustrative costs of `fraud_loss=500`, `review_cost=5`, and `false_block_cost=25`.

| Approve threshold | Block threshold | Approve rate | Review rate | Block rate | Block precision | Block recall | False block rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.04 | 0.05 | 74.32% | 6.45% | 19.23% | 9.57% | 56.46% | 17.98% |

With constraints of `max_false_block_rate=0.02`, `max_review_rate=0.30`, and
`min_block_precision=0.20`, the selected point changes:

| Approve threshold | Block threshold | Approve rate | Review rate | Block rate | Block precision | Block recall | False block rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.04 | 0.15 | 74.32% | 23.67% | 2.02% | 24.26% | 15.00% | 1.58% |

With isotonic-calibrated probabilities, the constrained point is:

| Approve threshold | Block threshold | Approve rate | Review rate | Block rate | Block precision | Block recall | False block rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.04 | 0.30 | 77.52% | 20.31% | 2.17% | 24.70% | 16.45% | 1.69% |

These operating points are useful for comparison, not deployment. Thresholds should be checked by
segment before promotion.
