# Feature Engineering

This document tracks the feature engineering choices used by the fraud detection platform, why they
were chosen, what has been tested so far, and what should be added next.

## Current Feature Set

The current IEEE-CIS training path uses a serving-safe feature set built from single-transaction
fields and first-pass derived features.

Numeric features:

- `TransactionAmt`
- `TransactionDT`
- `card1`
- `addr1`
- `TransactionAmt_log1p`
- `TransactionAmt_cents`
- `has_identity`
- `missing_addr1`
- `missing_P_emaildomain`
- `missing_DeviceType`
- `transaction_day`
- `transaction_hour`

Categorical features:

- `ProductCD`
- `P_emaildomain`
- `DeviceType`
- `id_31`

These are defined in `src/fraud_platform/training.py` as `NUMERIC_FEATURES`,
`CATEGORICAL_FEATURES`, and `MODEL_FEATURES`. The derivation logic lives in
`src/fraud_platform/features/transformers.py`.

## Current Transformations

The model pipeline applies the following preprocessing:

| Feature group | Transformation | Reason |
| --- | --- | --- |
| Numeric fields | Median imputation | Keeps training and scoring robust when optional fields are missing. |
| Numeric fields | Standard scaling | Required for the logistic baseline and harmless for current tree candidates. |
| Transaction amount | `log1p` transform and cent remainder | Reduces skew while preserving a simple decimal-pattern signal. |
| Identity/enrichment coverage | Missingness indicators and `has_identity` | Missing joins are meaningful in IEEE-CIS and available at scoring time. |
| Transaction time | Day and hour proxies from `TransactionDT` | Captures broad event-time drift without using future labels or aggregates. |
| Categorical fields | Constant missing-value imputation with `missing` | Treats missing identity/enrichment data as a real production state rather than dropping rows. |
| Categorical fields | One-hot encoding with unknown handling | Keeps the model bundle sklearn-compatible and safe for unseen categories at scoring time. |

The serving transformer selects raw columns, computes derived features, and casts categorical fields
to pandas `category`. The fitted model bundle includes this transformer inside the sklearn pipeline,
so loaded models can score raw transaction rows through the same feature path used in training.

## Why This First

The first feature set is intentionally conservative:

- It can be computed from a single transaction row.
- It avoids future-looking aggregate features.
- It matches fields available in replayed Kafka transaction events.
- It supports comparable baseline and candidate runs before more complex feature engineering changes
  the problem.

This keeps early model comparisons focused on the training/evaluation pipeline rather than hidden
feature leakage.

## Current Results

Before these derived features were added, using the most recent 100,000 time-ordered training rows
and the 88,581-row validation split:

| Candidate | ROC-AUC | PR-AUC | Brier score |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.7023 | 0.0977 | 0.0307 |
| CatBoost | 0.7260 | 0.1359 | 0.0300 |
| LightGBM | 0.7489 | 0.1498 | 0.0297 |

LightGBM was the strongest first-pass model on the minimal raw feature set. The derived feature set is
now implemented in code, but it still needs a fresh benchmark run against the same candidates before
updating the comparison table.

## Feature Engineering Notes

- Missing identity coverage is now represented by `has_identity` and field-level missingness flags.
- `ProductCD` has strongly different fraud rates by segment, especially `C` versus `W`.
- Fraud rate drifts over transaction time, so `TransactionDT`, `transaction_day`, and
  `transaction_hour` are included while validation remains time-ordered.
- The current feature set does not yet include percentile bins, frequency encodings, target encodings,
  or safe historical aggregates.

## Planned Additions

Next feature engineering work should add leakage-safe features in this order:

1. Amount refinements:
   - high-amount percentile bins fit on training data only
   - rounded amount indicators beyond the current cent remainder
   - amount/product interactions with fold-aware fitting

2. Segment features:
   - product-risk priors fit only on training folds
   - email-domain frequency buckets fit only on training folds

3. Safe historical aggregates:
   - card-level counts using only prior transactions
   - amount velocity using only prior transactions
   - product/email/card interactions with fold-aware fitting

## Leakage Rules

Any new feature must satisfy these rules:

- It must be available at scoring time.
- It must not use validation, calibration, replay, or future transaction information.
- If it is an aggregate, it must be fit inside each training fold during tuning.
- If it uses target rates, it must use out-of-fold or prior-only estimates.
- It must have a fallback behavior for missing or unseen values.

## Open Questions

- Should tree candidates use native categorical handling instead of one-hot encoding?
- Which identity fields add signal without creating brittle sparse features?
- Should high-cardinality features be frequency encoded, target encoded, or left out until the
  leakage-safe aggregate framework exists?
- Which features are worth exposing as analyst reason codes versus keeping internal to the model?
