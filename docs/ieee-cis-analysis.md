# IEEE-CIS Findings And Baseline Analysis

## Data Loaded

The IEEE-CIS Kaggle files were downloaded into `data/raw` and processed into time-ordered Parquet
splits under `data/processed`.

Raw training rows:

- `train_transaction.csv`: 590,540 transactions
- `train_identity.csv`: identity rows for 24.42% of transactions

Generated split files:

- `data/processed/train.parquet`
- `data/processed/calibration.parquet`
- `data/processed/validation.parquet`
- `data/processed/replay.parquet`
- `data/processed/split_summary.json`

## EDA Findings

The dataset is highly imbalanced:

- Fraud rows: 20,663
- Fraud rate: 3.50%

Identity coverage is sparse:

- Only 24.42% of transactions join to identity data.
- Missing identity is not an error; it is part of the production-like shape of this dataset.
- Models should treat identity fields as optional enrichment, not required inputs.

Product risk differs strongly by `ProductCD`:

| ProductCD | Rows | Fraud Rate |
| --- | ---: | ---: |
| C | 68,519 | 11.69% |
| S | 11,628 | 5.90% |
| H | 33,024 | 4.77% |
| R | 37,699 | 3.78% |
| W | 439,670 | 2.04% |

This matters because `ProductCD=C` is much riskier than the majority `W` traffic. A global threshold
can hide segment-specific false positive or false negative behavior.

Fraud rate drifts over time:

| Time Window | Rows | Fraud Rate |
| --- | ---: | ---: |
| 0 | 98,424 | 2.57% |
| 1 | 98,423 | 3.36% |
| 2 | 98,423 | 4.12% |
| 3 | 98,423 | 3.75% |
| 4 | 98,423 | 3.75% |
| 5 | 98,424 | 3.45% |

This supports the time-based split strategy. Random splitting would mix future distribution into
training and make validation look more stable than production replay.

## Time-Based Split

The processed split uses transaction time order:

| Split | Rows | Fraud Rate | Purpose |
| --- | ---: | ---: | --- |
| Train | 354,324 | 3.38% | Fit model parameters |
| Calibration | 88,581 | 4.04% | Later probability and conformal calibration |
| Validation | 88,581 | 3.26% | Model selection and threshold analysis |
| Replay | 59,054 | 3.75% | Kafka replay simulation |

The current baseline training command uses the most recent 100,000 train rows so the first real-data
iteration runs quickly while still respecting time order.

## Baseline Model

Command:

Requires local MLflow to be running at `http://localhost:5001` for the tracking step.

```bash
uv run fraud-train \
  --ieee-baseline \
  --processed-dir data/processed \
  --output-dir artifacts/model/latest \
  --max-train-rows 100000 \
  --mlflow-tracking-uri http://localhost:5001
```

The model bundle is always written locally. MLflow logging happens only when
`--mlflow-tracking-uri` is provided.

Model artifact:

- `model_version`: `ieee-logistic-baseline:1`
- `model_type`: `logistic_regression_ieee_baseline`
- `feature_schema_version`: `v1`
- `decision_policy_version`: `v1`
- MLflow experiment: `fraud-detection-ieee`
- MLflow run: `537422fa43054c8b9c58c0c49ab867f6`

Validation metrics:

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.7023 |
| PR-AUC | 0.0977 |
| Brier score | 0.0307 |
| Train fraud rate | 3.67% |
| Validation fraud rate | 3.26% |

## Interpretation

This model is a real-data baseline, not the final candidate.

What it proves:

- The project can now train from the actual IEEE-CIS files.
- The platform can produce a model artifact from real data, not just synthetic rows.
- The platform can log the baseline model, parameters, and validation metrics to MLflow.
- The validation score is meaningfully above random, so the selected features carry signal.
- The replay split exists, which unlocks Kafka replay with `data/processed/replay.parquet`.

What it does not prove yet:

- The logistic model is not expected to be the best performer for this dataset.
- PR-AUC is still low because fraud is rare and the feature set is intentionally small.
- Calibration has not been fit yet; the Brier score is only from raw logistic probabilities.
- Segment behavior, especially for `ProductCD=C`, still needs threshold and false-positive analysis.

## Recommended Next Modeling Steps

1. Train CatBoost on the same time-based splits with native categorical handling.
2. Train LightGBM with careful categorical encoding and missing-value handling.
3. Compare models using PR-AUC, ROC-AUC, recall at fixed precision, Brier score, and latency.
4. Fit probability calibration on the calibration split.
5. Replace the simple conformal prediction set with a validation-backed conformal method.
6. Add per-segment threshold analysis for `ProductCD`, especially `C` versus `W`.
