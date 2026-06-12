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

## Baseline Model And First Candidate Benchmarks

Baseline command:

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

Candidate commands add `--model-candidate catboost` or `--model-candidate lightgbm` and write to a
candidate-specific output directory during comparison.

Model artifacts:

- `feature_schema_version`: `v1`
- `decision_policy_version`: `v1`
- MLflow experiment: `fraud-detection-ieee`

Validation comparison:

| Candidate | Model version | MLflow run | ROC-AUC | PR-AUC | Brier score |
| --- | --- | --- | ---: | ---: | ---: |
| Logistic regression | `ieee-logistic-baseline:1` | `d1959d793cb74653ba634ec44859b323` | 0.7543 | 0.1111 | 0.0303 |
| CatBoost | `ieee-catboost:1` | `a7077a78e4eb47e7a5b3d1d518679203` | 0.7526 | 0.1309 | 0.0300 |
| LightGBM | `ieee-lightgbm:1` | `d27e4c3990234b6181d97ad606495352` | 0.7677 | 0.1503 | 0.0297 |

All three runs used the 16-feature leakage-safe transaction feature set, the most recent 100,000
training rows, and the same 88,581-row validation split. LightGBM remains the strongest first-pass
candidate on PR-AUC, ROC-AUC, and Brier score.

## Interpretation

These runs are first-pass real-data baselines and candidates, not final tuned models.

What it proves:

- The project can now train from the actual IEEE-CIS files.
- The platform can produce model artifacts from real data, not just synthetic rows.
- The platform can log candidate models, parameters, and validation metrics to MLflow.
- The validation scores are meaningfully above random, so the selected features carry signal.
- The first-pass derived transaction features improved ROC-AUC for all three candidates.
- The replay split exists, which unlocks Kafka replay with `data/processed/replay.parquet`.

What it does not prove yet:

- LightGBM is not final until threshold, calibration, latency, and segment behavior are checked.
- PR-AUC is still modest because fraud is rare and the feature set is still intentionally limited.
- Calibration has been fit offline, but the calibrator is not yet wired into runtime scoring.
- Segment behavior, especially for `ProductCD=C`, still needs threshold and false-positive analysis.

## Calibration

LightGBM calibration was fit on the dedicated 88,581-row calibration split and evaluated on the
88,581-row validation split:

| Method | Validation Brier | Calibration error |
| --- | ---: | ---: |
| Raw scores | 0.029665 | 0.005643 |
| Isotonic | 0.029763 | 0.003989 |
| Platt | 0.030220 | 0.007460 |

Isotonic improves calibration error but slightly worsens Brier score. Platt is worse on both metrics
in this first pass, so isotonic is the better candidate when threshold analysis needs calibrated
probabilities.

## Threshold Analysis

The first threshold report used the LightGBM candidate bundle, the 88,581-row validation split, and
illustrative cost assumptions of `fraud_loss=500`, `review_cost=5`, and `false_block_cost=25`.

Top utility point from the searched grid:

| Approve threshold | Block threshold | Approve rate | Review rate | Block rate | Block precision | Block recall | False block rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.04 | 0.05 | 74.32% | 6.45% | 19.23% | 9.57% | 56.46% | 17.98% |

Constrained point with `max_false_block_rate=0.02`, `max_review_rate=0.30`, and
`min_block_precision=0.20`:

| Approve threshold | Block threshold | Approve rate | Review rate | Block rate | Block precision | Block recall | False block rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.04 | 0.15 | 74.32% | 23.67% | 2.02% | 24.26% | 15.00% | 1.58% |

After isotonic calibration, the constrained point changes:

| Approve threshold | Block threshold | Approve rate | Review rate | Block rate | Block precision | Block recall | False block rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.04 | 0.30 | 77.52% | 20.31% | 2.17% | 24.70% | 16.45% | 1.69% |

This is not a production recommendation. It shows why threshold selection needs business constraints:
unconstrained utility optimization can choose an aggressive block policy with an unacceptable
false-block rate. The constrained calibrated point is more plausible, but it still needs segment
checks before promotion.

## Recommended Next Modeling Steps

1. Wire the selected calibrator into the scoring engine and API/consumer runtime.
2. Replace the simple conformal prediction set with a validation-backed conformal method.
3. Add per-segment threshold analysis for `ProductCD`, especially `C` versus `W`.
4. Measure candidate inference latency before promotion.
