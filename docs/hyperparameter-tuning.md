# Hyperparameter Tuning

This document tracks the model hyperparameters, tuning strategy, MLflow logging behavior, benchmark
results, and open tuning work for the IEEE-CIS fraud detection models.

## Current Candidates

The IEEE-CIS trainer supports these model candidates:

| Candidate flag | Model type | Purpose |
| --- | --- | --- |
| `logistic_regression` | `LogisticRegression` | Interpretable baseline and performance floor. |
| `catboost` | `CatBoostClassifier` | Tree candidate for mixed tabular data with missingness and categoricals. |
| `lightgbm` | `LGBMClassifier` | Fast tree candidate for fraud-style tabular data. |

Use `--model-candidate` with `fraud-train --ieee-baseline` to select a candidate.

## Default Hyperparameters

The current defaults are intentionally small so local iteration remains fast.

| Candidate | Default parameters |
| --- | --- |
| Logistic regression | `max_iter=1000` |
| CatBoost | `iterations=50`, `depth=4`, `learning_rate=0.1` |
| LightGBM | `n_estimators=50`, `learning_rate=0.1`, `num_leaves=7` |

These defaults are not final recommendations. They are starter values for repeatable local
benchmarking.

## MLflow Logging

When `--mlflow-tracking-uri` is provided, the trainer logs:

- model identity:
  - `model_version`
  - `model_type`
  - `model_candidate`
  - `feature_schema_version`
  - `decision_policy_version`
  - `max_train_rows`
- model hyperparameters with `model__` prefixes:
  - examples: `model__n_estimators`, `model__num_leaves`, `model__learning_rate`
- validation metrics:
  - `validation_roc_auc`
  - `validation_pr_auc`
  - `validation_brier_score`
  - fraud-rate metrics
- the sklearn-compatible model artifact

If hyperparameter tuning is enabled, MLflow also logs:

- `tuning_strategy`
- `tuning_splits`
- `tuning_trials`

Fold-level tuning details are stored in `training_summary.json`.

## Tuning Strategy

Use:

```bash
uv run fraud-train \
  --ieee-baseline \
  --processed-dir data/processed \
  --output-dir artifacts/model/candidates/lightgbm-tuned \
  --max-train-rows 100000 \
  --model-candidate lightgbm \
  --tune-hyperparameters \
  --mlflow-tracking-uri http://localhost:5001
```

The current tuning path uses `TimeSeriesSplit`, not random cross-validation.

Reason:

- Fraud data is time-dependent.
- Random folds can leak future distribution into model selection.
- Time-ordered folds better approximate model behavior on later transactions.

Selection currently optimizes mean validation PR-AUC across folds, using mean Brier score as a
tie-breaker.

## Current Search Spaces

CatBoost:

| `iterations` | `depth` | `learning_rate` |
| ---: | ---: | ---: |
| 50 | 4 | 0.1 |
| 100 | 4 | 0.05 |
| 100 | 6 | 0.05 |

LightGBM:

| `n_estimators` | `num_leaves` | `learning_rate` |
| ---: | ---: | ---: |
| 50 | 7 | 0.1 |
| 100 | 15 | 0.05 |
| 150 | 31 | 0.03 |

These grids are intentionally small. They confirm the tuning path works without turning local model
development into a long-running experiment.

## Current Benchmark Results

First-pass, untuned candidate comparison on the 16-feature transaction set and the most recent
100,000 training rows:

| Candidate | MLflow run | ROC-AUC | PR-AUC | Brier score |
| --- | --- | ---: | ---: | ---: |
| Logistic regression | `d1959d793cb74653ba634ec44859b323` | 0.7543 | 0.1111 | 0.0303 |
| CatBoost | `a7077a78e4eb47e7a5b3d1d518679203` | 0.7526 | 0.1309 | 0.0300 |
| LightGBM | `d27e4c3990234b6181d97ad606495352` | 0.7677 | 0.1503 | 0.0297 |

LightGBM is the strongest first-pass candidate so far.

## Notes

- These results are from fixed starter hyperparameters, not a full tuning campaign.
- `training_summary.json` records the MLflow run ID for each logged run.
- The tuning path is implemented and covered by tests, but a full real-data tuned benchmark has not
  yet been recorded in the model card.
- Candidate artifacts should be written under `artifacts/model/candidates/` during comparison.
- `artifacts/model/latest` should only be replaced during an explicit promotion step.

## Planned Tuning Work

1. Run tuned LightGBM and CatBoost benchmark jobs on the 100,000-row slice.
2. Log the tuned run IDs and selected parameters in this document and the model card.
3. Add segment-level threshold checks.
4. Replace the simple conformal prediction set with a validation-backed conformal method.
5. Measure inference latency for the best candidate bundle.
6. Revisit the search space after feature engineering improves the signal.

## Important Non-Goals For Current Tuning

- Do not use random cross-validation for final model selection.
- Do not select a model by ROC-AUC alone.
- Do not promote a tuned model until threshold, calibration, latency, and segment behavior are
  checked.
- Do not expand the search space significantly until leakage-safe feature engineering is in place.
