# Modeling Plan

## Objective

Train a robust tabular classifier that detects fraudulent transactions while controlling false positives and routing uncertain cases to review.

## Python Version

Use Python 3.11.

Rationale:

- modern and widely supported
- stable for ML packages
- compatible with CatBoost, LightGBM, FastAPI, MLflow, MAPIE, SHAP, and Evidently
- less risky than targeting the newest CPython release

## Package Manager

Use `uv` for Python environment and dependency management.

## Candidate Models

Detailed modeling logs:

- [Feature Engineering](feature-engineering.md)
- [Hyperparameter Tuning](hyperparameter-tuning.md)

### Baseline

- Logistic regression with simple preprocessing

Purpose:

- sanity check
- interpretable baseline
- performance floor

### Primary Candidates

- CatBoostClassifier
- LightGBMClassifier

CatBoost is a first-class candidate because the dataset has high-cardinality categoricals, missingness, and mixed tabular features. LightGBM remains a strong challenger because it is fast and often excellent on fraud-style tabular data.

The IEEE-CIS trainer supports these candidates with `--model-candidate catboost` and
`--model-candidate lightgbm`. Candidate runs use the same feature columns, artifact bundle format,
validation metrics, and optional MLflow logging path as the logistic baseline.

Use `--tune-hyperparameters` to run a small `TimeSeriesSplit` grid search before fitting the final
candidate on the full training slice. The current tuning path logs candidate hyperparameters,
selected parameters, fold metrics, and final validation metrics to MLflow when tracking is enabled.

### Optional Challenger

- XGBoostClassifier

Only add XGBoost if the first benchmark suggests meaningful upside.

## Metrics

Primary metrics:

- PR-AUC
- recall at fixed precision
- precision at fixed recall
- expected fraud loss saved
- false positive review/block rate

Secondary metrics:

- ROC-AUC
- Brier score
- calibration curve
- inference latency

## Calibration

Raw fraud probabilities should be calibrated before use in decision policy.

Candidate methods:

- Platt scaling
- isotonic regression

Calibration should be fit on a calibration split, not the training split.

## Conformal Prediction

Conformal prediction should be used as an uncertainty layer, not as a replacement for probability calibration or thresholding.

For classification, conformal prediction produces prediction sets such as:

- `{legit}`
- `{fraud}`
- `{legit, fraud}`

The project should use MAPIE or a lightweight split-conformal implementation.

Decision policy:

- `{legit}` with low fraud probability: `approve`
- `{fraud}` with high fraud probability: `block`
- `{legit, fraud}`: `review`
- empty or invalid set: `review`

This is especially valuable in fraud detection because uncertain cases should be escalated rather than silently approved or blocked.

Monitoring should include conformal coverage on delayed labels.

## Explainability

Use SHAP to generate:

- global feature importance
- per-transaction reason codes
- analyst-facing risk explanations

Reason codes should be short, stable, and safe for an operations dashboard. They should not expose raw SHAP internals directly.

## Model Selection

The winning model should not be selected by leaderboard-style AUC alone.

Selection criteria:

- validation PR-AUC
- calibrated probability quality
- business utility at selected thresholds
- acceptable false positive rate
- stable conformal behavior
- inference latency
- explainability quality

## Remaining Modeling Work

The current project has first-pass candidate training and small time-aware hyperparameter search.
The following modeling work is still important before treating any model as production-ready:

- richer feature engineering, including amount transformations, identity coverage indicators, and
  time-based transaction features
- leakage checks for any aggregate or identity-derived feature
- class-imbalance handling and threshold optimization for approve/review/block decisions
- probability calibration on the calibration split
- conformal uncertainty validation on held-out data
- segment analysis across product, device, email-domain, and missingness cohorts
- inference latency measurement for the candidate model bundle
- SHAP stability checks before exposing reason codes in the dashboard

## Artifacts

Each production candidate should produce:

- fitted preprocessing pipeline
- model artifact
- calibration artifact
- conformal calibration artifact
- feature schema
- model card
- decision policy config
- MLflow run metadata
