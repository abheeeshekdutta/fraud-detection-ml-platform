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
