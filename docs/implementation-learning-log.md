# Implementation Learning Log

This document records implementation task summaries for learning and review.

## Task 1: Project Scaffold

**What changed**

- Created the Python 3.11 project scaffold with `pyproject.toml`, `uv.lock`, `.env.example`, `.gitignore`, `Makefile`, package marker, synthetic pytest fixtures, and scaffold smoke test.
- Added placeholder console entrypoint modules so declared scripts resolve before later tasks implement real behavior.
- Hardened scaffold commands so `make test`, `make api`, and dependency sync have clear behavior.

**Problems faced**

- `pytest` exits with code 5 when no tests are collected, which made the initial scaffold verification red.
- Declared console scripts pointed to modules that did not exist yet.
- The first lockfile selected stale `numba`/`llvmlite` versions for Intel macOS.
- `train-smoke` initially looked successful even though real training was not implemented yet.

**Solutions applied**

- Added `tests/test_scaffold.py`.
- Added minimal placeholder `main()` functions for future service modules.
- Added `numba>=0.60.0` and regenerated `uv.lock`.
- Changed `train-smoke` to fail clearly until Task 5 owns training artifacts.
- Made `compose-down` non-destructive and added explicit `compose-reset`.

**Verification performed**

- `uv sync --extra dev`
- `uv run pytest`
- `uv run ruff check src tests`
- `uv lock --check`
- `uv build`
- `make api`
- `make train-smoke`
- `git diff --check`

**Reusable learnings**

- A scaffold should make advertised commands either pass truthfully or fail with a clear ownership message.
- Lockfiles for ML projects need platform sanity checks early, especially around native dependencies.
- Placeholder entrypoints are useful when future console scripts are declared before their real modules exist.

## Task 2: Data Contracts And Decision Policy

**What changed**

- Added strict Pydantic contracts for transaction events, decision events, reason codes, and alerts.
- Added decision policy config, YAML loading, and approve/review/block decision logic.
- Added default decision policy config in `configs/decision_policy.yaml`.
- Expanded tests around strict types, timezone-aware datetimes, JSON-safe maps, finite floats, policy boundaries, malformed YAML, invalid probabilities, and conformal prediction sets.

**Problems faced**

- Strict datetime fields rejected ISO/Z timestamp strings from already-decoded JSON dictionaries.
- Non-finite floats such as `inf` could validate and serialize to JSON as `null`.
- Flexible feature and metadata maps could accept values that later failed JSON serialization.
- Policy config initially ignored typoed YAML keys.
- Policy decisions initially allowed invalid probabilities and duplicate or unknown prediction labels.

**Solutions applied**

- Added explicit pre-validation for ISO datetime strings while still rejecting naive timestamps.
- Added finite-float validation for all contract float fields.
- Added recursive JSON-compatible validation for feature maps and alert metadata.
- Made `PolicyConfig` strict and rejected malformed or non-mapping YAML with clear errors.
- Validated policy probabilities and prediction sets at the decision boundary.

**Verification performed**

- `uv run pytest tests/test_contracts.py tests/test_policy.py -q`
- `uv run pytest -q`
- `uv run ruff check src tests`
- `git diff --check`
- Manual edge smoke for JSON timestamp parsing, JSON-safe maps, non-finite floats, and unsafe map values.

**Reusable learnings**

- Boundary contracts need both strict internal validation and a deliberate JSON ingress path.
- Anything accepted by API/Kafka/storage-facing models should be safely serializable before it crosses process boundaries.
- Decision policy validation should fail close to the decision point rather than relying on a later event schema to catch bad inputs.

## Task 3: IEEE-CIS Loading, Time Split, And Feature Transformations

**What changed**

- Added `configs/feature_schema_v1.yaml` to define the first serving-safe feature groups and excluded leakage fields.
- Added IEEE-CIS helper functions for transaction/identity joins, basic training-frame validation, time-ordered splits, and replay event construction.
- Added a shared `FraudFeatureTransformer` with stable feature column ordering and categorical conversion.
- Added focused tests for missing identity preservation, time-aware splits, validation, transformation output, and serving event mapping.

**Problems faced**

- The first focused test run failed as expected because the `fraud_platform.features` package did not exist yet.
- Ruff flagged a quoted return annotation in the transformer after implementation.
- Task 2's strict contracts meant event construction had to convert pandas/numpy scalar values into native JSON-safe Python values.

**Solutions applied**

- Created the feature package only after observing the red test.
- Removed the unnecessary quoted annotation and reran lint.
- Added `_clean_mapping()` to drop missing enrichment values and convert pandas/numpy scalar values with `.item()` before creating `TransactionEvent`.

**Verification performed**

- `uv run pytest tests/test_features.py -q`
- `uv run pytest -q`
- `uv run ruff check src tests`
- `git diff --check`
- Manual smoke to serialize a built `TransactionEvent`, inspect split ordering, and inspect transformer dtypes.

**Reusable learnings**

- Feature/event builders should normalize pandas scalar values before handing data to strict boundary contracts.
- Keeping offline and replay feature helpers small makes it easier to verify missing-enrichment behavior without the full IEEE-CIS dataset.
- Time-aware split tests should assert ordering between split windows, not just row counts.

## Task 4: Metrics, Calibration, And Conformal Utilities

**What changed**

- Added model metric helpers for recall at minimum precision, expected fraud utility, and calibration error.
- Added `ProbabilityCalibrator` with isotonic and Platt scaling modes.
- Added `SplitConformalClassifier` for split-conformal prediction sets over fraud probabilities.
- Added focused tests for metric calculations, calibration output bounds, and conformal prediction-set behavior.

**Problems faced**

- The first focused test run failed as expected because `metrics`, `calibration`, and `conformal` modules did not exist yet.
- The conformal helper needed to return operationally meaningful ambiguous sets for borderline probabilities, not just a single hard label.

**Solutions applied**

- Implemented small, dependency-light wrappers around scikit-learn primitives where appropriate.
- Kept conformal behavior explicit: fit stores a nonconformity threshold, and prediction returns `legit`, `fraud`, or `legit, fraud` sets.
- Added runtime guards so calibrator and conformal prediction calls fail clearly before fitting.

**Verification performed**

- `uv run pytest tests/test_metrics.py tests/test_calibration_conformal.py -q`
- `uv run pytest -q`
- `uv run ruff check src tests`
- `git diff --check`
- Manual smoke for invalid calibrator method, unfitted calibrator, invalid conformal alpha, and unfitted conformal predictor.

**Reusable learnings**

- Early metric helpers should stay small and deterministic so later model-selection logic can compose them without hidden state.
- Calibration and uncertainty wrappers should fail loudly when used before fitting.
- Prediction sets are a clearer handoff to policy logic than forcing uncertainty into a single probability threshold.

## Task 5: Model Artifacts, Training Smoke Pipeline, And Reason Codes

**What changed**

- Added model artifact metadata, save/load helpers, and a `ModelBundle` prediction wrapper.
- Replaced the initial `fraud-train` placeholder with a synthetic logistic-regression smoke training pipeline.
- Restored `make train-smoke` to run the real synthetic training command.
- Added a deterministic fallback reason-code helper for early analyst-facing explanations.
- Added tests for loadable model bundles, raw probability prediction, and stable reason codes.

**Problems faced**

- Task 1 intentionally made `train-smoke` fail until this task owned real training artifacts.
- The smoke pipeline needed to create a real serialized model without pulling in the full IEEE-CIS data path too early.
- Reason codes need to be analyst-readable now, while SHAP-based explanations remain a later project slice.

**Solutions applied**

- Used a tiny synthetic tabular dataset and a scikit-learn `Pipeline` with `ColumnTransformer`, `OneHotEncoder`, and `LogisticRegression`.
- Stored metadata separately from the pickled model so downstream services can inspect model/version/schema information.
- Implemented fallback reason codes from stable feature names and simple risk heuristics.
- Kept Venn-Abers out of this slice; it remains a later challenger calibration method once real validation/calibration splits exist.

**Verification performed**

- `uv run pytest tests/test_training.py tests/test_explain.py -q`
- `uv run pytest -q`
- `uv run ruff check src tests`
- `git diff --check`
- `uv run fraud-train --synthetic --output-dir artifacts/model/latest`
- `make train-smoke`

**Reusable learnings**

- Smoke training should produce real artifacts, even when the model is deliberately simple.
- Artifact metadata should be first-class so serving, monitoring, and dashboard slices can report model governance fields without loading training code.
- A deterministic fallback explanation path is useful before SHAP artifacts are available.

## Task 6: Scoring Engine Shared By API And Kafka

**What changed**

- Added `ScoringEngine` to load a model bundle and apply the decision policy to a transaction event.
- Added conversion from `TransactionEvent` payloads into model feature rows.
- Added shared decision-event construction with model version, feature schema version, decision policy version, raw/calibrated probability, conformal prediction set, reason codes, and latency.
- Added a focused scoring test using the synthetic model artifact from Task 5.

**Problems faced**

- The first focused test run failed as expected because `fraud_platform.scoring` did not exist yet.
- The scoring slice needed to stay simple: calibrated probability currently equals raw probability, and the prediction set is threshold-derived until later calibration/conformal artifacts are wired in.
- Ruff flagged one long test line after the first green run.

**Solutions applied**

- Implemented a small scoring engine that depends only on existing contracts, artifacts, reason-code fallback, and policy logic.
- Kept `_simple_prediction_set()` explicit so later calibration/conformal implementations can replace it cleanly.
- Wrapped the long policy construction line and reran verification.

**Verification performed**

- `uv run pytest tests/test_scoring.py -q`
- `uv run pytest -q`
- `uv run ruff check src tests`
- `git diff --check`
- Manual smoke to train a temporary synthetic model, score a transaction, and serialize the resulting `DecisionEvent`.

**Reusable learnings**

- A shared scoring engine keeps API and Kafka consumers from duplicating model loading, feature mapping, policy application, and governance metadata.
- Early scoring slices can use simple uncertainty logic if the interface preserves where calibrated and conformal artifacts will plug in later.
- Scoring output should be validated as a serializable contract object before downstream storage or dashboard work begins.

## Task 7: FastAPI Scoring And Operations Endpoints

**What changed**

- Replaced the initial `fraud-api` placeholder with a real FastAPI app factory.
- Added environment-backed settings for local model, policy, Kafka, Postgres, and MLflow defaults.
- Added `/health`, `/model-info`, `/score`, and `/metrics` endpoints.
- Added Prometheus request and scoring-latency metrics.
- Added API tests using a synthetic model-backed `ScoringEngine`.

**Problems faced**

- The first focused API test failed as expected because the scaffold placeholder did not expose `create_app`.
- `TestClient` emits a third-party Starlette deprecation warning about `httpx`; tests still pass.
- The API must support dependency injection for tests while still loading the configured model and policy in normal runtime.

**Solutions applied**

- Added `create_app(scoring_engine: ScoringEngine | None = None)` so tests can inject a temporary synthetic model while runtime can load from settings.
- Used the strict `TransactionEvent` and `DecisionEvent` contracts directly as request/response models.
- Added `/metrics` using `prometheus_client.generate_latest()`.

**Verification performed**

- `uv run pytest tests/test_api.py -q`
- `uv run pytest -q`
- `uv run ruff check src tests`
- `git diff --check`
- Manual API smoke for `/health`, `/score`, `/metrics`, and invalid score payload validation.

**Reusable learnings**

- App factories make service tests much cleaner because expensive runtime dependencies can be injected.
- Reusing strict contracts at the API boundary gives immediate 422 responses for malformed scoring payloads.
- Metrics endpoints should be present early so service behavior can be observed as soon as the app runs.
