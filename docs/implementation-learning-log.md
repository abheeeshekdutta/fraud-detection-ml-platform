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

## Task 8: Prediction And Alert Storage

**What changed**

- Added SQLAlchemy table definitions for prediction records and alert records.
- Added repository helpers for saving and reading latest prediction and alert rows.
- Added SQLite-backed unit tests for repository round trips.
- Added Task 8.5 to the implementation plan for IEEE-CIS EDA and data profiling before real model training.

**Problems faced**

- The first focused storage test failed as expected because repository/storage modules did not exist yet.
- Storage needs to preserve JSON fields such as conformal prediction sets, reason codes, and alert metadata for later dashboard and monitoring slices.
- The project plan needed an explicit EDA slice so full IEEE-CIS modeling is evidence-driven.

**Solutions applied**

- Used SQLAlchemy ORM models with JSON columns for structured prediction and alert metadata.
- Kept repository methods small: `save()` for upsert-like persistence and `latest()` for dashboard-friendly retrieval.
- Verified ordering by timestamp and JSON field round trips with a manual in-memory SQLite smoke.

**Verification performed**

- `uv run pytest tests/test_storage.py -q`
- `uv run pytest -q`
- `uv run ruff check src tests`
- `git diff --check`
- Manual SQLite smoke for prediction ordering, reason-code JSON, and alert metadata JSON.

**Reusable learnings**

- Prediction storage should persist governance metadata beside every score so downstream audit and dashboard views do not depend on model internals.
- JSON columns are useful for flexible reason codes and alert metadata, but strict upstream contracts are what keep them safe.
- EDA deserves its own reproducible slice before full benchmark modeling, not just an informal notebook afterthought.

## Task 8.5: IEEE-CIS EDA And Data Profiling

**What changed**

- Added a reusable `fraud_platform.data_profile` module for IEEE-CIS exploratory profiling.
- Added a CLI script, `scripts/profile_ieee_cis.py`, that reads raw IEEE-CIS CSVs from `data/raw`.
- Added generated-output support for EDA CSV tables, lightweight SVG charts, and `docs/data-profile.md`.
- Added tests for fraud imbalance, transaction amount summaries, ProductCD fraud rates, identity join coverage, missingness, categorical cardinality, time-window drift, leakage checks, missing-data handling, and output generation.

**Problems faced**

- The first focused EDA test failed as expected because the profiler module did not exist.
- The CLI initially printed a Python traceback when raw IEEE-CIS files were missing.
- A test import for `scripts/profile_ieee_cis.py` failed because root-level scripts are not installed as normal package modules.
- Ruff caught import-order and line-length issues during verification.

**Solutions applied**

- Moved the profiling calculations into an importable module under `src/fraud_platform`.
- Kept `scripts/profile_ieee_cis.py` as a thin command-line wrapper around the tested module.
- Converted missing raw data into a clean `SystemExit` message that tells the user which files are expected.
- Loaded the CLI script by file path in tests to match how the script is executed.

**Verification performed**

- `uv run pytest tests/test_data_profile.py -q`
- `uv run pytest -q`
- `uv run ruff check src tests scripts`
- `git diff --check`
- Manual missing-data CLI check with `uv run python scripts/profile_ieee_cis.py --raw-dir /tmp/fraud-platform-missing-data`

**Reusable learnings**

- EDA logic should be testable as ordinary Python functions, not locked inside notebooks or scripts.
- Profiling before full training helps choose CatBoost and LightGBM preprocessing strategies based on real missingness, cardinality, and drift.
- Missing-data failures are part of developer experience; clear CLI exits are better than stack traces for expected setup gaps.

## Task 9: Kafka Replay Producer And Fraud Consumer

**What changed**

- Added Kafka topic definitions for transaction events, fraud decisions, labels, model alerts, and dead-letter events.
- Added Pydantic event serialization and deserialization helpers.
- Replaced the replay placeholder with a Kafka replay producer that publishes processed transactions in `TransactionDT` order.
- Replaced the consumer placeholder with a scoring consumer that reads transaction events, scores them, and publishes fraud decisions.
- Added unit tests with fake Kafka producer/consumer clients so streaming behavior can be verified without Docker.
- Added a skipped Kafka integration test placeholder for the later Docker Compose stack.
- Added an execution runbook that explains data placement, script order, and expected outputs.

**Problems faced**

- The first streaming test failed as expected because `fraud_platform.streaming` did not exist yet.
- Replay and consumer tests then failed because the placeholder modules did not expose testable helpers.
- The first fake scoring engine returned a plain dictionary, but production serialization expects Pydantic contract objects.
- Ruff caught import-order formatting in the new streaming tests.

**Solutions applied**

- Added a small `streaming.py` helper around Pydantic JSON serialization.
- Kept replay logic in a reusable `replay_frame()` function, with Confluent Kafka only in the CLI wrapper path.
- Kept consumer logic in `consume_available_messages()` so unit tests can use fake clients while runtime uses real Kafka clients.
- Updated the fake scoring engine to return a real `DecisionEvent`, matching the production scoring contract.
- Documented the current execution chain in `docs/execution-runbook.md`.

**Verification performed**

- `uv run pytest tests/test_streaming.py -q`
- `uv run pytest -q`
- `uv run ruff check src tests scripts`
- `git diff --check`

**Reusable learnings**

- Streaming code is easier to test when Kafka clients are passed into small helper functions instead of hidden inside loops.
- Test doubles should honor real project contracts; returning a dict hid an unrealistic behavior that Pydantic serialization correctly rejected.
- A runbook should be created as soon as multiple scripts exist, otherwise the project becomes hard to operate even when the code works.

## Task 10: Monitoring Calculations And Alert Emission

**What changed**

- Replaced the monitoring placeholder with calculation helpers for missingness rate, conformal coverage, and decision-rate shift alerts.
- Added tests for column-level missingness, conformal coverage, positive alert emission, and quiet behavior below threshold.
- Kept the `fraud-monitor` entrypoint as a placeholder for the later scheduled worker/compose wiring.

**Problems faced**

- The first focused monitoring test failed as expected because the module did not expose the requested functions.
- Alert outputs need to be contract-compatible with the storage and dashboard slices, not just raw dictionaries.

**Solutions applied**

- Returned `AlertEvent` from the decision-rate shift detector so alerts can be saved through the existing repository later.
- Included reference and current review rates in alert metadata for analyst/debug visibility.
- Kept the calculations pure and dataframe-based so they can be reused by batch jobs, scheduled workers, or tests.

**Verification performed**

- `uv run pytest tests/test_monitoring.py -q`
- `uv run pytest -q`
- `uv run ruff check src tests scripts`
- `git diff --check`

**Reusable learnings**

- Monitoring checks are easiest to trust when the metric calculation and alert thresholding are separated into small pure functions.
- Alerts should use the same project contract as runtime events so storage, dashboards, and future Kafka publication do not need adapters.
- A conformal coverage monitor directly connects delayed labels back to the uncertainty layer, which is important for fraud review quality.

## Task 11: React Fraud Operations Dashboard

**What changed**

- Added a Vite, React, and TypeScript dashboard under `dashboard/`.
- Implemented the approved Analyst Queue layout with a KPI strip, decision feed, alerts panel, and transaction detail drawer.
- Added a dashboard API client for prediction and alert endpoints.
- Added fallback demo data so the UI remains usable before backend dashboard endpoints are wired.
- Added a Vitest render test for the dashboard shell.
- Updated the execution runbook with dashboard startup steps and expected UI outputs.

**Problems faced**

- The first dashboard test failed because Vitest needed a jsdom browser-like environment.
- TypeScript did not recognize Vite's `import.meta.env` until Vite client types were added.
- The dashboard initially produced unhandled fetch failures when the local API was not running.
- `npm audit` found a critical Vitest advisory in the original scaffold version.

**Solutions applied**

- Added `vite.config.ts` with the React plugin and jsdom test environment.
- Added Vite environment typing through `src/vite-env.d.ts`.
- Made the API client return empty arrays when backend requests fail, allowing fallback data to render cleanly.
- Upgraded Vitest to `4.1.8`, which cleared the audit report.
- Added `.superpowers/` to `.gitignore` because the visual companion creates local design-session files.

**Verification performed**

- `npm run test`
- `npm run build`
- `npm audit --audit-level=moderate`
- `uv run pytest -q`
- `uv run ruff check src tests scripts`
- `git diff --check`
- Browser preview at `http://127.0.0.1:5173/`
- Browser interaction check for opening the transaction detail drawer.

**Reusable learnings**

- Frontend tests need the same browser assumptions as the components; React render tests should declare jsdom explicitly.
- Local dashboards should handle offline backend endpoints as a normal development state, not as an unhandled error.
- Checking dependency audit output during scaffold creation is cheaper than cleaning up a vulnerable lockfile later.

## Task 12: Docker Compose And Observability

**What changed**

- Added backend and dashboard Dockerfiles.
- Added a Docker Compose stack for Kafka, Postgres, MLflow, fraud API, fraud consumer, replay producer, monitoring worker, Prometheus, Grafana, and dashboard.
- Added Postgres initialization SQL for prediction and alert storage tables.
- Added Prometheus scrape configuration for the fraud API metrics endpoint.
- Added Grafana datasource and dashboard provisioning for API request rate and p95 scoring latency.
- Added deployment configuration tests that parse Compose, Prometheus, Grafana, and Postgres config files.
- Added `.dockerignore` to keep local environments, node modules, generated data, and artifacts out of Docker build contexts.
- Updated deployment docs and the execution runbook with Compose startup guidance.

**Problems faced**

- The first focused deployment tests failed as expected because the Compose, Postgres, Prometheus, and Grafana files did not exist yet.
- Bitnami Kafka public image availability changed, so it was not a good default for a free local project.
- The local machine does not currently have `docker` or `promtool` available on PATH, so native Compose and Prometheus validators could not be run here.
- The full stack requires a model artifact before `fraud-api` can start successfully.

**Solutions applied**

- Used Confluent's `cp-kafka` image for a local KRaft Kafka broker.
- Added file-level tests for service names, ports, database URL wiring, SQL table definitions, Prometheus scrape targets, Grafana datasource, and dashboard expressions.
- Documented `uv run fraud-train --synthetic --output-dir artifacts/model/latest` as a prerequisite before `docker compose up --build`.
- Added mounted volumes for local artifacts and data so the API and replay producer can use host-generated outputs.

**Verification performed**

- `uv run pytest tests/test_deployment_config.py -q`
- `docker compose config` attempted but could not run because `docker` is not installed or not on PATH.
- `promtool check config monitoring/prometheus.yml` attempted but could not run because `promtool` is not installed or not on PATH.

**Reusable learnings**

- Deployment config should have lightweight parse tests even when Docker is unavailable in the current environment.
- Avoid image sources whose free/public availability has changed; deployment defaults should remain reproducible for learners.
- Compose docs should name startup prerequisites explicitly, especially when one service depends on a generated local artifact.

## Task 13: Documentation, End-To-End Smoke, And Project Polish

**What changed**

- Added a README quickstart with local startup commands and key service URLs.
- Updated README status to reflect the implemented smoke-path platform pieces.
- Added `docs/runbook.md` for local startup, health checks, and common operational issues.
- Added `docs/demo-script.md` for a recruiter/interviewer walkthrough.
- Updated the model card with implemented artifact metadata.
- Added deployment health checks and clarified that config tests still run when Docker is unavailable.

**Problems faced**

- Docker Compose could not be validated with `docker compose config` because `docker` is not installed or not on PATH in this environment.
- The docs needed to be honest that the current model is still a synthetic smoke artifact, not the full IEEE-CIS trained candidate.

**Solutions applied**

- Documented the smoke training command as the prerequisite before starting the full stack.
- Separated operator troubleshooting guidance from the recruiter-facing demo flow.
- Kept the README quickstart short and linked deeper docs for details.

**Verification performed**

- `uv run ruff check src tests scripts`
- `uv run pytest -q`
- `npm run build`
- `npm run test`
- `docker compose config` attempted but could not run because `docker` is not installed or not on PATH.

**Reusable learnings**

- A final polish slice should make the project teachable, not just runnable.
- Documentation should clearly distinguish smoke-test artifacts from final modeling goals so readers do not overestimate current model maturity.
- Demo scripts help turn engineering work into a coherent story for portfolio reviews and interviews.
