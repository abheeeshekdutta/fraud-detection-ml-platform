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
- Added `docs/demo-script.md` for a local system walkthrough.
- Updated the model card with implemented artifact metadata.
- Added deployment health checks and clarified that config tests still run when Docker is unavailable.

**Problems faced**

- Docker Compose could not be validated with `docker compose config` because `docker` is not installed or not on PATH in this environment.
- The docs needed to be honest that the current model is still a synthetic smoke artifact, not the full IEEE-CIS trained candidate.

**Solutions applied**

- Documented the smoke training command as the prerequisite before starting the full stack.
- Separated operator troubleshooting guidance from the local walkthrough flow.
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
- Walkthrough scripts help make system behavior easier to verify and explain.

## Task 14: IEEE-CIS Processing And Baseline Training Foundation

**What changed**

- Downloaded the IEEE-CIS Kaggle dataset into `data/raw`.
- Ran the EDA profiler and generated `docs/data-profile.md` plus report tables/charts under `reports/eda`.
- Added `prepare_ieee_cis_splits()` to create time-ordered train, calibration, validation, and replay Parquet splits.
- Added a real-data logistic baseline training path with `uv run fraud-train --ieee-baseline`.
- Trained the first IEEE-CIS baseline model artifact into `artifacts/model/latest`.
- Added `docs/ieee-cis-analysis.md` with EDA findings, split analysis, baseline metrics, and next modeling recommendations.
- Updated README, runbook, and model card with real-data baseline commands and metrics.

**Problems faced**

- The Kaggle CLI was not installed and the machine was not authenticated at first.
- The first `--prepare-ieee` CLI run wrote splits but exited incorrectly because the CLI did not return after preparation.
- The first logistic baseline emitted a convergence warning.
- Raw Kaggle data, processed Parquet files, and model artifacts are intentionally ignored, so the durable record of findings needed to live in docs and report files.

**Solutions applied**

- Used `uvx kaggle auth login` for OAuth authentication and `uvx kaggle competitions download` for dataset download.
- Added a regression test for the `--prepare-ieee` CLI mode and fixed the return path.
- Added numeric imputation/scaling and increased logistic regression iterations to produce a cleaner baseline fit.
- Captured the model results in `docs/ieee-cis-analysis.md` and the model card instead of committing ignored artifacts.

**Verification performed**

- `uv run pytest tests/test_ieee_pipeline.py tests/test_training.py -q`
- `uv run fraud-train --prepare-ieee --raw-dir data/raw --processed-dir data/processed`
- `uv run fraud-train --ieee-baseline --processed-dir data/processed --output-dir artifacts/model/latest --max-train-rows 100000`

**Reusable learnings**

- Real fraud datasets need time-based splits before model quality claims; random splits would hide drift.
- Identity coverage can be sparse in production-like fraud data, so identity features should be treated as optional enrichment.
- A quick logistic baseline is useful as a floor, but rare-event PR-AUC and segment risk show why CatBoost/LightGBM benchmarking is still needed.

## Task 15: MLflow Baseline Tracking

**What changed**

- Added optional MLflow tracking to the IEEE-CIS logistic baseline training path.
- Logged baseline parameters, validation metrics, and the sklearn model when
  `--mlflow-tracking-uri` is provided.
- Added a regression test that verifies training creates an MLflow experiment and run.
- Updated Docker Compose so the local MLflow server serves artifacts through HTTP instead of
  exposing a container-local artifact path to the host training process.
- Updated README, runbook, model card, and IEEE-CIS analysis docs with the MLflow-enabled command.

**Problems faced**

- The logistic regression model was not in MLflow because the previous baseline slice only saved the
  local model bundle and `training_summary.json`; MLflow run logging had not been implemented yet.
- The first live MLflow logging attempt created a run but failed while saving the model artifact.
- MLflow advertised `/mlflow/artifacts` as the artifact root, so the host-side training process tried
  to write to `/mlflow` on macOS and hit a read-only filesystem error.
- The recreated MLflow container took time to become ready because it installs MLflow at startup.

**Solutions applied**

- Added explicit MLflow setup in training: set tracking URI, set experiment, start a run, then log
  params, metrics, and the sklearn pipeline model.
- Used a SQLite MLflow backend in the unit test so the behavior can be verified without Docker.
- Switched the Compose MLflow command to `--serve-artifacts --artifacts-destination /mlflow/artifacts`
  so artifact upload goes through the tracking server.
- Verified readiness with a retry loop before rerunning live training.

**Verification performed**

- `uv run pytest tests/test_deployment_config.py tests/test_ieee_pipeline.py tests/test_training.py -q`
- `uv run ruff check src/fraud_platform/training.py tests/test_ieee_pipeline.py tests/test_deployment_config.py`
- `docker compose up -d --force-recreate mlflow`
- `curl -f http://localhost:5001`
- `uv run fraud-train --ieee-baseline --processed-dir data/processed --output-dir artifacts/model/latest --max-train-rows 100000 --mlflow-tracking-uri http://localhost:5001 --mlflow-experiment-name fraud-detection-ieee`
- Queried MLflow and confirmed run `537422fa43054c8b9c58c0c49ab867f6` finished with ROC-AUC 0.7023,
  PR-AUC 0.0977, and Brier score 0.0307.

**Reusable learnings**

- Saving a model artifact locally and logging a model to MLflow are separate responsibilities.
- Local Docker MLflow should proxy artifacts when training runs on the host, otherwise host processes
  may receive container-local paths they cannot write to.
- Experiment tracking should be added as soon as real baseline modeling begins so every comparison
  candidate has durable params, metrics, and artifacts.

## Task 16: IEEE-CIS Model Candidate Switch

**What changed**

- Added `--model-candidate` support to the IEEE-CIS training path.
- Kept `logistic_regression` as the default candidate.
- Added CatBoost and LightGBM candidate pipelines that use the same feature columns, artifact bundle,
  validation summary, and optional MLflow logging path.
- Added regression tests for CatBoost and LightGBM artifact generation.
- Updated README, runbook, modeling plan, and model card docs with the new candidate option.

**Problems faced**

- The first test run failed as expected because `train_ieee_baseline_model()` did not accept a model
  candidate argument yet.
- The first implementation pass briefly built the shared preprocessor with the wrong sklearn helper
  shape, which would not have worked as a column transformer.
- Ruff caught several long synthetic-data lines in the new regression test.

**Solutions applied**

- Added a shared model-candidate dispatch layer with explicit model version and model type mappings.
- Reused the existing preprocessing behavior for logistic regression and produced dense transformed
  features for CatBoost and LightGBM.
- Broke the test fixture values onto multiple lines and added an MLflow assertion for the logged
  model candidate.

**Verification performed**

- `uv run pytest tests/test_ieee_pipeline.py::test_train_ieee_baseline_model_supports_tree_candidates -q`
- `uv run pytest tests/test_deployment_config.py tests/test_ieee_pipeline.py tests/test_training.py -q`
- `uv run ruff check src/fraud_platform/training.py tests/test_ieee_pipeline.py tests/test_deployment_config.py`

**Reusable learnings**

- Adding model candidates through one artifact/logging path keeps benchmarking comparable and avoids
  separate one-off training scripts.
- For sklearn-compatible tree models, dense preprocessed features are the least surprising integration
  point when the existing baseline already owns categorical imputation and encoding.

## Task 17: First IEEE-CIS Candidate Benchmark Runs

**What changed**

- Ran CatBoost and LightGBM candidate training against the same 100,000-row IEEE-CIS training slice as
  the logistic baseline.
- Logged both candidate runs to the `fraud-detection-ieee` MLflow experiment.
- Wrote candidate artifacts under `artifacts/model/candidates/` for comparison without replacing the
  current `artifacts/model/latest` bundle.
- Updated `docs/ieee-cis-analysis.md` and `docs/model-card.md` with a candidate comparison table.

**Problems faced**

- Candidate artifacts should not overwrite the current promoted local model bundle during comparison.
- LightGBM emits a sklearn feature-name warning in this pipeline, but the run completes and produces
  validation metrics.

**Solutions applied**

- Used candidate-specific output directories:
  `artifacts/model/candidates/catboost` and `artifacts/model/candidates/lightgbm`.
- Kept `artifacts/model/latest` as the logistic baseline until a deliberate promotion step.
- Recorded MLflow run IDs in the analysis doc so the comparison is traceable.

**Verification performed**

- `uv run fraud-train --ieee-baseline --processed-dir data/processed --output-dir artifacts/model/candidates/catboost --max-train-rows 100000 --model-candidate catboost --mlflow-tracking-uri http://localhost:5001 --mlflow-experiment-name fraud-detection-ieee`
- `uv run fraud-train --ieee-baseline --processed-dir data/processed --output-dir artifacts/model/candidates/lightgbm --max-train-rows 100000 --model-candidate lightgbm --mlflow-tracking-uri http://localhost:5001 --mlflow-experiment-name fraud-detection-ieee`
- Queried MLflow and confirmed CatBoost run `1ae898956eea4b07b788ee3adc645ae0` and LightGBM run
  `bb9cd72672564d16bd2b6bef153129da`.
- LightGBM led the first comparison with ROC-AUC 0.7489, PR-AUC 0.1498, and Brier score 0.0297.

**Reusable learnings**

- Candidate benchmark docs should record both metrics and run IDs so future model-card updates are
  auditable.
- Do not promote a candidate just because it wins first-pass metrics; threshold, calibration, latency,
  and segment behavior still need checks.

## Task 18: Hyperparameter Logging And Time-Aware Tuning

**What changed**

- Logged model hyperparameters to MLflow with `model__` parameter names.
- Added `--tune-hyperparameters` and `--tuning-splits` to the IEEE-CIS training CLI.
- Added a small time-aware grid search using `TimeSeriesSplit`.
- Stored selected parameters and fold-level tuning metrics in `training_summary.json`.
- Updated README, runbook, and modeling plan docs with tuning guidance and remaining ML gaps.

**Problems faced**

- MLflow previously recorded model identity and validation metrics, but not model hyperparameters.
- The first tuning test failed because the training function had no tuning arguments.
- LightGBM emits feature-name warnings in small sklearn-pipeline fixtures.

**Solutions applied**

- Added explicit default parameter dictionaries for logistic regression, CatBoost, and LightGBM.
- Routed candidate construction through parameter dictionaries so defaults, tuning, summaries, and
  MLflow logging stay aligned.
- Used time-ordered folds rather than random cross-validation to respect fraud data chronology.
- Suppressed the known LightGBM fixture warning only in the affected tests.

**Verification performed**

- `uv run pytest tests/test_ieee_pipeline.py::test_train_ieee_baseline_model_logs_candidate_hyperparameters tests/test_ieee_pipeline.py::test_train_ieee_baseline_model_records_time_aware_tuning -q`
- `uv run pytest tests/test_deployment_config.py tests/test_ieee_pipeline.py tests/test_training.py -q`
- `uv run ruff check src/fraud_platform/training.py tests/test_ieee_pipeline.py tests/test_deployment_config.py`

**Reusable learnings**

- MLflow runs need both metrics and hyperparameters to support reproducible model comparison.
- Random cross-validation is usually the wrong default for transaction fraud; time-aware folds are a
  better baseline even before full backtesting is implemented.

## Task 19: Feature Engineering And Hyperparameter Documentation

**What changed**

- Added `docs/feature-engineering.md` to track current features, transformations, rationale, results,
  leakage rules, planned features, and open questions.
- Added `docs/hyperparameter-tuning.md` to track default parameters, MLflow logging, tuning strategy,
  search spaces, current benchmark results, and planned tuning work.
- Linked both docs from README and the modeling plan.

**Problems faced**

- Feature engineering and tuning decisions were spread across code, benchmark docs, and conversation
  context.
- The project needed a clear record of what has been implemented versus what remains planned.

**Solutions applied**

- Created living docs that separate current state, results, reasons, notes, and next steps.
- Explicitly documented that the current benchmark results are first-pass and untuned.
- Added leakage rules before implementing richer aggregate features.

**Verification performed**

- Reviewed current training code, feature transformer, modeling docs, and benchmark results.
- Ran markdown/diff checks before committing this documentation-only slice.

**Reusable learnings**

- ML projects need decision logs for features and tuning, not just metrics tables.
- Writing leakage rules before feature expansion helps prevent accidental future-information leakage.

## Task 20: Leakage-Safe Per-Transaction Features

**What changed**

- Added first-pass derived features for amount skew, decimal remainder, identity coverage,
  missingness, and event time.
- Moved feature derivation into the saved sklearn pipeline so training, tuning, and loaded-bundle
  prediction use the same raw transaction input path.
- Recorded feature counts and feature lists in `training_summary.json`.
- Updated the feature engineering and modeling docs to distinguish implemented features from planned
  aggregate and encoding work.

**Problems faced**

- The previous trainer selected model columns before pipeline fitting, which meant derived features
  would have to be duplicated in training and serving paths.
- Synthetic tests did not include `TransactionDT`, so the new event-time features needed fixture
  coverage.

**Solutions applied**

- Put `FraudFeatureTransformer` at the start of each candidate pipeline.
- Kept derived features limited to single-row values that are available at scoring time.
- Added regression coverage for the transformer output and feature lineage in the training summary.

**Verification performed**

- Added a failing transformer test before implementing the derived features.
- Reran focused feature, IEEE pipeline, and bundle prediction tests after implementation.

**Reusable learnings**

- Feature engineering belongs in the model bundle when the same logic must run in training and
  inference.
- Start with per-row features before historical aggregates; the leakage surface is smaller and easier
  to test.

## Task 21: Refreshed Candidate Benchmarks With Derived Features

**What changed**

- Reran logistic regression, CatBoost, and LightGBM on the 16-feature IEEE-CIS transaction feature
  set.
- Logged refreshed runs to the `fraud-detection-ieee` MLflow experiment.
- Fixed `training_summary.json` so it records the MLflow run ID after a logged training run.
- Updated benchmark tables in the IEEE-CIS analysis, model card, feature engineering log, and
  hyperparameter tuning log.

**Problems faced**

- The local training summaries did not include the MLflow run IDs, even though the runs were created.
- The docs still showed the raw-feature benchmark numbers after the derived feature pipeline landed.

**Solutions applied**

- Added a regression test that compares the summary run ID to the MLflow run ID.
- Returned the active MLflow run ID from the logging helper and rewrote the summary after logging.
- Regenerated all three candidate summaries before updating documentation.

**Verification performed**

- Logistic regression: ROC-AUC 0.7543, PR-AUC 0.1111, Brier score 0.0303, MLflow run
  `d1959d793cb74653ba634ec44859b323`.
- CatBoost: ROC-AUC 0.7526, PR-AUC 0.1309, Brier score 0.0300, MLflow run
  `a7077a78e4eb47e7a5b3d1d518679203`.
- LightGBM: ROC-AUC 0.7677, PR-AUC 0.1503, Brier score 0.0297, MLflow run
  `d27e4c3990234b6181d97ad606495352`.

**Reusable learnings**

- Benchmark docs should be regenerated from local summaries only after those summaries contain the
  run IDs needed for traceability.
- A small feature set can improve ranking metrics, but rare-event PR-AUC still needs tuning,
  calibration, threshold work, and stronger leakage-safe signal.

## Task 22: Threshold Analysis Report

**What changed**

- Added threshold-grid metrics for approve/review/block operating points.
- Added a `fraud-thresholds` CLI command that loads a saved model bundle, scores the validation split,
  and writes a local JSON threshold report.
- Ran an initial LightGBM threshold report using illustrative cost assumptions.
- Updated the runbook, IEEE-CIS analysis, model card, and tuning notes with threshold guidance.

**Problems faced**

- Generic thresholds such as `0.8` are not useful for the current uncalibrated rare-event model
  because most predicted probabilities are much lower.
- Pure utility sorting can prefer aggressive blocking when fraud-loss assumptions are high.

**Solutions applied**

- Chose threshold grids based on the observed LightGBM validation score distribution.
- Documented the top utility point as an analysis result, not a deployment recommendation.
- Called out the need for constraints on block precision, false-block rate, review capacity, and
  segment behavior.

**Verification performed**

- `uv run fraud-thresholds --processed-dir data/processed --model-dir artifacts/model/candidates/lightgbm-features --output-path reports/generated/lightgbm_threshold_analysis.json --approve-thresholds 0.01,0.02,0.03,0.04 --block-thresholds 0.05,0.08,0.10,0.15,0.20,0.30 --fraud-loss 500 --review-cost 5 --false-block-cost 25 --top-k 5`
- Top searched operating point: approve threshold 0.04, block threshold 0.05, approve rate 74.32%,
  review rate 6.45%, block rate 19.23%, block precision 9.57%, block recall 56.46%, false-block rate
  17.98%.

**Reusable learnings**

- Threshold grids should be based on the model's actual score distribution, especially before
  calibration.
- Utility-only threshold selection needs business guardrails; otherwise it can optimize for catching
  fraud while creating too many false blocks.

## Task 23: Constrained Threshold Selection

**What changed**

- Added constraint-aware threshold selection on top of the threshold-grid report.
- Added optional CLI guardrails for maximum false-block rate, maximum review rate, and minimum block
  precision.
- Reran the LightGBM threshold report with constraints and updated the runbook, IEEE-CIS analysis,
  model card, and tuning notes.

**Problems faced**

- The unconstrained utility winner blocked 19.23% of validation transactions and falsely blocked
  17.98% of legitimate transactions.
- The project needed a way to reject high-utility but operationally unacceptable threshold pairs.

**Solutions applied**

- Filtered sorted threshold reports by configurable business constraints.
- Kept both the unconstrained and constrained best points in the JSON report so the tradeoff is
  visible.
- Used constraints of `max_false_block_rate=0.02`, `max_review_rate=0.30`, and
  `min_block_precision=0.20` for the first constrained LightGBM pass.

**Verification performed**

- `uv run fraud-thresholds --processed-dir data/processed --model-dir artifacts/model/candidates/lightgbm-features --output-path reports/generated/lightgbm_threshold_analysis_constrained.json --approve-thresholds 0.01,0.02,0.03,0.04 --block-thresholds 0.05,0.08,0.10,0.15,0.20,0.30 --fraud-loss 500 --review-cost 5 --false-block-cost 25 --max-false-block-rate 0.02 --max-review-rate 0.30 --min-block-precision 0.20 --top-k 10`
- Constrained selected point: approve threshold 0.04, block threshold 0.15, approve rate 74.32%,
  review rate 23.67%, block rate 2.02%, block precision 24.26%, block recall 15.00%, false-block
  rate 1.58%.

**Reusable learnings**

- Keep unconstrained and constrained threshold choices side by side; the gap explains the business
  cost of guardrails.
- Constrained threshold results should be rerun after calibration because the current model scores are
  still raw probabilities.

## Task 24: Probability Calibration Artifact

**What changed**

- Added save/load helpers for `ProbabilityCalibrator`.
- Added a `fraud-calibrate` CLI command that fits a calibrator on `calibration.parquet`, saves the
  artifact, and writes a calibration summary.
- Updated threshold analysis to optionally use a saved calibrator.
- Ran isotonic and Platt calibration for the LightGBM candidate and documented the results.

**Problems faced**

- Calibration primitives existed, but there was no persisted artifact or CLI workflow.
- Calibration changed metrics differently: isotonic improved calibration error but slightly worsened
  Brier score, while Platt worsened both on this validation split.

**Solutions applied**

- Persisted fitted calibrators with a small typed load guard.
- Evaluated raw and calibrated scores on the validation split with Brier score and calibration error.
- Kept threshold analysis flexible by accepting an optional `--calibrator-path`.

**Verification performed**

- Isotonic: validation Brier 0.029763, calibration error 0.003989.
- Platt: validation Brier 0.030220, calibration error 0.007460.
- Raw LightGBM scores: validation Brier 0.029665, calibration error 0.005643.
- Isotonic constrained threshold point: approve threshold 0.04, block threshold 0.30, approve rate
  77.52%, review rate 20.31%, block rate 2.17%, block precision 24.70%, block recall 16.45%,
  false-block rate 1.69%.

**Reusable learnings**

- Calibration should be judged with more than one metric; improving calibration error can still worsen
  Brier score.
- Runtime scoring should not be wired to a calibrator until the artifact workflow and offline
  threshold reports are traceable.

## Task 25: Runtime Calibration Wiring

**What changed**

- Wired optional calibrator loading into `ScoringEngine`.
- Added `CALIBRATOR_PATH` settings support for the API.
- Added `--calibrator-path` support to the Kafka consumer CLI and Compose command.
- Updated docs to explain how runtime scoring loads the saved calibrator.

**Problems faced**

- The first consumer wiring test hung because `run_consumer()` is intentionally a long-running
  service loop.
- API and Kafka consumer startup paths needed to stay optional so local synthetic smoke runs still
  work without calibration artifacts.

**Solutions applied**

- Added focused scoring/API tests that assert calibrated probability comes from the loaded calibrator.
- Tested the consumer construction path by monkeypatching the long-running message loop.
- Kept calibrator loading optional; missing or empty `CALIBRATOR_PATH` preserves raw-probability
  scoring.

**Verification performed**

- `uv run pytest tests/test_scoring.py tests/test_api.py tests/test_streaming.py tests/test_deployment_config.py -q`

**Reusable learnings**

- Runtime artifact wiring needs tests at both the pure scoring layer and the app startup layer.
- Long-running service entry points should be tested by isolating construction from the polling loop.

## Task 26: Dashboard API Feed Endpoints And Decision Persistence

**What changed**

- Added `/predictions` and `/alerts` API endpoints backed by the existing repository layer.
- Persisted Kafka consumer scoring decisions through the existing prediction repository when
  `DATABASE_URL` is configured.
- Enabled CORS for the local dashboard origins.
- Updated README status and runbook troubleshooting notes for the live dashboard feed path.

**Problems faced**

- The React dashboard already called `/predictions` and `/alerts`, but the API did not expose those
  routes.
- The consumer emitted `fraud-decisions` to Kafka but did not save those decisions for the dashboard
  to read.
- Browser fetches from `localhost:5173` to `localhost:8000` needed explicit CORS handling.

**Solutions applied**

- Added focused API tests for stored prediction/alert responses and local dashboard CORS preflight.
- Added a streaming test that verifies scored consumer decisions are saved when a repository is
  configured.
- Kept `create_app()` injectable for tests while defaulting to SQLAlchemy repositories from
  `DATABASE_URL` in normal runtime.

**Verification performed**

- `uv run pytest tests/test_api.py -q`
- `uv run pytest tests/test_streaming.py tests/test_api.py -q`

**Reusable learnings**

- Dashboard fallback states should have matching API smoke endpoints so operators can distinguish
  empty data from connectivity problems quickly.
- Event-loop services are easier to extend when optional side effects, such as persistence, are
  injected into the message-processing helper rather than hidden in the polling loop.

## Task 27: Monitoring Worker Alert Persistence

**What changed**

- Replaced the placeholder `fraud-monitor` entrypoint with a real review-rate shift worker.
- Added monitoring settings for interval, prediction window size, reference review rate, and shift
  multiplier.
- Wired the Docker Compose monitoring-worker service with explicit monitoring environment values.
- Updated README, monitoring docs, and runbook notes for the implemented worker.

**Problems faced**

- The monitoring module had alert calculations, but the console entrypoint only printed a placeholder
  message.
- The raw Compose test reads YAML before environment interpolation, so deployment assertions need to
  check the configured default expression rather than the rendered value.

**Solutions applied**

- Added focused tests for one-shot monitoring checks that persist alerts and stay quiet when no
  predictions exist.
- Implemented `run_monitoring_check()` for testable business logic and `run_monitoring_loop()` for
  the long-running worker.
- Kept the first worker narrow: recent decision mix to `decision_rate_shift` alerts in Postgres.

**Verification performed**

- `uv run pytest tests/test_monitoring.py tests/test_deployment_config.py -q`
- `uv run ruff check src/fraud_platform/monitoring.py src/fraud_platform/config.py tests/test_monitoring.py tests/test_deployment_config.py`

**Reusable learnings**

- Long-running worker entrypoints should delegate to one-shot helpers so alert behavior can be tested
  without sleeping or starting Docker services.
- Compose tests that parse YAML directly should assert raw interpolation expressions; rendered values
  belong in `docker compose config` verification.

## Task 28: Delayed Label Replay Events

**What changed**

- Added a strict `FraudLabelEvent` contract for delayed fraud outcomes.
- Added optional label publishing to the replay producer when replay rows include `isFraud`.
- Wired the Compose transaction producer to publish labels to `fraud-labels` with
  `LABEL_DELAY_SECONDS`.
- Updated README, data contracts, and runbook notes for delayed label replay.

**Problems faced**

- The project declared a `fraud-labels` topic and delayed-label monitoring plans, but replay only
  emitted transaction events.
- Replay settings such as `REPLAY_DATA_PATH`, `REPLAY_SPEED_MULTIPLIER`, and `LABEL_DELAY_SECONDS`
  existed in `.env.example` but were not represented in typed settings.

**Solutions applied**

- Added failing streaming tests for label-event serialization and replay label publication.
- Kept label publication optional so replay files without labels still publish transactions normally.
- Added the missing replay settings to `Settings`.

**Verification performed**

- `uv run pytest tests/test_streaming.py tests/test_contracts.py tests/test_deployment_config.py -q`
- `uv run ruff check src/fraud_platform/contracts.py src/fraud_platform/replay.py src/fraud_platform/config.py tests/test_streaming.py tests/test_deployment_config.py`

**Reusable learnings**

- Declared Kafka topics should have at least a smoke producer or consumer path before they appear in
  operator docs.
- Environment variables in `.env.example` should be mirrored in typed settings to keep CLI defaults
  and Compose behavior aligned.

## Task 29: Model Alert Kafka Publication

**What changed**

- Added optional Kafka publication for monitoring alerts to `model-alerts`.
- Added typed `MODEL_ALERTS_TOPIC` settings support.
- Updated the Compose monitoring-worker command to pass Kafka bootstrap servers and alert topic.
- Updated monitoring docs and runbook language for dual Postgres/Kafka alert routing.

**Problems faced**

- The monitoring worker persisted alerts for the dashboard, but the declared `model-alerts` topic was
  still unused.
- The worker needed to remain testable without a real Kafka broker.

**Solutions applied**

- Added a focused monitoring test with a fake producer and verified serialized `AlertEvent` output.
- Kept producer creation optional and injected producer/topic into the one-shot helper.
- Flushed after alert publication so one-shot checks do not exit before the alert is handed to the
  client library.

**Verification performed**

- `uv run pytest tests/test_monitoring.py tests/test_deployment_config.py -q`
- `uv run ruff check src/fraud_platform/monitoring.py src/fraud_platform/config.py tests/test_monitoring.py tests/test_deployment_config.py`

**Reusable learnings**

- Kafka side effects should be injectable at the helper boundary so long-running services can be unit
  tested without broker dependencies.

## Task 30: Consumer Dead-Letter Routing

**What changed**

- Added a strict `DeadLetterEvent` contract.
- Updated the fraud consumer to publish malformed or unprocessable messages to `dead-letter-events`.
- Wired the Compose consumer command with `--dead-letter-topic dead-letter-events`.
- Updated README, data contracts, and runbook notes for dead-letter handling.

**Problems faced**

- The project declared a dead-letter topic, but consumer deserialization failures would crash or be
  skipped without an inspectable payload path.
- The consumer needed to commit failed messages after routing so a single bad payload would not stall
  local replay.

**Solutions applied**

- Added focused streaming tests for dead-letter serialization and invalid-payload routing.
- Wrapped message processing in a narrow `try` block and emitted `DeadLetterEvent` only when a topic
  is configured.
- Preserved existing success behavior and optional prediction persistence.

**Verification performed**

- `uv run pytest tests/test_streaming.py tests/test_contracts.py tests/test_deployment_config.py -q`
- `uv run ruff check src/fraud_platform/contracts.py src/fraud_platform/config.py src/fraud_platform/consumer.py tests/test_streaming.py tests/test_deployment_config.py`

**Reusable learnings**

- Stream consumers should make failure events explicit and serializable; skipped or crashing payloads
  are much harder to debug during local demos.

## Task 31: Conformal Artifact Workflow And Runtime Wiring

**What changed**

- Added save/load helpers for `SplitConformalClassifier`.
- Added `fit_conformal_artifact()` and the `fraud-conformal` CLI.
- Wired optional `CONFORMAL_PATH` settings into API and Kafka consumer scoring.
- Updated Compose, README, runbook, modeling docs, and model card notes.

**Problems faced**

- Conformal utilities existed, but runtime scoring still used threshold-derived prediction sets.
- The project needed a persisted conformal artifact workflow parallel to probability calibration.

**Solutions applied**

- Added failing tests for conformal artifact round-tripping and scoring-engine use.
- Fit conformal artifacts on the calibration split and summarize validation coverage.
- Kept `CONFORMAL_PATH` optional so synthetic smoke runs still work without extra artifacts.

**Verification performed**

- `uv run pytest tests/test_calibration_conformal.py tests/test_calibration_artifact.py tests/test_scoring.py tests/test_streaming.py tests/test_api.py tests/test_deployment_config.py -q`
- `uv run ruff check src/fraud_platform/conformal.py src/fraud_platform/scoring.py src/fraud_platform/config.py src/fraud_platform/api.py src/fraud_platform/consumer.py tests/test_calibration_conformal.py tests/test_calibration_artifact.py tests/test_scoring.py tests/test_streaming.py tests/test_deployment_config.py`

**Reusable learnings**

- Artifact workflows are easier to operate when calibration and uncertainty follow the same pattern:
  fit on calibration data, summarize validation behavior, then load optionally at runtime.

## Task 32: Global SHAP Explanation Artifact

**What changed**

- Added `fit_explanation_artifact()` and the `fraud-explain` CLI.
- Generated global SHAP feature-importance summaries for saved model bundles.
- Updated runbook, model card, and modeling docs to distinguish global SHAP artifacts from runtime
  deterministic reason codes.

**Problems faced**

- SHAP's permutation tabular masker attempted numeric closeness checks on string categorical columns.
- Runtime reason codes should not switch to SHAP output before stability and wording review.

**Solutions applied**

- Used SHAP KernelExplainer with a prediction wrapper that preserves DataFrame column names for the
  model pipeline.
- Kept the output compact as `global_shap_summary.json` with ranked mean absolute SHAP values.
- Left runtime reason codes deterministic while providing the global explanation artifact workflow.

**Verification performed**

- `uv run pytest tests/test_explain.py -q`
- `uv run ruff check src/fraud_platform/explain.py tests/test_explain.py pyproject.toml`

**Reusable learnings**

- Mixed categorical/numeric model pipelines need a SHAP wrapper that preserves the serving feature
  schema; generic tabular maskers can assume numeric arrays.

## Task 33: Local Monitoring Report Artifact

**What changed**

- Added `write_monitoring_report()` for local JSON drift and missingness summaries.
- Added the `fraud-monitor-report` CLI.
- Updated README, monitoring docs, and runbook guidance for offline monitoring reports.

**Problems faced**

- Evidently 0.7 uses a newer API than older examples, and the project needed a stable report artifact
  immediately.
- Operators still need a simple local report even before full scheduled Evidently templates are
  finalized.

**Solutions applied**

- Implemented a deterministic JSON report with row counts, missingness, numeric mean shifts, and
  categorical total variation distance.
- Kept Evidently documented as an available deeper reporting layer after the monitored production
  schema stabilizes.

**Verification performed**

- `uv run pytest tests/test_monitoring.py -q`
- `uv run ruff check src/fraud_platform/monitoring.py tests/test_monitoring.py pyproject.toml`

**Reusable learnings**

- For fast-moving monitoring libraries, a stable project-native report contract can keep operator
  workflows useful while richer third-party report templates evolve.
