# Validation record

Local verification on 2026-09-11 used Python 3.11 on macOS and the repository lockfiles.

| Check | Result |
| --- | --- |
| `uv run ruff check src tests` | Passed |
| `uv run pytest --cov=fraud_platform --cov-report=term-missing` | 130 passed, 1 opt-in integration test skipped |
| Python statement coverage | 89% |
| `npm test` in `dashboard` | 2 passed |
| `npm run build` in `dashboard` | Passed TypeScript and Vite production build |
| `npm audit` after lockfile refresh | 0 known vulnerabilities reported |
| `uv run fraud-demo --rows 20` | Generated model, replay parquet, and manifest |
| Combined base/demo `docker compose config --quiet` | Passed |
| Browser check | Connected feed with 20 scored synthetic transactions; transaction drawer opened |

The local persistence regression runs synthetic training, HTTP scoring, SQLAlchemy persistence,
event-ID retry, and dashboard feed reads against a temporary SQLite database. Failure tests verify
that scoring, storage, delivery, and dead-letter publication failures do not commit input offsets.
Feature-parity coverage compares the model probability before and after event serialization.

The Docker daemon was unavailable on this machine. A full Compose deployment and live Kafka test
were therefore not executed locally. `.github/workflows/ci.yml` runs the opt-in Kafka test against
a real broker, in addition to separate backend and dashboard jobs. Check the CI run for remote
results; this file records local results only.

Existing dependency deprecation warnings and SHAP convergence warnings on small synthetic fixtures
remain visible in pytest output. No IEEE-CIS data or trained real-data artifacts were available for
retraining here. Historical model metrics are retained with their original provenance and are not
presented as newly reproduced results.

The screenshot in the README was captured from the local API/React preview with generated
synthetic decisions. It verifies rendering, not Kafka throughput or predictive performance.
