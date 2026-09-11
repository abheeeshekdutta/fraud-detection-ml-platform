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
Feature-parity coverage compares offline model probabilities with scoring after event construction.

## Remote deployment verification

[GitHub Actions run 34600316707](https://github.com/abheeeshekdutta/fraud-detection-ml-platform/actions/runs/34600316707)
passed all four jobs for implementation commit `d63889e`: backend, dashboard, Kafka, and Compose.

The Docker daemon was unavailable locally, so deployment verification ran on the clean Ubuntu CI
runner. The Compose job built the application images and started the full synthetic stack.
`scripts/smoke_compose.py` verified ten replayed transactions reached the persisted API feed through
Kafka, the scoring consumer, and PostgreSQL. It also submitted an HTTP scoring request, verified
its persisted result, checked the alerts endpoint, and fetched the nginx-served dashboard.
The independent Kafka test verified decision delivery, dead-letter routing, and committed offsets.

The browser check additionally confirmed that stopping the preview API produces a visible stale-data
warning while retaining the last successful feed. Temporary local preview services were stopped
when verification finished.

Existing dependency deprecation warnings and SHAP convergence warnings on small synthetic fixtures
remain visible in pytest output. No IEEE-CIS data or trained real-data artifacts were available for
retraining here. Historical model metrics are retained with their original provenance and are not
presented as newly reproduced results.

The screenshot in the README was captured from the local API/React preview with generated
synthetic decisions. It verifies rendering, not Kafka throughput or predictive performance.

## Docker Hub pull failure and CI recovery

The documentation-only run `34600560002` failed before application startup because the Docker Hub
OAuth token connection was reset while pulling `postgres:17-alpine`. Other image pulls were then
cancelled by Compose. This was a registry transport failure, not a PostgreSQL or scoring failure.

CI now prepares images separately from startup. Service pulls are serialized in the Compose job;
pulls and builds have at most four attempts with 5/10/20-second backoff. A persistent failure still
fails the job with the final exit code. Startup uses `--no-build --pull never`, and application
startup and tests are not retried. The Kafka job uses the same bounded image-pull recovery.
The retry helper has regression tests for immediate success, transient recovery, and exhaustion.

Python jobs share the restored uv cache, with only the backend job saving it to avoid parallel
cache-reservation warnings. Action versions were updated to remove Node.js 20 deprecation warnings.
