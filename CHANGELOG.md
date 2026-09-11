# Changelog

## 2026-09-11 — End-to-end reliability and reproducibility

- Added a deterministic synthetic demo and Compose overlay with isolated model and replay assets.
- Persisted synchronous API decisions so both scoring paths populate the operations console.
- Preserved dataset-relative transaction time through replay and online feature generation.
- Prevented enrichment maps from overriding canonical amount, product, and time fields.
- Required confirmed decision/dead-letter delivery before manual Kafka offset commits.
- Kept scoring and infrastructure failures uncommitted for recovery.
- Corrected the finite-sample conformal rank and preserved the raw-score scale at inference.
- Replaced fabricated dashboard fallback data with loading, empty, stale, and connection-error states.
- Added backend, frontend, persistence, delivery-failure, and opt-in live Kafka regression coverage.
- Added GitHub Actions checks, locked backend image installation, dependency health checks, and
  durable PostgreSQL storage; refreshed frontend dependencies.
- Reworked project documentation around architecture, measured results, reproducibility, and
  operating boundaries. Removed superseded planning and session-process documents; their history
  remains available in Git.

Existing real-data model metrics were not regenerated. Refit conformal artifacts to use the corrected
finite-sample quantile, and re-evaluate online behavior after the transaction-time parity fix.
