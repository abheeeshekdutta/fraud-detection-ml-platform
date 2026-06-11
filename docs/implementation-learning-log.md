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
