# Fraud Detection Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the local, production-shaped fraud detection ML platform described in the committed design docs, from repo scaffold through offline modeling, real-time scoring, monitoring, and analyst dashboard.

**Architecture:** Use a Python 3.11 package as the shared backend core for data contracts, feature transforms, model artifacts, decision policy, storage, streaming, API, and monitoring. Implement the platform in vertical slices so contracts and decision behavior are tested first, then reused by FastAPI, Kafka services, model training, and the React operations console.

**Tech Stack:** Python 3.11, uv, pytest, Pandas, scikit-learn, CatBoost, LightGBM, MAPIE, SHAP, MLflow OSS, FastAPI, SQLAlchemy, PostgreSQL, confluent-kafka, Apache Kafka KRaft, Evidently OSS, Prometheus, Grafana OSS, React, Vite, TypeScript, Docker Compose.

---

## Source Requirements

This plan implements the requirements from:

- `README.md`
- `docs/superpowers/specs/2026-06-10-fraud-detection-platform-design.md`
- `docs/architecture.md`
- `docs/data-contracts.md`
- `docs/modeling.md`
- `docs/monitoring.md`
- `docs/deployment.md`
- `docs/model-card.md`

The current repository is docs-only, so Task 1 creates the working project scaffold.

## Planned File Structure

### Python Project And Tooling

- Create: `.python-version` - pins Python `3.11`.
- Create: `pyproject.toml` - uv project metadata, runtime dependencies, dev dependencies, pytest config, Ruff config.
- Create: `.env.example` - local defaults for Kafka, Postgres, MLflow, model paths, policy paths, and replay speed.
- Create: `.gitignore` - excludes local data, artifacts, virtualenvs, caches, node modules, and generated reports.
- Create: `Makefile` - common local commands for install, test, lint, train-smoke, compose-up, compose-down.

### Backend Package

- Create: `src/fraud_platform/__init__.py` - package marker and version.
- Create: `src/fraud_platform/config.py` - typed settings loaded from environment.
- Create: `src/fraud_platform/contracts.py` - Pydantic event, decision, reason-code, and alert schemas.
- Create: `src/fraud_platform/policy.py` - decision policy config and approve/review/block logic.
- Create: `src/fraud_platform/features/__init__.py` - feature package marker.
- Create: `src/fraud_platform/features/ieee.py` - IEEE-CIS loading, joining, time split, and replay-event conversion.
- Create: `src/fraud_platform/features/transformers.py` - shared offline/online feature transformation.
- Create: `src/fraud_platform/metrics.py` - PR-AUC, recall-at-precision, expected utility, calibration error, latency helpers.
- Create: `src/fraud_platform/calibration.py` - Platt/isotonic calibration wrapper.
- Create: `src/fraud_platform/conformal.py` - split conformal prediction-set wrapper.
- Create: `src/fraud_platform/explain.py` - SHAP reason-code adapter with deterministic fallback for tests.
- Create: `src/fraud_platform/artifacts.py` - model bundle load/save and metadata contract.
- Create: `src/fraud_platform/scoring.py` - scoring engine used by API and Kafka consumer.
- Create: `src/fraud_platform/training.py` - train/evaluate/package pipeline entrypoint.
- Create: `src/fraud_platform/storage.py` - SQLAlchemy engine/session setup and table definitions.
- Create: `src/fraud_platform/repositories.py` - prediction and alert persistence helpers.
- Create: `src/fraud_platform/api.py` - FastAPI app with score, health, model-info, predictions, alerts, metrics.
- Create: `src/fraud_platform/streaming.py` - Kafka producer/consumer helpers and dead-letter behavior.
- Create: `src/fraud_platform/replay.py` - transaction replay producer.
- Create: `src/fraud_platform/consumer.py` - fraud scoring Kafka consumer.
- Create: `src/fraud_platform/monitoring.py` - monitoring calculations and alert emission.

### Config, Schemas, And Operations

- Create: `configs/decision_policy.yaml` - thresholds, conformal labels, and policy version.
- Create: `configs/kafka_topics.yaml` - topic names, partitions, replication factor.
- Create: `configs/feature_schema_v1.yaml` - serving-safe feature groups and target-leakage exclusions.
- Create: `docker-compose.yml` - Kafka, Postgres, MLflow, fraud API, fraud consumer, transaction producer, monitoring worker, Prometheus, Grafana, dashboard.
- Create: `docker/backend.Dockerfile` - backend service image.
- Create: `docker/dashboard.Dockerfile` - dashboard service image.
- Create: `docker/postgres/init.sql` - local prediction and alert tables if migrations are not introduced.
- Create: `monitoring/prometheus.yml` - Prometheus scrape configuration.
- Create: `monitoring/grafana/provisioning/datasources/prometheus.yml` - Grafana datasource.
- Create: `monitoring/grafana/provisioning/dashboards/dashboards.yml` - Grafana dashboard loader.
- Create: `monitoring/grafana/dashboards/fraud-platform.json` - starter operational dashboard.

### Dashboard

- Create: `dashboard/package.json` - React/Vite app scripts and dependencies.
- Create: `dashboard/tsconfig.json` - strict TypeScript config.
- Create: `dashboard/index.html` - Vite mount point.
- Create: `dashboard/src/main.tsx` - React root.
- Create: `dashboard/src/App.tsx` - operations console shell and data loading.
- Create: `dashboard/src/api.ts` - typed API client.
- Create: `dashboard/src/types.ts` - dashboard DTOs aligned to backend contracts.
- Create: `dashboard/src/styles.css` - restrained financial operations visual system.
- Create: `dashboard/src/components/DecisionFeed.tsx` - live transaction table.
- Create: `dashboard/src/components/TransactionDrawer.tsx` - transaction detail and reason codes.
- Create: `dashboard/src/components/KpiStrip.tsx` - approve/review/block, latency, throughput cards.
- Create: `dashboard/src/components/AlertPanel.tsx` - model and service alerts.

### Tests

- Create: `tests/conftest.py` - shared fixtures and synthetic IEEE-CIS-like data.
- Create: `tests/test_contracts.py` - schema validation tests.
- Create: `tests/test_policy.py` - decision policy tests.
- Create: `tests/test_features.py` - time split, join, validation, and transformation tests.
- Create: `tests/test_metrics.py` - model metric and utility tests.
- Create: `tests/test_calibration_conformal.py` - calibration and prediction-set tests.
- Create: `tests/test_scoring.py` - scoring engine smoke tests.
- Create: `tests/test_api.py` - FastAPI contract tests.
- Create: `tests/test_storage.py` - repository tests using SQLite for fast unit coverage.
- Create: `tests/test_streaming.py` - serialization/dead-letter unit tests.
- Create: `tests/test_monitoring.py` - drift, missingness, rate-change, coverage, and alert tests.
- Create: `tests/integration/test_kafka_flow.py` - optional Docker Compose Kafka flow, marked `integration`.
- Create: `dashboard/src/App.test.tsx` - dashboard render smoke test.

## Task 1: Project Scaffold

**Files:**

- Create: `.python-version`
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `Makefile`
- Create: `src/fraud_platform/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create Python version pin**

Create `.python-version`:

```text
3.11
```

- [ ] **Step 2: Create Python package configuration**

Create `pyproject.toml`:

```toml
[project]
name = "fraud-detection-ml-platform"
version = "0.1.0"
description = "Local production-shaped fraud detection ML platform"
readme = "README.md"
requires-python = ">=3.11,<3.12"
dependencies = [
  "catboost>=1.2.8",
  "confluent-kafka>=2.8.0",
  "evidently>=0.6.7",
  "fastapi>=0.115.0",
  "lightgbm>=4.5.0",
  "mapie>=0.9.0",
  "mlflow>=2.18.0",
  "numpy>=1.26.0",
  "pandas>=2.2.0",
  "prometheus-client>=0.21.0",
  "psycopg[binary]>=3.2.0",
  "pydantic>=2.10.0",
  "pydantic-settings>=2.6.0",
  "python-dotenv>=1.0.1",
  "pyyaml>=6.0.2",
  "scikit-learn>=1.5.0",
  "shap>=0.46.0",
  "sqlalchemy>=2.0.0",
  "uvicorn[standard]>=0.32.0",
]

[project.optional-dependencies]
dev = [
  "httpx>=0.28.0",
  "pytest>=8.3.0",
  "pytest-cov>=6.0.0",
  "ruff>=0.8.0",
]

[project.scripts]
fraud-api = "fraud_platform.api:main"
fraud-train = "fraud_platform.training:main"
fraud-replay = "fraud_platform.replay:main"
fraud-consumer = "fraud_platform.consumer:main"
fraud-monitor = "fraud_platform.monitoring:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/fraud_platform"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
markers = [
  "integration: requires Docker Compose services",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

- [ ] **Step 3: Create local environment example**

Create `.env.example`:

```bash
APP_ENV=local
LOG_LEVEL=INFO

KAFKA_BOOTSTRAP_SERVERS=localhost:9092
TRANSACTION_EVENTS_TOPIC=transaction-events
FRAUD_DECISIONS_TOPIC=fraud-decisions
FRAUD_LABELS_TOPIC=fraud-labels
MODEL_ALERTS_TOPIC=model-alerts
DEAD_LETTER_EVENTS_TOPIC=dead-letter-events

DATABASE_URL=postgresql+psycopg://fraud:fraud@localhost:5432/fraud
MLFLOW_TRACKING_URI=http://localhost:5000

MODEL_BUNDLE_PATH=artifacts/model/latest
DECISION_POLICY_PATH=configs/decision_policy.yaml
FEATURE_SCHEMA_PATH=configs/feature_schema_v1.yaml

REPLAY_DATA_PATH=data/processed/replay.parquet
REPLAY_SPEED_MULTIPLIER=60
LABEL_DELAY_SECONDS=30
```

- [ ] **Step 4: Create ignore rules**

Create `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/

data/raw/
data/interim/
data/processed/
artifacts/
mlruns/
reports/generated/

node_modules/
dashboard/dist/
```

- [ ] **Step 5: Create common commands**

Create `Makefile`:

```makefile
.PHONY: install test lint format train-smoke api compose-up compose-down dashboard

install:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

train-smoke:
	uv run fraud-train --synthetic --output-dir artifacts/model/latest

api:
	uv run uvicorn fraud_platform.api:create_app --factory --reload --host 0.0.0.0 --port 8000

compose-up:
	docker compose up --build

compose-down:
	docker compose down -v

dashboard:
	cd dashboard && npm install && npm run dev -- --host 0.0.0.0
```

- [ ] **Step 6: Create package marker**

Create `src/fraud_platform/__init__.py`:

```python
__version__ = "0.1.0"
```

- [ ] **Step 7: Create synthetic test fixtures**

Create `tests/conftest.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def synthetic_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "TransactionDT": [10, 20, 30, 40, 50, 60],
            "TransactionAmt": [20.0, 200.0, 35.0, 500.0, 75.0, 900.0],
            "ProductCD": ["W", "C", "W", "R", "H", "C"],
            "card1": [1001, 1002, 1001, 1003, 1004, 1002],
            "addr1": [100.0, 200.0, 100.0, None, 300.0, 200.0],
            "P_emaildomain": ["a.test", "b.test", None, "c.test", "a.test", "b.test"],
            "isFraud": [0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def synthetic_identity() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 4],
            "DeviceType": ["desktop", "mobile", "mobile"],
            "id_31": ["chrome", "safari", "firefox"],
        }
    )
```

- [ ] **Step 8: Verify scaffold**

Run:

```bash
uv sync --extra dev
uv run pytest
```

Expected:

```text
no tests ran
```

- [ ] **Step 9: Commit**

```bash
git add .python-version pyproject.toml .env.example .gitignore Makefile src/fraud_platform/__init__.py tests/conftest.py
git commit -m "chore: scaffold python project"
```

## Task 2: Data Contracts And Decision Policy

**Files:**

- Create: `src/fraud_platform/contracts.py`
- Create: `src/fraud_platform/policy.py`
- Create: `configs/decision_policy.yaml`
- Test: `tests/test_contracts.py`
- Test: `tests/test_policy.py`

- [ ] **Step 1: Write contract tests**

Create `tests/test_contracts.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from fraud_platform.contracts import (
    AlertEvent,
    DecisionEvent,
    ReasonCode,
    TransactionEvent,
)


def test_transaction_event_accepts_production_safe_payload() -> None:
    event = TransactionEvent(
        event_id="evt-1",
        transaction_id=2987000,
        event_time=datetime(2026, 6, 10, 12, tzinfo=UTC),
        amount=68.5,
        product_cd="W",
        card_features={"card1": 1001},
        address_features={"addr1": 100.0},
        email_domain_features={"P_emaildomain": "example.test"},
        identity_features={},
        schema_version="v1",
    )

    assert event.transaction_id == 2987000
    assert event.schema_version == "v1"


def test_transaction_event_rejects_negative_amount() -> None:
    with pytest.raises(ValidationError):
        TransactionEvent(
            event_id="evt-1",
            transaction_id=1,
            event_time=datetime(2026, 6, 10, 12, tzinfo=UTC),
            amount=-1.0,
            product_cd="W",
            schema_version="v1",
        )


def test_decision_event_contains_governance_metadata() -> None:
    decision = DecisionEvent(
        event_id="evt-1",
        transaction_id=2987000,
        scored_at=datetime(2026, 6, 10, 12, 0, 1, tzinfo=UTC),
        model_version="fraud-model:1",
        feature_schema_version="v1",
        decision_policy_version="v1",
        fraud_probability=0.82,
        calibrated_probability=0.76,
        conformal_prediction_set=["fraud"],
        uncertainty="low",
        decision="block",
        reason_codes=[ReasonCode(feature="TransactionAmt", direction="increases_risk")],
        latency_ms=42.0,
    )

    assert decision.decision == "block"
    assert decision.reason_codes[0].direction == "increases_risk"


def test_decision_event_rejects_invalid_decision() -> None:
    with pytest.raises(ValidationError):
        DecisionEvent(
            event_id="evt-1",
            transaction_id=2987000,
            scored_at=datetime(2026, 6, 10, 12, 0, 1, tzinfo=UTC),
            model_version="fraud-model:1",
            feature_schema_version="v1",
            decision_policy_version="v1",
            fraud_probability=0.5,
            calibrated_probability=0.5,
            conformal_prediction_set=["legit", "fraud"],
            uncertainty="high",
            decision="escalate",
            latency_ms=42.0,
        )


def test_alert_event_schema() -> None:
    alert = AlertEvent(
        alert_id="alert-1",
        created_at=datetime(2026, 6, 10, 12, tzinfo=UTC),
        severity="warning",
        alert_type="review_rate_shift",
        message="Review rate increased from 0.10 to 0.25",
        metadata={"current_rate": 0.25},
    )

    assert alert.severity == "warning"
```

- [ ] **Step 2: Write policy tests**

Create `tests/test_policy.py`:

```python
from __future__ import annotations

from fraud_platform.policy import DecisionPolicy, PolicyConfig


def test_policy_approves_low_risk_legit_set() -> None:
    policy = DecisionPolicy(
        PolicyConfig(
            version="v1",
            approve_threshold=0.2,
            block_threshold=0.8,
            conformal_alpha=0.1,
        )
    )

    result = policy.decide(calibrated_probability=0.05, prediction_set=["legit"])

    assert result.decision == "approve"
    assert result.uncertainty == "low"


def test_policy_blocks_high_risk_fraud_set() -> None:
    policy = DecisionPolicy(PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8))

    result = policy.decide(calibrated_probability=0.95, prediction_set=["fraud"])

    assert result.decision == "block"
    assert result.uncertainty == "low"


def test_policy_reviews_ambiguous_conformal_set() -> None:
    policy = DecisionPolicy(PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8))

    result = policy.decide(calibrated_probability=0.5, prediction_set=["legit", "fraud"])

    assert result.decision == "review"
    assert result.uncertainty == "high"


def test_policy_reviews_empty_conformal_set() -> None:
    policy = DecisionPolicy(PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8))

    result = policy.decide(calibrated_probability=0.95, prediction_set=[])

    assert result.decision == "review"
    assert result.uncertainty == "high"


def test_policy_reviews_probability_threshold_mismatch() -> None:
    policy = DecisionPolicy(PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8))

    result = policy.decide(calibrated_probability=0.7, prediction_set=["fraud"])

    assert result.decision == "review"
    assert result.uncertainty == "medium"
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_contracts.py tests/test_policy.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'fraud_platform.contracts'`.

- [ ] **Step 4: Implement schemas**

Create `src/fraud_platform/contracts.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["approve", "review", "block"]
Uncertainty = Literal["low", "medium", "high"]
ReasonDirection = Literal["increases_risk", "decreases_risk"]
Severity = Literal["info", "warning", "critical"]


class TransactionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    transaction_id: int = Field(gt=0)
    event_time: datetime
    amount: float = Field(ge=0)
    product_cd: str = Field(min_length=1)
    card_features: dict[str, Any] = Field(default_factory=dict)
    address_features: dict[str, Any] = Field(default_factory=dict)
    email_domain_features: dict[str, Any] = Field(default_factory=dict)
    identity_features: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = Field(min_length=1)


class ReasonCode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str = Field(min_length=1)
    direction: ReasonDirection
    contribution: float | None = None


class DecisionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    transaction_id: int = Field(gt=0)
    scored_at: datetime
    model_version: str = Field(min_length=1)
    feature_schema_version: str = Field(min_length=1)
    decision_policy_version: str = Field(min_length=1)
    fraud_probability: float = Field(ge=0, le=1)
    calibrated_probability: float = Field(ge=0, le=1)
    conformal_prediction_set: list[Literal["legit", "fraud"]]
    uncertainty: Uncertainty
    decision: Decision
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    latency_ms: float = Field(ge=0)


class AlertEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alert_id: str = Field(min_length=1)
    created_at: datetime
    severity: Severity
    alert_type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 5: Implement policy**

Create `src/fraud_platform/policy.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from fraud_platform.contracts import Decision, Uncertainty


class PolicyConfig(BaseModel):
    version: str = Field(min_length=1)
    approve_threshold: float = Field(default=0.2, ge=0, le=1)
    block_threshold: float = Field(default=0.8, ge=0, le=1)
    conformal_alpha: float = Field(default=0.1, gt=0, lt=1)


@dataclass(frozen=True)
class PolicyDecision:
    decision: Decision
    uncertainty: Uncertainty


class DecisionPolicy:
    def __init__(self, config: PolicyConfig) -> None:
        if config.approve_threshold >= config.block_threshold:
            raise ValueError("approve_threshold must be lower than block_threshold")
        self.config = config

    def decide(self, calibrated_probability: float, prediction_set: list[str]) -> PolicyDecision:
        labels = set(prediction_set)
        if labels == {"legit"} and calibrated_probability <= self.config.approve_threshold:
            return PolicyDecision(decision="approve", uncertainty="low")
        if labels == {"fraud"} and calibrated_probability >= self.config.block_threshold:
            return PolicyDecision(decision="block", uncertainty="low")
        if labels != {"legit"} and labels != {"fraud"}:
            return PolicyDecision(decision="review", uncertainty="high")
        return PolicyDecision(decision="review", uncertainty="medium")


def load_policy(path: str | Path) -> DecisionPolicy:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    return DecisionPolicy(PolicyConfig(**payload))
```

- [ ] **Step 6: Create default policy config**

Create `configs/decision_policy.yaml`:

```yaml
version: v1
approve_threshold: 0.20
block_threshold: 0.80
conformal_alpha: 0.10
```

- [ ] **Step 7: Run tests**

Run:

```bash
uv run pytest tests/test_contracts.py tests/test_policy.py -q
```

Expected:

```text
9 passed
```

- [ ] **Step 8: Commit**

```bash
git add src/fraud_platform/contracts.py src/fraud_platform/policy.py configs/decision_policy.yaml tests/test_contracts.py tests/test_policy.py
git commit -m "feat: add fraud event contracts and decision policy"
```

## Task 3: IEEE-CIS Loading, Time Split, And Feature Transformations

**Files:**

- Create: `configs/feature_schema_v1.yaml`
- Create: `src/fraud_platform/features/__init__.py`
- Create: `src/fraud_platform/features/ieee.py`
- Create: `src/fraud_platform/features/transformers.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write feature tests**

Create `tests/test_features.py`:

```python
from __future__ import annotations

import pandas as pd

from fraud_platform.features.ieee import (
    build_transaction_event,
    join_transaction_identity,
    time_ordered_split,
    validate_training_frame,
)
from fraud_platform.features.transformers import FraudFeatureTransformer


def test_join_preserves_transactions_with_missing_identity(
    synthetic_transactions: pd.DataFrame,
    synthetic_identity: pd.DataFrame,
) -> None:
    joined = join_transaction_identity(synthetic_transactions, synthetic_identity)

    assert len(joined) == len(synthetic_transactions)
    assert joined.loc[joined["TransactionID"] == 3, "DeviceType"].isna().all()


def test_time_ordered_split_uses_transaction_dt(synthetic_transactions: pd.DataFrame) -> None:
    splits = time_ordered_split(
        synthetic_transactions,
        train_fraction=0.50,
        calibration_fraction=0.17,
        validation_fraction=0.17,
    )

    assert splits.train["TransactionDT"].max() < splits.calibration["TransactionDT"].min()
    assert splits.calibration["TransactionDT"].max() < splits.validation["TransactionDT"].min()
    assert splits.validation["TransactionDT"].max() < splits.replay["TransactionDT"].min()


def test_validate_training_frame_requires_expected_columns(
    synthetic_transactions: pd.DataFrame,
) -> None:
    result = validate_training_frame(synthetic_transactions)

    assert result.valid is True
    assert result.errors == []


def test_transformer_produces_stable_feature_columns(
    synthetic_transactions: pd.DataFrame,
    synthetic_identity: pd.DataFrame,
) -> None:
    joined = join_transaction_identity(synthetic_transactions, synthetic_identity)
    transformer = FraudFeatureTransformer()

    features = transformer.fit_transform(joined)

    assert list(features.columns) == [
        "TransactionAmt",
        "ProductCD",
        "card1",
        "addr1",
        "P_emaildomain",
        "DeviceType",
        "id_31",
    ]
    assert features["ProductCD"].dtype.name == "category"


def test_build_transaction_event_maps_serving_safe_groups(
    synthetic_transactions: pd.DataFrame,
    synthetic_identity: pd.DataFrame,
) -> None:
    joined = join_transaction_identity(synthetic_transactions, synthetic_identity)
    event = build_transaction_event(joined.iloc[0], event_time_base="2026-06-10T12:00:00Z")

    assert event.amount == 20.0
    assert event.card_features == {"card1": 1001}
    assert event.identity_features == {"DeviceType": "desktop", "id_31": "chrome"}
    assert event.schema_version == "v1"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_features.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'fraud_platform.features'`.

- [ ] **Step 3: Create feature schema config**

Create `configs/feature_schema_v1.yaml`:

```yaml
version: v1
label_column: isFraud
event_time_column: TransactionDT
join_key: TransactionID
amount_column: TransactionAmt
product_column: ProductCD
serving_features:
  base:
    - TransactionAmt
    - ProductCD
  card:
    - card1
  address:
    - addr1
  email_domain:
    - P_emaildomain
  identity:
    - DeviceType
    - id_31
excluded_from_serving:
  - isFraud
  - TransactionDT
  - TransactionID
```

- [ ] **Step 4: Implement IEEE helpers**

Create `src/fraud_platform/features/__init__.py`:

```python
"""Feature loading and transformation helpers."""
```

Create `src/fraud_platform/features/ieee.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import pandas as pd

from fraud_platform.contracts import TransactionEvent


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str]


@dataclass(frozen=True)
class TimeSplits:
    train: pd.DataFrame
    calibration: pd.DataFrame
    validation: pd.DataFrame
    replay: pd.DataFrame


REQUIRED_COLUMNS = {"TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "isFraud"}


def join_transaction_identity(
    transactions: pd.DataFrame,
    identity: pd.DataFrame | None,
) -> pd.DataFrame:
    if identity is None or identity.empty:
        return transactions.copy()
    if identity["TransactionID"].duplicated().any():
        raise ValueError("identity TransactionID values must be unique")
    return transactions.merge(identity, on="TransactionID", how="left", validate="one_to_one")


def validate_training_frame(frame: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        errors.append(f"missing required columns: {missing}")
    if "isFraud" in frame and not set(frame["isFraud"].dropna().unique()).issubset({0, 1}):
        errors.append("isFraud must contain only 0 and 1")
    if "TransactionDT" in frame and not frame["TransactionDT"].is_monotonic_increasing:
        ordered = frame.sort_values("TransactionDT")["TransactionDT"].tolist()
        if ordered != frame["TransactionDT"].tolist():
            errors.append("TransactionDT must be sorted for final split inputs")
    return ValidationResult(valid=not errors, errors=errors)


def time_ordered_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.15,
    validation_fraction: float = 0.15,
) -> TimeSplits:
    if train_fraction + calibration_fraction + validation_fraction >= 1:
        raise ValueError("fractions must leave at least one replay segment")
    ordered = frame.sort_values("TransactionDT").reset_index(drop=True)
    n_rows = len(ordered)
    train_end = max(1, int(n_rows * train_fraction))
    calibration_end = max(train_end + 1, int(n_rows * (train_fraction + calibration_fraction)))
    validation_end = max(
        calibration_end + 1,
        int(n_rows * (train_fraction + calibration_fraction + validation_fraction)),
    )
    validation_end = min(validation_end, n_rows - 1)
    return TimeSplits(
        train=ordered.iloc[:train_end].copy(),
        calibration=ordered.iloc[train_end:calibration_end].copy(),
        validation=ordered.iloc[calibration_end:validation_end].copy(),
        replay=ordered.iloc[validation_end:].copy(),
    )


def _clean_mapping(values: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in values.items():
        if pd.isna(value):
            continue
        cleaned[key] = value.item() if hasattr(value, "item") else value
    return cleaned


def build_transaction_event(row: pd.Series, event_time_base: str) -> TransactionEvent:
    base_time = datetime.fromisoformat(event_time_base.replace("Z", "+00:00"))
    event_time = base_time + timedelta(seconds=int(row["TransactionDT"]))
    return TransactionEvent(
        event_id=str(uuid4()),
        transaction_id=int(row["TransactionID"]),
        event_time=event_time,
        amount=float(row["TransactionAmt"]),
        product_cd=str(row["ProductCD"]),
        card_features=_clean_mapping({"card1": row.get("card1")}),
        address_features=_clean_mapping({"addr1": row.get("addr1")}),
        email_domain_features=_clean_mapping({"P_emaildomain": row.get("P_emaildomain")}),
        identity_features=_clean_mapping({"DeviceType": row.get("DeviceType"), "id_31": row.get("id_31")}),
        schema_version="v1",
    )
```

- [ ] **Step 5: Implement shared transformer**

Create `src/fraud_platform/features/transformers.py`:

```python
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FraudFeatureTransformer(BaseEstimator, TransformerMixin):
    feature_columns = [
        "TransactionAmt",
        "ProductCD",
        "card1",
        "addr1",
        "P_emaildomain",
        "DeviceType",
        "id_31",
    ]
    categorical_columns = ["ProductCD", "P_emaildomain", "DeviceType", "id_31"]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FraudFeatureTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        features = X.reindex(columns=self.feature_columns).copy()
        for column in self.categorical_columns:
            features[column] = features[column].fillna("missing").astype("category")
        return features
```

- [ ] **Step 6: Run tests**

Run:

```bash
uv run pytest tests/test_features.py -q
```

Expected:

```text
5 passed
```

- [ ] **Step 7: Commit**

```bash
git add configs/feature_schema_v1.yaml src/fraud_platform/features tests/test_features.py
git commit -m "feat: add ieee feature pipeline foundation"
```

## Task 4: Metrics, Calibration, And Conformal Utilities

**Files:**

- Create: `src/fraud_platform/metrics.py`
- Create: `src/fraud_platform/calibration.py`
- Create: `src/fraud_platform/conformal.py`
- Test: `tests/test_metrics.py`
- Test: `tests/test_calibration_conformal.py`

- [ ] **Step 1: Write metric tests**

Create `tests/test_metrics.py`:

```python
from __future__ import annotations

import numpy as np

from fraud_platform.metrics import (
    calibration_error,
    expected_fraud_utility,
    recall_at_min_precision,
)


def test_recall_at_min_precision_returns_best_supported_recall() -> None:
    y_true = np.array([0, 1, 1, 0, 1])
    scores = np.array([0.05, 0.90, 0.80, 0.40, 0.70])

    recall = recall_at_min_precision(y_true, scores, min_precision=0.75)

    assert recall == 1.0


def test_expected_fraud_utility_rewards_caught_fraud_and_penalizes_false_blocks() -> None:
    y_true = np.array([0, 1, 1, 0])
    decisions = np.array(["approve", "block", "review", "block"])

    utility = expected_fraud_utility(
        y_true,
        decisions,
        fraud_loss=100.0,
        review_cost=5.0,
        false_block_cost=25.0,
    )

    assert utility == 70.0


def test_calibration_error_bins_probabilities() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    error = calibration_error(y_true, probabilities, n_bins=2)

    assert error < 0.2
```

- [ ] **Step 2: Write calibration and conformal tests**

Create `tests/test_calibration_conformal.py`:

```python
from __future__ import annotations

import numpy as np

from fraud_platform.calibration import ProbabilityCalibrator
from fraud_platform.conformal import SplitConformalClassifier


def test_probability_calibrator_maps_scores_to_probabilities() -> None:
    raw_scores = np.array([0.05, 0.20, 0.80, 0.95])
    labels = np.array([0, 0, 1, 1])
    calibrator = ProbabilityCalibrator(method="isotonic")

    calibrator.fit(raw_scores, labels)
    probabilities = calibrator.predict(raw_scores)

    assert probabilities.min() >= 0
    assert probabilities.max() <= 1
    assert probabilities[-1] >= probabilities[0]


def test_split_conformal_classifier_returns_prediction_sets() -> None:
    probabilities = np.array([0.05, 0.95, 0.50, 0.60])
    labels = np.array([0, 1, 0, 1])
    conformal = SplitConformalClassifier(alpha=0.25)

    conformal.fit(probabilities, labels)
    prediction_sets = conformal.predict_sets(np.array([0.05, 0.95, 0.50]))

    assert prediction_sets[0] == ["legit"]
    assert prediction_sets[1] == ["fraud"]
    assert prediction_sets[2] == ["legit", "fraud"]
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_metrics.py tests/test_calibration_conformal.py -q
```

Expected: FAIL with missing modules.

- [ ] **Step 4: Implement metrics**

Create `src/fraud_platform/metrics.py`:

```python
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def recall_at_min_precision(y_true: np.ndarray, scores: np.ndarray, min_precision: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    supported = recall[precision >= min_precision]
    return float(supported.max()) if supported.size else 0.0


def expected_fraud_utility(
    y_true: np.ndarray,
    decisions: np.ndarray,
    fraud_loss: float,
    review_cost: float,
    false_block_cost: float,
) -> float:
    utility = 0.0
    for label, decision in zip(y_true, decisions, strict=True):
        if decision == "block" and label == 1:
            utility += fraud_loss
        elif decision == "block" and label == 0:
            utility -= false_block_cost
        elif decision == "review":
            utility -= review_cost
    return utility


def calibration_error(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(probabilities)
    error = 0.0
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        mask = (probabilities >= lower) & (probabilities <= upper if upper == 1 else probabilities < upper)
        if not mask.any():
            continue
        bin_confidence = float(probabilities[mask].mean())
        bin_accuracy = float(y_true[mask].mean())
        error += (mask.sum() / total) * abs(bin_accuracy - bin_confidence)
    return float(error)
```

- [ ] **Step 5: Implement calibration wrapper**

Create `src/fraud_platform/calibration.py`:

```python
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    def __init__(self, method: str = "isotonic") -> None:
        if method not in {"isotonic", "platt"}:
            raise ValueError("method must be 'isotonic' or 'platt'")
        self.method = method
        self._model: IsotonicRegression | LogisticRegression | None = None

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> "ProbabilityCalibrator":
        if self.method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(raw_scores, labels)
        else:
            model = LogisticRegression()
            model.fit(raw_scores.reshape(-1, 1), labels)
        self._model = model
        return self

    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("calibrator must be fit before predict")
        if self.method == "isotonic":
            return self._model.predict(raw_scores)
        return self._model.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
```

- [ ] **Step 6: Implement conformal wrapper**

Create `src/fraud_platform/conformal.py`:

```python
from __future__ import annotations

import numpy as np


class SplitConformalClassifier:
    def __init__(self, alpha: float = 0.1) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.threshold_: float | None = None

    def fit(self, fraud_probabilities: np.ndarray, labels: np.ndarray) -> "SplitConformalClassifier":
        true_class_probability = np.where(labels == 1, fraud_probabilities, 1 - fraud_probabilities)
        nonconformity = 1 - true_class_probability
        self.threshold_ = float(np.quantile(nonconformity, 1 - self.alpha, method="higher"))
        return self

    def predict_sets(self, fraud_probabilities: np.ndarray) -> list[list[str]]:
        if self.threshold_ is None:
            raise RuntimeError("conformal classifier must be fit before predict_sets")
        prediction_sets: list[list[str]] = []
        for probability in fraud_probabilities:
            labels: list[str] = []
            if 1 - probability >= 1 - self.threshold_:
                labels.append("legit")
            if probability >= 1 - self.threshold_:
                labels.append("fraud")
            prediction_sets.append(labels or ["legit", "fraud"])
        return prediction_sets
```

- [ ] **Step 7: Run tests**

Run:

```bash
uv run pytest tests/test_metrics.py tests/test_calibration_conformal.py -q
```

Expected:

```text
5 passed
```

- [ ] **Step 8: Commit**

```bash
git add src/fraud_platform/metrics.py src/fraud_platform/calibration.py src/fraud_platform/conformal.py tests/test_metrics.py tests/test_calibration_conformal.py
git commit -m "feat: add model metrics calibration and conformal utilities"
```

## Task 5: Model Artifacts, Training Smoke Pipeline, And Reason Codes

**Files:**

- Create: `src/fraud_platform/artifacts.py`
- Create: `src/fraud_platform/explain.py`
- Create: `src/fraud_platform/training.py`
- Test: `tests/test_training.py`
- Test: `tests/test_explain.py`

- [ ] **Step 1: Write artifact and training smoke tests**

Create `tests/test_training.py`:

```python
from __future__ import annotations

import pandas as pd

from fraud_platform.artifacts import ModelBundle, load_model_bundle
from fraud_platform.training import train_synthetic_model


def test_train_synthetic_model_writes_loadable_bundle(tmp_path) -> None:
    output_dir = tmp_path / "model"

    metadata = train_synthetic_model(output_dir)
    bundle = load_model_bundle(output_dir)

    assert metadata.model_version == "synthetic-fraud-model:1"
    assert isinstance(bundle, ModelBundle)
    assert bundle.feature_schema_version == "v1"


def test_loaded_bundle_predicts_probability(tmp_path) -> None:
    output_dir = tmp_path / "model"
    train_synthetic_model(output_dir)
    bundle = load_model_bundle(output_dir)
    rows = pd.DataFrame(
        {
            "TransactionAmt": [20.0],
            "ProductCD": ["W"],
            "card1": [1001],
            "addr1": [100.0],
            "P_emaildomain": ["a.test"],
            "DeviceType": ["desktop"],
            "id_31": ["chrome"],
        }
    )

    probability = bundle.predict_raw_probability(rows)[0]

    assert 0 <= probability <= 1
```

- [ ] **Step 2: Write reason-code tests**

Create `tests/test_explain.py`:

```python
from __future__ import annotations

import pandas as pd

from fraud_platform.explain import fallback_reason_codes


def test_fallback_reason_codes_are_stable_and_analyst_readable() -> None:
    rows = pd.DataFrame(
        {
            "TransactionAmt": [900.0],
            "ProductCD": ["C"],
            "card1": [1002],
        }
    )

    reason_codes = fallback_reason_codes(rows, max_reasons=2)

    assert reason_codes[0].feature == "TransactionAmt"
    assert reason_codes[0].direction == "increases_risk"
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_training.py tests/test_explain.py -q
```

Expected: FAIL with missing modules.

- [ ] **Step 4: Implement model artifact wrapper**

Create `src/fraud_platform/artifacts.py`:

```python
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel


class ModelMetadata(BaseModel):
    model_version: str
    feature_schema_version: str
    decision_policy_version: str
    model_type: str


@dataclass
class ModelBundle:
    model: Any
    metadata: ModelMetadata

    @property
    def model_version(self) -> str:
        return self.metadata.model_version

    @property
    def feature_schema_version(self) -> str:
        return self.metadata.feature_schema_version

    def predict_raw_probability(self, features: pd.DataFrame) -> list[float]:
        probabilities = self.model.predict_proba(features)[:, 1]
        return [float(value) for value in probabilities]


def save_model_bundle(bundle: ModelBundle, path: str | Path) -> None:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    (target / "metadata.json").write_text(bundle.metadata.model_dump_json(indent=2))
    with (target / "model.pkl").open("wb") as handle:
        pickle.dump(bundle.model, handle)


def load_model_bundle(path: str | Path) -> ModelBundle:
    target = Path(path)
    metadata = ModelMetadata.model_validate_json((target / "metadata.json").read_text())
    with (target / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    return ModelBundle(model=model, metadata=metadata)
```

- [ ] **Step 5: Implement reason-code fallback**

Create `src/fraud_platform/explain.py`:

```python
from __future__ import annotations

import pandas as pd

from fraud_platform.contracts import ReasonCode


RISK_FEATURE_ORDER = ["TransactionAmt", "card1", "ProductCD", "P_emaildomain", "DeviceType"]


def fallback_reason_codes(features: pd.DataFrame, max_reasons: int = 3) -> list[ReasonCode]:
    row = features.iloc[0]
    reason_codes: list[ReasonCode] = []
    for feature in RISK_FEATURE_ORDER:
        if feature not in row or pd.isna(row[feature]):
            continue
        if feature == "TransactionAmt" and float(row[feature]) <= 100:
            continue
        reason_codes.append(ReasonCode(feature=feature, direction="increases_risk"))
        if len(reason_codes) == max_reasons:
            break
    return reason_codes
```

- [ ] **Step 6: Implement synthetic training entrypoint**

Create `src/fraud_platform/training.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fraud_platform.artifacts import ModelBundle, ModelMetadata, save_model_bundle


def _synthetic_training_frame() -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame(
        {
            "TransactionAmt": [20.0, 200.0, 35.0, 500.0, 75.0, 900.0, 15.0, 650.0],
            "ProductCD": ["W", "C", "W", "R", "H", "C", "W", "C"],
            "card1": [1001, 1002, 1001, 1003, 1004, 1002, 1001, 1003],
            "addr1": [100.0, 200.0, 100.0, 300.0, 300.0, 200.0, 100.0, 300.0],
            "P_emaildomain": ["a.test", "b.test", "a.test", "c.test", "a.test", "b.test", "a.test", "c.test"],
            "DeviceType": ["desktop", "mobile", "desktop", "mobile", "desktop", "mobile", "desktop", "mobile"],
            "id_31": ["chrome", "safari", "chrome", "firefox", "chrome", "safari", "chrome", "firefox"],
        }
    )
    labels = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    return features, labels


def train_synthetic_model(output_dir: str | Path) -> ModelMetadata:
    features, labels = _synthetic_training_frame()
    categorical = ["ProductCD", "P_emaildomain", "DeviceType", "id_31"]
    numeric = ["TransactionAmt", "card1", "addr1"]
    model = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
                        ("numeric", "passthrough", numeric),
                    ]
                ),
            ),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )
    model.fit(features, labels)
    metadata = ModelMetadata(
        model_version="synthetic-fraud-model:1",
        feature_schema_version="v1",
        decision_policy_version="v1",
        model_type="logistic_regression_smoke",
    )
    save_model_bundle(ModelBundle(model=model, metadata=metadata), output_dir)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/model/latest")
    args = parser.parse_args()
    if not args.synthetic:
        raise SystemExit("Only --synthetic is implemented in the first training slice")
    train_synthetic_model(args.output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run tests and smoke command**

Run:

```bash
uv run pytest tests/test_training.py tests/test_explain.py -q
uv run fraud-train --synthetic --output-dir artifacts/model/latest
```

Expected:

```text
3 passed
```

The smoke command should create:

```text
artifacts/model/latest/metadata.json
artifacts/model/latest/model.pkl
```

- [ ] **Step 8: Commit**

```bash
git add src/fraud_platform/artifacts.py src/fraud_platform/explain.py src/fraud_platform/training.py tests/test_training.py tests/test_explain.py
git commit -m "feat: add training smoke pipeline and model artifacts"
```

## Task 6: Scoring Engine Shared By API And Kafka

**Files:**

- Create: `src/fraud_platform/scoring.py`
- Test: `tests/test_scoring.py`

- [ ] **Step 1: Write scoring tests**

Create `tests/test_scoring.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

from fraud_platform.contracts import TransactionEvent
from fraud_platform.policy import DecisionPolicy, PolicyConfig
from fraud_platform.scoring import ScoringEngine
from fraud_platform.training import train_synthetic_model


def test_scoring_engine_returns_decision_event(tmp_path) -> None:
    model_dir = tmp_path / "model"
    train_synthetic_model(model_dir)
    engine = ScoringEngine.from_paths(
        model_path=model_dir,
        policy=DecisionPolicy(PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8)),
    )
    event = TransactionEvent(
        event_id="evt-1",
        transaction_id=1,
        event_time=datetime(2026, 6, 10, 12, tzinfo=UTC),
        amount=900.0,
        product_cd="C",
        card_features={"card1": 1002},
        address_features={"addr1": 200.0},
        email_domain_features={"P_emaildomain": "b.test"},
        identity_features={"DeviceType": "mobile", "id_31": "safari"},
        schema_version="v1",
    )

    decision = engine.score(event)

    assert decision.event_id == "evt-1"
    assert decision.model_version == "synthetic-fraud-model:1"
    assert decision.feature_schema_version == "v1"
    assert decision.decision in {"approve", "review", "block"}
    assert decision.latency_ms >= 0
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest tests/test_scoring.py -q
```

Expected: FAIL with missing `fraud_platform.scoring`.

- [ ] **Step 3: Implement scoring engine**

Create `src/fraud_platform/scoring.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from fraud_platform.artifacts import ModelBundle, load_model_bundle
from fraud_platform.contracts import DecisionEvent, TransactionEvent
from fraud_platform.explain import fallback_reason_codes
from fraud_platform.policy import DecisionPolicy


class ScoringEngine:
    def __init__(self, bundle: ModelBundle, policy: DecisionPolicy) -> None:
        self.bundle = bundle
        self.policy = policy

    @classmethod
    def from_paths(cls, model_path: str | Path, policy: DecisionPolicy) -> "ScoringEngine":
        return cls(bundle=load_model_bundle(model_path), policy=policy)

    def score(self, event: TransactionEvent) -> DecisionEvent:
        started = perf_counter()
        features = self._features_from_event(event)
        raw_probability = self.bundle.predict_raw_probability(features)[0]
        calibrated_probability = raw_probability
        prediction_set = self._simple_prediction_set(calibrated_probability)
        policy_decision = self.policy.decide(calibrated_probability, prediction_set)
        latency_ms = (perf_counter() - started) * 1000
        return DecisionEvent(
            event_id=event.event_id,
            transaction_id=event.transaction_id,
            scored_at=datetime.now(UTC),
            model_version=self.bundle.model_version,
            feature_schema_version=self.bundle.feature_schema_version,
            decision_policy_version=self.policy.config.version,
            fraud_probability=raw_probability,
            calibrated_probability=calibrated_probability,
            conformal_prediction_set=prediction_set,
            uncertainty=policy_decision.uncertainty,
            decision=policy_decision.decision,
            reason_codes=fallback_reason_codes(features),
            latency_ms=latency_ms,
        )

    def _features_from_event(self, event: TransactionEvent) -> pd.DataFrame:
        payload = {
            "TransactionAmt": event.amount,
            "ProductCD": event.product_cd,
            **event.card_features,
            **event.address_features,
            **event.email_domain_features,
            **event.identity_features,
        }
        return pd.DataFrame([payload])

    def _simple_prediction_set(self, probability: float) -> list[str]:
        if probability <= self.policy.config.approve_threshold:
            return ["legit"]
        if probability >= self.policy.config.block_threshold:
            return ["fraud"]
        return ["legit", "fraud"]
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/test_scoring.py -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/fraud_platform/scoring.py tests/test_scoring.py
git commit -m "feat: add shared scoring engine"
```

## Task 7: FastAPI Scoring And Operations Endpoints

**Files:**

- Create: `src/fraud_platform/config.py`
- Create: `src/fraud_platform/api.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write API tests**

Create `tests/test_api.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from fraud_platform.api import create_app
from fraud_platform.policy import DecisionPolicy, PolicyConfig
from fraud_platform.scoring import ScoringEngine
from fraud_platform.training import train_synthetic_model


def test_health_endpoint(tmp_path) -> None:
    model_dir = tmp_path / "model"
    train_synthetic_model(model_dir)
    engine = ScoringEngine.from_paths(
        model_dir,
        DecisionPolicy(PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8)),
    )
    client = TestClient(create_app(scoring_engine=engine))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_score_endpoint_returns_decision(tmp_path) -> None:
    model_dir = tmp_path / "model"
    train_synthetic_model(model_dir)
    engine = ScoringEngine.from_paths(
        model_dir,
        DecisionPolicy(PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8)),
    )
    client = TestClient(create_app(scoring_engine=engine))
    payload = {
        "event_id": "evt-1",
        "transaction_id": 1,
        "event_time": datetime(2026, 6, 10, 12, tzinfo=UTC).isoformat(),
        "amount": 900.0,
        "product_cd": "C",
        "card_features": {"card1": 1002},
        "address_features": {"addr1": 200.0},
        "email_domain_features": {"P_emaildomain": "b.test"},
        "identity_features": {"DeviceType": "mobile", "id_31": "safari"},
        "schema_version": "v1",
    }

    response = client.post("/score", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["event_id"] == "evt-1"
    assert body["decision"] in {"approve", "review", "block"}


def test_model_info_endpoint(tmp_path) -> None:
    model_dir = tmp_path / "model"
    train_synthetic_model(model_dir)
    engine = ScoringEngine.from_paths(
        model_dir,
        DecisionPolicy(PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8)),
    )
    client = TestClient(create_app(scoring_engine=engine))

    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model_version"] == "synthetic-fraud-model:1"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_api.py -q
```

Expected: FAIL with missing `fraud_platform.api`.

- [ ] **Step 3: Implement settings**

Create `src/fraud_platform/config.py`:

```python
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    kafka_bootstrap_servers: str = "localhost:9092"
    database_url: str = "postgresql+psycopg://fraud:fraud@localhost:5432/fraud"
    mlflow_tracking_uri: str = "http://localhost:5000"
    model_bundle_path: str = "artifacts/model/latest"
    decision_policy_path: str = "configs/decision_policy.yaml"
```

- [ ] **Step 4: Implement FastAPI app**

Create `src/fraud_platform/api.py`:

```python
from __future__ import annotations

import uvicorn
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from fraud_platform.config import Settings
from fraud_platform.contracts import DecisionEvent, TransactionEvent
from fraud_platform.policy import load_policy
from fraud_platform.scoring import ScoringEngine

REQUEST_COUNT = Counter("fraud_api_requests_total", "Fraud API request count", ["endpoint"])
SCORING_LATENCY = Histogram("fraud_api_scoring_latency_ms", "Scoring latency in milliseconds")


def create_app(scoring_engine: ScoringEngine | None = None) -> FastAPI:
    settings = Settings()
    if scoring_engine is None:
        scoring_engine = ScoringEngine.from_paths(
            model_path=settings.model_bundle_path,
            policy=load_policy(settings.decision_policy_path),
        )
    app = FastAPI(title="Fraud Detection API", version="0.1.0")
    app.state.scoring_engine = scoring_engine

    @app.get("/health")
    def health() -> dict[str, str]:
        REQUEST_COUNT.labels(endpoint="/health").inc()
        return {"status": "ok"}

    @app.get("/model-info")
    def model_info() -> dict[str, str]:
        REQUEST_COUNT.labels(endpoint="/model-info").inc()
        bundle = app.state.scoring_engine.bundle
        return bundle.metadata.model_dump()

    @app.post("/score", response_model=DecisionEvent)
    def score(event: TransactionEvent) -> DecisionEvent:
        REQUEST_COUNT.labels(endpoint="/score").inc()
        decision = app.state.scoring_engine.score(event)
        SCORING_LATENCY.observe(decision.latency_ms)
        return decision

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def main() -> None:
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
```

- [ ] **Step 5: Run tests**

Run:

```bash
uv run pytest tests/test_api.py -q
```

Expected:

```text
3 passed
```

- [ ] **Step 6: Commit**

```bash
git add src/fraud_platform/config.py src/fraud_platform/api.py tests/test_api.py
git commit -m "feat: add fraud scoring api"
```

## Task 8: Prediction And Alert Storage

**Files:**

- Create: `src/fraud_platform/storage.py`
- Create: `src/fraud_platform/repositories.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write storage tests**

Create `tests/test_storage.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

from fraud_platform.contracts import AlertEvent, DecisionEvent
from fraud_platform.repositories import AlertRepository, PredictionRepository
from fraud_platform.storage import create_session_factory, create_tables


def test_prediction_repository_round_trips_decision() -> None:
    session_factory = create_session_factory("sqlite+pysqlite:///:memory:")
    create_tables(session_factory)
    decision = DecisionEvent(
        event_id="evt-1",
        transaction_id=1,
        scored_at=datetime(2026, 6, 10, 12, tzinfo=UTC),
        model_version="fraud-model:1",
        feature_schema_version="v1",
        decision_policy_version="v1",
        fraud_probability=0.9,
        calibrated_probability=0.85,
        conformal_prediction_set=["fraud"],
        uncertainty="low",
        decision="block",
        latency_ms=10.0,
    )

    PredictionRepository(session_factory).save(decision)
    rows = PredictionRepository(session_factory).latest(limit=10)

    assert rows[0].event_id == "evt-1"
    assert rows[0].decision == "block"


def test_alert_repository_round_trips_alert() -> None:
    session_factory = create_session_factory("sqlite+pysqlite:///:memory:")
    create_tables(session_factory)
    alert = AlertEvent(
        alert_id="alert-1",
        created_at=datetime(2026, 6, 10, 12, tzinfo=UTC),
        severity="warning",
        alert_type="dead_letter_rate",
        message="Dead letter rate exceeded threshold",
    )

    AlertRepository(session_factory).save(alert)
    rows = AlertRepository(session_factory).latest(limit=10)

    assert rows[0].alert_type == "dead_letter_rate"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_storage.py -q
```

Expected: FAIL with missing storage modules.

- [ ] **Step 3: Implement SQLAlchemy tables**

Create `src/fraud_platform/storage.py`:

```python
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    transaction_id: Mapped[int] = mapped_column(Integer, index=True)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    model_version: Mapped[str] = mapped_column(String)
    feature_schema_version: Mapped[str] = mapped_column(String)
    decision_policy_version: Mapped[str] = mapped_column(String)
    fraud_probability: Mapped[float] = mapped_column(Float)
    calibrated_probability: Mapped[float] = mapped_column(Float)
    conformal_prediction_set: Mapped[list[str]] = mapped_column(JSON)
    uncertainty: Mapped[str] = mapped_column(String)
    decision: Mapped[str] = mapped_column(String, index=True)
    reason_codes: Mapped[list[dict]] = mapped_column(JSON, default=list)
    latency_ms: Mapped[float] = mapped_column(Float)


class AlertRecord(Base):
    __tablename__ = "alerts"

    alert_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    severity: Mapped[str] = mapped_column(String, index=True)
    alert_type: Mapped[str] = mapped_column(String, index=True)
    message: Mapped[str] = mapped_column(String)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


def create_session_factory(database_url: str):
    engine = create_engine(database_url, future=True)
    return sessionmaker(bind=engine, expire_on_commit=False)


def create_tables(session_factory) -> None:
    Base.metadata.create_all(session_factory.kw["bind"])
```

- [ ] **Step 4: Implement repositories**

Create `src/fraud_platform/repositories.py`:

```python
from __future__ import annotations

from fraud_platform.contracts import AlertEvent, DecisionEvent
from fraud_platform.storage import AlertRecord, PredictionRecord


class PredictionRepository:
    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory

    def save(self, decision: DecisionEvent) -> None:
        with self.session_factory() as session:
            session.merge(
                PredictionRecord(
                    event_id=decision.event_id,
                    transaction_id=decision.transaction_id,
                    scored_at=decision.scored_at,
                    model_version=decision.model_version,
                    feature_schema_version=decision.feature_schema_version,
                    decision_policy_version=decision.decision_policy_version,
                    fraud_probability=decision.fraud_probability,
                    calibrated_probability=decision.calibrated_probability,
                    conformal_prediction_set=decision.conformal_prediction_set,
                    uncertainty=decision.uncertainty,
                    decision=decision.decision,
                    reason_codes=[reason.model_dump() for reason in decision.reason_codes],
                    latency_ms=decision.latency_ms,
                )
            )
            session.commit()

    def latest(self, limit: int = 100) -> list[PredictionRecord]:
        with self.session_factory() as session:
            return list(
                session.query(PredictionRecord)
                .order_by(PredictionRecord.scored_at.desc())
                .limit(limit)
                .all()
            )


class AlertRepository:
    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory

    def save(self, alert: AlertEvent) -> None:
        with self.session_factory() as session:
            session.merge(
                AlertRecord(
                    alert_id=alert.alert_id,
                    created_at=alert.created_at,
                    severity=alert.severity,
                    alert_type=alert.alert_type,
                    message=alert.message,
                    metadata_json=alert.metadata,
                )
            )
            session.commit()

    def latest(self, limit: int = 100) -> list[AlertRecord]:
        with self.session_factory() as session:
            return list(
                session.query(AlertRecord)
                .order_by(AlertRecord.created_at.desc())
                .limit(limit)
                .all()
            )
```

- [ ] **Step 5: Run tests**

Run:

```bash
uv run pytest tests/test_storage.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 6: Commit**

```bash
git add src/fraud_platform/storage.py src/fraud_platform/repositories.py tests/test_storage.py
git commit -m "feat: add prediction and alert storage"
```

## Task 9: Kafka Replay Producer And Fraud Consumer

**Files:**

- Create: `configs/kafka_topics.yaml`
- Create: `src/fraud_platform/streaming.py`
- Create: `src/fraud_platform/replay.py`
- Create: `src/fraud_platform/consumer.py`
- Test: `tests/test_streaming.py`
- Test: `tests/integration/test_kafka_flow.py`

- [ ] **Step 1: Write streaming unit tests**

Create `tests/test_streaming.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

from fraud_platform.contracts import TransactionEvent
from fraud_platform.streaming import deserialize_event, serialize_event


def test_serialize_deserialize_transaction_event() -> None:
    event = TransactionEvent(
        event_id="evt-1",
        transaction_id=1,
        event_time=datetime(2026, 6, 10, 12, tzinfo=UTC),
        amount=20.0,
        product_cd="W",
        schema_version="v1",
    )

    payload = serialize_event(event)
    restored = deserialize_event(payload, TransactionEvent)

    assert restored == event
```

- [ ] **Step 2: Write optional integration test**

Create `tests/integration/test_kafka_flow.py`:

```python
from __future__ import annotations

import pytest


@pytest.mark.integration
def test_kafka_flow_documented_for_compose_stack() -> None:
    pytest.skip("Run after docker compose stack exists: replay publishes transaction-events and consumer emits fraud-decisions.")
```

- [ ] **Step 3: Run unit test to verify failure**

Run:

```bash
uv run pytest tests/test_streaming.py -q
```

Expected: FAIL with missing `fraud_platform.streaming`.

- [ ] **Step 4: Create topic config**

Create `configs/kafka_topics.yaml`:

```yaml
topics:
  transaction-events:
    partitions: 3
    replication_factor: 1
  fraud-decisions:
    partitions: 3
    replication_factor: 1
  fraud-labels:
    partitions: 1
    replication_factor: 1
  model-alerts:
    partitions: 1
    replication_factor: 1
  dead-letter-events:
    partitions: 1
    replication_factor: 1
```

- [ ] **Step 5: Implement streaming helpers**

Create `src/fraud_platform/streaming.py`:

```python
from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def serialize_event(event: BaseModel) -> bytes:
    return event.model_dump_json().encode("utf-8")


def deserialize_event(payload: bytes, model: type[T]) -> T:
    return model.model_validate_json(payload.decode("utf-8"))
```

- [ ] **Step 6: Implement replay producer skeleton**

Create `src/fraud_platform/replay.py`:

```python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from confluent_kafka import Producer

from fraud_platform.features.ieee import build_transaction_event
from fraud_platform.streaming import serialize_event


def replay_transactions(
    replay_path: str | Path,
    bootstrap_servers: str,
    topic: str,
    speed_multiplier: float = 60.0,
) -> int:
    producer = Producer({"bootstrap.servers": bootstrap_servers})
    frame = pd.read_parquet(replay_path).sort_values("TransactionDT")
    previous_dt: float | None = None
    count = 0
    for _, row in frame.iterrows():
        current_dt = float(row["TransactionDT"])
        if previous_dt is not None:
            delay = max(0.0, (current_dt - previous_dt) / speed_multiplier)
            time.sleep(min(delay, 1.0))
        event = build_transaction_event(row, event_time_base="2026-06-10T12:00:00Z")
        producer.produce(topic, key=str(event.transaction_id), value=serialize_event(event))
        producer.poll(0)
        previous_dt = current_dt
        count += 1
    producer.flush()
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-path", default="data/processed/replay.parquet")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default="transaction-events")
    parser.add_argument("--speed-multiplier", type=float, default=60.0)
    args = parser.parse_args()
    replay_transactions(args.replay_path, args.bootstrap_servers, args.topic, args.speed_multiplier)
```

- [ ] **Step 7: Implement fraud consumer skeleton**

Create `src/fraud_platform/consumer.py`:

```python
from __future__ import annotations

import argparse

from confluent_kafka import Consumer, Producer

from fraud_platform.contracts import TransactionEvent
from fraud_platform.policy import load_policy
from fraud_platform.scoring import ScoringEngine
from fraud_platform.streaming import deserialize_event, serialize_event


def run_consumer(
    bootstrap_servers: str,
    input_topic: str,
    output_topic: str,
    group_id: str,
    model_path: str,
    policy_path: str,
) -> None:
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
        }
    )
    producer = Producer({"bootstrap.servers": bootstrap_servers})
    engine = ScoringEngine.from_paths(model_path, load_policy(policy_path))
    consumer.subscribe([input_topic])
    try:
        while True:
            message = consumer.poll(1.0)
            if message is None:
                continue
            if message.error():
                continue
            event = deserialize_event(message.value(), TransactionEvent)
            decision = engine.score(event)
            producer.produce(output_topic, key=str(decision.transaction_id), value=serialize_event(decision))
            producer.poll(0)
            consumer.commit(message)
    finally:
        consumer.close()
        producer.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--input-topic", default="transaction-events")
    parser.add_argument("--output-topic", default="fraud-decisions")
    parser.add_argument("--group-id", default="fraud-consumer")
    parser.add_argument("--model-path", default="artifacts/model/latest")
    parser.add_argument("--policy-path", default="configs/decision_policy.yaml")
    args = parser.parse_args()
    run_consumer(
        args.bootstrap_servers,
        args.input_topic,
        args.output_topic,
        args.group_id,
        args.model_path,
        args.policy_path,
    )
```

- [ ] **Step 8: Run tests**

Run:

```bash
uv run pytest tests/test_streaming.py -q
```

Expected:

```text
1 passed
```

- [ ] **Step 9: Commit**

```bash
git add configs/kafka_topics.yaml src/fraud_platform/streaming.py src/fraud_platform/replay.py src/fraud_platform/consumer.py tests/test_streaming.py tests/integration/test_kafka_flow.py
git commit -m "feat: add kafka replay and scoring consumer"
```

## Task 10: Monitoring Calculations And Alert Emission

**Files:**

- Create: `src/fraud_platform/monitoring.py`
- Test: `tests/test_monitoring.py`

- [ ] **Step 1: Write monitoring tests**

Create `tests/test_monitoring.py`:

```python
from __future__ import annotations

import pandas as pd

from fraud_platform.monitoring import (
    conformal_coverage,
    decision_rate_shift_alert,
    missingness_rate,
)


def test_missingness_rate_by_column() -> None:
    frame = pd.DataFrame({"identity": [None, "mobile", None], "amount": [1.0, 2.0, 3.0]})

    rates = missingness_rate(frame)

    assert rates["identity"] == 2 / 3
    assert rates["amount"] == 0.0


def test_conformal_coverage_counts_true_label_inside_set() -> None:
    frame = pd.DataFrame(
        {
            "is_fraud": [0, 1, 1],
            "conformal_prediction_set": [["legit"], ["fraud"], ["legit", "fraud"]],
        }
    )

    coverage = conformal_coverage(frame)

    assert coverage == 1.0


def test_decision_rate_shift_alert_when_review_rate_doubles() -> None:
    alert = decision_rate_shift_alert(
        reference_rates={"review": 0.10},
        current_rates={"review": 0.25},
        threshold_multiplier=2.0,
    )

    assert alert is not None
    assert alert.alert_type == "decision_rate_shift"
    assert alert.severity == "warning"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_monitoring.py -q
```

Expected: FAIL with missing `fraud_platform.monitoring`.

- [ ] **Step 3: Implement monitoring calculations**

Create `src/fraud_platform/monitoring.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd

from fraud_platform.contracts import AlertEvent


def missingness_rate(frame: pd.DataFrame) -> dict[str, float]:
    return {column: float(frame[column].isna().mean()) for column in frame.columns}


def conformal_coverage(frame: pd.DataFrame) -> float:
    covered = []
    for _, row in frame.iterrows():
        label = "fraud" if int(row["is_fraud"]) == 1 else "legit"
        covered.append(label in row["conformal_prediction_set"])
    return float(sum(covered) / len(covered)) if covered else 0.0


def decision_rate_shift_alert(
    reference_rates: dict[str, float],
    current_rates: dict[str, float],
    threshold_multiplier: float,
) -> AlertEvent | None:
    reference_review = reference_rates.get("review", 0.0)
    current_review = current_rates.get("review", 0.0)
    if reference_review == 0:
        return None
    if current_review >= reference_review * threshold_multiplier:
        return AlertEvent(
            alert_id=str(uuid4()),
            created_at=datetime.now(UTC),
            severity="warning",
            alert_type="decision_rate_shift",
            message=f"Review rate increased from {reference_review:.3f} to {current_review:.3f}",
            metadata={"reference_review_rate": reference_review, "current_review_rate": current_review},
        )
    return None


def main() -> None:
    print("Monitoring worker entrypoint will run scheduled checks in the compose stack.")
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/test_monitoring.py -q
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/fraud_platform/monitoring.py tests/test_monitoring.py
git commit -m "feat: add monitoring calculation foundation"
```

## Task 11: React Fraud Operations Dashboard

**Files:**

- Create: `dashboard/package.json`
- Create: `dashboard/tsconfig.json`
- Create: `dashboard/index.html`
- Create: `dashboard/src/main.tsx`
- Create: `dashboard/src/App.tsx`
- Create: `dashboard/src/api.ts`
- Create: `dashboard/src/types.ts`
- Create: `dashboard/src/styles.css`
- Create: `dashboard/src/components/DecisionFeed.tsx`
- Create: `dashboard/src/components/TransactionDrawer.tsx`
- Create: `dashboard/src/components/KpiStrip.tsx`
- Create: `dashboard/src/components/AlertPanel.tsx`
- Create: `dashboard/src/App.test.tsx`

- [ ] **Step 1: Create dashboard package**

Create `dashboard/package.json`:

```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "test": "vitest run"
  },
  "dependencies": {
    "@vitejs/plugin-react": "^4.3.4",
    "vite": "^6.0.0",
    "typescript": "^5.7.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "lucide-react": "^0.468.0"
  },
  "devDependencies": {
    "@testing-library/react": "^16.1.0",
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "vitest": "^2.1.0",
    "jsdom": "^25.0.0"
  }
}
```

- [ ] **Step 2: Create TypeScript and Vite shell files**

Create `dashboard/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "useDefineForClassFields": true,
    "lib": ["DOM", "DOM.Iterable", "ES2022"],
    "allowJs": false,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "module": "ESNext",
    "moduleResolution": "Node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx"
  },
  "include": ["src"]
}
```

Create `dashboard/index.html`:

```html
<div id="root"></div>
<script type="module" src="/src/main.tsx"></script>
```

- [ ] **Step 3: Create dashboard types and API client**

Create `dashboard/src/types.ts`:

```ts
export type Decision = "approve" | "review" | "block";
export type Severity = "info" | "warning" | "critical";

export interface ReasonCode {
  feature: string;
  direction: "increases_risk" | "decreases_risk";
  contribution?: number | null;
}

export interface DecisionEvent {
  event_id: string;
  transaction_id: number;
  scored_at: string;
  model_version: string;
  feature_schema_version: string;
  decision_policy_version: string;
  fraud_probability: number;
  calibrated_probability: number;
  conformal_prediction_set: string[];
  uncertainty: "low" | "medium" | "high";
  decision: Decision;
  reason_codes: ReasonCode[];
  latency_ms: number;
}

export interface AlertEvent {
  alert_id: string;
  created_at: string;
  severity: Severity;
  alert_type: string;
  message: string;
  metadata: Record<string, unknown>;
}
```

Create `dashboard/src/api.ts`:

```ts
import type { AlertEvent, DecisionEvent } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function fetchDecisions(): Promise<DecisionEvent[]> {
  const response = await fetch(`${API_BASE}/predictions`);
  if (!response.ok) return [];
  return response.json();
}

export async function fetchAlerts(): Promise<AlertEvent[]> {
  const response = await fetch(`${API_BASE}/alerts`);
  if (!response.ok) return [];
  return response.json();
}
```

- [ ] **Step 4: Implement app and components**

Create `dashboard/src/main.tsx`:

```tsx
import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

Create `dashboard/src/App.tsx`:

```tsx
import { useEffect, useMemo, useState } from "react";
import { AlertPanel } from "./components/AlertPanel";
import { DecisionFeed } from "./components/DecisionFeed";
import { KpiStrip } from "./components/KpiStrip";
import { TransactionDrawer } from "./components/TransactionDrawer";
import { fetchAlerts, fetchDecisions } from "./api";
import type { AlertEvent, DecisionEvent } from "./types";

export default function App() {
  const [decisions, setDecisions] = useState<DecisionEvent[]>([]);
  const [alerts, setAlerts] = useState<AlertEvent[]>([]);
  const [selected, setSelected] = useState<DecisionEvent | null>(null);

  useEffect(() => {
    void fetchDecisions().then(setDecisions);
    void fetchAlerts().then(setAlerts);
    const timer = window.setInterval(() => {
      void fetchDecisions().then(setDecisions);
      void fetchAlerts().then(setAlerts);
    }, 5000);
    return () => window.clearInterval(timer);
  }, []);

  const fallbackDecisions = useMemo<DecisionEvent[]>(
    () =>
      decisions.length
        ? decisions
        : [
            {
              event_id: "sample-1",
              transaction_id: 2987000,
              scored_at: new Date().toISOString(),
              model_version: "synthetic-fraud-model:1",
              feature_schema_version: "v1",
              decision_policy_version: "v1",
              fraud_probability: 0.82,
              calibrated_probability: 0.76,
              conformal_prediction_set: ["fraud"],
              uncertainty: "low",
              decision: "block",
              reason_codes: [{ feature: "TransactionAmt", direction: "increases_risk" }],
              latency_ms: 42,
            },
          ],
    [decisions]
  );

  return (
    <main className="app-shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">Fraud operations</p>
          <h1>Live transaction decisions</h1>
        </div>
        <div className="model-pill">Model {fallbackDecisions[0]?.model_version ?? "loading"}</div>
      </section>
      <KpiStrip decisions={fallbackDecisions} />
      <section className="workspace">
        <DecisionFeed decisions={fallbackDecisions} onSelect={setSelected} />
        <AlertPanel alerts={alerts} />
      </section>
      <TransactionDrawer decision={selected} onClose={() => setSelected(null)} />
    </main>
  );
}
```

Create `dashboard/src/components/DecisionFeed.tsx`:

```tsx
import type { DecisionEvent } from "../types";

interface DecisionFeedProps {
  decisions: DecisionEvent[];
  onSelect: (decision: DecisionEvent) => void;
}

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

export function DecisionFeed({ decisions, onSelect }: DecisionFeedProps) {
  return (
    <section className="panel decision-feed">
      <div className="panel-header">
        <h2>Decision feed</h2>
        <span>{decisions.length} transactions</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>Transaction</th>
            <th>Decision</th>
            <th>Probability</th>
            <th>Uncertainty</th>
            <th>Latency</th>
            <th>Model</th>
          </tr>
        </thead>
        <tbody>
          {decisions.map((decision) => (
            <tr key={decision.event_id} onClick={() => onSelect(decision)}>
              <td>{decision.transaction_id}</td>
              <td>
                <span className={`status ${decision.decision}`}>{decision.decision}</span>
              </td>
              <td>{formatPercent(decision.calibrated_probability)}</td>
              <td>{decision.uncertainty}</td>
              <td>{Math.round(decision.latency_ms)} ms</td>
              <td>{decision.model_version}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
```

Create `dashboard/src/components/TransactionDrawer.tsx`:

```tsx
import { X } from "lucide-react";
import type { DecisionEvent } from "../types";

interface TransactionDrawerProps {
  decision: DecisionEvent | null;
  onClose: () => void;
}

export function TransactionDrawer({ decision, onClose }: TransactionDrawerProps) {
  if (!decision) return null;

  return (
    <aside className="drawer" aria-label="Transaction detail">
      <button className="icon-button" onClick={onClose} aria-label="Close transaction detail">
        <X size={18} />
      </button>
      <p className="eyebrow">Transaction {decision.transaction_id}</p>
      <h2>{decision.decision}</h2>
      <dl className="detail-grid">
        <div>
          <dt>Calibrated probability</dt>
          <dd>{Math.round(decision.calibrated_probability * 100)}%</dd>
        </div>
        <div>
          <dt>Conformal set</dt>
          <dd>{decision.conformal_prediction_set.join(", ")}</dd>
        </div>
        <div>
          <dt>Policy</dt>
          <dd>{decision.decision_policy_version}</dd>
        </div>
        <div>
          <dt>Feature schema</dt>
          <dd>{decision.feature_schema_version}</dd>
        </div>
      </dl>
      <h3>Reason codes</h3>
      <ul className="reason-list">
        {decision.reason_codes.map((reason) => (
          <li key={`${reason.feature}-${reason.direction}`}>
            <span>{reason.feature}</span>
            <span>{reason.direction.replace("_", " ")}</span>
          </li>
        ))}
      </ul>
    </aside>
  );
}
```

Create `dashboard/src/components/KpiStrip.tsx`:

```tsx
import type { Decision, DecisionEvent } from "../types";

interface KpiStripProps {
  decisions: DecisionEvent[];
}

function rate(decisions: DecisionEvent[], decision: Decision) {
  if (!decisions.length) return "0%";
  const count = decisions.filter((item) => item.decision === decision).length;
  return `${Math.round((count / decisions.length) * 100)}%`;
}

function p95Latency(decisions: DecisionEvent[]) {
  if (!decisions.length) return "0 ms";
  const sorted = [...decisions].map((item) => item.latency_ms).sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.95) - 1);
  return `${Math.round(sorted[index])} ms`;
}

export function KpiStrip({ decisions }: KpiStripProps) {
  const items = [
    ["Approve rate", rate(decisions, "approve")],
    ["Review rate", rate(decisions, "review")],
    ["Block rate", rate(decisions, "block")],
    ["p95 latency", p95Latency(decisions)],
  ];

  return (
    <section className="kpi-strip">
      {items.map(([label, value]) => (
        <div className="kpi" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </section>
  );
}
```

Create `dashboard/src/components/AlertPanel.tsx`:

```tsx
import type { AlertEvent } from "../types";

interface AlertPanelProps {
  alerts: AlertEvent[];
}

export function AlertPanel({ alerts }: AlertPanelProps) {
  return (
    <section className="panel alert-panel">
      <div className="panel-header">
        <h2>Alerts</h2>
        <span>{alerts.length} active</span>
      </div>
      {alerts.length === 0 ? (
        <p className="empty-state">No active model or service alerts.</p>
      ) : (
        <ul>
          {alerts.map((alert) => (
            <li key={alert.alert_id} className={`alert ${alert.severity}`}>
              <strong>{alert.alert_type}</strong>
              <span>{alert.message}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
```

Create `dashboard/src/styles.css`:

```css
:root {
  color: #17202a;
  background: #eef2f6;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
}

button,
table {
  font: inherit;
}

.app-shell {
  min-height: 100vh;
  padding: 24px;
}

.topbar,
.workspace,
.kpi-strip {
  max-width: 1440px;
  margin: 0 auto 16px;
}

.topbar {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 16px;
}

.eyebrow {
  margin: 0 0 4px;
  color: #607080;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}

h1,
h2,
h3 {
  margin: 0;
  letter-spacing: 0;
}

h1 {
  font-size: 28px;
}

h2 {
  font-size: 16px;
}

.model-pill,
.status,
.kpi,
.panel,
.drawer {
  border: 1px solid #d5dde5;
  border-radius: 8px;
  background: #ffffff;
}

.model-pill {
  padding: 8px 10px;
  color: #425466;
  font-size: 13px;
}

.kpi-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.kpi {
  padding: 14px;
}

.kpi span {
  display: block;
  color: #607080;
  font-size: 12px;
}

.kpi strong {
  display: block;
  margin-top: 6px;
  font-size: 24px;
}

.workspace {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  gap: 16px;
}

.panel {
  overflow: hidden;
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 16px;
  border-bottom: 1px solid #d5dde5;
}

.panel-header span,
.empty-state {
  color: #607080;
  font-size: 13px;
}

table {
  width: 100%;
  border-collapse: collapse;
}

th,
td {
  padding: 11px 16px;
  border-bottom: 1px solid #edf1f5;
  text-align: left;
  font-size: 13px;
}

th {
  color: #607080;
  font-weight: 700;
}

tbody tr {
  cursor: pointer;
}

tbody tr:hover {
  background: #f7f9fb;
}

.status {
  display: inline-block;
  min-width: 64px;
  padding: 4px 8px;
  text-align: center;
  font-size: 12px;
  font-weight: 700;
}

.approve {
  color: #176b45;
  background: #e8f6ef;
  border-color: #bde3cf;
}

.review {
  color: #7a4d00;
  background: #fff4dc;
  border-color: #f2d08a;
}

.block {
  color: #9f1f2f;
  background: #fdecef;
  border-color: #f4bbc4;
}

.alert-panel {
  min-height: 320px;
}

.alert-panel ul,
.reason-list {
  margin: 0;
  padding: 12px;
  list-style: none;
}

.alert,
.reason-list li {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 10px;
  border-bottom: 1px solid #edf1f5;
  font-size: 13px;
}

.empty-state {
  margin: 16px;
}

.drawer {
  position: fixed;
  top: 0;
  right: 0;
  width: min(420px, 100vw);
  height: 100vh;
  padding: 24px;
  box-shadow: -12px 0 30px rgb(20 30 40 / 12%);
}

.icon-button {
  display: grid;
  width: 34px;
  height: 34px;
  margin-left: auto;
  place-items: center;
  border: 1px solid #d5dde5;
  border-radius: 8px;
  background: #ffffff;
  cursor: pointer;
}

.detail-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin: 20px 0;
}

dt {
  color: #607080;
  font-size: 12px;
}

dd {
  margin: 4px 0 0;
  font-weight: 700;
}

@media (max-width: 900px) {
  .topbar,
  .workspace {
    display: block;
  }

  .kpi-strip {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .alert-panel {
    margin-top: 16px;
  }
}
```

- [ ] **Step 5: Add dashboard render test**

Create `dashboard/src/App.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import App from "./App";

describe("App", () => {
  it("renders the operations dashboard", () => {
    render(<App />);
    expect(screen.getByText("Live transaction decisions")).toBeTruthy();
    expect(screen.getByText("Fraud operations")).toBeTruthy();
  });
});
```

- [ ] **Step 6: Run dashboard checks**

Run:

```bash
cd dashboard
npm install
npm run build
npm run test
```

Expected:

```text
vite build succeeds
1 test passes
```

- [ ] **Step 7: Commit**

```bash
git add dashboard
git commit -m "feat: add fraud operations dashboard"
```

## Task 12: Docker Compose And Observability

**Files:**

- Create: `docker-compose.yml`
- Create: `docker/backend.Dockerfile`
- Create: `docker/dashboard.Dockerfile`
- Create: `docker/postgres/init.sql`
- Create: `monitoring/prometheus.yml`
- Create: `monitoring/grafana/provisioning/datasources/prometheus.yml`
- Create: `monitoring/grafana/provisioning/dashboards/dashboards.yml`
- Create: `monitoring/grafana/dashboards/fraud-platform.json`

- [ ] **Step 1: Create backend Dockerfile**

Create `docker/backend.Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir uv && uv pip install --system .
COPY configs ./configs
CMD ["fraud-api"]
```

- [ ] **Step 2: Create dashboard Dockerfile**

Create `docker/dashboard.Dockerfile`:

```dockerfile
FROM node:22-slim AS build
WORKDIR /app
COPY dashboard/package.json dashboard/package-lock.json* ./
RUN npm install
COPY dashboard ./
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
```

- [ ] **Step 3: Create Postgres init SQL**

Create `docker/postgres/init.sql`:

```sql
CREATE TABLE IF NOT EXISTS predictions (
  event_id TEXT PRIMARY KEY,
  transaction_id INTEGER NOT NULL,
  scored_at TIMESTAMPTZ NOT NULL,
  model_version TEXT NOT NULL,
  feature_schema_version TEXT NOT NULL,
  decision_policy_version TEXT NOT NULL,
  fraud_probability DOUBLE PRECISION NOT NULL,
  calibrated_probability DOUBLE PRECISION NOT NULL,
  conformal_prediction_set JSONB NOT NULL,
  uncertainty TEXT NOT NULL,
  decision TEXT NOT NULL,
  reason_codes JSONB NOT NULL DEFAULT '[]'::jsonb,
  latency_ms DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
  alert_id TEXT PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL,
  severity TEXT NOT NULL,
  alert_type TEXT NOT NULL,
  message TEXT NOT NULL,
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);
```

- [ ] **Step 4: Create Compose stack**

Create `docker-compose.yml` with services and ports:

```yaml
services:
  kafka:
    image: apache/kafka:3.9.0
    ports:
      - "9092:9092"
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka:9093
      KAFKA_LISTENERS: PLAINTEXT://:29092,PLAINTEXT_HOST://:9092,CONTROLLER://:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  postgres:
    image: postgres:17-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: fraud
      POSTGRES_PASSWORD: fraud
      POSTGRES_DB: fraud
    volumes:
      - ./docker/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql:ro

  mlflow:
    image: python:3.11-slim
    ports:
      - "5000:5000"
    command: >
      sh -c "pip install --no-cache-dir mlflow && mlflow server
      --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
      --default-artifact-root /mlflow/artifacts"

  fraud-api:
    build:
      context: .
      dockerfile: docker/backend.Dockerfile
    ports:
      - "8000:8000"
    env_file: .env.example
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:29092
      DATABASE_URL: postgresql+psycopg://fraud:fraud@postgres:5432/fraud
      MLFLOW_TRACKING_URI: http://mlflow:5000
    command: ["fraud-api"]
    volumes:
      - ./artifacts:/app/artifacts
    depends_on:
      - postgres
      - kafka

  fraud-consumer:
    build:
      context: .
      dockerfile: docker/backend.Dockerfile
    env_file: .env.example
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:29092
      DATABASE_URL: postgresql+psycopg://fraud:fraud@postgres:5432/fraud
    command: ["fraud-consumer", "--bootstrap-servers", "kafka:29092"]
    volumes:
      - ./artifacts:/app/artifacts
    depends_on:
      - kafka
      - postgres

  transaction-producer:
    build:
      context: .
      dockerfile: docker/backend.Dockerfile
    env_file: .env.example
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:29092
    command: ["fraud-replay", "--bootstrap-servers", "kafka:29092"]
    volumes:
      - ./data:/app/data
    depends_on:
      - kafka

  monitoring-worker:
    build:
      context: .
      dockerfile: docker/backend.Dockerfile
    env_file: .env.example
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:29092
      DATABASE_URL: postgresql+psycopg://fraud:fraud@postgres:5432/fraud
    command: ["fraud-monitor"]
    depends_on:
      - postgres
      - kafka

  prometheus:
    image: prom/prometheus:v3.0.0
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro

  grafana:
    image: grafana/grafana-oss:11.4.0
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro

  dashboard:
    build:
      context: .
      dockerfile: docker/dashboard.Dockerfile
    ports:
      - "5173:80"
    depends_on:
      - fraud-api
```

- [ ] **Step 5: Create Prometheus config**

Create `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: fraud-api
    metrics_path: /metrics
    static_configs:
      - targets: ["fraud-api:8000"]
```

- [ ] **Step 6: Create Grafana provisioning**

Create `monitoring/grafana/provisioning/datasources/prometheus.yml`:

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

Create `monitoring/grafana/provisioning/dashboards/dashboards.yml`:

```yaml
apiVersion: 1
providers:
  - name: Fraud Platform
    orgId: 1
    folder: Fraud Platform
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    options:
      path: /var/lib/grafana/dashboards
```

Create `monitoring/grafana/dashboards/fraud-platform.json`:

```json
{
  "annotations": {
    "list": []
  },
  "editable": true,
  "fiscalYearStartMonth": 0,
  "graphTooltip": 0,
  "id": null,
  "links": [],
  "panels": [
    {
      "datasource": {
        "type": "prometheus",
        "uid": "Prometheus"
      },
      "fieldConfig": {
        "defaults": {
          "unit": "reqps"
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "id": 1,
      "targets": [
        {
          "expr": "sum(rate(fraud_api_requests_total[5m]))",
          "legendFormat": "requests/sec",
          "refId": "A"
        }
      ],
      "title": "Fraud API Request Rate",
      "type": "timeseries"
    },
    {
      "datasource": {
        "type": "prometheus",
        "uid": "Prometheus"
      },
      "fieldConfig": {
        "defaults": {
          "unit": "ms"
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 0
      },
      "id": 2,
      "targets": [
        {
          "expr": "histogram_quantile(0.95, sum(rate(fraud_api_scoring_latency_ms_bucket[5m])) by (le))",
          "legendFormat": "p95 latency",
          "refId": "A"
        }
      ],
      "title": "Scoring Latency p95",
      "type": "timeseries"
    }
  ],
  "refresh": "15s",
  "schemaVersion": 40,
  "tags": ["fraud", "ml"],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-1h",
    "to": "now"
  },
  "timezone": "browser",
  "title": "Fraud Platform",
  "uid": "fraud-platform",
  "version": 1,
  "weekStart": ""
}
```

- [ ] **Step 7: Verify Compose stack**

Run:

```bash
docker compose config
docker compose up --build -d kafka postgres mlflow fraud-api prometheus grafana dashboard
curl -f http://localhost:8000/health
curl -f http://localhost:9090/-/ready
curl -f http://localhost:3000/api/health
curl -f http://localhost:5173
docker compose down
```

Expected:

```text
compose config succeeds
fraud API health returns {"status":"ok"}
Prometheus, Grafana, and dashboard HTTP checks succeed
```

- [ ] **Step 8: Commit**

```bash
git add docker-compose.yml docker monitoring
git commit -m "feat: add local compose and observability stack"
```

## Task 13: Documentation, End-To-End Smoke, And Project Polish

**Files:**

- Modify: `README.md`
- Modify: `docs/deployment.md`
- Modify: `docs/model-card.md`
- Create: `docs/runbook.md`
- Create: `docs/demo-script.md`

- [ ] **Step 1: Update README status and quickstart**

Modify `README.md` to include:

```markdown
## Quickstart

1. Install Python 3.11 and uv.
2. Copy `.env.example` to `.env`.
3. Run `uv sync --extra dev`.
4. Run `uv run fraud-train --synthetic --output-dir artifacts/model/latest`.
5. Run `docker compose up --build`.
6. Open:
   - dashboard: `http://localhost:5173`
   - fraud API: `http://localhost:8000/docs`
   - MLflow: `http://localhost:5000`
   - Grafana: `http://localhost:3000`
   - Prometheus: `http://localhost:9090`
```

- [ ] **Step 2: Create operator runbook**

Create `docs/runbook.md`:

```markdown
# Fraud Platform Runbook

## Local Startup

Run `uv run fraud-train --synthetic --output-dir artifacts/model/latest` before starting the API if no model artifact exists.

Run `docker compose up --build` to start the local stack.

## Health Checks

- API: `curl http://localhost:8000/health`
- API metrics: `curl http://localhost:8000/metrics`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Dashboard: `http://localhost:5173`

## Common Issues

- If Kafka is unavailable, restart `docker compose up kafka`.
- If the API cannot load a model, rerun the synthetic training command.
- If Postgres tables are missing, run `docker compose down -v` and start the stack again.
```

- [ ] **Step 3: Create recruiter demo script**

Create `docs/demo-script.md`:

```markdown
# Demo Script

1. Show the architecture diagram in `README.md`.
2. Start the local stack with Docker Compose.
3. Open the dashboard and point out live decisions, reason codes, and model metadata.
4. Open `/docs` on the FastAPI service and submit a sample `/score` request.
5. Open MLflow and show the smoke model run or later IEEE-CIS benchmark runs.
6. Open Grafana and show request volume and latency.
7. Explain how uncertainty routes ambiguous transactions to review.
```

- [ ] **Step 4: Update model card with implemented artifact metadata**

Modify `docs/model-card.md` to add:

```markdown
## Implemented Artifact Metadata

Every packaged model bundle includes:

- `model_version`
- `feature_schema_version`
- `decision_policy_version`
- `model_type`

The initial synthetic model is a smoke-test artifact, not the final IEEE-CIS production candidate.
```

- [ ] **Step 5: Run full verification**

Run:

```bash
uv run ruff check src tests
uv run pytest
cd dashboard && npm run build && npm run test
docker compose config
```

Expected:

```text
Ruff passes
pytest passes, except integration tests are skipped unless explicitly selected
dashboard build and tests pass
compose config passes
```

- [ ] **Step 6: Commit**

```bash
git add README.md docs/deployment.md docs/model-card.md docs/runbook.md docs/demo-script.md
git commit -m "docs: add local runbook and demo workflow"
```

## Later Expansion Tasks

These are deliberately outside the first implementation slice but should be planned after the platform runs end-to-end with the synthetic model:

- Replace synthetic training with full IEEE-CIS ingestion from `data/raw`.
- Add CatBoost and LightGBM benchmark runs with MLflow tracking.
- Fit probability calibration on the dedicated calibration split.
- Replace simple conformal prediction sets with MAPIE or a validated split-conformal implementation over validation data.
- Add SHAP global and local explanation artifacts for the winning model.
- Persist Kafka consumer decisions into Postgres.
- Add delayed-label publishing and monitoring joins.
- Add Evidently reports under `reports/generated`.
- Add Playwright or browser-based visual checks for the dashboard.

## Self-Review

### Spec Coverage

- Real-time Kafka scoring: Tasks 9 and 12 create Kafka topics, replay producer, scoring consumer, and Compose services.
- Synchronous scoring API: Tasks 6 and 7 build the shared scoring engine and FastAPI endpoints.
- Offline training and model artifacts: Tasks 3, 4, and 5 create feature transforms, metrics, calibration/conformal primitives, and a smoke training artifact.
- Calibrated probabilities and conformal uncertainty: Tasks 4 and 6 add the first utilities and wire a simple prediction-set policy into scoring; later expansion replaces the simple set logic with full calibration artifacts.
- SHAP/reason codes: Task 5 creates a stable reason-code interface and deterministic fallback; later expansion adds SHAP artifacts.
- Monitoring and alerts: Task 10 adds monitoring calculations; Task 12 provisions Prometheus and Grafana.
- React operations dashboard: Task 11 builds the console with live feed, detail drawer, KPIs, and alerts.
- Docker Compose local deployment: Task 12 creates the free local stack with Kafka, Postgres, MLflow, API, Prometheus, Grafana, and dashboard.
- Required tests: Tasks 2 through 12 cover contracts, feature transforms, model smoke, API, Kafka serialization, decision policy, storage, and monitoring calculations.

### Placeholder Scan

The plan avoids placeholder language and vague test instructions. The only deferred items are listed under "Later Expansion Tasks" after the first runnable vertical slice, because they require the full IEEE-CIS dataset and benchmark evidence.

### Type Consistency

- `TransactionEvent`, `DecisionEvent`, `ReasonCode`, and `AlertEvent` are defined in Task 2 and reused consistently by scoring, API, streaming, storage, and dashboard.
- Policy labels use `approve`, `review`, `block`; conformal labels use `legit`, `fraud`.
- Model metadata uses `model_version`, `feature_schema_version`, `decision_policy_version`, and `model_type` across artifact, API, dashboard, and model-card tasks.
