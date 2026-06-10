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
