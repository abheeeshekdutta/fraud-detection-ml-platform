from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

Decision = Literal["approve", "review", "block"]
Uncertainty = Literal["low", "medium", "high"]
ReasonDirection = Literal["increases_risk", "decreases_risk"]
Severity = Literal["info", "warning", "critical"]


def _parse_aware_datetime(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("datetime string must be a valid ISO 8601 datetime") from exc

    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise ValueError("datetime string must include timezone information")
    return parsed


class TransactionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    event_id: str = Field(min_length=1)
    transaction_id: int = Field(gt=0)
    event_time: AwareDatetime
    amount: float = Field(ge=0)
    product_cd: str = Field(min_length=1)
    card_features: dict[str, Any] = Field(default_factory=dict)
    address_features: dict[str, Any] = Field(default_factory=dict)
    email_domain_features: dict[str, Any] = Field(default_factory=dict)
    identity_features: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = Field(min_length=1)

    @field_validator("event_time", mode="before")
    @classmethod
    def parse_event_time(cls, value: Any) -> Any:
        return _parse_aware_datetime(value)


class ReasonCode(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    feature: str = Field(min_length=1)
    direction: ReasonDirection
    contribution: float | None = None


class DecisionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    event_id: str = Field(min_length=1)
    transaction_id: int = Field(gt=0)
    scored_at: AwareDatetime
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

    @field_validator("scored_at", mode="before")
    @classmethod
    def parse_scored_at(cls, value: Any) -> Any:
        return _parse_aware_datetime(value)

    @field_validator("conformal_prediction_set")
    @classmethod
    def reject_duplicate_prediction_labels(
        cls, prediction_set: list[Literal["legit", "fraud"]]
    ) -> list[Literal["legit", "fraud"]]:
        if len(prediction_set) != len(set(prediction_set)):
            raise ValueError("conformal_prediction_set cannot contain duplicate labels")
        return prediction_set


class AlertEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    alert_id: str = Field(min_length=1)
    created_at: AwareDatetime
    severity: Severity
    alert_type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at", mode="before")
    @classmethod
    def parse_created_at(cls, value: Any) -> Any:
        return _parse_aware_datetime(value)
