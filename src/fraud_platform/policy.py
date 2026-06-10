from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from fraud_platform.contracts import Decision, Uncertainty


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

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
        if not math.isfinite(calibrated_probability) or not 0 <= calibrated_probability <= 1:
            raise ValueError("calibrated_probability must be finite and within [0, 1]")
        if len(prediction_set) != len(set(prediction_set)):
            raise ValueError("prediction_set cannot contain duplicate labels")

        labels = set(prediction_set)
        if unknown_labels := labels - {"legit", "fraud"}:
            raise ValueError(f"prediction_set contains invalid labels: {sorted(unknown_labels)}")
        if labels == {"legit"} and calibrated_probability <= self.config.approve_threshold:
            return PolicyDecision(decision="approve", uncertainty="low")
        if labels == {"fraud"} and calibrated_probability >= self.config.block_threshold:
            return PolicyDecision(decision="block", uncertainty="low")
        if labels != {"legit"} and labels != {"fraud"}:
            return PolicyDecision(decision="review", uncertainty="high")
        return PolicyDecision(decision="review", uncertainty="medium")


def load_policy(path: str | Path) -> DecisionPolicy:
    payload = yaml.safe_load(Path(path).read_text())
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise ValueError("policy YAML must contain a mapping")
    return DecisionPolicy(PolicyConfig(**payload))
