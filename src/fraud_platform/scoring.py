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
    def from_paths(cls, model_path: str | Path, policy: DecisionPolicy) -> ScoringEngine:
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
