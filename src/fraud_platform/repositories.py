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
