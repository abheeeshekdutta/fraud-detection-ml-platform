from __future__ import annotations

from datetime import datetime
from typing import Any

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
    reason_codes: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    latency_ms: Mapped[float] = mapped_column(Float)


class AlertRecord(Base):
    __tablename__ = "alerts"

    alert_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    severity: Mapped[str] = mapped_column(String, index=True)
    alert_type: Mapped[str] = mapped_column(String, index=True)
    message: Mapped[str] = mapped_column(String)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


def create_session_factory(database_url: str):
    engine = create_engine(database_url, future=True)
    return sessionmaker(bind=engine, expire_on_commit=False)


def create_tables(session_factory) -> None:
    Base.metadata.create_all(session_factory.kw["bind"])
