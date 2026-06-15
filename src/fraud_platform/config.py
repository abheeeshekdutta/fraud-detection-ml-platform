from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    kafka_bootstrap_servers: str = "localhost:9092"
    model_alerts_topic: str = "model-alerts"
    database_url: str = "postgresql+psycopg://fraud:fraud@localhost:5432/fraud"
    mlflow_tracking_uri: str = "http://localhost:5001"
    model_bundle_path: str = "artifacts/model/latest"
    decision_policy_path: str = "configs/decision_policy.yaml"
    calibrator_path: str | None = None
    replay_data_path: str = "data/processed/replay.parquet"
    replay_speed_multiplier: float = 60.0
    label_delay_seconds: float = 30.0
    monitoring_interval_seconds: float = 60.0
    monitoring_prediction_limit: int = 500
    monitoring_reference_review_rate: float = 0.10
    monitoring_review_rate_multiplier: float = 2.0
