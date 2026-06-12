from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def test_compose_stack_defines_expected_services_and_ports() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())

    services = compose["services"]

    assert {
        "kafka",
        "postgres",
        "mlflow",
        "fraud-api",
        "fraud-consumer",
        "transaction-producer",
        "monitoring-worker",
        "prometheus",
        "grafana",
        "dashboard",
    }.issubset(services)
    assert "8000:8000" in services["fraud-api"]["ports"]
    assert "5173:80" in services["dashboard"]["ports"]
    assert "5001:5000" in services["mlflow"]["ports"]
    assert "5000:5000" not in services["mlflow"]["ports"]
    mlflow_command = services["mlflow"]["command"]
    assert "--serve-artifacts" in mlflow_command
    assert "--artifacts-destination /mlflow/artifacts" in mlflow_command
    assert "--default-artifact-root /mlflow/artifacts" not in mlflow_command
    assert services["fraud-api"]["environment"]["DATABASE_URL"].endswith("@postgres:5432/fraud")
    for service_name in (
        "fraud-api",
        "fraud-consumer",
        "transaction-producer",
        "monitoring-worker",
    ):
        assert services[service_name]["environment"]["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
    assert services["fraud-consumer"]["command"][0] == "fraud-consumer"


def test_postgres_init_matches_storage_tables() -> None:
    init_sql = (ROOT / "docker" / "postgres" / "init.sql").read_text()

    assert "CREATE TABLE IF NOT EXISTS predictions" in init_sql
    assert "CREATE TABLE IF NOT EXISTS alerts" in init_sql
    assert "conformal_prediction_set JSONB NOT NULL" in init_sql
    assert "metadata_json JSONB NOT NULL" in init_sql


def test_prometheus_and_grafana_provisioning_targets_fraud_api() -> None:
    prometheus = yaml.safe_load((ROOT / "monitoring" / "prometheus.yml").read_text())
    datasource = yaml.safe_load(
        (ROOT / "monitoring" / "grafana" / "provisioning" / "datasources" / "prometheus.yml")
        .read_text()
    )
    dashboard = json.loads(
        (ROOT / "monitoring" / "grafana" / "dashboards" / "fraud-platform.json").read_text()
    )

    assert prometheus["scrape_configs"][0]["static_configs"][0]["targets"] == ["fraud-api:8000"]
    assert datasource["datasources"][0]["url"] == "http://prometheus:9090"
    assert dashboard["uid"] == "fraud-platform"
    assert "fraud_api_requests_total" in dashboard["panels"][0]["targets"][0]["expr"]
