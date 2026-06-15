.PHONY: install test lint format train-smoke calibrate conformal explain monitor-report api compose-up compose-down compose-reset dashboard

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

calibrate:
	uv run fraud-calibrate --processed-dir data/processed --model-dir artifacts/model/latest --output-dir artifacts/calibration/latest

conformal:
	uv run fraud-conformal --processed-dir data/processed --model-dir artifacts/model/latest --output-dir artifacts/conformal/latest

explain:
	uv run fraud-explain --processed-dir data/processed --model-dir artifacts/model/latest --output-dir artifacts/explain/latest

monitor-report:
	uv run fraud-monitor-report --reference-path data/processed/validation.parquet --current-path data/processed/replay.parquet --output-path reports/generated/monitoring_report.json

api:
	uv run fraud-api

compose-up:
	docker compose up --build

compose-down:
	docker compose down

compose-reset:
	docker compose down -v

dashboard:
	cd dashboard && npm install && npm run dev -- --host 0.0.0.0
