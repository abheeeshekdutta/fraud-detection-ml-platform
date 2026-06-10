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
