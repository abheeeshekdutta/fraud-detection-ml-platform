.PHONY: install test lint format train-smoke api compose-up compose-down compose-reset dashboard

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
	uv run fraud-api

compose-up:
	docker compose up --build

compose-down:
	docker compose down

compose-reset:
	docker compose down -v

dashboard:
	cd dashboard && npm install && npm run dev -- --host 0.0.0.0
