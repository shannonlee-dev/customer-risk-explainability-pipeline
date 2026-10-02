.PHONY: setup check format test smoke build run

setup:
	uv sync --frozen

check:
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .
	uv run --frozen python scripts/check.py

format:
	uv run --frozen ruff check --fix .
	uv run --frozen ruff format .

test:
	uv run --frozen pytest

smoke:
	uv run --frozen pytest -m smoke

build:
	uv build

run:
	uv run --frozen customer-risk-cluster $(ARGS)
	uv run --frozen customer-risk-explain $(ARGS)
