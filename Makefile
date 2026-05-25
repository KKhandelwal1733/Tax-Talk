.PHONY: help setup infra-up infra-down ingest serve eval test lint format clean

help:
	@echo "Available commands:"
	@echo "  make setup        Install dependencies via uv"
	@echo "  make infra-up     Start Qdrant (and Langfuse if uncommented)"
	@echo "  make infra-down   Stop infrastructure containers"
	@echo "  make ingest       Run ingestion pipeline (chunk + embed corpus)"
	@echo "  make download-corpus  Download corpus sources into data/raw"
	@echo "  make serve        Start FastAPI app on http://localhost:8000"
	@echo "  make eval         Run RAGAS eval suite against golden dataset"
	@echo "  make test         Run pytest"
	@echo "  make lint         Run ruff + mypy"
	@echo "  make format       Auto-format with ruff"
	@echo "  make clean        Remove caches and build artifacts"

setup:
	uv sync --extra dev
	@echo "✓ Run 'cp .env.example .env' and add your API keys."

infra-up:
	docker compose up -d
	@echo "✓ Qdrant: http://localhost:6333/dashboard"

infra-down:
	docker compose down

download-corpus:
	uv run python scripts/download_corpus.py

ingest:
	uv run python -m tax_talk.ingestion.run

serve:
	uv run uvicorn tax_talk.api.main:app --reload --host 0.0.0.0 --port 8000

eval:
	uv run python -m tax_talk.evals.run

test:
	uv run pytest -v

lint:
	uv run ruff check src tests
	uv run mypy src

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
