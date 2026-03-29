install:
	uv sync

dev:
	uv run langgraph dev --no-browser --allow-blocking --no-reload

server:
	PYTHONPATH=. uv run uvicorn app.server.app:app --host 0.0.0.0 --port 2024

gateway:
	PYTHONPATH=. uv run uvicorn app.gateway.app:app --host 0.0.0.0 --port 8001

test:
	PYTHONPATH=. uv run pytest tests/ -v

lint:
	uvx ruff check .
	uvx ruff format --check .

format:
	uvx ruff check . --fix && uvx ruff format .
