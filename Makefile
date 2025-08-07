.PHONY: help install install-dev test format lint clean run-example

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies with uv"
	@echo "  make install-dev   - Install with development dependencies"
	@echo "  make test          - Run tests"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Lint code with ruff"
	@echo "  make clean         - Clean cache and temporary files"
	@echo "  make run-example   - Run basic inference example"

install:
	uv pip install -r requirements.txt

install-dev:
	uv pip install -e ".[dev]"

test:
	pytest tests/ -v

format:
	black src/ tests/ examples/ scripts/
	ruff check src/ tests/ examples/ scripts/ --fix

lint:
	ruff check src/ tests/ examples/ scripts/
	mypy src/ --ignore-missing-imports

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

run-example:
	python examples/basic_inference.py
