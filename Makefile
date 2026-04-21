.PHONY: help install install-dev lint format test run run-ui clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	uv sync

install-dev: ## Install all dependencies including dev tools
	uv sync --all-extras

lint: ## Run linter and type checker
	uv run ruff check src/ tests/
	uv run mypy src/

format: ## Auto-format code
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

test: ## Run tests with coverage
	uv run pytest tests/ -v --cov=ai_researcher --cov-report=term-missing

run: ## Run the CLI chat interface
	uv run ai-researcher --mode cli

run-ui: ## Launch the Streamlit web interface
	uv run ai-researcher --mode ui

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
