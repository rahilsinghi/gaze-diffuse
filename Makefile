.PHONY: test test-unit test-integration test-slow lint format typecheck clean setup-hpc download-data download-checkpoints

test:
	python3 -m pytest tests/ -v

test-unit:
	python3 -m pytest tests/ -m unit -v

test-integration:
	python3 -m pytest tests/ -m integration -v

test-slow:
	python3 -m pytest tests/ -m slow -v

lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	isort src/ tests/

typecheck:
	mypy src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage .pytest_cache *.egg-info
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

setup-hpc:
	bash scripts/setup_hpc.sh

download-data:
	bash scripts/download_data.sh

download-checkpoints:
	bash scripts/download_checkpoints.sh
