.PHONY: install test lint format clean setup data experiments

# Installation and setup
install:
	pip install -r requirements.txt

setup:
	pip install -e .
	mkdir -p data/sample_data data/prompts results logs
	cp .env.example .env

# Data preparation
data:
	python -c "from src.data.sample_generator import create_sample_dataset; create_sample_dataset(200)"

# Testing
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=src --cov-report=html

# Code quality
lint:
	flake8 src/ tests/ experiments/
	black --check src/ tests/ experiments/

format:
	black src/ tests/ experiments/
	isort src/ tests/ experiments/

# Experiments
experiments:
	python experiments/run_experiments.py

baseline:
	python experiments/baseline_comparison.py

# Visualization
visualize:
	python src/visualization/plotting.py results/experiments/ results/figures/

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf .coverage htmlcov/

# Full pipeline
all: setup data experiments visualize