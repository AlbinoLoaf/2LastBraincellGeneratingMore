.PHONY: help install update format lint typecheck check run train evaluate notebook clean

help:
	@echo "Available commands:"
	@echo "  make install    Create the Conda environment"
	@echo "  make update     Update the Conda environment"
	@echo "  make format     Format Python code with Black"
	@echo "  make lint       Run Flake8 and Pylint"
	@echo "  make typecheck  Run mypy"
	@echo "  make check      Run formatting checks, linting, and type checking"
	@echo "  make run        Run the main script"
	@echo "  make train      Run the training script"
	@echo "  make evaluate   Run the evaluation script"
	@echo "  make notebook   Start Jupyter Notebook"
	@echo "  make clean      Remove generated Python files"

install:
	conda env create -f environment.yml
	conda run -n src_rp python -m pip install -e .

update:
	conda env update -f environment.yml --prune
	conda run -n src_rp python -m pip install -e .

format:
	black src scripts

lint:
	flake8 src scripts
	pylint src scripts

typecheck:
	mypy src scripts

check:
	black --check src scripts
	flake8 src scripts
	pylint src scripts
	mypy src scripts

run:
	python scripts/diff_main.py

train:
	python scripts/diff_trainer.py

evaluate:
	python scripts/diff_evaluator.py

notebook:
	jupyter notebook

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
