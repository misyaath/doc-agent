.PHONY: quality lint typecheck test format

PYTHON ?= .venv/bin/python

quality:
	$(PYTHON) quality

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy .

test:
	$(PYTHON) -m pytest

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .
