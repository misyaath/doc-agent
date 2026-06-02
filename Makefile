.PHONY: quality lint typecheck format

PYTHON ?= .venv/bin/python

quality: lint typecheck

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .
