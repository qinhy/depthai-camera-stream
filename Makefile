.PHONY: test lint check

test:
	PYTHONPATH=src pytest

lint:
	ruff check .

check: test lint
