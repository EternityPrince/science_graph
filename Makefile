.PHONY: test test-llm test-all

test:
	uv run pytest -m "not llm"

test-llm:
	uv run pytest -m llm

test-all:
	uv run pytest
