.PHONY: check format typecheck test gate clean run

check:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src

test:
	uv run pytest -q

gate: check typecheck test
	@echo "🎉 [GATE PASSED] All lints, types, and unit tests satisfied in <10s."

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__ dist build
