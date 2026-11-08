# Thin wrapper over tasks.py (the cross-platform runner). Windows users: `uv run python tasks.py <target>`.
.PHONY: setup test test-all lint data
setup:
	uv sync --frozen --extra dev
test test-all lint data:
	uv run python tasks.py $@
