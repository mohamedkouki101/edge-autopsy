"""Shared data-layer exception."""


class DataError(ValueError):
    """Raised when raw or processed data violates a documented rule (PLAN §5.3)."""
