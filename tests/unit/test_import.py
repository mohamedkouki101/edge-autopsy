"""Smoke test: the package imports and exposes the version from pyproject."""

import importlib.metadata

import edge_autopsy


def test_package_version_matches_metadata():
    assert edge_autopsy.__version__ == importlib.metadata.version("edge-autopsy")
