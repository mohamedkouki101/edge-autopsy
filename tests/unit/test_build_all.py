"""D6: orchestration guards + real end-to-end run (data marker)."""

from pathlib import Path

import pandas as pd
import pytest

from edge_autopsy.data.build import build_all, check_sample_start
from edge_autopsy.data.errors import DataError


def test_sample_start_unset_is_written(tmp_path):
    y = tmp_path / "base.yaml"
    y.write_text("data:\n  sample_start: null  # c\n")
    assert check_sample_start(pd.Timestamp("1963-07-05"), None, y) == "written"
    assert "sample_start: 1963-07-05  # c" in y.read_text()


def test_sample_start_match_is_ok(tmp_path):
    y = tmp_path / "base.yaml"
    assert check_sample_start(pd.Timestamp("1963-07-05"), pd.Timestamp("1963-07-05"), y) == "ok"


def test_sample_start_moved_by_data_revision_raises(tmp_path):
    with pytest.raises(DataError, match="sample_start"):
        check_sample_start(pd.Timestamp("1963-07-12"), pd.Timestamp("1963-07-05"), tmp_path / "x")


@pytest.mark.data
def test_make_data_end_to_end_offline():
    """`tasks.py data --offline` on the real raw files passes validation."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "data" / "raw" / "MANIFEST.json").exists():
        pytest.skip("raw data not downloaded")
    report = build_all(root, offline=True)
    assert report["validation"]["n_rows"] > 1_000_000
    assert report["sample_start"] == "1963-07-05"
