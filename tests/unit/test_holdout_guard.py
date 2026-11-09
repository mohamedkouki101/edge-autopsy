"""V3: the hold-out (2017-01-02 ->) opens only with UNSEAL=I_ACCEPT *and* an unchanged, frozen
pre-registration (git-free substitute for the `prereg-v1` tag, approved 2026-11-08).
Every code path that can touch hold-out dates goes through `holdout_guard()`."""

import datetime as dt

import pandas as pd
import pytest

from edge_autopsy.config import SplitCfg
from edge_autopsy.data import load as load_mod
from edge_autopsy.data.calendar import rebalance_dates
from edge_autopsy.prereg import freeze
from edge_autopsy.splits import HoldoutSealedError, holdout_guard, walk_forward
from tests.unit.test_load import _panel


@pytest.fixture
def frozen(tmp_path):
    doc, lock = tmp_path / "PREREGISTRATION.md", tmp_path / "PREREGISTRATION.lock"
    doc.write_text("# Pre-registration\n\nH1: IC t > 2.0\n", encoding="utf-8")
    freeze(doc, lock)
    return doc, lock


@pytest.fixture
def spy(monkeypatch):
    calls = []
    monkeypatch.setattr(load_mod, "holdout_guard", lambda: calls.append(1) or "sha")
    return calls


def test_holdout_guard_raises(frozen, monkeypatch):
    """V3 acceptance: no UNSEAL, or any value but the exact phrase, keeps the hold-out shut."""
    for value in (None, "", "yes", "i_accept", "I_ACCEPT "):
        if value is None:
            monkeypatch.delenv("UNSEAL", raising=False)
        else:
            monkeypatch.setenv("UNSEAL", value)
        with pytest.raises(HoldoutSealedError, match="UNSEAL=I_ACCEPT"):
            holdout_guard(*frozen)


def test_unseal_is_not_enough_if_the_preregistration_changed(frozen, monkeypatch):
    doc, lock = frozen
    monkeypatch.setenv("UNSEAL", "I_ACCEPT")
    doc.write_text(doc.read_text().replace("2.0", "1.5"), encoding="utf-8")
    with pytest.raises(HoldoutSealedError, match="pre-registration"):
        holdout_guard(doc, lock)


def test_unseal_is_not_enough_without_a_freeze(tmp_path, monkeypatch):
    doc = tmp_path / "PREREGISTRATION.md"
    doc.write_text("# draft\n", encoding="utf-8")
    monkeypatch.setenv("UNSEAL", "I_ACCEPT")
    with pytest.raises(HoldoutSealedError, match="not frozen"):
        holdout_guard(doc, tmp_path / "PREREGISTRATION.lock")


def test_guard_opens_with_unseal_and_a_verified_preregistration(frozen, monkeypatch):
    monkeypatch.setenv("UNSEAL", "I_ACCEPT")
    assert len(holdout_guard(*frozen)) == 64  # the frozen sha256


def test_sealed_error_is_a_permission_error():
    assert issubclass(HoldoutSealedError, PermissionError)
    assert load_mod.HoldoutSealedError is HoldoutSealedError  # one error type everywhere


def test_default_guard_uses_the_project_preregistration(monkeypatch):
    monkeypatch.setenv("UNSEAL", "I_ACCEPT")
    import json
    from pathlib import Path

    lock = Path(__file__).resolve().parents[2] / "PREREGISTRATION.lock"
    assert holdout_guard() == json.loads(lock.read_text())["versions"][-1]["sha256"]


# ---------------------------------------------------------------- loaders call it


def test_loader_asks_the_guard_for_holdout_dates(tmp_path, spy):
    u = load_mod.load_universe(
        _panel(tmp_path), "ff49", end="2017-01-06", holdout_start="2017-01-02"
    )
    assert spy == [1]
    assert u.returns.index.max() == pd.Timestamp("2017-01-06")


def test_loader_does_not_ask_the_guard_for_dev_dates(tmp_path, spy):
    load_mod.load_universe(_panel(tmp_path), "ff49", end="2016-12-31", holdout_start="2017-01-02")
    assert spy == []


def test_loader_stays_sealed_without_unseal(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSEAL", raising=False)
    with pytest.raises(HoldoutSealedError):
        load_mod.load_universe(
            _panel(tmp_path), "ff49", end="2017-01-06", holdout_start="2017-01-02"
        )


def _cfg() -> SplitCfg:
    return SplitCfg(
        scheme="expanding",
        dev_test_start=dt.date(2000, 1, 1),
        dev_test_end=dt.date(2002, 12, 31),
        holdout_start=dt.date(2003, 1, 2),
        holdout_end=dt.date(2004, 6, 30),
        min_train_years=3,
        rolling_years=3,
        refit_every=52,
        purge_days=6,
        embargo_weeks=1,
        inner_folds=3,
        inner_fold_years=1,
    )


def test_walk_forward_holdout_period_needs_the_guard(monkeypatch):
    cal = pd.DatetimeIndex(pd.bdate_range("1995-01-02", "2004-06-30"), name="date")
    weekly = rebalance_dates(cal)
    monkeypatch.delenv("UNSEAL", raising=False)
    with pytest.raises(HoldoutSealedError):
        walk_forward(weekly, cal, _cfg(), period="holdout")
    monkeypatch.setenv("UNSEAL", "I_ACCEPT")  # real project prereg verifies
    folds = walk_forward(weekly, cal, _cfg(), period="holdout")
    assert folds[0].test_dates[0] >= pd.Timestamp("2003-01-02")
    assert folds[-1].test_dates[-1] <= pd.Timestamp("2004-06-30")
    assert folds[0].train_dates.max() < pd.Timestamp("2003-01-02")  # trains on the past only


def test_walk_forward_dev_period_never_calls_the_guard(monkeypatch):
    import edge_autopsy.splits as splits

    cal = pd.DatetimeIndex(pd.bdate_range("1995-01-02", "2004-06-30"), name="date")
    monkeypatch.setattr(splits, "holdout_guard", lambda: pytest.fail("guard called for dev"))
    walk_forward(rebalance_dates(cal), cal, _cfg())


def test_log_run_holdout_rows_go_through_the_guard(tmp_path, monkeypatch):
    import edge_autopsy.runner as runner

    calls = []
    monkeypatch.setattr(runner, "holdout_guard", lambda: calls.append(1) or "sha")
    runner.log_run(tmp_path / "e.jsonl", {"phase": "holdout"})
    assert calls == [1]


def test_loader_guards_the_first_holdout_day_itself(tmp_path, spy):
    """end == holdout_start is already hold-out data (boundary)."""
    u = load_mod.load_universe(
        _panel(tmp_path), "ff49", end="2017-01-02", holdout_start="2017-01-02"
    )
    assert spy == [1]
    assert u.returns.index.max() == pd.Timestamp("2017-01-02")
