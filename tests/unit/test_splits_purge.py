"""V2 / K3: purged walk-forward splits (PLAN §5.5).

A sample dated t (a rebalance date) uses features from closes <= t and a label that ends,
at the latest, `purge_days` = lag + horizon trading days later (fill t+1, hold <= 5 days).
The K3 rule: for every fold, max(label end of any training sample) < min(test date).
"""

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from edge_autopsy.config import SplitCfg
from edge_autopsy.data.calendar import rebalance_dates
from edge_autopsy.splits import (
    Fold,
    embargo,
    inner_folds,
    label_end,
    purge,
    split_diagram,
    walk_forward,
)


def _cfg(**over) -> SplitCfg:
    base = dict(
        scheme="expanding",
        dev_test_start=dt.date(2000, 1, 1),
        dev_test_end=dt.date(2004, 12, 31),
        holdout_start=dt.date(2005, 1, 3),
        holdout_end=dt.date(2006, 12, 29),
        min_train_years=3,
        rolling_years=3,
        refit_every=52,
        purge_days=6,
        embargo_weeks=1,
        inner_folds=3,
        inner_fold_years=1,
    )
    return SplitCfg(**{**base, **over})


def _calendar(start="1995-01-02", end="2004-12-31", drop_every=0):
    cal = pd.bdate_range(start, end)
    if drop_every:
        cal = cal.delete(slice(None, None, drop_every))  # holidays
    return pd.DatetimeIndex(cal, name="date")


# ---------------------------------------------------------------- label_end / purge / embargo


def test_label_end_is_purge_days_trading_days_later_and_nat_past_the_end():
    cal = _calendar("2000-01-03", "2000-01-31")  # Mon 3 .. Mon 31
    ends = label_end(pd.DatetimeIndex(["2000-01-07", "2000-01-28"]), cal, 6)
    assert ends[0] == pd.Timestamp("2000-01-17")  # Fri + 6 trading days (skips a weekend)
    assert pd.isna(ends[1])  # label not observable inside the calendar


def test_label_end_rejects_dates_not_in_the_calendar():
    cal = _calendar("2000-01-03", "2000-01-31")
    with pytest.raises(ValueError, match="trading days"):
        label_end(pd.DatetimeIndex(["2000-01-08"]), cal, 6)  # a Saturday


def test_purge_drops_training_samples_whose_label_reaches_the_test_block():
    cal = _calendar("2000-01-03", "2000-03-31")
    weekly = rebalance_dates(cal)
    test = weekly[(weekly >= "2000-02-04") & (weekly <= "2000-02-25")]
    before = weekly[weekly < "2000-02-04"]
    kept, purged = purge(before, test, cal, purge_days=6)
    # Fri 28 Jan + 6 trading days = Mon 7 Feb >= first test date Fri 4 Feb -> purged.
    assert list(purged) == [pd.Timestamp("2000-01-28")]
    assert kept.max() == pd.Timestamp("2000-01-21")  # its label ends Mon 31 Jan < 4 Feb
    assert label_end(kept, cal, 6).max() < test.min()


def test_purge_also_drops_training_samples_inside_the_test_label_span():
    """Train *after* test (CPCV, inner folds): a sample dated before the last test label ends
    overlaps the test labels too."""
    cal = _calendar("2000-01-03", "2000-03-31")
    weekly = rebalance_dates(cal)
    test = weekly[(weekly >= "2000-01-07") & (weekly <= "2000-01-21")]
    after = weekly[weekly > "2000-01-21"]
    kept, purged = purge(after, test, cal, purge_days=6)
    # last test label ends Fri 21 Jan + 6 = Mon 31 Jan; Fri 28 Jan is inside it.
    assert list(purged) == [pd.Timestamp("2000-01-28")]
    assert kept.min() == pd.Timestamp("2000-02-04")


def test_embargo_drops_samples_just_after_the_test_labels_and_nothing_before():
    cal = _calendar("2000-01-03", "2000-03-31")
    weekly = rebalance_dates(cal)
    test = weekly[(weekly >= "2000-01-07") & (weekly <= "2000-01-21")]
    train = weekly[(weekly < "2000-01-07") | (weekly > "2000-01-28")]
    kept, emb = embargo(train, test, cal, purge_days=6, embargo_weeks=1)
    # test labels end Mon 31 Jan; embargo = 5 more trading days -> through Mon 7 Feb.
    assert list(emb) == [pd.Timestamp("2000-02-04")]
    assert pd.Timestamp("2000-02-11") in kept
    assert kept.union(emb).equals(train)


# ---------------------------------------------------------------- walk_forward, worked example


def test_walk_forward_expanding_worked_example():
    cal = _calendar()
    weekly = rebalance_dates(cal)
    folds = walk_forward(weekly, cal, _cfg())
    test_dates = weekly[(weekly >= "2000-01-01") & (weekly <= "2004-12-31")]
    assert all(isinstance(f, Fold) for f in folds)
    assert len(folds) == -(-len(test_dates) // 52)  # ceil
    assert pd.DatetimeIndex(pd.concat([f.test_dates.to_series() for f in folds])).equals(test_dates)
    f0 = folds[0]
    assert f0.test_dates[0] == pd.Timestamp("2000-01-07")
    assert f0.train_dates[0] == weekly[0]  # expanding: starts at the first sample
    assert list(f0.purged_dates) == [pd.Timestamp("1999-12-31")]
    assert f0.train_dates[-1] == pd.Timestamp("1999-12-24")
    assert len(f0.embargoed_dates) == 0  # train never follows test in walk-forward


def test_walk_forward_rolling_keeps_a_fixed_lookback():
    cal = _calendar()
    weekly = rebalance_dates(cal)
    folds = walk_forward(weekly, cal, _cfg(scheme="rolling", rolling_years=3))
    for f in folds:
        lo = f.test_dates[0] - pd.DateOffset(years=3)
        assert f.train_dates[0] >= lo
        assert (f.train_dates[0] - lo).days < 7  # starts at the first sample in the window
    assert folds[-1].train_dates[0] > folds[0].train_dates[0]


def test_walk_forward_refuses_too_little_training_history():
    cal = _calendar("1998-01-02")  # only 2 years before 2000
    with pytest.raises(ValueError, match="min_train_years"):
        walk_forward(rebalance_dates(cal), cal, _cfg(min_train_years=3))


def test_walk_forward_refuses_a_rolling_window_shorter_than_min_train():
    cal = _calendar()
    with pytest.raises(ValueError, match="rolling_years"):
        walk_forward(rebalance_dates(cal), cal, _cfg(scheme="rolling", rolling_years=2))


def test_walk_forward_refuses_dates_outside_the_calendar():
    cal = _calendar()
    weekly = rebalance_dates(cal).union(pd.DatetimeIndex(["2001-06-02"]))  # a Saturday
    with pytest.raises(ValueError, match="trading days"):
        walk_forward(weekly, cal, _cfg())


def test_walk_forward_never_tests_on_holdout_dates():
    """The calendar may run past the dev window (e.g. synthetic data); tests stop at dev end."""
    cal = _calendar(end="2006-12-29")
    weekly = rebalance_dates(cal)
    folds = walk_forward(weekly, cal, _cfg())
    assert max(f.test_dates.max() for f in folds) <= pd.Timestamp("2004-12-31")


def test_walk_forward_purges_samples_whose_label_is_not_observable_yet():
    """The dev loader stops the calendar at dev_test_end, so near the end a training label can
    run past the data (label_end = NaT). Unknown must count as overlapping, never as safe."""
    cal = _calendar(end="2004-12-31")  # calendar ends on the last test date
    weekly = rebalance_dates(cal)
    folds = walk_forward(weekly, cal, _cfg(refit_every=1))
    last = folds[-1]
    assert last.test_dates[-1] == pd.Timestamp("2004-12-31")
    assert pd.Timestamp("2004-12-24") in last.purged_dates  # its label would end 2005-01-03
    assert label_end(last.train_dates, cal, 6).notna().all()


def test_split_diagram_lists_every_fold():
    cal = _calendar()
    folds = walk_forward(rebalance_dates(cal), cal, _cfg())
    text = split_diagram(folds)
    lines = [ln for ln in text.splitlines() if ln.startswith("fold")]
    assert len(lines) == len(folds)
    assert "purged" in text and "#" in text and "=" in text


# ---------------------------------------------------------------- property test (K3)


@st.composite
def _setups(draw):
    start_year = draw(st.integers(1990, 1994))
    test_start_year = start_year + draw(st.integers(2, 5))
    n_test_years = draw(st.integers(1, 4))
    cal = _calendar(
        f"{start_year}-01-0{draw(st.integers(1, 9))}",
        f"{test_start_year + n_test_years + 1}-06-30",
        drop_every=draw(st.sampled_from([0, 7, 11, 23])),
    )
    freq = draw(st.sampled_from(["W-FRI", "W-WED", "M"]))
    dates = rebalance_dates(cal, freq)
    gap = test_start_year - start_year  # >= 2 full-ish years of history before the test
    min_train = draw(st.integers(1, gap - 1))
    cfg = _cfg(
        scheme=draw(st.sampled_from(["expanding", "rolling"])),
        dev_test_start=dt.date(test_start_year, draw(st.integers(1, 12)), 1),
        dev_test_end=dt.date(test_start_year + n_test_years, 12, 31),
        holdout_start=dt.date(test_start_year + n_test_years + 1, 1, 2),
        holdout_end=dt.date(test_start_year + n_test_years + 1, 6, 30),
        min_train_years=min_train,
        rolling_years=draw(st.integers(min_train, min_train + 2)),
        refit_every=draw(st.integers(1, 60)),
        purge_days=draw(st.integers(0, 12)),
    )
    return cal, dates, cfg


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_setups())
def test_splits_purge(setup):
    """K3 property: no training label window reaches a test date, in any fold, ever."""
    cal, dates, cfg = setup
    folds = walk_forward(dates, cal, cfg)
    dev = dates[
        (dates >= pd.Timestamp(cfg.dev_test_start)) & (dates <= pd.Timestamp(cfg.dev_test_end))
    ]
    assert len(folds) >= 1
    # test blocks partition the dev test dates, in order, refit_every at a time
    assert pd.DatetimeIndex(pd.concat([f.test_dates.to_series() for f in folds])).equals(dev)
    assert all(len(f.test_dates) == cfg.refit_every for f in folds[:-1])
    assert 1 <= len(folds[-1].test_dates) <= cfg.refit_every
    prev_train = None
    for f in folds:
        t0 = f.test_dates.min()
        ends = label_end(f.train_dates, cal, cfg.purge_days)
        assert ends.notna().all()
        assert ends.max() < t0  # the K3 rule
        assert f.train_dates.max() < t0
        assert f.train_dates.intersection(f.test_dates).empty
        assert f.train_dates.intersection(f.purged_dates).empty
        assert len(f.embargoed_dates) == 0
        # purge is exact: everything purged really did reach the test block
        assert (label_end(f.purged_dates, cal, cfg.purge_days).fillna(t0) >= t0).all()
        assert len(f.purged_dates) <= cfg.purge_days
        lo = t0 - pd.DateOffset(years=cfg.rolling_years)
        if cfg.scheme == "expanding":
            assert f.train_dates[0] == dates[0]
            if prev_train is not None:
                assert prev_train.isin(f.train_dates).all()  # expanding window only grows
        else:
            assert f.train_dates[0] >= lo
            assert f.train_dates.union(f.purged_dates).equals(dates[(dates >= lo) & (dates < t0)])
        prev_train = f.train_dates


# ---------------------------------------------------------------- inner folds (V4)


def test_inner_folds_are_the_last_three_years_of_train_one_year_each():
    cal = _calendar()
    weekly = rebalance_dates(cal)
    outer = walk_forward(weekly, cal, _cfg())[0]  # train ..1999-12-24, test 2000
    inner = inner_folds(outer.train_dates, cal, _cfg())
    assert len(inner) == 3
    end = outer.train_dates[-1]
    for j, f in enumerate(inner):
        lo = end - pd.DateOffset(years=3 - j)
        hi = end - pd.DateOffset(years=2 - j)
        assert f.test_dates.min() > lo and f.test_dates.max() <= hi
        assert f.test_dates.equals(
            outer.train_dates[(outer.train_dates > lo) & (outer.train_dates <= hi)]
        )
    assert inner[-1].test_dates[-1] == end
    # validation blocks tile the last 3 years exactly once
    vals = pd.DatetimeIndex(pd.concat([f.test_dates.to_series() for f in inner]))
    assert vals.equals(outer.train_dates[outer.train_dates > end - pd.DateOffset(years=3)])


def test_inner_folds_refuse_a_training_window_too_short_for_them():
    cal = _calendar("1997-01-06", "1999-12-31")
    with pytest.raises(ValueError, match="inner"):
        inner_folds(rebalance_dates(cal), cal, _cfg())  # 3 years = all validation, no train


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_setups())
def test_inner_folds_stay_inside_train_and_are_purged(setup):
    cal, dates, cfg = setup
    cfg = cfg.model_copy(update={"inner_folds": 2, "inner_fold_years": 1, "min_train_years": 3})
    try:
        folds = walk_forward(dates, cal, cfg)
    except ValueError:
        return  # not enough history for this draw
    for outer in folds:
        try:
            inner = inner_folds(outer.train_dates, cal, cfg)
        except ValueError:
            continue
        for f in inner:
            assert f.train_dates.isin(outer.train_dates).all()
            assert f.test_dates.isin(outer.train_dates).all()
            assert not f.test_dates.isin(outer.test_dates).any()
            assert label_end(f.train_dates, cal, cfg.purge_days).max() < f.test_dates.min()
            assert label_end(f.test_dates, cal, cfg.purge_days).max() < outer.test_dates.min()


# ---------------------------------------------------------------- real data (V2 acceptance)

_PANEL = Path(__file__).resolve().parents[2]


@pytest.mark.data
@pytest.mark.skipif(
    not (_PANEL / "data" / "processed" / "panel.parquet").exists(), reason="no local panel"
)
def test_tasks_splits_prints_the_real_dev_diagram(capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location("tasks", _PANEL / "tasks.py")
    tasks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tasks)
    assert tasks.main(["splits"]) == 0
    out = capsys.readouterr().out
    folds = [ln for ln in out.splitlines() if ln.startswith("fold")]
    assert len(folds) >= 37  # 1980-2016 in blocks of 52 weekly rebalances
    assert "test 1980-01-04" in folds[0] and "train 1963-07-05" in folds[0]
    assert "2016-12-30" in folds[-1]
    assert "rolling" in out and "expanding" in out
