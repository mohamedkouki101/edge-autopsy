"""One definition of every reported metric (PLAN §2.1, task B4).

Portfolio metrics take daily simple returns. Forecast metrics take long Series indexed by
(date, asset). Lo-adjusted Sharpe and PSR live in stats/sharpe.py (P5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS = 252


def _excess(r: pd.Series, rf: pd.Series) -> np.ndarray:
    return (r - rf.reindex(r.index)).to_numpy(dtype=float)


def sharpe(r: pd.Series, rf: pd.Series) -> float:
    """mean(daily excess) / std(daily excess, ddof=1) * sqrt(252)."""
    ex = _excess(r, rf)
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(PERIODS))


def sortino(r: pd.Series, rf: pd.Series) -> float:
    """mean(daily excess) / RMS(min(excess, 0)) over all days * sqrt(252).
    NaN (undefined) when there is no downside at all."""
    ex = _excess(r, rf)
    dd = np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2))
    if dd == 0.0:
        return float("nan")
    return float(ex.mean() / dd * np.sqrt(PERIODS))


def ann_return(r: pd.Series) -> float:
    """Geometric: prod(1 + r) ** (252 / T) - 1."""
    return float(np.prod(1.0 + r.to_numpy()) ** (PERIODS / len(r)) - 1.0)


def ann_vol(r: pd.Series) -> float:
    return float(r.std(ddof=1) * np.sqrt(PERIODS))


def _wealth(r: pd.Series) -> np.ndarray:
    return np.concatenate([[1.0], np.cumprod(1.0 + r.to_numpy())])  # W0 = 1 included


def max_drawdown(r: pd.Series) -> float:
    """Worst peak-to-trough loss of compounded wealth (negative number); W0 = 1 counts."""
    w = _wealth(r)
    return float((w / np.maximum.accumulate(w)).min() - 1.0)


def drawdown_duration(r: pd.Series) -> int:
    """Longest peak-to-recovery span in trading days (to the end if never recovered)."""
    w = _wealth(r)
    peak, peak_i, longest = w[0], 0, 0
    for i in range(1, len(w)):
        if w[i] >= peak:
            if i - peak_i > 1:  # was under water in between: recovered on day i
                longest = max(longest, i - peak_i)
            peak, peak_i = w[i], i
    if peak_i < len(w) - 1:
        longest = max(longest, len(w) - 1 - peak_i)
    return int(longest)


def annual_turnover(turnover: pd.Series, n_days: int) -> float:
    """Sum of one-sided turnover per rebalance, scaled to a 252-day year."""
    return float(turnover.sum() * PERIODS / n_days)


def hit_rate_vs(r: pd.Series, benchmark: pd.Series, freq: str = "W-FRI") -> float:
    """Share of weeks in which the strategy's compounded return beats the benchmark's."""
    df = pd.concat({"s": r, "b": benchmark.reindex(r.index)}, axis=1)
    wk = (1.0 + df).groupby(df.index.to_period(freq)).prod() - 1.0
    return float((wk["s"] > wk["b"]).mean())


def summarize(
    r: pd.Series,
    rf: pd.Series,
    turnover: pd.Series | None = None,
    benchmark: pd.Series | None = None,
) -> dict:
    if r.isna().any():
        raise ValueError("returns contain NaN; slice to the evaluation window first")
    out = {
        "start": f"{r.index[0]:%Y-%m-%d}",
        "end": f"{r.index[-1]:%Y-%m-%d}",
        "n_days": len(r),
        "ann_return": ann_return(r),
        "ann_vol": ann_vol(r),
        "sharpe": sharpe(r, rf),
        "sortino": sortino(r, rf),
        "mdd": max_drawdown(r),
        "mdd_duration_days": drawdown_duration(r),
        "turnover_ann": None,
        "hit_rate_vs_bench": None,
    }
    if turnover is not None:
        t = turnover.loc[r.index[0] : r.index[-1]]
        out["turnover_ann"] = annual_turnover(t, len(r))
    if benchmark is not None:
        out["hit_rate_vs_bench"] = hit_rate_vs(r, benchmark)
    return out


# ------------------------------------------------------------------ forecast metrics


def _aligned(pred: pd.Series, realized: pd.Series) -> pd.DataFrame:
    return pd.concat({"p": pred, "y": realized}, axis=1, join="inner").dropna()


def rank_ic(pred: pd.Series, realized: pd.Series, min_assets: int = 3) -> pd.Series:
    """Per-date Spearman correlation of prediction vs realised; NaN if < min_assets pairs."""
    df = pd.concat({"p": pred, "y": realized}, axis=1, join="outer")
    dates = df.index.get_level_values("date").unique().sort_values()
    df = df.dropna()

    def ic(g: pd.DataFrame) -> float:
        if len(g) < min_assets:
            return np.nan
        return float(g["p"].rank().corr(g["y"].rank()))

    out = df.groupby(level="date").apply(ic).reindex(dates)
    out.index.name = "date"
    return out.rename("rank_ic")


def ic_summary(ic: pd.Series, nw_lags: int = 4) -> dict:
    """Mean IC and its Newey-West (HAC) t-stat with `nw_lags` lags (PLAN §2.1)."""
    import statsmodels.api as sm

    x = ic.dropna().to_numpy(dtype=float)
    fit = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
    return {"ic_mean": float(x.mean()), "ic_t_nw": float(fit.tvalues[0]), "n": len(x)}


def sign_accuracy(pred: pd.Series, realized: pd.Series) -> float:
    """Pooled share of (date, asset) where the sign of the cross-sectionally demeaned
    prediction equals the sign of the demeaned realised return."""
    df = _aligned(pred, realized)
    dm = df - df.groupby(level="date").transform("mean")
    return float((np.sign(dm["p"]) == np.sign(dm["y"])).mean())


def oos_r2(pred: pd.Series, realized: pd.Series, benchmark_pred: pd.Series) -> float:
    """1 - SSE(model) / SSE(benchmark forecast), Goyal-Welch style (S33)."""
    df = pd.concat({"p": pred, "y": realized, "b": benchmark_pred}, axis=1, join="inner")
    df = df.dropna()
    return float(1.0 - ((df.y - df.p) ** 2).sum() / ((df.y - df.b) ** 2).sum())
