# Decisions log

Dated record of every "why X not Y". The canonical, running log lives in `CLAUDE.md` (Decisions log); entries that matter for interviews are copied here at phase exits.

## P0–P1 (2026-11-07)

- **Sample start 1963-07-05, not 1927.** PLAN §5.5's rule (first week with ≥ 30 industries)
  gave 1927-05-06. The 1930s portfolios are thin: all 19 daily moves of 50 % or more are in
  1930–35, and the 1987 crash ranks 139th by |return|. With a 1963-07-01 floor the crash ranks
  4th/5th and there are still 16.5 years of training before the first 1980 test year.
  Rejected: 1927 (noisy), 1970-07 all-49 (breaks the 10-year minimum training window).
- **ETF total return from close + dividends, not Yahoo's Adj Close.** Adj Close implies
  Close/(prev − div) − 1 on ex-dates, up to 1.3e-4 away from the CRSP convention French uses.
- **CPU-only torch; uv lock; tasks.py runner** (Windows has no make).
- **Strict configs** (`extra="forbid"`) and a key-order-independent `config_hash`.
- **Data revisions can't move the sample silently** (sha256 manifest, archived old files, diff report, sample_start guard).

## P2 (2026-11-08)

- **Own engine, every timing rule tested and mutation-tested.** A planted same-bar leak, turnover
  measured against targets instead of drifted weights, borrow charged on the fill day, cost
  charged a day late and a disabled band rule: each is caught by a specific test.
- **Metrics match QuantStats exactly** (Sharpe, Sortino, vol, MDD; diff 0.0) and the textbook formulas.
- **The hold-out is unreachable from P2 code**: the loader raises for any date ≥ 2017-01-02.
- **First numbers (dev 1980–2016, 10 bps):** see CLAUDE.md §5. Industry momentum beats equal-weight
  buy-and-hold gross (0.650 vs 0.520) and keeps a smaller lead net (0.582 vs 0.518); 1-week reversal
  dies on costs (0.554 → 0.149).
- **Label = holding period (decision 2026-11-08, Kouki).** PLAN §3.2 says the label is "the 5-day
  return starting at the fill, so the model learns exactly what the strategy earns". Those two
  halves disagree in the 17.2 % of weeks with fewer than 5 trading days (holidays). We kept the
  second half: the label at rebalance t is the compounded return from the fill at close t+1 to
  the next fill. `horizon = 5` is now the maximum hold (the code raises if exceeded), so the
  6-day purge (PLAN §5.5) is unchanged. A test checks label = backtest earnings; on real dev data
  the max difference is 0.0. The fixed 5-day label stays available (`label: fixed_horizon`)
  for daily-sampled diagnostics such as leak lab L5. Baseline IC changes slightly
  (momentum 0.0378 → 0.0361, t 5.55 → 5.38); portfolio metrics are unchanged.
