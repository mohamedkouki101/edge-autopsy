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

## P3 (2026-11-08)

- **Git-free provenance (Kouki: no git in AI sessions).** Runs carry `code_hash` (sha256 of src/,
  configs/, pyproject.toml, uv.lock, tasks.py; CRLF→LF) instead of git commit/dirty. The
  pre-registration is frozen by a sha256 in `PREREGISTRATION.lock`, not a `prereg-v1` tag; the
  public timestamp is Kouki's push of that file.
- **V2 purge rule, exactly.** A sample dated t has label end = the trading day `purge_days`
  (= lag + horizon = 6) after t. A training sample before a test block is kept only if its
  label end is strictly before the first test date (K3); a label end past the available data
  counts as overlapping. Training samples after a block (inner folds, CPCV) are purged up to the
  last test label end and embargoed for `embargo_weeks` × 5 more trading days. The config
  refuses `purge_days < lag + horizon`.
- **Refit blocks count rebalances (52), as PLAN §5.5 says**, rather than calendar years. They
  drift about a week per year, giving 38 dev folds, the last one 7 weeks long. Every dev week is
  tested exactly once either way.
- **V3 hold-out guard, git-free.** `holdout_guard()` opens the hold-out only if UNSEAL is exactly
  `I_ACCEPT` and `PREREGISTRATION.md` still matches its lock (instead of "the `prereg-v1` tag
  exists"). This is stricter than the tag check: a tag can exist while the file has been edited
  since. The loader, the hold-out walk-forward and hold-out log rows all go through it.
- **V4 inner folds are forward-only.** PLAN §5.5 says "the last 3 years are split into 3 one-year
  inner validation folds (each purged)" without saying whether training may include dates after
  a validation year. We train each inner fold only on dates before it (like the outer
  walk-forward), so the model never learns from the future relative to what it is scored on.
  Ties in mean inner IC go to the first config in the declared grid; a config with an undefined
  IC (constant predictions) can never be chosen.
- **V5 one pipeline for every model.** `run_experiment` takes a rule model (scores from data
  <= t) or a learned model (walk-forward + nested tuning + refit) and runs both through the same
  portfolio, backtest, metrics and log code as the B5 baselines. Check: B2-B4 rerun through it
  reproduce their logged rows exactly. Learned models train on the demeaned label
  (PREREGISTRATION §3) of available assets only, and hold positions only on test dates.
- **Trial accounting logged both ways.** Each row has `n_inner_trials` (configs compared) and
  `n_inner_fits` (folds x configs x inner folds). Which one enters the Deflated Sharpe N is
  decided in P8 and disclosed; logging both means no rerun is needed.

## P4 (2026-11-08)

- **Feature count: 11 per-asset features, 14 in total (PREREGISTRATION Amendment 1).** PLAN §5.4
  and the frozen pre-registration list 11 per-asset features but call them "10" (and the input
  a "13-feature vector"). Kouki chose to keep the list as written. Alternatives were to drop
  `ret_252` (Spearman 0.955 with `mom_12_1` on dev data) or `ret_5` (close to baseline B4).
  The amendment was appended, not edited in, before any feature matrix was built.
- **Market z-scores start at the 1963-07-01 sample floor** with 252 past days required
  (Kouki), consistent with excluding the thin 1926–63 portfolios (D3). The alternative,
  starting in 1926, would let 1930s thin-portfolio dispersion set the scale.
- **Exact feature formulas are pre-registered** (Amendment 1 table): daily volatility (not
  annualised; ranks are scale-free), beta on raw daily returns against (Mkt-RF)+RF, and drawdown
  measured from the window start.
- **XGBoost final refit tree count = mean early-stopping round of the inner folds.** The plan
  asks for early stopping on the inner validation fold and then a refit on the whole training
  window, which has no validation data. The refit uses the selected config's mean best
  iteration (+1) over its 3 inner folds; it is logged per fold in `trials.json`.
- **Synthetic planted leak = the first earned day's return (`shift(-2)` with lag 1).** PLAN
  §8.2's L2 (`shift(-1)`) is the close t → t+1 return, which a lag-1 strategy never earns; on
  noise data it gives IC −0.001 (no inflation), while `shift(-2)` gives IC 0.43. Both are caught
  by the K2 check. The L2 definition for the leak lab is to be confirmed with Kouki in P6.
- **P4 dev runs are the 4 pre-registered trials, run once** (ridge and XGB, expanding and
  rolling; `tasks.py dev-models`). Determinism was checked on real data without logging extra
  trials (fold-1 XGB fit twice, bit-identical). The dev table is generated from the log, and
  each row is checked against its logged net Sharpe before it is printed.
