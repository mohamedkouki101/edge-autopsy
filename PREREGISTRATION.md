# Edge Autopsy: Pre-registration (v1)

Frozen on 2026-11-08, before any machine-learning model was trained or evaluated and before
any predictive or strategy measurement on hold-out dates (2017-01-02 onwards).

**How the freeze works.** `PREREGISTRATION.lock` stores the sha256 of this file
(CRLF normalised to LF). `edge_autopsy.prereg.verify()` fails if one byte changes, and the
hold-out guard (task V3) refuses to open the hold-out unless `verify()` passes. Changes after
the freeze are only possible as dated `## Amendment` sections appended at the end of this
file, each with a new sha256 added to the lock. Earlier versions stay listed; nothing is
rewritten. (Git-free replacement for the planned `git tag prereg-v1`, approved by Kouki on
2026-11-08.)

Source of every rule: `../phase2/PLAN.md` (sections cited) and `configs/base.yaml`, whose
validated hash at freeze time is `143cac943876ebe9ce16dd92ab53a715c7055ee3e82aa4f1092866e8a5e4068e`.

---

## 1. Question

Does an ML forecast of next-week industry returns (Ken French 49 industries, daily,
value-weighted) give a strategy that beats buy-and-hold after realistic costs, with no
look-ahead, and after correcting for the number of things tried? If not, where does the
apparent edge die? Tradable replication: the 9 original Select Sector SPDR ETFs.

## 2. Data and sample (PLAN §5.1–5.3, §5.5; D3 decision)

| Item | Frozen value |
|---|---|
| Universe (primary) | French 49 industry portfolios, value-weighted, daily (`ff49`) |
| Universe (replication) | XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY (`etf9`); benchmark SPY |
| Risk-free rate | Fama-French daily RF |
| Data version | `data/raw/MANIFEST.json` hash `4632037fcda6…` (CRSP vintage 202609; ETF pull 2026-11-07) |
| Availability | asset available on day *t* if its return is observed and it has ≥ 253 non-missing days up to *t* |
| Sample start | 1963-07-05 (first weekly rebalance ≥ floor 1963-07-01 with ≥ 30 industries; 43 available). Earlier data is used only as feature look-back |
| Sample end | 2026-09-30 (latest French table) |
| Development out-of-sample (dev OOS) | test years 1980-01-01 → 2016-12-31 |
| **Sealed hold-out** | **2017-01-02 → 2026-09-30**, evaluated once with the frozen configuration |
| ETF replication | same frozen pipeline, trained on ETF data 1999–2016 (expanding), evaluated 2017+ (the ETF dev period overlaps the French dev period in calendar time) |

## 3. Timing, label and target (PLAN §3.2, §5.4; label decision 2026-11-08)

- Signal at close *t* (*t* = last trading day of the week, `W-FRI`); features use data ≤ close *t*.
- Fill at close *t+1* (lag 1). Holdings at the close of day *d−1* earn the return of day *d*,
  so the position first earns day *t+2*.
- **Label** at rebalance date *t*: the compounded return from the fill at close *t+1* to the
  next fill at close *t'+1* (*t'* = next rebalance date), i.e. exactly what the strategy
  earns on the position. Normally days *t+2…t+6*; shorter in holiday weeks. The maximum
  hold is 5 trading days (code raises otherwise), so the 6-day purge covers every label.
- **Target** for all models: the label minus its cross-sectional mean on that date
  (demeaned), so models learn relative performance.

## 4. Features (PLAN §5.4; identical inputs for every model)

Per asset (10): cumulative return over 5, 21, 63, 126, 252 days; 12-1 momentum (252 days,
skipping the last 21); realised volatility 21 and 63 days; 252-day beta to the market;
63-day maximum drawdown; 21-day skewness. Market-level (3; same value for all assets on a
date): market 21-day return, market 21-day volatility, cross-sectional dispersion of
21-day returns.

Normalisation: asset features → per-date cross-sectional rank mapped to [−0.5, 0.5];
market features → z-score with an expanding mean/std computed only on past dates.

## 5. Validation design (PLAN §5.5)

- **Primary:** expanding window, minimum 10 years of training. **Comparison:** rolling
  10-year window (pre-registered pair, ablation A1).
- **Refit:** every 52 rebalances (annual). Sensitivity: every 13 (quarterly, A2).
- **Purge:** training samples whose label window overlaps the first test date are removed
  (gap = 6 trading days between the last training label end and the test start).
  **Embargo:** 1 week after each test block.
- **Nested tuning:** inside each training window, the last 3 years form 3 one-year inner
  validation folds (each purged). Hyperparameters are chosen by mean inner rank IC over the
  pre-declared grid (§7), then the model is refit on the whole training window. The outer
  test block is never passed to the tuning code.
- **Hold-out:** reachable only with `UNSEAL=I_ACCEPT` and a passing `verify()` of this
  file. The hold-out walk-forward still refits annually using only past data.
- **Robustness:** skfolio `CombinatorialPurgedCV` on the dev period, for PBO and for the
  distribution of Sharpe across paths.

## 6. Portfolio, costs and metrics (PLAN §5.6, §2.1)

- **Primary portfolio:** long-only, equal weight on the top quintile of predicted
  (demeaned) return among available assets, n = max(1, round(0.2 × n_eligible)), i.e. ≈ 10
  of 49 industries and 2 of 9 ETFs; ties broken by asset name. Weekly rebalance.
- **Costs per side** = half-spread + slippage on traded notional Σ|Δw|: base **10 bps**
  (5 + 5), grid {0, 2, 5, 10, 25, 50}. Break-even cost = cost at which net Sharpe = B0 Sharpe.
- **Long-short variant** (A7): +1/−1 gross, dollar neutral; borrow fee on short notional
  accrued daily (fee/252), base 25 bps/yr, stress 100 and 300; short proceeds earn RF.
  On ff49 it is a research variant (French portfolios can't be shorted); ETF long-short is
  the tradable version.
- **Capacity (ETF only):** maximum AUM such that each rebalance trade ≤ 1 % of 20-day
  average dollar volume; reported, not optimised.
- **Sharpe** = mean(daily excess over RF) / std(daily excess, ddof=1) × √252; reported with
  the Lo (2002) autocorrelation-adjusted annualisation and the Probabilistic Sharpe Ratio.
- **Sortino** = mean(daily excess) / RMS(min(excess, 0)) over all days × √252.
- **Annualised return** geometric; **annualised volatility** = daily std × √252.
- **Maximum drawdown** on compounded net wealth (initial wealth included), with duration.
- **Turnover** (one-sided) per rebalance = ½ Σ|w_new − w_drifted|, annualised.
- **Hit rate:** (a) share of weeks the strategy beats the benchmark; (b) sign accuracy of
  the demeaned prediction vs the demeaned realised label, pooled.
- **Forecast metrics:** weekly rank IC (Spearman of prediction vs realised label), averaged;
  t-stat Newey-West with 4 lags. OOS R² vs the historical-mean forecast (Goyal-Welch).

## 7. Models, grids and order (PLAN §2.2, §7.1)

Nothing more complex is run until the simpler rung is logged.

| Rung | Model | Pre-declared grid |
|---|---|---|
| B0 | Equal-weight buy-and-hold of all available assets, monthly rebalance | none |
| B1 | French value-weighted market (Mkt-RF + RF) | none |
| B2 | Historical mean (expanding mean of each asset; "no information" ranker) | none |
| B3 | 12-1 industry momentum rule (Moskowitz–Grinblatt) | none |
| B4 | 1-week reversal rule | none |
| B5 | Ridge regression on the full feature set | α ∈ {0.1, 1, 10, 100, 1000} |
| Primary | **XGBoost** `XGBRegressor`, `objective="reg:squarederror"`, `tree_method="hist"` | `max_depth` {2, 3, 4} × `learning_rate` {0.01, 0.05} × `min_child_weight` {50, 200}; `subsample` 0.8, `colsample_bytree` 0.8; `n_estimators` ≤ 2000 with early stopping (50 rounds) on the inner validation fold (12 configs) |
| — | **LSTM** (PyTorch, CPU), the same 13-feature vector at each of the last L rebalance dates | L ∈ {4, 12} × hidden {16, 32}; 1 layer, dropout 0.2, Adam lr 1e-3, MSE on the demeaned target, early stopping on the inner fold (patience 10), `batch_first=True`; 5 seeds, mean and range reported (4 configs) |

**Fairness rules:** same feature matrix, labels, folds, tuning budget (≤ 24 configs per
model per refit), portfolio construction and costs. The LSTM's only extra input is the
sequence of past feature vectors. Fairness ablation A9: LSTM L = 1 vs an MLP with the same
hidden size, and XGBoost on the flattened lagged feature vectors (L = 4).

## 8. Hypotheses and pass thresholds (PLAN §2.3)

Primary model = XGBoost, long-only top quintile, base cost 10 bps/side.

| ID | Hypothesis | Pass threshold | Evaluated on |
|---|---|---|---|
| H1 (forecast) | Mean weekly rank IC > 0 | Newey-West t > 2.0 in dev OOS **and** IC > 0 in hold-out | Dev OOS 1980–2016, hold-out 2017+ |
| H2 (beats momentum) | XGBoost net Sharpe > B3 net Sharpe | Paired stationary-bootstrap 95 % CI of the Sharpe difference excludes 0 | Hold-out |
| H3 (beats buy-and-hold) | Net Sharpe > B0 Sharpe | Same paired-bootstrap test **and** Deflated Sharpe ≥ 0.95 with N = all logged trials | Hold-out |
| H4 (not overfit) | PBO < 0.5 across the logged configuration set | CSCV with 16 blocks | Dev OOS |
| H5 (survives costs) | Break-even cost ≥ 10 bps/side | Point estimate | Hold-out |
| H6 (replicates) | Same sign of net active return on the 9 ETFs | Point estimate + CI reported | ETF hold-out 2017+ |

- A model **"has a signal"** only if H1 passes; it **"is a strategy"** only if H1, H3 and H5 pass.
- Harvey–Liu–Zhu's t > 3 bar is reported alongside as the stricter reading.
- Bootstrap: stationary bootstrap (`arch`), 10,000 replications, block length from
  `optimal_block_length`. SPA test (`arch`) across all models vs B0.
- **Trial count N for the Deflated Sharpe** = number of distinct configurations
  (distinct `config_hash`) logged with `phase = dev`, plus all inner tuning trials.
  Identical re-runs (same `config_hash`) are not new trials (interpretation of PLAN §7.4,
  which counts rows; see §11).

## 9. Ablations and sensitivities (PLAN §7.2, §7.3)

**Ablations** (each one experiment YAML): A1 expanding vs rolling 10 y · A2 annual vs
quarterly refit · A3 drop momentum features (does XGBoost just rediscover B3?) · A4 drop
market-level features · A5 equal vs inverse-63-day-vol weights · A6 buy/hold band on/off
(enter at rank ≤ 10, exit at rank > 15) · A7 long-only vs long-short · A8 raw returns vs
demeaned target · A9 LSTM L = 1 vs MLP vs L = 4/12 (fairness) · A10 top quintile vs top tercile.

**Sensitivities:** cost grid {0, 2, 5, 10, 25, 50} bps → break-even; execution lag
{0, 1, 2} days; borrow {0, 25, 100, 300} bps; rebalance day (Friday vs Wednesday, to rule
out day-of-week luck); hold-out start ±1 year (exploratory only, never used to choose).

## 10. What a null or negative result looks like, and how it is reported (PLAN §2.4)

- **Null:** IC t-stat < 2, or net Sharpe CI overlaps B0, or DSR < 0.95. This is the
  **expected** outcome.
- **Negative:** net Sharpe below B0 with the CI excluding 0 (likely only at high cost levels).
- **Presentation (pre-committed):** the headline is the waterfall (gross, lag 0 → lag 1 →
  after costs → after borrow → DSR probability) plus the leak-lab inflation table, not the
  Sharpe. The README's first line states the result in plain words. The synthetic
  known-answer test is what makes a null credible: it shows the pipeline can find a signal
  when one exists.
- **Never:** change the hold-out, the universe, the primary model or the cost base after
  unsealing. Any post-hoc analysis goes in a section titled "Exploratory (not pre-registered)".

## 11. Deviations from the plan's wording (each decided before this freeze)

| Plan says | Frozen choice | Why | Decided |
|---|---|---|---|
| Sample = first rebalance with ≥ 30 industries (≈ 1960s) | Floor 1963-07-01 → start 1963-07-05 | The literal rule gave 1927-05-06 and kept thin 1930s portfolios (all 19 daily moves ≥ 50 % are 1930–35) | Kouki, 2026-11-07 |
| Label = "the 5-day return starting at the fill" | Label = fill → next fill (≤ 5 days) | 17.2 % of dev weeks have < 5 trading days; the plan's other half ("the model learns exactly what the strategy earns") is kept | Kouki, 2026-11-08 |
| `git tag prereg-v1`; runs stamped with git commit | sha256 lock (this file); runs stamped with `code_hash` (sha256 of code, configs and lock file) | Kouki: the build never runs git | Kouki, 2026-11-08 |
| DSR N = number of dev rows + inner trials | Distinct dev `config_hash` + inner trials | An identical re-run (same config, same numbers) is not a new trial | judgement, disclosed here |
| ETF total return | (Close + Div + CapGain)/prev Close − 1, not Yahoo Adj Close | Adj Close differs from the CRSP convention by up to 1.3e-4 on ex-dates (PLAN §5.3 rule 7 fallback) | D5, 2026-11-07 |

## 12. What had been seen before this freeze

- **Hold-out dates:** data-quality checks only (schema, missing values, extreme-return
  flags, SPY vs FF market correlation on 1999–2016). No forecast, IC or strategy return was
  computed on any date ≥ 2017-01-02; the dev loader raises for those dates.
- **Baselines B0–B4 on dev OOS** (fixed rules, nothing tuned), logged with
  `pre_registered = false`, base 10 bps, holding-period label:

| Baseline | Net Sharpe | Gross Sharpe | Turnover/yr | IC mean | IC t (NW4) |
|---|---|---|---|---|---|
| B0 equal-weight buy-and-hold | 0.518 | 0.520 | 0.20 | — | — |
| B1 value-weighted market | 0.468 | 0.468 | — | — | — |
| B2 historical mean | 0.447 | 0.460 | 1.13 | −0.0031 | −0.91 |
| B3 12-1 momentum | 0.582 | 0.650 | 6.21 | 0.0361 | 5.38 |
| B4 1-week reversal | 0.149 | 0.554 | 39.57 | 0.0140 | 2.66 |

- No feature matrix had been built and no Ridge, XGBoost or LSTM had been fitted.

## Amendment 1 (2026-11-08): feature count and exact feature definitions

Made before any feature matrix was built and before any Ridge, XGBoost or LSTM model was
fitted (task M1). The text above is unchanged; this section takes precedence where they differ.

**Why.** §4 lists 11 per-asset features (cumulative return over 5, 21, 63, 126 and 252 days,
12-1 momentum, realised volatility over 21 and 63 days, 252-day beta, 63-day maximum drawdown,
21-day skewness) but labels them "(10)", and §7 calls the input a "13-feature vector". The
count is wrong, not the list (PLAN §5.4 has the same slip). Decision (Kouki, 2026-11-08): keep
all 11. The model input has **14 features: 11 per asset + 3 market-level**. Wherever this
document says "10" per-asset features or a "13-feature vector", read 11 and 14.

**Exact definitions** (r = daily simple return of the asset, m = market return
(Mkt-RF) + RF; every window ends at close t, includes t, and uses trading days):

| Feature | Definition |
|---|---|
| `ret_5`, `ret_21`, `ret_63`, `ret_126`, `ret_252` | compounded return over the last w days; missing unless all w returns are observed |
| `mom_12_1` | compounded return over days t−251 … t−21 (same as baseline B3) |
| `vol_21`, `vol_63` | sample standard deviation (ddof 1) of daily returns over w days |
| `beta_252` | cov(r, m) / var(m) over the last 252 days, all observed |
| `mdd_63` | maximum drawdown (≤ 0) of the compounded path over the last 63 days, starting from 1 at the window start |
| `skew_21` | bias-corrected sample skewness of daily returns over 21 days |
| `mkt_ret_21`, `mkt_vol_21` | `ret_21` and `vol_21` computed on m |
| `disp_21` | cross-sectional standard deviation (ddof 1) of `ret_21` across assets with a value on t |

**Normalisation.**
- Per-asset features: on each date, rank (ties averaged) among assets with a value, then map
  to (rank − 1)/(n − 1) − 0.5 ∈ [−0.5, 0.5]; 0 if n = 1.
- Market features: z_t = (x_t − mean of x_s) / std of x_s over past days s < t only, with s
  starting at the sample floor 1963-07-01 (consistent with excluding the thin 1926–63
  portfolios, decision D3), and missing until 252 past days exist. Decision (Kouki,
  2026-11-08). So market features are missing in the first sample year (1963-07 to 1964-07).
  This affects training data only, never a test year.

Nothing else changes: same models, grids, splits, hypotheses, metrics and presentation rules.
