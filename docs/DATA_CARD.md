# DATA_CARD

Status: **final for P1 (D6, 2026-11-07)**. Data version (manifest hash) `4632037fcda6…`; CRSP vintage 202609; data through 2026-09-30. Tags: [H] read directly on the source page,
[M] inferred from the source, **Unverified** = not stated by any source read.

## 1. Sources and licences

| Dataset | File | Source | Licence / terms | Committed? |
|---|---|---|---|---|
| 49 Industry Portfolios [Daily], value-weighted block | `49_Industry_Portfolios_daily_CSV.zip` | Kenneth R. French Data Library (N15) | No terms on the page (U4). Cite the source; don't redistribute raw files | No (raw, processed); derived results only |
| Fama/French 3 Factors [Daily] (Mkt-RF, SMB, HML, RF) | `F-F_Research_Data_Factors_daily_CSV.zip` | same | same | same |
| Momentum Factor (Mom) [Daily] | `F-F_Momentum_Factor_daily_CSV.zip` | same | same | same |
| 9 Select Sector SPDRs + SPY, daily bars and actions | `data/raw/etf/<T>.parquet` | yfinance 1.8.0 (Yahoo Finance) | Library Apache-2.0; Yahoo data **personal use only** (S34) | Never: no raw prices, no per-day ETF returns; only strategy-level aggregates |
| NBER recession dates | (added in P8, regimes) | N29 | public page; cite | yes (P8) |

Downloaded 2026-11-07, CRSP vintage **202609** (first line of each file), data through
**2026-09-30**. sha256s are in `data/raw/MANIFEST.json`.

## 2. What the French data is (U1, U2)

| Question | Answer | Evidence | Tag |
|---|---|---|---|
| Value- vs equal-weighted | The daily file has two blocks: "Average Value Weighted Returns -- Daily" (used) and "Average Equal Weighted Returns -- Daily" (ignored). `clean_french_ind49` selects VW by header text and asserts 49 columns | File header; `test_vw_block_selected` | [H] |
| Dividends (U1) | Returns **include dividends**. The library lists a separate "49 Industry Portfolios [ex. Dividends]" file, and states that since the January 2025 release returns are built from CRSP CIZ files "with dividends reinvested on their ex-dates". No page says in so many words that the *daily* industry file includes dividends | data_library.html, read 2026-11-07 | [M] |
| Industry assignment (U2) | "We assign each NYSE, AMEX, and NASDAQ stock to an industry portfolio at the end of June of year t based on its four-digit SIC code at that time. We use Compustat SIC codes for the fiscal year ending in calendar year t-1. Whenever Compustat SIC codes are not available, we use CRSP SIC codes for June of year t." Assignment is therefore **point-in-time** (no classification look-ahead at the firm level) | det_49_ind_port.html; file header "The portfolios are constructed at the end of June" | [H] |
| Residual classification risk (K10) | (a) The 49-industry *scheme* was designed after the fact and has been revised (e.g. a Software industry was added; car rental moved). This mapping is applied to all history. (b) Whether pre-1987 Compustat SIC codes are historical or the current "header" SIC is not stated; if header, early assignments carry some look-ahead | changes_ind.html | **Unverified** (b); disclosed as a limitation |
| Survivorship (K7) | Portfolios contain all NYSE/AMEX/NASDAQ stocks with data at each date. In daily files produced since May 2015, "stocks are dropped from a portfolio immediately after their CRSP delist date", so firms that later fail are in the data until they delist. Whether the final delisting return is included is **not stated** | data_library.html | [H] / delisting return **Unverified** |
| Weights | Value weights; portfolios reconstituted annually at end of June ("the other research portfolios are reconstituted annually") | data_library.html | [H] |
| History changes (K9) | "We reconstruct the full history of returns each month when we update the portfolios. (Historical returns can change, for example, if CRSP revises its database.)" → every download is hashed; a changed re-download is archived and logged in `MANIFEST.json["revisions"]` | data_library.html | [H] |
| Market return | Mkt-RF is the VW return of US-incorporated CRSP firms on NYSE/AMEX/NASDAQ with share code 10/11. We use mkt = (Mkt-RF) + RF | f-f_factors.html | [H] |
| RF | Daily T-bill rate that compounds to the 1-month T-bill rate; Ibbotson until 2024-05, ICE from 2024-06 | FF3 file header | [H] |

## 3. Cleaning rules (code: `src/edge_autopsy/data/clean.py`)

1. Select the VW daily block by header text; assert 49 columns + date.
2. −99.99 and −999 → NaN (matched with tolerance 1e-9, not float `==`); then ÷100.
3. Dates `YYYYMMDD` → datetime; strictly increasing and unique, else `DataError`.
4. No forward-filling.
5. FF3 and 49-industry files must have identical trading-day calendars (checked; true for
   vintage 202609). Momentum starts 1926-11-03; earlier rows are NaN.

## 4. Facts observed in the 202609 vintage (data quality only)

- 26,338 trading days, 1926-07-01 → 2026-09-30, in all three French files (momentum: 26,237 from 1926-11-03).
- 67,250 missing cells in the VW block, all coded `-99.99` (no `-999` present); equal to the number of NaNs after cleaning.
- RF ≥ 0 on every day (min 0.0).
- **Precision:** values are percent with 2 decimals, so daily returns and RF are quantised to 0.01 % (1 bp). RF steps are ~2.5 %/yr annualised.
- **Extreme values:** the largest |daily return| are in thin 1930s portfolios (Paper +150.0 % on 1932-08-11; MedEq −53.6 % then +111.8 % on 1931-03-26/27, a reversal pair typical of a bad tick or one-stock portfolio). Flagged, not dropped (PLAN §5.3 rule 6); they fall before the modelling sample (D3).
- Sanity: correlation of the cross-industry average return with FF `mkt` = 0.967.

## 5. Universe and sample (D3)

- **Entry rule:** an industry is `available` on day *t* if its return at *t* is observed and
  it has ≥ 253 non-missing days up to and including *t* (data ≤ *t* only; tested).
- **Sample floor 1963-07-01 → sample start 1963-07-05** (first rebalance date ≥ floor with
  ≥ 30 available; 43 available that day; all 49 from 1970-07-02). Decision 2026-11-07: the
  literal PLAN §5.5 rule gave 1927-05-06, contrary to the plan's own "≈ 1960s" expectation,
  and kept thin 1930s portfolios (all 19 cells with |ret| ≥ 50 % are 1930–35). Data before the
  start is used only as feature look-back.
- Industries entering late: Soda, FabPr, Guns, Gold (1963-07-01), Softw (1965-07-01), Hlth (1969-07-01).
- In-sample missing returns: one cell (Softw, 1971-03-11).
- 10 largest in-sample |returns| include 1987-10-19 (Fun −24.1 %, Steel −24.0 %); largest is
  FabPr −30.65 % on 2022-11-09 (no reversal next day; treated as genuine). Reversal-pair screen
  (|r| > 15 % then > 80 % back) finds 3 in-sample cases: Softw 1973-05-14/15, Coal 2008-11-20/21,
  Coal 2020-03-17/18; kept (crisis volatility / thin early portfolio).
- Generated files (gitignored): `data/processed/data_audit.md`, `flags.csv`,
  `nonmissing_by_year.csv`, `ff49_available.parquet`.

## 6. ETFs (D4, D5)

| Item | Value | Tag |
|---|---|---|
| Tickers | XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY (+ SPY benchmark) | N18 [M]; first dates verified in data |
| Source | yfinance 1.8.0 `Ticker.history(start=1998-12-01, end=2026-10-01, auto_adjust=False, actions=True)` | [H] |
| Licence | yfinance Apache-2.0; Yahoo data "personal use only" (S34) → raw and per-day ETF returns never committed; only strategy-level aggregates | [H] |
| First trading day | All nine SPDRs 1998-12-22 (≤ 1999-01-04 ✓); SPY 1998-12-01 | data |
| Coverage | 6,985 days per SPDR through 2026-09-30; no NaN; calendar identical across the nine and identical to French trading days | data |
| Total return (U5) | **r_t = (Close_t + Div_t + CapGain_t)/Close_{t−1} − 1**, the CRSP convention used by French. yfinance `Adj Close` is *not* used: on ex-dates it gives Close_t/(Close_{t−1} − Div_t) − 1 (matches to < 1.5e-6), which differs from the CRSP definition by up to 1.3e-4 (XLK, 85 ex-dates) and so fails the 1e-6 test (PLAN §5.3 rule 7 fallback) | `test_etf_total_return_on_exdate` |
| Splits | yfinance `Close`, `Dividends` and `Volume` are split-adjusted. 2:1 splits 2025-12-05 (XLB, XLE, XLK, XLU, XLY) show no jump (XLK +0.73 % that day) | test + data |
| XLRE spin-off | Yahoo encodes the 2016-09-19 XLRE distribution from XLF as a 1.231 "split"; XLF return that day +0.64 % vs French Banks +0.52 %, Fin +0.42 % → value preserved. Convention disclosed; it assumes the spin-off value equals the ratio Yahoo applied | data; **Unverified** exactness |
| Capital gains | None paid by these ETFs in the sample (column all 0, still included in the formula) | data |
| Volume | `volume_usd = Close × Volume` (both split-adjusted, so the product is unaffected). XLI has one zero-volume day (1999-01-26) | data |
| Cross-checks (1999–2016 only) | corr(SPY, FF mkt) 0.977 overall: 0.92–0.98 per year in 1999–2008, 0.99+ from 2009. Big gaps come in offsetting pairs on consecutive days (e.g. 2000-12-08/11), i.e. non-synchronous closes in early Yahoo ETF data (possible cause: AMEX ETF trading until 4:15 pm, **Unverified**). XLE~Oil 0.975, XLK~tech 0.975, XLF~financials 0.966, XLV~health 0.79 | data |

## 7. Panel (`data/processed/panel.parquet`, D6)

| Column | Type | Notes |
|---|---|---|
| `date` | datetime64[ns] | trading day (French calendar; ETF calendar identical from 1998-12-22) |
| `asset` | category | 49 industry names or 9 ETF tickers (SPY is **not** an asset) |
| `universe` | category | `ff49` or `etf9` |
| `ret` | float64 | simple daily total return, decimal; NaN = missing (never filled) |
| `available` | bool | ret observed and ≥ 253 non-missing days to date (same rule for ETFs) |
| `rf` | float64 | FF daily RF for both universes |
| `mkt` | float64 | ff49: (Mkt-RF) + RF; etf9: SPY total return |
| `volume_usd` | float64 | etf9: Close × Volume; ff49: NaN |

Rows: ff49 26,338 days × 49 = 1,290,562 (from 1926-07-01, available=False before entry,
kept as feature look-back); etf9 7,000 days × 9 = 63,000 (from 1998-12-01, the first SPY
date; the 135 rows before 1998-12-22 have NaN `ret`). Total 1,353,562; 1,271,552 available.
The modelling sample starts at `sample_start` 1963-07-05 (ff49); rows before it exist only
for look-back.

`validate_panel()` raises `DataError` on: wrong columns or dtypes; unknown universe;
duplicate (universe, date, asset); dates not increasing within an asset; wrong asset set;
inf values; `available` with NaN `ret`; NaN `rf`/`mkt` on available rows; `rf` < 0;
`available` earlier than the 253-day rule; last date ≠ `sample_end`. It **flags** (doesn't
raise) |ret| ≥ 0.5: 19 cells, all ff49 in 1930–1935, all before `sample_start`.
`build_all` additionally raises if a data revision changes the computed `sample_start`.

Independent checks run on 2026-11-07: 300 random ff49 cells vs the raw CSV text (0 mismatches);
etf9 `rf` = FF RF, `mkt` = SPY total return, XLK `ret` = raw (close+div)/prev close, all exact.

## 8. All cleaning and construction rules (one place)

1. French: VW daily block by header text; 49 columns asserted (`clean_french_ind49`).
2. −99.99 / −999 → NaN (tolerance 1e-9); ÷100.
3. Dates YYYYMMDD; strictly increasing and unique.
4. No forward-filling anywhere; NaN return → not available that day.
5. Entry: available from the 253rd non-missing day (data ≤ t only).
6. |ret| ≥ 0.5 flagged in `flags.csv`, not dropped.
7. ETF total return = (Close + Div + CapGain)/prev Close − 1; `Adj Close` not used.
8. Every raw file sha256-hashed in `MANIFEST.json`; changed French re-downloads archived and
   diffed (`revision_report.md`); `sample_start` cannot move silently.
9. Sample floor 1963-07-01 → sample_start 1963-07-05; data ≤ sample_end 2026-09-30.
10. ETF tickers fixed ex ante (launched 1998-12-16 per N18; first trade 1998-12-22 in data).

## 9. Open questions from the plan (U1–U5)

| ID | Status | Resolution |
|---|---|---|
| U1 dividends / VW-EW | **Resolved [M]** | VW block selected and tested; dividends included per French's library page (inferred: "[ex. Dividends]" variant exists; CIZ "dividends reinvested on ex-dates") |
| U2 classification | **Resolved [H], with disclosed residual risk** | Point-in-time SIC assignment at end of June t (Compustat FY t−1, else CRSP). Residual: ex-post industry scheme; pre-1987 Compustat SIC type Unverified |
| U3 early missing data | **Resolved** | 9 late-starting industries; sample floor 1963-07-01 → start 1963-07-05 with 43 available; all 49 from 1970-07-02; 1 in-sample NaN |
| U4 French terms | **Disclosed** | No terms on the page; raw files never committed; cite source; only derived results shown |
| U5 ETF total returns | **Resolved** | Adj Close ≠ CRSP convention on ex-dates (up to 1.3e-4) → explicit formula, tested within 1e-6 on 3 XLK ex-dates |

## 10. Known issues (data)

- Pre-1952 Saturday trading: 252 trading days < 1 calendar year (affects look-back before 1952 only).
- RF and returns quantised to 1 bp (2-decimal percent).
- Early Yahoo ETF closes non-synchronous with the 4 pm close (1999–2008).
- XLRE spin-off encoded as a split in XLF.
- Delisting-return treatment in French daily portfolios not stated.
- French history changes monthly; results are tied to the data version.
