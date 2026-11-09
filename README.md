# Edge Autopsy

Honest ML return forecasting with walk-forward backtesting: can a forecast of
next-week industry returns beat buy-and-hold after realistic costs, without
look-ahead, and after correcting for how many things were tried? If not, where
does the apparent edge die?

**Status: under construction (build phase P1, Data). No results yet.** This README
will lead with the result, in plain words, once the sealed hold-out is opened
(task H2). Until then there are no performance numbers here, by design.
## Pre-registration

Hypotheses, thresholds, dates and model grids were frozen on 2026-11-08, before any ML model
was trained: [`PREREGISTRATION.md`](PREREGISTRATION.md). Changes are appended as dated
amendments, never edits; `PREREGISTRATION.lock` keeps every version
(check with `uv run python tasks.py prereg`).

| Version | sha256 | What |
|---|---|---|
| v1 | `12baff06bf23602267e02ab99ca1c6a68d621e9a9af16e2dd00491a390b08d1b` | original freeze |
| amendment-1 (current) | `b6ca9ec52f425444dcb2f8bb6bda365b19bba24c9602c9d711b2d4f7fcd0b6df` | feature count 13 → 14 and exact feature definitions, before any feature matrix was built |

## Disclaimer

Educational and research project. All performance figures produced by this code are
**hypothetical** backtest results, not investment advice and not an offer of any
product. Past or simulated performance does not predict future results.

## Data

Code is MIT-licensed. Data is not included and not covered by the licence:
Kenneth R. French Data Library (cite the source) and Yahoo Finance via yfinance
(personal use only). Download scripts are provided; see `docs/DATA_CARD.md`.
