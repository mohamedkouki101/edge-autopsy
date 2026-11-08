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
