# data/ (gitignored)

Nothing in this folder is committed except this file.

- `raw/french/` — Ken French zips, as downloaded (never edited by hand)
- `raw/etf/` — yfinance downloads for the 9 Select Sector SPDRs + SPY (personal use only)
- `raw/MANIFEST.json` — sha256, download time, source URL, latest data date of each raw file
- `processed/` — cleaned parquet files, `panel.parquet`, `flags.csv`, `data_audit.md`

Rebuild everything with `uv run python tasks.py data`. If downloads are blocked,
see "Manual download" in `CLAUDE.md`.
