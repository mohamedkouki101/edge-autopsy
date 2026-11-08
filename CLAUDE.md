# CLAUDE.md — Edge Autopsy project memory

> **Standing rule:** read this file at the start of every session (human or AI).
> Update it at the end of every session: what was done, what changed, results, next.
> Source of truth for *what* to build: `../phase2/PLAN.md` and `../phase2/docs/ROADMAP.md`
> (read-only; never edit `../phase1/` or `../phase2/`). This file records *how far we are*
> and *what we learned*.

## 1. Project summary

Edge Autopsy asks whether an ML forecast of next-week industry returns (Ken French 49
industries, daily, value-weighted) gives a strategy that beats buy-and-hold after
realistic costs, with no look-ahead, and after correcting for the number of trials.
If it doesn't, the project shows where the edge disappears (waterfall: gross → lag →
costs → borrow → Deflated Sharpe). Tradable replication on the 9 original Select
Sector SPDR ETFs. The hold-out is 2017-01-02 → latest and stays sealed until P9.

| Field | Value |
|---|---|
| Current phase | P1 (Data) complete → waiting for go-ahead to start P2 |
| Current task | none (next: B1) |
| Status | P0 done (P0.2 repo and P0.4 CI handled by Kouki, outside this build); P1 done |
| Last updated | 2026-11-07 |

### Task status (ROADMAP.md IDs)

| ID | Task | Status | Commit / note |
|---|---|---|---|
| P0.1 | Repo skeleton | done | tree checked vs PLAN §4 (47 paths) |
| P0.2 | Attach GitHub remote | handed to Kouki (manages the repo himself) | |
| P0.3 | Pin environment | done | `uv sync --frozen` + import test passed (py 3.12.14, torch 2.15.0+cpu) |
| P0.4 | CI | removed 2026-11-08 at Kouki's request | `.github/workflows/ci.yml` deleted; run `tasks.py lint` + `tasks.py test` locally before each commit |
| P0.5 | Config + seeds | done | 11 tests pass (hash stable, order-independent, field-sensitive; strict keys; seeds) |
| D1 | Download French files | done | 3 zips + manifest; CRSP vintage 202609; latest date 2026-09-30; 8 tests pass |
| D2 | Parse + document | done | VW block verified on real file (26,338 days; 67,250 NaN = raw -99.99 count); U1 [M], U2 [H] in DATA_CARD; 9 tests |
| D3 | Audit + universe rule | done | sample_start 1963-07-05 (43 available) in base.yaml; 1987-10-19 in in-sample top-10 |
| D4 | ETF download | done | 9 SPDRs first date 1998-12-22, SPY 1998-12-01, all ≤ 1999-01-04; through 2026-09-30; tests pass |
| D5 | ETF total returns | done | explicit (close+div)/prev close; Adj Close differs up to 1.3e-4 on ex-dates (U5 resolved); 5 tests |
| D6 | Build + validate panel | done | `tasks.py data` end-to-end OK: 1,353,562 rows, 19 flags (all pre-1936); 68/68 tests incl. data+network |

## 2. How to run

All commands run from the `edge-autopsy/` folder and work in Windows PowerShell,
cmd, macOS and Linux. `tasks.py` is the task runner (Windows has no `make`; a thin
`Makefile` wraps the same targets for people who have `make`).

```powershell
# one-time: install uv  (https://docs.astral.sh/uv/getting-started/installation/)
#   PowerShell: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync --frozen --extra dev            # creates .venv with Python 3.12 and the exact locked versions
uv run python tasks.py test             # fast tests (excludes @slow and @network)
uv run python tasks.py test-all         # all tests incl. slow + network
uv run python tasks.py lint             # ruff check + ruff format --check
uv run python tasks.py data             # download (if missing) + clean + validate -> data/processed/panel.parquet
uv run python tasks.py data --offline   # same, but never touches the network (uses files in data/raw/)
```

### Manual download (if the network blocks the scripts)

Save these files **unchanged** (don't unzip, don't rename) and then run
`uv run python tasks.py data --offline`. The manifest records them as "manual".

| Save as | URL |
|---|---|
| `data/raw/french/49_Industry_Portfolios_daily_CSV.zip` | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip |
| `data/raw/french/F-F_Research_Data_Factors_daily_CSV.zip` | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip |
| `data/raw/french/F-F_Momentum_Factor_daily_CSV.zip` | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip |
| `data/raw/etf/<TICKER>.csv` | For XLB XLE XLF XLI XLK XLP XLU XLV XLY SPY: in Python, `yfinance.Ticker(T).history(start="1998-12-01", end="2026-10-01", auto_adjust=False, actions=True).to_csv(f"data/raw/etf/{T}.csv")` |

Later phases add: `dev`, `leaklab`, `unseal`, `etf`, `figures`, `app` (see PLAN §4).

**Note for AI sessions running in WSL on this Windows folder:** set
`UV_PROJECT_ENVIRONMENT=$HOME/.venvs/edge-autopsy` before `uv` commands so the Linux
venv lives outside the repo and does not clash with the Windows `.venv`.

## 3. Repo map

| Path | Purpose |
|---|---|
| `CLAUDE.md` | This file: status, how-to, decisions, results, gotchas, next tasks |
| `README.md` | Public summary; result-first once P9 is done (no numbers before then) |
| `LICENSE` | MIT for code; data explicitly excluded |
| `tasks.py` / `Makefile` | Task runner (cross-platform) / thin wrapper |
| `pyproject.toml`, `uv.lock`, `.python-version` | Pinned environment (Python 3.12) |
| `configs/base.yaml` | Every global choice: dates, universe, costs, seed, horizon, lag |
| `configs/models/`, `experiments/`, `leaklab/` | Per-model grids, per-experiment overrides, leak configs (later phases) |
| `src/edge_autopsy/config.py` | Pydantic config models + `config_hash()` |
| `src/edge_autopsy/seeds.py` | `set_all_seeds()` |
| `src/edge_autopsy/data/download.py` | French zips + yfinance → `data/raw/` + `MANIFEST.json` |
| `src/edge_autopsy/data/clean.py` | Parse French CSV blocks, missing codes → NaN, % → decimal; ETF returns |
| `src/edge_autopsy/data/validate.py` | `validate_panel()` raises `DataError` |
| `src/edge_autopsy/data/calendar.py` | Trading-day helpers (rebalance dates arrive in B1) |
| `src/edge_autopsy/data/build.py` | Assemble long `panel.parquet` (PLAN §5.2) |
| other `src/edge_autopsy/*.py` | Stubs for later phases (features, labels, splits, models, backtest, stats…) |
| `tests/unit/` | Unit tests; `tests/synthetic/`, `tests/regression/` later |
| `docs/DATA_CARD.md` | Sources, licences, schema, cleaning rules, known issues |
| `docs/DECISIONS.md` | Interview-grade copy of the decisions log |
| `data/` | Gitignored. `raw/` downloads, `processed/` outputs |
| `results/`, `experiments.jsonl` | Committed derived results and trial log (empty until P2) |

## 4. Decisions log

| Date | Decision | Reason | Rejected alternative | Source |
|---|---|---|---|---|
| 2026-11-07 | `tasks.py` (pure Python) is the task runner; `Makefile` only wraps it | User is on Windows (no `make`); PLAN §4 asks for Makefile targets | Makefile only; bash scripts | User brief; PLAN §4 |
| 2026-11-07 | torch from the PyTorch CPU index on Linux/Windows (`[tool.uv.sources]`) | PLAN §4 says CPU wheel; PyPI Linux torch pulls multi-GB CUDA wheels; LSTM runs on CPU anyway (N28 determinism) | Default PyPI torch | PLAN §4, N28 |
| 2026-11-07 | Pinned pyyaml==6.0.3, matplotlib==3.11.2 | PLAN §4 lists them unpinned; every dep must be `==` | Unpinned | PyPI 2026-11-07 |
| 2026-11-07 | Experiment YAMLs use `inherits:` + deep merge; all pydantic models `extra="forbid"` | A typo (e.g. `sprad_bps`) must raise, not silently use the default | Hydra composition | PLAN §3.4 (YAML+pydantic) |
| 2026-11-07 | `config_hash` = sha256 of canonical JSON (sorted keys) of the *validated* config | Key order / YAML formatting must not change identity; defaults are included in the hash | Hashing the YAML text | PLAN §4 |
| 2026-11-07 | **Sample floor 1963-07-01 → sample_start 1963-07-05** (Kouki's choice) | Literal §5.5 rule gave 1927-05-06 (plan expected ≈1960s); D3 acceptance (1987-10-19 among extremes) failed with 1927 (rank 139); thin 1930–35 portfolios hold all 19 |ret|≥50 % cells; 16.5 y of training before 1980 | 1927-05-06 literal rule; 1970-07-03 all-49 (breaks min_train_years=10 for 1980) | D3 audit; PLAN §5.5 |
| 2026-11-07 | D3 extremes check applied to the in-sample period | Full-history top-10 is all 1930s | Full history | D3 audit |
| 2026-11-07 | `rebalance_dates` implemented in D3 (pulled forward from B1) | Needed to define sample_start | Defining start on trading days | ROADMAP D3/B1 |
| 2026-11-07 | ETF total return = (Close + Div + CapGain)/prev Close − 1; never `Adj Close` | Adj Close on ex-dates = Close/(prev − div) − 1, off by up to 1.3e-4 vs CRSP/French convention; fails the 1e-6 test | yfinance Adj Close | PLAN §5.3 rule 7; D5 test |
| 2026-11-07 | Panel keeps ff49 history from 1926 (available=False before entry) | Features need look-back before sample_start | Truncating at sample_start | PLAN §5.2/§5.4 |
| 2026-11-07 | SPY is the etf9 `mkt` column, not an asset row; ETFs use the same 253-day entry rule | Benchmark must not enter the tradable universe | SPY as a 10th asset | PLAN §5.2 |
| 2026-11-07 | `validate_panel` flags |ret|≥0.5 but raises on schema/logic errors; `build_all` raises if `sample_start` would move | Rule 6 says flag; a silent sample change would contaminate comparisons | Raise on large returns | PLAN §5.3 |
| 2026-11-07 | Ruff PD010 disabled | `pivot_table` silently averages duplicate keys; `pivot` raises | Following the lint rule | engineering |
| 2026-11-08 | Removed GitHub Actions CI (`.github/`); repo managed by Kouki | Kouki's request | Keep ci.yml (PLAN §4 / P0.4) | User instruction |
| 2026-11-07 | `.gitattributes` forces LF line endings | Repo edited from both Windows and WSL | Git autocrlf defaults | Unverified (engineering judgement) |

## 5. Results log

Real numbers only, each with date, config hash, seed, data version. **Empty: no
experiments have been run.** Data-quality facts (not results) are in `docs/DATA_CARD.md`.

Current data version: manifest hash `4632037fcda6…` (2026-11-07; CRSP vintage 202609; ETF
pulled 2026-11-07). Current base config hash: `44809c05…` (after sample_start was written).

| Date | What | Value | Config hash | Seed | Data version (manifest hash) |
|---|---|---|---|---|---|

## 6. Known issues and gotchas

- **No remote and no CI** (Kouki manages the GitHub repo). AI sessions commit locally and never
  push or create repos. With no CI, nothing checks a commit automatically: always run
  `uv run python tasks.py lint` and `uv run python tasks.py test` before committing.

- **XGBoost pulls `nvidia-nccl-cu13` on Linux** (its own dependency, ~200 MB). Harmless; not
  installed on Windows. The Linux venv is ~2.1 GB in total.
- **First `uv sync` is slow** on this machine (~8 min on 2026-11-07, mostly the torch wheel).

- **PYTHONHASHSEED** set by `set_all_seeds` only affects subprocesses started later; never let
  results depend on set/dict ordering of strings.

- **French values are percent with 2 decimals.** Daily RF is therefore quantised to 0.01 %
  (1 bp/day ≈ 2.5 %/yr steps). Fine for Sharpe over long samples; noted in DATA_CARD.
- **French history is rebuilt monthly** (CRSP revisions). Results are tied to the manifest
  hash; re-downloading can change numbers. Use `refresh=True` deliberately and check
  `MANIFEST.json["revisions"]`.
- **Missing codes matched with tolerance**, not `==` (float-parser rounding risk). pandas 3.1
  deprecated `read_csv(float_precision=...)`.
- **1930s extremes:** Paper +150 % (1932-08-11), MedEq −53.6 %/+111.8 % (1931-03-26/27). Before the sample start.
- **U2 caveat:** industry *scheme* is ex-post and revised; pre-1987 Compustat SIC may be header SIC (Unverified).
- **ETF corporate actions (yfinance):** 2:1 splits on 2025-12-05 (XLB, XLE, XLK, XLU, XLY);
  XLF "split" 1.231 on 2016-09-19 = XLRE spin-off encoded as a split. yfinance `Close` is
  split-adjusted but not dividend-adjusted. XLI has one zero-volume day.
- **Early Yahoo ETF closes look non-synchronous** (SPY vs FF mkt corr 0.92–0.98 in 1999–2008,
  0.99+ after). Offsetting next-day gaps. Matters for the ETF dev period, not the 2017+ hold-out.
- **Yahoo data is personal-use only:** `data/raw/etf/` is gitignored; ETF revisions are logged
  but old files are not archived.
- **Saturday trading before 1952:** the 253rd trading day of 1926–27 is 1927-05-05 (39
  Saturdays). Trading-day windows (252 days) are shorter than a calendar year pre-1952; only
  look-back data is affected now that the sample starts 1963-07-05.
- **etf9 panel rows start 1998-12-01** (SPY's first date); the nine SPDRs have NaN `ret`
  until 1998-12-23 and become `available` on 1999-12-23.
- **`data`-marked tests skip** when `data/raw` is empty (e.g. CI). Run `tasks.py test-all`
  locally after `tasks.py data` to exercise them.
- **French file header:** "The portfolios are constructed at the end of June" (relevant to U2).

## 7. Next tasks (in order)

1. B1 calendar + labels: `build_labels(horizon=5, lag=1)`; label at t = Π(1+r) over t+2…t+6 − 1;
   `rebalance_dates` already exists (pulled forward in D3) — add the labels hand-example test.
2. B2 backtest engine: drift, lag, turnover vs drifted weights; `test_backtest_timing`.
3. B3 cost model: 10 bps hand example; borrow over 252 days = fee.
4. B4 metrics vs closed form and QuantStats (1e-6).
5. B5/B6 baselines B0–B4 on the dev period only (≤ 2016-12-31) + portfolio construction.

## 8. Rules (non-negotiable)

1. Features use data ≤ close *t*. Fills at close *t+1*. Labels = return from close *t+1* to close *t+6*.
2. Seeds always set (`set_all_seeds`); seed lives in `configs/base.yaml`.
3. Tests are written before or alongside the code they test; run them and paste real output.
4. The hold-out (2017-01-02 → latest) is untouched until P9. Data-quality checks on
   hold-out dates are allowed; anything that measures predictability or strategy
   performance on them is not.
5. Never report a number that was not produced by running code in this repo.
6. Report nulls as nulls.
7. Never commit raw data, secrets, or `.venv`.
8. `../phase1/` and `../phase2/` are read-only references.
9. A contradiction or error in the plan → stop and ask; don't silently deviate.
