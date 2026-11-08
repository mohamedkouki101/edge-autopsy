"""Downloads into data/raw/ with a sha256 manifest (PLAN §5.1, §5.3 rule 8, task D1/D4).

Rules:
- Raw files are never edited. A re-download whose bytes differ (French history can change
  with CRSP revisions, N15) archives the old file under `raw/french/archive/` and appends a
  revision record to MANIFEST.json instead of silently overwriting.
- `offline=True` never touches the network: it registers files the user placed by hand.
- The network is injected (`fetch`) so tests run without internet.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path

import pandas as pd

Fetch = Callable[[str], bytes]

USER_AGENT = "edge-autopsy-research/0.1 (+https://github.com/mohamedkouki101/edge-autopsy)"
MANIFEST = "MANIFEST.json"
_DATE_ROW = re.compile(r"^\s*(\d{8})\s*,")
_VINTAGE = re.compile(r"(\d{6}) CRSP database")


def http_fetch(url: str, timeout: float = 120.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URLs
        return resp.read()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_manifest(raw_dir: Path) -> dict:
    p = Path(raw_dir) / MANIFEST
    if not p.exists():
        return {"files": {}, "revisions": []}
    return json.loads(p.read_text(encoding="utf-8"))


def save_manifest(raw_dir: Path, manifest: dict) -> None:
    p = Path(raw_dir) / MANIFEST
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(p)


def manifest_hash(raw_dir: Path) -> str:
    """Data version: sha256 over the (name, sha256) pairs of all current raw files."""
    files = load_manifest(raw_dir)["files"]
    canon = json.dumps({k: v["sha256"] for k, v in sorted(files.items())}, sort_keys=True)
    return sha256_bytes(canon.encode("utf-8"))


def _read_single_csv(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if len(names) != 1:
            raise ValueError(f"{path.name}: expected one file in zip, found {names}")
        return names[0], zf.read(names[0]).decode("latin-1")


def inspect_french_zip(path: Path) -> dict:
    """CSV name, CRSP vintage (YYYYMM) and first/latest data date of a French daily zip."""
    csv_name, text = _read_single_csv(Path(path))
    dates = [m.group(1) for line in text.splitlines() if (m := _DATE_ROW.match(line))]
    if not dates:
        raise ValueError(f"{Path(path).name}: no YYYYMMDD data rows found")
    vint = _VINTAGE.search(text[:500])

    def iso(d: str) -> str:
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"

    return {
        "csv_name": csv_name,
        "crsp_vintage": vint.group(1) if vint else None,
        "first_date": iso(min(dates)),
        "latest_date": iso(max(dates)),
    }


def download_french(
    name: str,
    filename: str,
    base_url: str,
    raw_dir: Path,
    *,
    fetch: Fetch = http_fetch,
    refresh: bool = False,
    offline: bool = False,
) -> Path:
    """Ensure `raw_dir/french/<filename>` exists and is recorded in MANIFEST.json."""
    raw_dir = Path(raw_dir)
    target = raw_dir / "french" / filename
    url = base_url + filename
    manifest = load_manifest(raw_dir)

    if offline or (target.exists() and not refresh):
        if not target.exists():
            raise FileNotFoundError(
                f"Offline mode and {target} is missing. Download {url} by hand and save it "
                f"there unchanged (see CLAUDE.md, 'Manual download')."
            )
        prev = manifest["files"].get(name)
        data = target.read_bytes()
        if prev is None or prev["sha256"] != sha256_bytes(data):
            # File placed (or replaced) by hand: record it, stamped with its mtime.
            mtime = dt.datetime.fromtimestamp(target.stat().st_mtime, dt.UTC)
            when = f"manual (file mtime {mtime:%Y-%m-%dT%H:%M:%SZ})"
            _record(manifest, name, target, raw_dir, url, data, when)
            save_manifest(raw_dir, manifest)
        return target

    data = fetch(url)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        old = target.read_bytes()
        if sha256_bytes(old) != sha256_bytes(data):
            stamp = _utc_now().replace(":", "").replace("-", "")
            archived = target.parent / "archive" / f"{target.stem}.{stamp}{target.suffix}"
            archived.parent.mkdir(parents=True, exist_ok=True)
            archived.write_bytes(old)
            manifest["revisions"].append(
                {
                    "name": name,
                    "detected_at": _utc_now(),
                    "old_sha256": sha256_bytes(old),
                    "new_sha256": sha256_bytes(data),
                    "archived_as": archived.relative_to(raw_dir).as_posix(),
                }
            )
    target.write_bytes(data)
    with zipfile.ZipFile(target) as zf:  # fail fast on a truncated / HTML error download
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"{filename}: corrupt member {bad}")
    _record(manifest, name, target, raw_dir, url, data, _utc_now())
    save_manifest(raw_dir, manifest)
    return target


def _record(
    manifest: dict, name: str, target: Path, raw_dir: Path, url: str, data: bytes, when: str
) -> None:
    info = inspect_french_zip(target)
    manifest["files"][name] = {
        "file": target.relative_to(raw_dir).as_posix(),
        "url": url,
        "sha256": sha256_bytes(data),
        "bytes": len(data),
        "downloaded_at": when,
        **info,
    }


def download_french_all(
    files: Mapping[str, str],
    base_url: str,
    raw_dir: Path,
    *,
    fetch: Fetch = http_fetch,
    refresh: bool = False,
    offline: bool = False,
) -> dict[str, Path]:
    return {
        name: download_french(
            name, fn, base_url, raw_dir, fetch=fetch, refresh=refresh, offline=offline
        )
        for name, fn in files.items()
    }


# --------------------------------------------------------------------------- ETFs (D4)

ETF_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
    "Capital Gains",
]
HistoryFetch = Callable[[str, str, str], pd.DataFrame]


def yf_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Raw (unadjusted for dividends) daily bars + actions from yfinance. `end` is exclusive."""
    import yfinance as yf

    h = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False, actions=True)
    if h.empty:
        raise ValueError(f"yfinance returned no rows for {ticker}")
    return h


def _normalise_etf(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)  # keep exchange-local calendar date
    df.index = idx.normalize().rename("date")
    for c in ETF_COLUMNS:
        if c not in df.columns:
            if c in ("Dividends", "Stock Splits", "Capital Gains"):
                df[c] = 0.0  # yfinance omits action columns when there are none
            else:
                raise ValueError(f"missing column {c!r}")
    df = df[ETF_COLUMNS].astype("float64").sort_index()
    if not df.index.is_unique:
        raise ValueError("duplicate dates in ETF history")
    return df


def load_etf_raw(raw_dir: Path, ticker: str) -> pd.DataFrame:
    """Read `etf/<T>.parquet` (script download) or `etf/<T>.csv` (manual placement)."""
    d = Path(raw_dir) / "etf"
    pq, csv = d / f"{ticker}.parquet", d / f"{ticker}.csv"
    if pq.exists():
        return _normalise_etf(pd.read_parquet(pq))
    if csv.exists():
        return _normalise_etf(pd.read_csv(csv, index_col=0, parse_dates=True))
    raise FileNotFoundError(
        f"{ticker}: neither {pq} nor {csv} exists. Run `tasks.py data` with network, or save "
        f"yfinance output there (see CLAUDE.md, 'Manual download')."
    )


def download_etfs(
    tickers: list[str],
    start: str,
    end: str,
    raw_dir: Path,
    *,
    fetch: HistoryFetch = yf_history,
    refresh: bool = False,
    offline: bool = False,
) -> dict[str, Path]:
    """Ensure data/raw/etf/<T>.parquet (or a manual .csv) exists for each ticker; record it."""
    raw_dir = Path(raw_dir)
    d = raw_dir / "etf"
    manifest = load_manifest(raw_dir)
    out: dict[str, Path] = {}
    for t in tickers:
        pq, csv = d / f"{t}.parquet", d / f"{t}.csv"
        existing = pq if pq.exists() else csv if csv.exists() else None
        if offline or (existing is not None and not refresh):
            df = load_etf_raw(raw_dir, t)  # raises with instructions if missing
            path = existing
            when = manifest["files"].get(f"etf_{t}", {}).get("downloaded_at") or "manual"
        else:
            df = _normalise_etf(fetch(t, start, end))
            d.mkdir(parents=True, exist_ok=True)
            df.to_parquet(pq)
            path, when = pq, _utc_now()
        assert path is not None
        data = path.read_bytes()
        prev = manifest["files"].get(f"etf_{t}")
        if prev and prev["sha256"] != sha256_bytes(data):
            manifest["revisions"].append(
                {
                    "name": f"etf_{t}",
                    "detected_at": _utc_now(),
                    "old_sha256": prev["sha256"],
                    "new_sha256": sha256_bytes(data),
                    "archived_as": None,  # Yahoo data: not archived (personal use only)
                }
            )
        manifest["files"][f"etf_{t}"] = {
            "file": path.relative_to(raw_dir).as_posix(),
            "url": f"yfinance {t} history(start={start}, end={end}, auto_adjust=False)",
            "sha256": sha256_bytes(data),
            "bytes": len(data),
            "downloaded_at": when,
            "rows": len(df),
            "first_date": f"{df.index[0]:%Y-%m-%d}",
            "latest_date": f"{df.index[-1]:%Y-%m-%d}",
        }
        out[t] = path
    save_manifest(raw_dir, manifest)
    return out


def check_etf_first_dates(raw_dir: Path, tickers: list[str], max_first: pd.Timestamp) -> dict:
    """Raise DataError unless every ticker's first trading date is <= max_first (D4)."""
    from edge_autopsy.data.errors import DataError

    firsts = {t: load_etf_raw(raw_dir, t).index[0] for t in tickers}
    late = {t: f"{d:%Y-%m-%d}" for t, d in firsts.items() if d > max_first}
    if late:
        raise DataError(f"tickers start after {max_first:%Y-%m-%d}: {late}")
    return firsts
