"""D1 acceptance: French zips land in data/raw/french, unzip, and are recorded in MANIFEST.json;
a changed re-download is archived and logged as a revision, never silently overwritten."""

import json
import zipfile

import pytest

from edge_autopsy.data.download import (
    download_french,
    download_french_all,
    inspect_french_zip,
    load_manifest,
)
from tests.unit.french_fixtures import default_ind49_zip, ff3_csv, mom_csv, zip_bytes

BASE = "https://example.test/ftp/"
FILES = {
    "ind49_daily": "49_Industry_Portfolios_daily_CSV.zip",
    "ff3_daily": "F-F_Research_Data_Factors_daily_CSV.zip",
    "mom_daily": "F-F_Momentum_Factor_daily_CSV.zip",
}


def _payloads():
    return {
        BASE + FILES["ind49_daily"]: default_ind49_zip(),
        BASE + FILES["ff3_daily"]: zip_bytes(
            "F-F_Research_Data_Factors_daily.csv",
            ff3_csv({"20260929": [0.1, 0.2, 0.3, 0.01], "20260930": [0.1, 0.2, 0.3, 0.01]}),
        ),
        BASE + FILES["mom_daily"]: zip_bytes(
            "F-F_Momentum_Factor_daily.csv", mom_csv({"20260929": 0.5, "20260930": 0.32})
        ),
    }


def _fetcher(payloads, calls=None):
    def fetch(url: str) -> bytes:
        if calls is not None:
            calls.append(url)
        return payloads[url]

    return fetch


def test_inspect_reads_vintage_and_date_range(tmp_path):
    p = tmp_path / "x.zip"
    p.write_bytes(default_ind49_zip())
    info = inspect_french_zip(p)
    assert info["crsp_vintage"] == "202609"
    assert info["first_date"] == "2026-09-28"
    assert info["latest_date"] == "2026-09-30"
    assert info["csv_name"] == "49_Industry_Portfolios_Daily.csv"


def test_download_all_writes_three_files_and_manifest(tmp_path):
    raw = tmp_path / "raw"
    paths = download_french_all(FILES, BASE, raw, fetch=_fetcher(_payloads()))
    assert len(paths) == 3
    for p in paths.values():
        assert p.parent == raw / "french"
        with zipfile.ZipFile(p) as zf:  # unzips
            assert zf.testzip() is None
    m = load_manifest(raw)
    assert set(m["files"]) == set(FILES)
    e = m["files"]["mom_daily"]
    assert len(e["sha256"]) == 64
    assert e["url"] == BASE + FILES["mom_daily"]
    assert e["latest_date"] == "2026-09-30"
    assert e["crsp_vintage"] == "202609"
    assert e["downloaded_at"].endswith("Z")
    assert m["revisions"] == []


def test_existing_file_not_refetched_unless_refresh(tmp_path):
    raw = tmp_path / "raw"
    calls: list[str] = []
    fetch = _fetcher(_payloads(), calls)
    download_french("mom_daily", FILES["mom_daily"], BASE, raw, fetch=fetch)
    download_french("mom_daily", FILES["mom_daily"], BASE, raw, fetch=fetch)
    assert len(calls) == 1
    download_french("mom_daily", FILES["mom_daily"], BASE, raw, fetch=fetch, refresh=True)
    assert len(calls) == 2
    assert load_manifest(raw)["revisions"] == []  # identical bytes: no revision


def test_changed_history_is_archived_and_logged(tmp_path):
    raw = tmp_path / "raw"
    payloads = _payloads()
    url = BASE + FILES["mom_daily"]
    download_french("mom_daily", FILES["mom_daily"], BASE, raw, fetch=_fetcher(payloads))
    old_sha = load_manifest(raw)["files"]["mom_daily"]["sha256"]
    payloads[url] = zip_bytes(  # history revised: 2026-09-29 value changes
        "F-F_Momentum_Factor_daily.csv", mom_csv({"20260929": 0.9, "20260930": 0.32})
    )
    download_french(
        "mom_daily", FILES["mom_daily"], BASE, raw, fetch=_fetcher(payloads), refresh=True
    )
    m = load_manifest(raw)
    assert len(m["revisions"]) == 1
    rev = m["revisions"][0]
    assert rev["name"] == "mom_daily" and rev["old_sha256"] == old_sha
    assert rev["new_sha256"] == m["files"]["mom_daily"]["sha256"] != old_sha
    archived = raw / rev["archived_as"]
    assert (
        archived.exists()
        and archived.read_bytes() != (raw / "french" / FILES["mom_daily"]).read_bytes()
    )


def test_offline_uses_manually_placed_file(tmp_path):
    raw = tmp_path / "raw"
    (raw / "french").mkdir(parents=True)
    (raw / "french" / FILES["mom_daily"]).write_bytes(_payloads()[BASE + FILES["mom_daily"]])

    def no_network(url):  # pragma: no cover - must not be called
        raise AssertionError("network used in offline mode")

    download_french("mom_daily", FILES["mom_daily"], BASE, raw, fetch=no_network, offline=True)
    e = load_manifest(raw)["files"]["mom_daily"]
    assert e["downloaded_at"].startswith("manual")


def test_offline_missing_file_raises_with_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="F-F_Momentum_Factor_daily_CSV.zip"):
        download_french("mom_daily", FILES["mom_daily"], BASE, tmp_path, offline=True)


def test_manifest_is_valid_json_on_disk(tmp_path):
    raw = tmp_path / "raw"
    download_french_all(FILES, BASE, raw, fetch=_fetcher(_payloads()))
    json.loads((raw / "MANIFEST.json").read_text(encoding="utf-8"))


@pytest.mark.network
def test_real_french_download(tmp_path):
    from edge_autopsy.config import load_config

    cfg = load_config("configs/base.yaml")
    paths = download_french_all(cfg.data.french_files, cfg.data.french_base_url, tmp_path)
    for name, p in paths.items():
        info = inspect_french_zip(p)
        assert info["first_date"] < "1927-01-01", name
        assert info["latest_date"] >= str(cfg.data.sample_end), name
